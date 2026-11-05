// Aerodynamic force and torque of the foam plate under the whoop: the equations of docs/plate_model.md, term by term,
// so that the simulator and the identification (sysid/) share one model. Everything is in the body frame of L2F
// (x forward, y left, z up) except the inputs velocity and wind, which are world-frame like the L2F state.
//
// The plate is a flat rectangle of any orientation in the body: its unit normal n and an in-plane unit vector e_1
// (chord_direction) along its edge of length c_1 (chord_x); the other edge, of length c_2 (height_z), is along
// e_2 = e_1 x n. The nominal plate is vertical: n = y, e_1 = x, e_2 = z. A plate tilted by theta about body x has
// n = (0, cos theta, sin theta), e_1 = x, e_2 = (0, -sin theta, cos theta); theta = 90 deg is horizontal (n = z).
//
// Only RLtools math functions are used (no std::), so that the same code can run in the CUDA path of L2F later.
#pragma once

#include <rl_tools/rl/environments/l2f/quaternion_helper.h>
#include <rl_tools/utils/generic/vector_operations.h>

namespace whoop{
    template <typename T>
    struct Plate{
        // area S [m^2]; S = 0 means no plate, and the dynamics are then exactly those of L2F
        T area;
        T chord_x;      // edge c_1 [m], along chord_direction (body x for every plate here)
        T height_z;     // edge c_2 [m], along chord_direction x normal (body z for the vertical plate)
        T normal[3];    // unit normal n, body frame
        T chord_direction[3];  // unit e_1 in the plate plane (n . e_1 = 0, checked by tools/generate_params.py)
        T centroid[3];  // r_c relative to the centre of mass, body frame [m]
        T CN90;         // normal-force coefficient at 90 deg (plateau of the saturating model)
        T Kp_x;         // slope dC_N/dalpha at small alpha for the in-plane wind along e_1 (Helmbold-type prior)
        T Kp_z;         // same for the in-plane wind along e_2
        T CT;           // in-plane friction coefficient
        T e0;           // centre-of-pressure shift toward the leading edge at alpha = 0, in chords
        T air_density;  // rho [kg/m^3]
        T wind[3];      // world frame [m/s]; zero indoors [assumption]
        // crude propeller downwash on the plate (docs/plate_model.md): k_w times the momentum-theory induced velocity
        // of the rotors, along body +z in the plate's velocity relative to the air. k_w = 0 (nominal): no downwash,
        // and the code path is the one without it [assumption: the plate under the rotor discs, all of it in the wake]
        T downwash_factor;   // k_w
        T rotor_disc_area;   // A, total area of the rotor discs [m^2]; only used when k_w > 0
    };

    template <typename T>
    struct PlateAerodynamics{
        T force[3];               // body frame [N]
        T torque[3];              // body frame, about the centre of mass [N m]
        T center_of_pressure[3];  // r_cp, body frame [m]
        T alpha;                  // angle between the plate and the relative wind [rad], 0 ... pi/2
        T airspeed;               // V = sqrt(|v|^2 + eps) [m/s]
        T downwash;               // k_w v_i added along body z [m/s] (0 without downwash)
    };

    // eps of docs/plate_model.md: keeps the square roots and the direction v_t / V_t defined at zero airspeed
    template <typename T>
    constexpr T PLATE_EPSILON = (T)1e-6;

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T plate_dot(const T a[3], const T b[3]){
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    // induced velocity of the rotors from momentum theory with Glauert's forward-flight correction, neglecting the
    // axial flow through the discs: v_h^2 = T / (2 rho A), v_i^2 (v_i^2 + V_par^2) = v_h^4, V_par the speed in the
    // disc plane. Written in the cancellation-free form v_i^2 = 2 v_h^4 / (V_par^2 + sqrt(V_par^4 + 4 v_h^4))
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T rotor_induced_velocity(DEVICE& device, T thrust, T air_density, T disc_area, T in_plane_speed_squared){
        namespace math = rl_tools::math;
        if(!(thrust > 0) || !(disc_area > 0)){
            return 0;
        }
        T vh2 = thrust / (2 * air_density * disc_area);
        T vi2 = 2 * vh2 * vh2 / (in_plane_speed_squared + math::sqrt(device.math, in_plane_speed_squared * in_plane_speed_squared + 4 * vh2 * vh2));
        return math::sqrt(device.math, vi2);
    }

    // rotor_thrust: the total rotor thrust along body z [N], only used for the downwash (k_w > 0)
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void plate_aerodynamics(DEVICE& device, const Plate<T>& plate, const T orientation[4], const T linear_velocity_world[3], const T angular_velocity_body[3], T rotor_thrust, PlateAerodynamics<T>& out){
        namespace math = rl_tools::math;
        namespace vo = rl_tools::utils::vector_operations;
        using TI = typename DEVICE::index_t;
        const auto& m = device.math;
        constexpr T eps = PLATE_EPSILON<T>;

        // velocity of the plate relative to the air in the body frame: v = R^T (v_world - wind) + omega x r_c.
        // R^T is the rotation by the conjugate quaternion, the same operation L2F uses to rotate the thrust
        T q_conjugate[4] = {orientation[0], -orientation[1], -orientation[2], -orientation[3]};
        T relative_world[3];
        for(TI i = 0; i < 3; i++){
            relative_world[i] = linear_velocity_world[i] - plate.wind[i];
        }
        T v[3];
        rl_tools::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conjugate, relative_world, v);
        out.downwash = 0;
        if(plate.downwash_factor > 0){
            // the wake moves down (body -z) at v_i, so the plate moves up relative to it: + k_w v_i z. V_par from the
            // centre of mass's velocity in the disc plane (body x, y), before the rotation term
            T v_par2 = v[0] * v[0] + v[1] * v[1];
            out.downwash = plate.downwash_factor * rotor_induced_velocity(device, rotor_thrust, plate.air_density, plate.rotor_disc_area, v_par2);
        }
        vo::cross_product_accumulate<DEVICE, T>(angular_velocity_body, plate.centroid, v);
        if(plate.downwash_factor > 0){
            v[2] += out.downwash;
        }

        // normal and in-plane parts
        T v_n = plate_dot<T>(plate.normal, v);
        T v_t[3];
        for(TI i = 0; i < 3; i++){
            v_t[i] = v[i] - v_n * plate.normal[i];
        }
        T V2 = plate_dot<T>(v, v) + eps;
        T V = math::sqrt(m, V2);
        T V_t = math::sqrt(m, plate_dot<T>(v_t, v_t) + eps);
        T alpha = math::atan2(m, math::abs(m, v_n), V_t);

        // in-plane wind direction: c^2 = 1 for wind along e_1, 0 along e_2 (for e_1 = x this is v_t,x^2 / V_t^2 exactly:
        // the products with the zero components of e_1 are exact zeros)
        T v_t1 = plate_dot<T>(v_t, plate.chord_direction);
        T c2 = v_t1 * v_t1 / (V_t * V_t);
        T K_p = plate.Kp_x * c2 + plate.Kp_z * (1 - c2);
        // Model B of research/flat_plate_aero.md: slope K_p at small alpha, saturating at C_N90
        T C_N = plate.CN90 * math::tanh(m, K_p * alpha / plate.CN90);

        T q_S = (T)0.5 * plate.air_density * plate.area;
        T sign_v_n = (v_n > 0) ? (T)1 : ((v_n < 0) ? (T)-1 : (T)0);
        for(TI i = 0; i < 3; i++){
            out.force[i] = -q_S * C_N * V2 * sign_v_n * plate.normal[i] - q_S * plate.CT * V * v_t[i];
        }

        // centre of pressure toward the leading edge, which is on the +v_t side (the plate moves into the air there);
        // the shift vanishes at alpha = 90 deg (cos alpha = 0) and when there is no in-plane wind (v_t = 0)
        T chord = plate.chord_x * c2 + plate.height_z * (1 - c2);
        T shift = plate.e0 * math::cos(m, alpha) * chord / V_t;
        for(TI i = 0; i < 3; i++){
            out.center_of_pressure[i] = plate.centroid[i] + shift * v_t[i];
        }
        vo::cross_product<DEVICE, T>(out.center_of_pressure, out.force, out.torque);
        out.alpha = alpha;
        out.airspeed = V;
    }

    // without the rotor thrust: no downwash term (k_w must be 0 for this to be the model)
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void plate_aerodynamics(DEVICE& device, const Plate<T>& plate, const T orientation[4], const T linear_velocity_world[3], const T angular_velocity_body[3], PlateAerodynamics<T>& out){
        plate_aerodynamics(device, plate, orientation, linear_velocity_world, angular_velocity_body, (T)0, out);
    }
}
