// Aerodynamic force and torque of the vertical foam plate under the whoop: the equations of docs/plate_model.md, term
// by term, so that the simulator and the identification (sysid/) share one model. Everything is in the body frame of
// L2F (x forward, y left, z up) except the inputs velocity and wind, which are world-frame like the L2F state.
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
        T chord_x;      // horizontal edge c_x [m] (along body x)
        T height_z;     // vertical edge c_z [m]
        T normal[3];    // unit normal n, body frame; the plate contains body x (checked by tools/generate_params.py)
        T centroid[3];  // r_c relative to the centre of mass, body frame [m]
        T CN90;         // normal-force coefficient at 90 deg (plateau of the saturating model)
        T Kp_x;         // slope dC_N/dalpha at small alpha for the in-plane wind along x (Helmbold-type prior)
        T Kp_z;         // same for the in-plane wind along z
        T CT;           // in-plane friction coefficient
        T e0;           // centre-of-pressure shift toward the leading edge at alpha = 0, in chords
        T air_density;  // rho [kg/m^3]
        T wind[3];      // world frame [m/s]; zero indoors [assumption]
    };

    template <typename T>
    struct PlateAerodynamics{
        T force[3];               // body frame [N]
        T torque[3];              // body frame, about the centre of mass [N m]
        T center_of_pressure[3];  // r_cp, body frame [m]
        T alpha;                  // angle between the plate and the relative wind [rad], 0 ... pi/2
        T airspeed;               // V = sqrt(|v|^2 + eps) [m/s]
    };

    // eps of docs/plate_model.md: keeps the square roots and the direction v_t / V_t defined at zero airspeed
    template <typename T>
    constexpr T PLATE_EPSILON = (T)1e-6;

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T plate_dot(const T a[3], const T b[3]){
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void plate_aerodynamics(DEVICE& device, const Plate<T>& plate, const T orientation[4], const T linear_velocity_world[3], const T angular_velocity_body[3], PlateAerodynamics<T>& out){
        namespace math = rl_tools::math;
        namespace vo = rl_tools::utils::vector_operations;
        using TI = typename DEVICE::index_t;
        const auto& m = device.math;
        constexpr T eps = PLATE_EPSILON<T>;

        // velocity of the plate relative to the air in the body frame: v = R^T (v_world - wind) + omega x r_c.
        // R^T is the rotation by the conjugate quaternion, the same operation L2F uses to rotate the thrust
        // (the downwash term k_w v_i z of the model is zero for now; generate_params.py refuses k_w != 0)
        T q_conjugate[4] = {orientation[0], -orientation[1], -orientation[2], -orientation[3]};
        T relative_world[3];
        for(TI i = 0; i < 3; i++){
            relative_world[i] = linear_velocity_world[i] - plate.wind[i];
        }
        T v[3];
        rl_tools::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conjugate, relative_world, v);
        vo::cross_product_accumulate<DEVICE, T>(angular_velocity_body, plate.centroid, v);

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

        // in-plane wind direction: c^2 = 1 for wind along x, 0 along z
        T c2 = v_t[0] * v_t[0] / (V_t * V_t);
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
}
