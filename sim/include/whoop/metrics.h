// What the evaluation measures of a flight state, to see whether the plate carries part of the weight:
//   plate_lift_to_weight    the world-z component of the plate force divided by the weight m g (> 0: the plate pushes up)
//   plate_force_to_weight   |F_plate| / m g, the magnitude (mostly side force and drag when the plate is not banked)
//   rotor_thrust_to_weight  the sum of the four rotor thrusts, from the rotor states (the lagged motor states that the
//                           dynamics use, not the commands), divided by m g; 1 in hover
// m is the mass of the vehicle being flown (with the plate when it is on), so a whoop that flies with the plate
// carrying 10 % of its weight, level, shows a rotor thrust of 0.9.
#pragma once

#include <whoop/plate.h>

#include <cmath>
#include <type_traits>

namespace whoop{
    template <typename T>
    struct FlightMetrics{
        T plate_lift_to_weight;
        T plate_force_to_weight;
        T rotor_thrust_to_weight;
        PlateAerodynamics<T> aero;  // the plate force (body frame), angle of attack and airspeed; zero without the plate
    };

    template <typename DEVICE, typename PARAMETERS, typename STATE>
    auto flight_metrics(DEVICE& device, const PARAMETERS& p, const STATE& s){
        using T = std::decay_t<decltype(p.dynamics.mass)>;
        const auto& g = p.dynamics.gravity;
        T weight = p.dynamics.mass * (T)std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        FlightMetrics<T> m{};
        T thrust = 0;
        for(int i = 0; i < 4; i++){
            const auto& c = p.dynamics.rotor_thrust_coefficients[i];
            // the thrust along each rotor's axis; for the whoop all four point along body z
            thrust += c[0] + c[1] * s.rpm[i] + c[2] * s.rpm[i] * s.rpm[i];
        }
        m.rotor_thrust_to_weight = thrust / weight;
        if(p.plate.area > 0){
            // the thrust is also what the downwash term needs (k_w > 0): all four rotors point along body z
            plate_aerodynamics(device, p.plate, s.orientation, s.linear_velocity, s.angular_velocity, thrust, m.aero);
            T f_world[3];
            rl_tools::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(s.orientation, m.aero.force, f_world);
            m.plate_lift_to_weight = f_world[2] / weight;
            m.plate_force_to_weight = (T)std::sqrt(m.aero.force[0] * m.aero.force[0] + m.aero.force[1] * m.aero.force[1] + m.aero.force[2] * m.aero.force[2]) / weight;
        }
        return m;
    }
}
