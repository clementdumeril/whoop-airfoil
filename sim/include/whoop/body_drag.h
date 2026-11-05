// Aerodynamic drag of the whoop's own body (frame, ducts, battery), without the plate: the model of the identification
// (sysid/model.py, body_drag_force), linear in the velocity relative to the air and acting at the centre of mass, so it
// gives no torque:
//   F = -diag(d) v_rel,   v_rel = R^T (v_world - wind)   (body frame)
// d comes from params/whoop_nominal.json ("body_drag", kg/s). At 3 m/s along x the nominal d_x = 0.008 kg/s gives
// 0.024 N, 8 % of the weight.
#pragma once

#include <rl_tools/rl/environments/l2f/quaternion_helper.h>

namespace whoop{
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void body_drag_force(DEVICE& device, const T drag[3], const T orientation[4], const T linear_velocity_world[3], const T wind[3], T force_body[3]){
        using TI = typename DEVICE::index_t;
        // R^T is the rotation by the conjugate quaternion, as for the plate (whoop/plate.h)
        T q_conjugate[4] = {orientation[0], -orientation[1], -orientation[2], -orientation[3]};
        T relative_world[3];
        for(TI i = 0; i < 3; i++){
            relative_world[i] = linear_velocity_world[i] - wind[i];
        }
        T v[3];
        rl_tools::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conjugate, relative_world, v);
        for(TI i = 0; i < 3; i++){
            force_body[i] = -drag[i] * v[i];
        }
    }
}
