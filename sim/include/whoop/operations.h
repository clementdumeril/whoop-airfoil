// The L2F operations that change for the whoop, as overloads on ParametersWhoop (whoop/environment.h).
//
// How this works without touching RLtools: L2F calls its operations unqualified (multirotor_dynamics inside the
// integrator, sample_initial_state / reward / log_reward in the off-policy runner), so argument-dependent lookup also
// finds functions in the namespace of the parameter type, here whoop. Each overload below takes ParametersWhoop<...>&
// where L2F takes any PARAMETERS&, so C++ prefers it (more specialized), and it calls back into L2F's own code for
// everything it does not change. The overloads are:
//   multirotor_dynamics  L2F's rigid-body and rotor equations, then + the plate force and torque when S > 0
//   sample_initial_parameters  L2F's, then the plate and task parameters, and the velocity reference of the episode
//   sample_initial_state  L2F's (rotors, action history, random force), then the rigid-body initial state of the L2F
//                         paper: the current RLtools samples the tilt uniformly in [0, 1] rad whatever max_angle says
//   reward, log_reward  the reward of the L2F paper (Reward in environment.h)
// Include this after <rl_tools/operations/cpu_mux.h> (or another device), as any RLtools environment.
#pragma once

#include <whoop/environment.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <type_traits>

namespace whoop{
    namespace rlt = rl_tools;

    template <typename DEVICE, typename STATE_SPEC, typename WHOOP_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const ParametersWhoop<WHOOP_SPEC>& params, const l2f::StateBase<STATE_SPEC>& state, const T* action, l2f::StateBase<STATE_SPEC>& state_change){
        // L2F's equations with the L2F part of the parameters (exactly the same code and numbers as without the plate)
        using L2F_PARAMETERS = typename WHOOP_SPEC::NEXT_COMPONENT;
        l2f::multirotor_dynamics(device, static_cast<const L2F_PARAMETERS&>(params), state, action, state_change);
        if(params.plate.area > 0){
            // L2F sums the rotor forces and torques in the body frame, then a = R F / m + g and
            // domega/dt = J^-1 (tau - omega x J omega). Both are linear in F and tau, so adding R F_plate / m and
            // J^-1 tau_plate afterwards is the same as adding the plate to L2F's sums (up to rounding)
            PlateAerodynamics<T> aero;
            plate_aerodynamics(device, params.plate, state.orientation, state.linear_velocity, state.angular_velocity, aero);
            T force_world[3];
            l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, aero.force, force_world);
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                state_change.linear_velocity[i] += force_world[i] / params.dynamics.mass;
            }
            T angular_acceleration[3];
            rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, aero.torque, angular_acceleration);
            rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(angular_acceleration, state_change.angular_velocity);
        }
    }

    template <typename DEVICE, typename WHOOP_SPEC, typename RNG>
    void sample_velocity_reference(DEVICE& device, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        // horizontal straight line through the start, random heading and speed; the reference accelerates uniformly
        // from rest during ramp_time so that a start near hover can follow it [assumption: horizontal only]
        using T = typename WHOOP_SPEC::T;
        using TI = typename DEVICE::index_t;
        using TRAJECTORY = decltype(parameters.trajectory);
        const auto& task = parameters.velocity_task;
        T heading = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, 2 * rlt::math::PI<T>, rng);
        T speed = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), task.speed_min, task.speed_max, rng);
        T direction[2] = {rlt::math::cos(device.math, heading), rlt::math::sin(device.math, heading)};
        for(TI step_i = 0; step_i < TRAJECTORY::LENGTH; step_i++){
            T t = step_i * TRAJECTORY::DT;
            T s, d;  // speed and distance along the line at t
            if(t < task.ramp_time){
                s = speed * t / task.ramp_time;
                d = speed * t * t / (2 * task.ramp_time);
            }
            else{
                s = speed;
                d = speed * (t - task.ramp_time / 2);
            }
            auto& reference = parameters.trajectory.steps[step_i];
            for(TI i = 0; i < 2; i++){
                reference.position[i] = d * direction[i];
                reference.linear_velocity[i] = s * direction[i];
            }
            reference.position[2] = 0;
            reference.linear_velocity[2] = 0;
            reference.yaw = 0;
            reference.yaw_velocity = 0;
        }
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rlt::rl::environments::Multirotor<SPEC>& env, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        rlt::sample_initial_parameters(device, env, static_cast<typename WHOOP_SPEC::NEXT_COMPONENT&>(parameters), rng);
        // no domain randomization of the plate yet: the nominal values of the environment (L2F: none either)
        parameters.plate = env.parameters.plate;
        parameters.velocity_task = env.parameters.velocity_task;
        if constexpr(WHOOP_SPEC::TASK == Task::VELOCITY){
            sample_velocity_reference(device, parameters, rng);
        }
    }

    template <typename DEVICE, typename T, typename TI, typename INIT, typename STATE_SPEC, typename RNG>
    void sample_rigid_body_l2f_paper(DEVICE& device, const INIT& init, l2f::StateBase<STATE_SPEC>& state, RNG& rng){
        // learning-to-fly simulator/operations_generic.h sample_initial_state: "guidance" starts (probability
        // init.guidance) at the origin and upright, the others anywhere in +-max_position with an attitude drawn
        // uniformly on SO(3) and kept if its angle is within max_angle; the velocities are random in both cases
        typename DEVICE::SPEC::RANDOM random;
        bool guidance = rlt::random::uniform_real_distribution(random, (T)0, (T)1, rng) < init.guidance;
        for(TI i = 0; i < 3; i++){
            state.position[i] = guidance ? (T)0 : rlt::random::uniform_real_distribution(random, -init.max_position, init.max_position, rng);
        }
        if(init.max_angle > 0 && !guidance){
            do{
                T u[3];
                for(TI i = 0; i < 3; i++){
                    u[i] = rlt::random::uniform_real_distribution(random, (T)0, (T)1, rng);
                }
                state.orientation[0] = rlt::math::sqrt(device.math, 1 - u[0]) * rlt::math::sin(device.math, 2 * rlt::math::PI<T> * u[1]);
                state.orientation[1] = rlt::math::sqrt(device.math, 1 - u[0]) * rlt::math::cos(device.math, 2 * rlt::math::PI<T> * u[1]);
                state.orientation[2] = rlt::math::sqrt(device.math, u[0]) * rlt::math::sin(device.math, 2 * rlt::math::PI<T> * u[2]);
                state.orientation[3] = rlt::math::sqrt(device.math, u[0]) * rlt::math::cos(device.math, 2 * rlt::math::PI<T> * u[2]);
            } while(rlt::math::abs(device.math, 2 * rlt::math::acos(device.math, rlt::math::clamp(device.math, state.orientation[0], (T)-1, (T)1))) > init.max_angle);
        }
        else{
            state.orientation[0] = 1;
            state.orientation[1] = 0;
            state.orientation[2] = 0;
            state.orientation[3] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.linear_velocity[i] = rlt::random::uniform_real_distribution(random, -init.max_linear_velocity, init.max_linear_velocity, rng);
        }
        for(TI i = 0; i < 3; i++){
            state.angular_velocity[i] = rlt::random::uniform_real_distribution(random, -init.max_angular_velocity, init.max_angular_velocity, rng);
        }
    }

    // only for the whole state of the environment: L2F calls sample_initial_state again on each part of the state,
    // and those calls must go to L2F's own function
    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename STATE, typename RNG>
    auto sample_initial_state(DEVICE& device, rlt::rl::environments::Multirotor<SPEC>& env, ParametersWhoop<WHOOP_SPEC>& parameters, STATE& state, RNG& rng)
        -> std::enable_if_t<std::is_same_v<STATE, typename rlt::rl::environments::Multirotor<SPEC>::State>>{
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        l2f::_sample_initial_state(device, env, parameters, state, rng);
        sample_rigid_body_l2f_paper<DEVICE, T, TI>(device, parameters.mdp.init, state, rng);
    }

    template <typename T>
    T action_baseline(T hovering_throttle_relative){
        // hover command in the normalized action [-1, 1] of the policy
        return 2 * hovering_throttle_relative - 1;
    }

    template <typename T>
    T reward_from_errors(const Reward<T>& r, const T position_error[3], T orientation_cost, const T linear_velocity_error[3], const T angular_velocity[3], const T* action, T baseline, bool terminated){
        if(terminated){
            return r.termination_penalty;
        }
        T position = 0, linear_velocity = 0, angular_velocity_cost = 0, action_cost = 0;
        for(int i = 0; i < 3; i++){
            position += position_error[i] * position_error[i];
            linear_velocity += linear_velocity_error[i] * linear_velocity_error[i];
            angular_velocity_cost += angular_velocity[i] * angular_velocity[i];
        }
        for(int i = 0; i < 4; i++){
            action_cost += (action[i] - baseline) * (action[i] - baseline);
        }
        T cost = r.position * position + r.orientation * orientation_cost + r.linear_velocity * linear_velocity
               + r.angular_velocity * angular_velocity_cost + r.action * action_cost;
        return r.constant - r.scale * cost;
    }

    template <typename T>
    T reward_from_privileged_observation(const Reward<T>& r, const T* observation, const T* action, T baseline, bool terminated){
        // the same reward from the critic's observation of the state before the step (position error, rotation
        // matrix, velocity error, angular velocity: environment.h), to recompute the replay buffer after a curriculum
        // step as L2F does. 1 - q_w^2 = (3 - trace R) / 4
        T orientation_cost = (3 - (observation[3] + observation[7] + observation[11])) / 4;
        return reward_from_errors(r, observation, orientation_cost, observation + 12, observation + 15, action, baseline, terminated);
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rlt::rl::environments::Multirotor<SPEC>& env, ParametersWhoop<WHOOP_SPEC>& parameters, const STATE& state, const rlt::Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        STATE desired_state;
        rlt::get_desired_state(device, env, parameters, state, desired_state, rng);
        T position_error[3], linear_velocity_error[3], a[4];
        for(TI i = 0; i < 3; i++){
            position_error[i] = state.position[i] - desired_state.position[i];
            linear_velocity_error[i] = state.linear_velocity[i] - desired_state.linear_velocity[i];
        }
        for(TI i = 0; i < 4; i++){
            a[i] = rlt::get(action, 0, i);
        }
        T orientation_cost = 1 - state.orientation[0] * state.orientation[0];
        bool terminated = rlt::terminated(device, env, parameters, next_state, rng);
        return reward_from_errors(parameters.mdp.reward, position_error, orientation_cost, linear_velocity_error, state.angular_velocity, a,
                                  action_baseline(parameters.dynamics.hovering_throttle_relative), terminated);
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE&, const rlt::rl::environments::Multirotor<SPEC>&, ParametersWhoop<WHOOP_SPEC>&, const STATE&, const rlt::Matrix<ACTION_SPEC>&, const STATE&, RNG&, typename DEVICE::index_t = 1){
        // the off-policy runner logs the reward terms to the device logger; train.cpp writes its own progress instead
    }
}
