// The L2F operations that change for the whoop, as overloads on ParametersWhoop (whoop/environment.h).
//
// How this works without touching RLtools: L2F calls its operations unqualified (multirotor_dynamics inside the
// integrator, sample_initial_state / reward / log_reward in the off-policy runner), so argument-dependent lookup also
// finds functions in the namespace of the parameter type, here whoop. Each overload below takes ParametersWhoop<...>&
// where L2F takes any PARAMETERS&, so C++ prefers it (more specialized), and it calls back into L2F's own code for
// everything it does not change. The overloads are:
//   multirotor_dynamics  L2F's rigid-body and rotor equations, then + the body drag (whoop/body_drag.h) and the plate
//                        force and torque when S > 0
//   sample_initial_parameters  L2F's, then the plate, drag and task parameters, the randomization of the aerodynamics,
//                              the velocity reference and the motor cut of the episode
//   sample_initial_state  L2F's (rotors, action history, random force), then the rigid-body initial state of the L2F
//                         paper: the current RLtools samples the tilt uniformly in [0, 1] rad whatever max_angle says
//   step                 L2F's, with the motor commands forced to 0 during a motor cut
//   reward, log_reward  the reward of the L2F paper (Reward in environment.h)
// Include this after <rl_tools/operations/cpu_mux.h> (or another device), as any RLtools environment.
#pragma once

#include <whoop/environment.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <cmath>
#include <type_traits>

namespace whoop{
    namespace rlt = rl_tools;

    template <typename DEVICE, typename STATE_SPEC, typename WHOOP_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const ParametersWhoop<WHOOP_SPEC>& params, const l2f::StateBase<STATE_SPEC>& state, const T* action, l2f::StateBase<STATE_SPEC>& state_change){
        // L2F's equations with the L2F part of the parameters (exactly the same code and numbers as without the plate)
        using TI = typename DEVICE::index_t;
        using L2F_PARAMETERS = typename WHOOP_SPEC::NEXT_COMPONENT;
        l2f::multirotor_dynamics(device, static_cast<const L2F_PARAMETERS&>(params), state, action, state_change);
        bool plate = params.plate.area > 0;
        bool drag = params.body_drag[0] > 0 || params.body_drag[1] > 0 || params.body_drag[2] > 0;
        // with neither, nothing is added: the code path is L2F's alone (the tests check it bit for bit)
        if(plate || drag){
            // L2F sums the rotor forces and torques in the body frame, then a = R F / m + g and
            // domega/dt = J^-1 (tau - omega x J omega). Both are linear in F and tau, so adding R F_aero / m and
            // J^-1 tau_aero afterwards is the same as adding the aerodynamics to L2F's sums (up to rounding)
            T force_body[3] = {0, 0, 0};
            if(drag){
                // at the centre of mass: a force, no torque
                body_drag_force(device, params.body_drag, state.orientation, state.linear_velocity, params.plate.wind, force_body);
            }
            PlateAerodynamics<T> aero;
            if(plate){
                // the total rotor thrust along body z, only for the downwash on the plate (k_w > 0); action holds
                // the rotor states here (L2F passes state.rpm), from which L2F computes the same thrusts
                T rotor_thrust = 0;
                if(params.plate.downwash_factor > 0){
                    for(TI i = 0; i < 4; i++){
                        const auto& c = params.dynamics.rotor_thrust_coefficients[i];
                        rotor_thrust += (c[0] + c[1] * action[i] + c[2] * action[i] * action[i]) * params.dynamics.rotor_thrust_directions[i][2];
                    }
                }
                plate_aerodynamics(device, params.plate, state.orientation, state.linear_velocity, state.angular_velocity, rotor_thrust, aero);
                for(TI i = 0; i < 3; i++){
                    force_body[i] += aero.force[i];
                }
            }
            T force_world[3];
            l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, force_body, force_world);
            for(TI i = 0; i < 3; i++){
                state_change.linear_velocity[i] += force_world[i] / params.dynamics.mass;
            }
            if(plate){
                T angular_acceleration[3];
                rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, aero.torque, angular_acceleration);
                rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(angular_acceleration, state_change.angular_velocity);
            }
        }
    }

    template <typename DEVICE, typename WHOOP_SPEC>
    void set_velocity_reference(DEVICE& device, ParametersWhoop<WHOOP_SPEC>& parameters, typename WHOOP_SPEC::T heading, typename WHOOP_SPEC::T speed){
        // horizontal straight line through the origin at the given heading [rad] and speed [m/s]; the reference
        // accelerates uniformly from rest during ramp_time so that a start near hover can follow it
        using T = typename WHOOP_SPEC::T;
        using TI = typename DEVICE::index_t;
        using TRAJECTORY = decltype(parameters.trajectory);
        const auto& task = parameters.velocity_task;
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

    template <typename DEVICE, typename WHOOP_SPEC, typename RNG>
    void sample_velocity_reference(DEVICE& device, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        // random heading and speed [assumption: horizontal only]
        using T = typename WHOOP_SPEC::T;
        const auto& task = parameters.velocity_task;
        T heading = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, 2 * rlt::math::PI<T>, rng);
        T speed = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), task.speed_min, task.speed_max, rng);
        set_velocity_reference(device, parameters, heading, speed);
    }

    template <typename DEVICE, typename WHOOP_SPEC, typename RNG>
    void sample_aerodynamics(DEVICE& device, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        // one independent factor U(1 - r, 1 + r) per coefficient. The plate geometry (S, c_x, c_z, r_c) is measured,
        // so only the coefficients that the identification has to estimate are randomized
        using T = typename WHOOP_SPEC::T;
        const auto& r = parameters.randomization;
        auto factor = [&](T range){
            return rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), 1 - range, 1 + range, rng);
        };
        auto& plate = parameters.plate;
        plate.CN90 *= factor(r.plate);
        plate.Kp_x *= factor(r.plate);
        plate.Kp_z *= factor(r.plate);
        plate.CT *= factor(r.plate);
        plate.e0 *= factor(r.plate);
        for(int i = 0; i < 3; i++){
            parameters.body_drag[i] *= factor(r.body_drag);
        }
    }

    template <typename DEVICE, typename WHOOP_SPEC, typename RNG>
    void sample_motor_cut(DEVICE& device, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        using T = typename WHOOP_SPEC::T;
        using TI = typename WHOOP_SPEC::TI;
        auto& cut = parameters.motor_cut;
        typename DEVICE::SPEC::RANDOM random;
        cut.start_step = 0;
        cut.end_step = 0;
        if(rlt::random::uniform_real_distribution(random, (T)0, (T)1, rng) < cut.probability){
            T start = rlt::random::uniform_real_distribution(random, cut.start_min, cut.start_max, rng);
            T duration = rlt::random::uniform_real_distribution(random, cut.duration_min, cut.duration_max, rng);
            T dt = parameters.integration.dt;
            cut.start_step = (TI)(start / dt + (T)0.5);
            cut.end_step = (TI)((start + duration) / dt + (T)0.5);
        }
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rlt::rl::environments::Multirotor<SPEC>& env, ParametersWhoop<WHOOP_SPEC>& parameters, RNG& rng){
        rlt::sample_initial_parameters(device, env, static_cast<typename WHOOP_SPEC::NEXT_COMPONENT&>(parameters), rng);
        parameters.plate = env.parameters.plate;
        parameters.velocity_task = env.parameters.velocity_task;
        for(int i = 0; i < 3; i++){
            parameters.body_drag[i] = env.parameters.body_drag[i];
        }
        parameters.randomization = env.parameters.randomization;
        parameters.motor_cut = env.parameters.motor_cut;
        if constexpr(WHOOP_SPEC::TASK == Task::VELOCITY){
            sample_velocity_reference(device, parameters, rng);
        }
        // random numbers are drawn only for the options that are on, so that with both off the episodes (and the
        // fixed test flights of train.cpp) are the same as without these options
        if(parameters.randomization.enabled){
            sample_aerodynamics(device, parameters, rng);
        }
        parameters.motor_cut.start_step = 0;
        parameters.motor_cut.end_step = 0;
        if constexpr(WHOOP_SPEC::TASK == Task::VELOCITY){
            // the cut is timed by the trajectory step, which only the velocity task has
            if(parameters.motor_cut.probability > 0){
                sample_motor_cut(device, parameters, rng);
            }
        }
    }

    template <typename WHOOP_SPEC, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT bool motors_cut(const ParametersWhoop<WHOOP_SPEC>& parameters, const STATE& state){
        // true while the episode's motor cut is on, for the step from this state
        if constexpr(WHOOP_SPEC::TASK == Task::VELOCITY){
            return state.trajectory_step >= parameters.motor_cut.start_step && state.trajectory_step < parameters.motor_cut.end_step;
        }
        else{
            return false;
        }
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rlt::rl::environments::Multirotor<SPEC>& env, ParametersWhoop<WHOOP_SPEC>& parameters, const STATE& state, const rlt::Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng){
        // L2F's step, except during a motor cut: then every command is -1 (motor command 0) whatever the policy asks.
        // The forced command also goes into the action history, as it would on the drone, where the flight controller
        // knows that the motors are off; the replay buffer and the reward keep the policy's own action
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if(motors_cut(parameters, state)){
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_SPEC::COLS, false>> off;
            for(TI i = 0; i < ACTION_SPEC::COLS; i++){
                rlt::set(off, 0, i, (T)-1);
            }
            return rlt::step(device, env, parameters, state, off, next_state, rng);
        }
        return rlt::step(device, env, parameters, state, action, next_state, rng);
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

    template <typename DYNAMICS>
    auto rotor_power_model(const DYNAMICS& dynamics){
        // the four rotors share one thrust curve in the whoop parameters (rotor 0 stands for all)
        using T = std::decay_t<decltype(dynamics.mass)>;
        RotorPower<T> model;
        for(int j = 0; j < 3; j++) model.thrust_coefficients[j] = dynamics.rotor_thrust_coefficients[0][j];
        T g = std::sqrt(dynamics.gravity[0] * dynamics.gravity[0] + dynamics.gravity[1] * dynamics.gravity[1] + dynamics.gravity[2] * dynamics.gravity[2]);
        model.hover_thrust = dynamics.mass * g / 4;
        return model;
    }

    template <typename T>
    T rotor_thrust_from_command(const RotorPower<T>& model, T u){
        // thrust of one rotor at the normalized motor command u in [0, 1], never negative (an identified c0 < 0 would
        // otherwise make the power of an idle rotor undefined)
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
        T f = model.thrust_coefficients[0] + model.thrust_coefficients[1] * u + model.thrust_coefficients[2] * u * u;
        return f > 0 ? f : 0;
    }

    template <typename T>
    T power_cost(const RotorPower<T>& model, const T* action){
        // mean_i (f_i / f_hover)^1.5 - 1 for the commanded thrusts; the action is in [-1, 1], the command (a + 1) / 2
        // as in L2F's step for an action limit of [0, 1]. Zero at the hover command
        T sum = 0;
        for(int i = 0; i < 4; i++){
            T ratio = rotor_thrust_from_command(model, (action[i] + 1) / 2) / model.hover_thrust;
            sum += ratio * std::sqrt(ratio);
        }
        return sum / 4 - 1;
    }

    template <typename T>
    T reward_from_errors(const Reward<T>& r, const T position_error[3], T orientation_cost, const T linear_velocity_error[3], const T angular_velocity[3], const T* action, T baseline, const RotorPower<T>& power, bool terminated){
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
        if(r.thrust != 0){
            // only when the option is on, so that the paper's reward is computed exactly as before
            cost += r.thrust * power_cost(power, action);
        }
        return r.constant - r.scale * cost;
    }

    template <typename T>
    T reward_from_privileged_observation(const Reward<T>& r, const T* observation, const T* action, T baseline, const RotorPower<T>& power, bool terminated){
        // the same reward from the critic's observation of the state before the step (position error, rotation
        // matrix, velocity error, angular velocity: environment.h), to recompute the replay buffer after a curriculum
        // step as L2F does. 1 - q_w^2 = (3 - trace R) / 4; the tilt alone is (1 - R_zz) / 2. The power cost needs
        // only the stored action
        T orientation_cost = r.tilt_only ? (1 - observation[11]) / 2 : (3 - (observation[3] + observation[7] + observation[11])) / 4;
        return reward_from_errors(r, observation, orientation_cost, observation + 12, observation + 15, action, baseline, power, terminated);
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
        const auto& r = parameters.mdp.reward;
        T orientation_cost = r.tilt_only ? state.orientation[1] * state.orientation[1] + state.orientation[2] * state.orientation[2]
                                         : 1 - state.orientation[0] * state.orientation[0];
        bool terminated = rlt::terminated(device, env, parameters, next_state, rng);
        return reward_from_errors(r, position_error, orientation_cost, linear_velocity_error, state.angular_velocity, a,
                                  action_baseline(parameters.dynamics.hovering_throttle_relative), rotor_power_model(parameters.dynamics), terminated);
    }

    template <typename DEVICE, typename SPEC, typename WHOOP_SPEC, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE&, const rlt::rl::environments::Multirotor<SPEC>&, ParametersWhoop<WHOOP_SPEC>&, const STATE&, const rlt::Matrix<ACTION_SPEC>&, const STATE&, RNG&, typename DEVICE::index_t = 1){
        // the off-policy runner logs the reward terms to the device logger; train.cpp writes its own progress instead
    }
}
