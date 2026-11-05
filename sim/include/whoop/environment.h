// The whoop environment: the L2F multirotor of RLtools (include/rl_tools/rl/environments/l2f) with the whoop's
// parameters (params/whoop_nominal.json), the plate of docs/plate_model.md, and the training setup of the L2F paper
// (Eschmann et al., "Learning to Fly in Seconds", RA-L 2024; github.com/arplaboratory/learning-to-fly src/config).
//
// The RLtools checkout is not modified. L2F composes its parameters as a chain of structs (ParametersBase,
// ParametersDisturbances, ...) and dispatches its operations on the types of that chain. This file adds one more link
// on top, ParametersWhoop, which carries the plate and the task; whoop/operations.h then overloads the few L2F
// operations that must change for that type (dynamics, initial state, reward). Everything else (integration, rotor
// model, observations, termination, disturbances) is L2F's own code.
//
// Two tasks:
//   HOVER    L2F's position hold at the origin from random initial states (the paper's task)
//   VELOCITY fly along a horizontal straight line at a constant speed of 1-4 m/s, where the plate matters: the
//            reference ramps up from rest, and its position and velocity are L2F's trajectory reference
#pragma once

#include <rl_tools/rl/environments/l2f/multirotor.h>
#include <rl_tools/utils/generic/typing.h>

#include <whoop/params.h>  // generated from params/whoop_nominal.json (tools/generate_params.py)
#include <whoop/plate.h>

namespace whoop{
    namespace l2f = rl_tools::rl::environments::l2f;

    enum class Task{
        HOVER,
        VELOCITY
    };

    // reward of the L2F paper (learning-to-fly simulator/parameters/reward_functions/squared.h), which differs from the
    // "Squared" reward of the current RLtools (norms instead of squares, yaw-only orientation cost):
    //   r = constant - scale (position |p - p_ref|^2 + orientation (1 - q_w^2) + linear_velocity |v - v_ref|^2
    //                         + angular_velocity |omega|^2 + action |a - a_hover|^2),  termination_penalty if terminated
    // evaluated on the state before the step, as in L2F. a_hover is the hover command in the normalized action [-1, 1]
    // (0.334 for the Crazyflie in L2F), taken from the dynamics so that it follows the mass of each variant.
    template <typename T>
    struct Reward{
        T scale;
        T constant;
        T termination_penalty;
        T position;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T action;
    };

    template <typename T>
    struct VelocityTask{
        T speed_min;  // [m/s]
        T speed_max;
        T ramp_time;  // [s] the reference accelerates uniformly from rest to the speed in this time
    };

    template <typename T_T, typename T_TI, Task T_TASK, typename T_NEXT_COMPONENT>
    struct ParametersWhoopSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr Task TASK = T_TASK;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_WHOOP_SPEC>
    struct ParametersWhoop: T_WHOOP_SPEC::NEXT_COMPONENT{
        // named WHOOP_SPEC, not SPEC: ParametersBase::SPEC must stay visible to the L2F code
        using WHOOP_SPEC = T_WHOOP_SPEC;
        using NEXT_COMPONENT = typename WHOOP_SPEC::NEXT_COMPONENT;
        static constexpr Task TASK = WHOOP_SPEC::TASK;
        Plate<typename WHOOP_SPEC::T> plate;
        VelocityTask<typename WHOOP_SPEC::T> velocity_task;
    };

    // L2F observations are tagged privileged (noise-free for the critic) through their specification; the trajectory
    // tracking ones have no privileged variant in RLtools, so the critic would get the actor's noise without these
    template <typename T, typename TI, typename NEXT = l2f::observation::LastComponent<TI>>
    struct TrackingPositionPrivileged: l2f::observation::TrajectoryTrackingPositionSpecification<T, TI, NEXT>{
        static constexpr bool PRIVILEGED = true;
    };
    template <typename T, typename TI, typename NEXT = l2f::observation::LastComponent<TI>>
    struct TrackingLinearVelocityPrivileged: l2f::observation::TrajectoryTrackingLinearVelocitySpecification<T, TI, NEXT>{
        static constexpr bool PRIVILEGED = true;
    };

    template <typename T, typename TI, bool PLATE>
    constexpr l2f::parameters::Dynamics<T, TI, 4> whoop_dynamics(){
        // params.h holds doubles; L2F wants its own struct in the simulation type. With the plate on, the sheet's mass
        // and inertia are added (tools/generate_params.py, [assumption])
        l2f::parameters::Dynamics<T, TI, 4> d{};
        for(TI i = 0; i < 4; i++){
            for(TI j = 0; j < 3; j++){
                d.rotor_positions[i][j] = (T)params::rotor_positions[i][j];
                d.rotor_thrust_directions[i][j] = (T)params::rotor_thrust_directions[i][j];
                d.rotor_torque_directions[i][j] = (T)params::rotor_torque_directions[i][j];
                d.rotor_thrust_coefficients[i][j] = (T)params::rotor_thrust_coefficients[j];
            }
            d.rotor_torque_constants[i] = (T)params::rotor_torque_constant;
            d.rotor_time_constants_rising[i] = (T)params::rotor_time_constant_rising;
            d.rotor_time_constants_falling[i] = (T)params::rotor_time_constant_falling;
        }
        d.mass = (T)(PLATE ? params::plate::mass_total : params::mass);
        for(TI j = 0; j < 3; j++){
            d.gravity[j] = (T)params::gravity[j];
            for(TI k = 0; k < 3; k++){
                d.J[j][k] = (T)(PLATE ? params::plate::J_total[j][k] : params::J[j][k]);
                d.J_inv[j][k] = (T)(PLATE ? params::plate::J_total_inv[j][k] : params::J_inv[j][k]);
            }
        }
        d.hovering_throttle_relative = (T)(PLATE ? params::plate::hovering_throttle_total : params::hovering_throttle);
        d.action_limit.min = 0;  // normalized motor command, as the RLtools Crazyflie model
        d.action_limit.max = 1;
        return d;
    }

    template <typename T, bool PLATE>
    constexpr Plate<T> whoop_plate(){
        Plate<T> p{};
        p.area = PLATE ? (T)params::plate::area : (T)0;
        p.chord_x = (T)params::plate::chord_x;
        p.height_z = (T)params::plate::height_z;
        for(int i = 0; i < 3; i++){
            p.normal[i] = (T)params::plate::normal[i];
            p.centroid[i] = (T)params::plate::centroid[i];
            p.wind[i] = 0;
        }
        p.CN90 = (T)params::plate::CN90;
        p.Kp_x = (T)params::plate::Kp_x;
        p.Kp_z = (T)params::plate::Kp_z;
        p.CT = (T)params::plate::CT;
        p.e0 = (T)params::plate::e0;
        p.air_density = (T)params::plate::air_density;
        return p;
    }

    // L2F paper reward weights (learning-to-fly reward_squared_position_only_torque) and the curriculum targets
    // (reward_squared_position_only_torque_curriculum_target, src/steps/curriculum.h)
    template <typename T>
    constexpr Reward<T> reward_initial = {
        0.5,   // scale
        2.0,   // constant
        0.0,   // termination penalty
        5.0,   // position
        5.0,   // orientation
        0.01,  // linear velocity
        0.0,   // angular velocity
        0.01,  // action
    };
    template <typename T>
    struct Curriculum{
        // every PERIOD steps (not at step 0): weight *= factor, capped
        static constexpr unsigned long PERIOD = 100000;
        static constexpr T POSITION_FACTOR = 1.2, POSITION_LIMIT = 40;
        static constexpr T LINEAR_VELOCITY_FACTOR = 1.4, LINEAR_VELOCITY_LIMIT = 1;
        static constexpr T ACTION_FACTOR = 1.4, ACTION_LIMIT = 1;
        static Reward<T> next(Reward<T> r){
            auto grow = [](T w, T factor, T limit){ w *= factor; return w > limit ? limit : w; };
            r.position = grow(r.position, POSITION_FACTOR, POSITION_LIMIT);
            r.linear_velocity = grow(r.linear_velocity, LINEAR_VELOCITY_FACTOR, LINEAR_VELOCITY_LIMIT);
            r.action = grow(r.action, ACTION_FACTOR, ACTION_LIMIT);
            return r;
        }
    };

    constexpr int SIMULATION_FREQUENCY = 100;  // L2F: 100 Hz, one RK4 step per action

    // outside the factory: a constexpr member function cannot initialize a static member of its own class
    template <typename T, typename TI, bool PLATE, bool VELOCITY, typename L2F_PARAMETERS>
    constexpr L2F_PARAMETERS make_l2f_parameters(){
        L2F_PARAMETERS p{};
        p.dynamics = whoop_dynamics<T, TI, PLATE>();
        p.integration.dt = (T)1 / (T)SIMULATION_FREQUENCY;
        // initial states of the paper (learning-to-fly init/default.h orientation_biggest_angle): 10 % at the
        // origin and upright, otherwise p ~ U(+-0.2 m) and attitude within 90 deg; v ~ U(+-1 m/s) and
        // omega ~ U(+-1 rad/s) always; motors at half command
        p.mdp.init.guidance = (T)0.1;
        p.mdp.init.max_position = (T)0.2;
        p.mdp.init.max_angle = (T)(90.0 / 180.0 * 3.14);
        p.mdp.init.max_linear_velocity = 1;
        p.mdp.init.max_angular_velocity = 1;
        p.mdp.init.relative_rpm = true;
        p.mdp.init.min_rpm = 0;
        p.mdp.init.max_rpm = 0;
        p.mdp.reward = reward_initial<T>;
        // sensor noise of the paper (Gaussian std, actor only): position, orientation, velocities
        p.mdp.observation_noise.position = (T)0.001;
        p.mdp.observation_noise.orientation = (T)0.001;
        p.mdp.observation_noise.linear_velocity = (T)0.002;
        p.mdp.observation_noise.angular_velocity = (T)0.002;
        p.mdp.observation_noise.imu_acceleration = 0;
        p.mdp.action_noise.normalized_rpm = 0;
        // termination of the paper: |p - p_ref| > 0.6 m on any axis; velocity limits effectively off. The
        // velocity task allows 1 m, for the lag while the reference accelerates [assumption]
        p.mdp.termination.enabled = true;
        p.mdp.termination.position_threshold = VELOCITY ? (T)1.0 : (T)0.6;
        p.mdp.termination.linear_velocity_threshold = 1000;
        p.mdp.termination.angular_velocity_threshold = 1000;
        p.mdp.termination.position_integral_threshold = 0;
        p.mdp.termination.orientation_integral_threshold = 0;
        // constant random force and torque per episode, Gaussian (released L2F code): m g / 20 and m g / 10000,
        // with this vehicle's mass [assumption: the same ratios as for the Crazyflie]
        T weight = p.dynamics.mass * (T)9.81;
        p.disturbances.random_force = {0, weight / 20};
        p.disturbances.random_torque = {0, weight / 10000};
        if constexpr(VELOCITY){
            // the L2F trajectory sampler needs a valid type; whoop/operations.h overwrites the steps per episode
            p.trajectory_parameters.type = l2f::parameters::trajectories::Type::LISSAJOUS;
            p.trajectory_parameters.parameters.lissajous = l2f::parameters::trajectories::lissajous::default_parameters<T>;
        }
        return p;
    }
    template <typename T_T, typename T_TI, bool T_PLATE, Task T_TASK>
    struct EnvironmentFactory{
        using T = T_T;
        using TI = T_TI;
        static constexpr bool PLATE = T_PLATE;
        static constexpr Task TASK = T_TASK;
        static constexpr bool VELOCITY = TASK == Task::VELOCITY;
        static constexpr TI SIMULATION_FREQUENCY = whoop::SIMULATION_FREQUENCY;
        static constexpr TI EPISODE_STEP_LIMIT = 500;    // 5 s
        static constexpr TI ACTION_HISTORY_LENGTH = 32;  // L2F paper (the RLtools zoo uses 16)

        using REWARD_FUNCTION = Reward<T>;
        using BASE_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
        using PARAMETERS_DISTURBANCES = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<BASE_SPEC>>>;
        // the velocity task reuses L2F's trajectory reference (one step per simulation step), filled per episode
        using PARAMETERS_TRAJECTORY = l2f::ParametersTrajectory<l2f::ParametersTrajectorySpecification<T, TI, EPISODE_STEP_LIMIT, 1000000 / SIMULATION_FREQUENCY, PARAMETERS_DISTURBANCES>>;
        using L2F_PARAMETERS = rl_tools::utils::typing::conditional_t<VELOCITY, PARAMETERS_TRAJECTORY, PARAMETERS_DISTURBANCES>;
        using PARAMETERS = ParametersWhoop<ParametersWhoopSpecification<T, TI, TASK, L2F_PARAMETERS>>;

        static constexpr L2F_PARAMETERS l2f_nominal_parameters = make_l2f_parameters<T, TI, PLATE, VELOCITY, L2F_PARAMETERS>();
        static constexpr VelocityTask<T> velocity_task = {1, 4, 1};  // [assumption] speed range and ramp
        static constexpr PARAMETERS nominal_parameters = {l2f_nominal_parameters, whoop_plate<T, PLATE>(), velocity_task};

        // state: L2F's rigid body, random force, rotor speeds (first-order motors) and the action history; the
        // velocity task adds the trajectory step
        using STATE_HOVER = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, false, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateBase<l2f::StateSpecification<T, TI>>>>>>;
        using STATE_VELOCITY = l2f::StateTrajectory<l2f::StateSpecification<T, TI, STATE_HOVER>>;

        // actor: the paper's position, rotation matrix, linear velocity, angular velocity and action history (146
        // values). For hovering at the origin these are the tracking errors, and the linear velocity is the absolute
        // one (still air: the airspeed the plate sees, up to the rotation). The velocity task observes the errors to
        // the reference and, in addition, the absolute linear velocity (149 values)
        using ACTION_HISTORY = l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>;
        using OBSERVATION_HOVER =
            l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
            l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
            l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
            l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
            ACTION_HISTORY>>>>>>>>;
        using OBSERVATION_VELOCITY =
            l2f::observation::TrajectoryTrackingPosition<l2f::observation::TrajectoryTrackingPositionSpecification<T, TI,
            l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
            l2f::observation::TrajectoryTrackingLinearVelocity<l2f::observation::TrajectoryTrackingLinearVelocitySpecification<T, TI,
            l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
            l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
            ACTION_HISTORY>>>>>>>>>>;
        // critic (asymmetric actor-critic of the paper): the same without noise and without the action history, plus
        // the random force and torque and the rotor speeds. The first 18 values are always the position error,
        // rotation matrix, velocity error and angular velocity: train.cpp recomputes the rewards from them
        using PRIVILEGED_TAIL = l2f::observation::RandomForce<l2f::observation::RandomForceSpecification<T, TI,
            l2f::observation::RotorSpeeds<l2f::observation::RotorSpeedsSpecification<T, TI>>>>;
        using OBSERVATION_PRIVILEGED_HOVER =
            l2f::observation::Position<l2f::observation::PositionSpecificationPrivileged<T, TI,
            l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
            l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
            l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI,
            PRIVILEGED_TAIL>>>>>>>>;
        using OBSERVATION_PRIVILEGED_VELOCITY =
            l2f::observation::TrajectoryTrackingPosition<TrackingPositionPrivileged<T, TI,
            l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
            l2f::observation::TrajectoryTrackingLinearVelocity<TrackingLinearVelocityPrivileged<T, TI,
            l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI,
            l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
            PRIVILEGED_TAIL>>>>>>>>>>;
        // index of the absolute linear velocity in the actor observation
        static constexpr TI OBSERVATION_LINEAR_VELOCITY_INDEX = VELOCITY ? 18 : 12;

        struct STATIC_PARAMETERS_COMMON{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = EnvironmentFactory::ACTION_HISTORY_LENGTH;
            static constexpr TI EPISODE_STEP_LIMIT = EnvironmentFactory::EPISODE_STEP_LIMIT;
            static constexpr bool CLOSED_FORM = false;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<VELOCITY, STATE_VELOCITY, STATE_HOVER>;
            using OBSERVATION_TYPE = rl_tools::utils::typing::conditional_t<VELOCITY, OBSERVATION_VELOCITY, OBSERVATION_HOVER>;
            using OBSERVATION_TYPE_PRIVILEGED = rl_tools::utils::typing::conditional_t<VELOCITY, OBSERVATION_PRIVILEGED_VELOCITY, OBSERVATION_PRIVILEGED_HOVER>;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };
        struct STATIC_PARAMETERS: STATIC_PARAMETERS_COMMON{
            using PARAMETERS = EnvironmentFactory::PARAMETERS;
            static constexpr PARAMETERS PARAMETER_VALUES = nominal_parameters;
        };
        // the same environment without the whoop link of the parameter chain: plain L2F operations everywhere. Only
        // used by the tests, to check that the plate code leaves the L2F dynamics untouched when S = 0
        struct STATIC_PARAMETERS_L2F: STATIC_PARAMETERS_COMMON{
            using PARAMETERS = L2F_PARAMETERS;
            static constexpr PARAMETERS PARAMETER_VALUES = l2f_nominal_parameters;
        };
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS>>;
        using ENVIRONMENT_L2F = rl_tools::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS_L2F>>;
    };
}
