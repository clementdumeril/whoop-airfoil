// Training of the whoop policy with the setup of "Learning to Fly in Seconds" (L2F, Eschmann et al., RA-L 2024; the
// released code, github.com/arplaboratory/learning-to-fly src/config): actor and critics 64 x 64 (tanh), batch 256,
// critics trained every 10 steps and the actor every 20, Adam 1e-3 with weight decay 1e-4, critic warmup 15k steps
// and actor warmup 30k, 100 Hz, 500-step episodes, 300k steps, asymmetric actor-critic (the critic sees the noise-free
// state, the random force and the rotor speeds), the paper's reward and its curriculum (every 100k steps the position
// weight x1.2, the velocity and action weights x1.4, then all stored rewards recomputed).
//
// SAC by default: in the Fixed-Wing project the same intervals and batch gave smooth commands with SAC and bang-bang
// commands with TD3, the algorithm of the paper; the whoop_train_td3_* targets train with TD3 (WHOOP_TD3).
//
// The aerodynamics (plate coefficients and body drag) are randomized per episode unless --no-randomization; with
// --motor-cuts P (velocity task) a fraction P of the training episodes has all motors cut for 0.2-0.5 s.
//
// whoop_train_<sac|td3>_<hover|velocity>_<noplate|plate> <out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum]
//                                                         [--no-randomization] [--motor-cuts P]
//   out_dir gets config.json; progress.csv (episode length and return of the last 50k steps, every 10k steps, with the
//   wall clock); evaluation_progress.csv (the fixed test flights every N steps, default 50000); evaluation.json and
//   flights.csv (the final actor on the fixed test flights: 100 starts sampled once with seed 12345, nominal
//   aerodynamics, the first 10 written step by step); evaluation_randomized.json (the same starts with randomized
//   aerodynamics, when training randomized them); actor.h (the trained actor as C++ code, the checkpoint format of
//   L2F) and actor.tar (the same weights, read back by --scenario); runner_checks.json (evidence from the replay
//   buffer that the RLtools runner used the whoop initial states, references and motor cuts)
//
// whoop_train_<sac|td3>_velocity_<noplate|plate> <out_dir> --scenario <actor.tar> [--scenario-seeds N] [--speed V]
//                                                [--cut-time S] [--cut-duration S] [--duration S]
//   no training: the motor-cut scenario with a trained actor. The whoop starts level at rest at the origin and
//   follows the reference of the velocity task straight along world x at V m/s (default 3, 1 s ramp); at --cut-time
//   (default 1.5 s) all motor commands are forced to 0 for --cut-duration (default 0.4 s), then the policy flies again
//   until --duration (default 5 s, at most the 5 s episode). Nominal aerodynamics; the N seeds (default 5) change the
//   random force and torque and the sensor noise. out_dir gets config.json, flights.csv (the format of training plus
//   a motors_cut column) and scenario.json (altitude lost and recovery, per seed)
//
// Options for every mode (they change the parameters every environment of the run starts from):
//   --params FILE     the whoop and plate parameters from a JSON in the format of params/whoop_nominal.json instead of
//                     the compiled ones (to train on identified parameters, or to fly a policy on a "true" drone)
//   --plate-z Z       the plate centroid height relative to the centre of mass [m] (nominal -0.04); the inertia follows
//   --plate-tilt DEG  the plate tilted by DEG about body x: normal (0, cos, sin), chord along body x (0 = the nominal
//                     vertical plate, 90 = horizontal); the inertia follows
//   --downwash K      the crude downwash term on the plate, k_w = K (docs/plate_model.md; nominal 0 = none)
//   --thrust-cost W   a rotor power cost of weight W in the reward (whoop::Reward::thrust, environment.h)
//   --free-yaw        the orientation cost on the tilt only, so that the policy may choose its heading to the path
//   --speed-min V, --speed-max V   the speed range of the velocity task (default 1-4 m/s)
//
// whoop_train_<sac|td3>_velocity_<noplate|plate> <out_dir> --evaluate <actor.tar> [options above]
//   no training: the test flights of training (evaluation.json, flights.csv) and the speed sweep (sweep_flights.csv)
//   with a trained actor, on the parameters given by the options (for example a policy trained on identified
//   parameters, flown on the true drone)
//
// Every evaluation also flies a speed sweep: level starts at rest, references at 2, 2.5, 3, 3.5 and 4 m/s in 8
// headings, no random force; evaluation.json gives per speed the mean rotor thrust and plate lift over t = 2-5 s, the
// angle between body x and the path, the roll and the plate angle of attack
#include <rl_tools/operations/cpu_mux.h>

#include <whoop/operations.h>
#include <whoop/params_runtime.h>
#include <whoop/metrics.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifndef WHOOP_TD3
#define WHOOP_TD3 0
#endif
#ifndef WHOOP_TASK_VELOCITY
#define WHOOP_TASK_VELOCITY 0
#endif
#ifndef WHOOP_PLATE
#define WHOOP_PLATE 1
#endif
#if WHOOP_TD3
#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#else
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#endif

// export of the trained actor as C++ code (weights included), the checkpoint format of L2F
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
// and as a tar file of the weights (RLtools' dependency-free checkpoint backend), which the scenario mode reads back
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;  // networks and simulation, as L2F
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;
constexpr whoop::Task TASK = WHOOP_TASK_VELOCITY ? whoop::Task::VELOCITY : whoop::Task::HOVER;
using FACTORY = whoop::EnvironmentFactory<T, TI, WHOOP_PLATE, TASK>;
using ENVIRONMENT = FACTORY::ENVIRONMENT;

// the parameters every environment of this run starts from: the compiled nominal ones, changed once in run() by the
// command-line options (--params, --plate-z, --plate-tilt, --downwash, --thrust-cost, --free-yaw, --speed-min, --speed-max) before anything is
// sampled. Without those options this is exactly FACTORY::nominal_parameters, so the default runs are unchanged
ENVIRONMENT::Parameters base_parameters = FACTORY::nominal_parameters;

void init_environment(DEVICE& device, ENVIRONMENT& env){
    rlt::init(device, env);
    env.parameters = base_parameters;
}

// RLtools stops training silently past STEP_LIMIT: keep it above any step count passed on the command line
constexpr TI STEP_LIMIT_MAX = 10000001;
// L2F keeps every transition of its 300k-step run (replay buffer = step limit). 1M transitions take about 1.4 GB here
#ifndef WHOOP_REPLAY_BUFFER
#define WHOOP_REPLAY_BUFFER 1000000
#endif
constexpr TI BATCH = 256;
constexpr TI CRITIC_INTERVAL = 10;
constexpr TI ACTOR_INTERVAL = 20;
constexpr TI WARMUP_CRITIC = 15000;
constexpr TI WARMUP_ACTOR = 30000;
constexpr T LEARNING_RATE = 1e-3;
constexpr T WEIGHT_DECAY = 1e-4;

struct OPTIMIZER_PARAMETERS_L2F: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    // learning-to-fly src/config/actor_and_critic.h: Adam 1e-3, weight decay 1e-4 on every layer
    static constexpr T ALPHA = LEARNING_RATE;
    static constexpr bool ENABLE_WEIGHT_DECAY = true;
    static constexpr T WEIGHT_DECAY = ::WEIGHT_DECAY;
    static constexpr T WEIGHT_DECAY_INPUT = ::WEIGHT_DECAY;
    static constexpr T WEIGHT_DECAY_OUTPUT = ::WEIGHT_DECAY;
};

#if WHOOP_TD3
// TD3 as the released L2F code: critic targets updated with the critics, actor target with the actor, target-action
// noise 0.2 clipped at 0.5, Gaussian exploration noise 0.1. The paper's runner acts with the (untrained) actor plus noise
// from the start; RLtools' loop acts uniformly at random for N_WARMUP_STEPS, set here to the critic warmup as for SAC
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    struct TD3_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<TYPE_POLICY, TI>{
        static constexpr T GAMMA = 0.99;
        static constexpr TI ACTOR_BATCH_SIZE = BATCH;
        static constexpr TI CRITIC_BATCH_SIZE = BATCH;
        static constexpr TI CRITIC_TRAINING_INTERVAL = CRITIC_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = ACTOR_INTERVAL;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = CRITIC_INTERVAL;
        static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = ACTOR_INTERVAL;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
    };
    static constexpr TI N_WARMUP_STEPS = WARMUP_CRITIC;
    static constexpr TI N_WARMUP_STEPS_CRITIC = WARMUP_CRITIC;
    static constexpr TI N_WARMUP_STEPS_ACTOR = WARMUP_ACTOR;
    static constexpr TI STEP_LIMIT = STEP_LIMIT_MAX;
    static constexpr TI REPLAY_BUFFER_CAP = WHOOP_REPLAY_BUFFER;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI ACTOR_NUM_LAYERS = 3;  // two hidden layers
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr T EXPLORATION_NOISE = 0.1;
    using OPTIMIZER_PARAMETERS = OPTIMIZER_PARAMETERS_L2F;
};
using LOOP_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
#else
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr T GAMMA = 0.99;
        static constexpr TI ACTOR_BATCH_SIZE = BATCH;
        static constexpr TI CRITIC_BATCH_SIZE = BATCH;
        static constexpr TI CRITIC_TRAINING_INTERVAL = CRITIC_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = ACTOR_INTERVAL;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = CRITIC_INTERVAL;
        // entropy: RLtools defaults (initial coefficient 0.5, adapted toward a target entropy of -4)
    };
    // uniform random actions until the critics start training
    static constexpr TI N_WARMUP_STEPS = WARMUP_CRITIC;
    static constexpr TI N_WARMUP_STEPS_CRITIC = WARMUP_CRITIC;
    static constexpr TI N_WARMUP_STEPS_ACTOR = WARMUP_ACTOR;
    static constexpr TI STEP_LIMIT = STEP_LIMIT_MAX;
    static constexpr TI REPLAY_BUFFER_CAP = WHOOP_REPLAY_BUFFER;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI ACTOR_NUM_LAYERS = 3;  // two hidden layers
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    using ACTOR_OPTIMIZER_PARAMETERS = OPTIMIZER_PARAMETERS_L2F;
    using CRITIC_OPTIMIZER_PARAMETERS = OPTIMIZER_PARAMETERS_L2F;
    struct ALPHA_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
        static constexpr T ALPHA = LEARNING_RATE;
    };
};
using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
#endif
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;
static_assert(LOOP_CORE_PARAMETERS::N_ENVIRONMENTS == 1, "one whoop, as L2F");
// the actor for one observation at a time, without gradients: what is exported and what the scenario flies
using ACTOR_FORWARD = LOOP_CONFIG::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;

template <typename BUFFER>
TI filled(const BUFFER& rb){ return rb.full ? BUFFER::CAPACITY : rb.position; }

// ----- curriculum: L2F src/steps/curriculum.h ---------------------------------------------------------------------------
template <typename RUNNER>
void apply_reward(DEVICE& device, RUNNER& runner, const whoop::Reward<T>& reward){
    // the new weights for the next episodes and for the current one (L2F changes them in place), then every stored
    // reward recomputed with them from the critic's observation of the state before the step
    auto& env = rlt::get(runner.envs, 0, 0);
    auto& parameters = rlt::get(runner.env_parameters, 0, 0);
    env.parameters.mdp.reward = reward;
    parameters.mdp.reward = reward;
    T baseline = whoop::action_baseline(env.parameters.dynamics.hovering_throttle_relative);
    // the thrust curve and mass are not randomized, so one power model holds for every stored transition
    auto power = whoop::rotor_power_model(env.parameters.dynamics);
    auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    constexpr TI N = ENVIRONMENT::ObservationPrivileged::DIM;
    T o[N], a[ENVIRONMENT::ACTION_DIM];
    for(TI i = 0; i < filled(rb); i++){
        for(TI j = 0; j < N; j++) o[j] = rlt::get(rb.observations_privileged, i, j);
        for(TI j = 0; j < ENVIRONMENT::ACTION_DIM; j++) a[j] = rlt::get(rb.actions, i, j);
        bool terminated = rlt::get(rb.terminated, i, 0) > T(0.5);
        rlt::set(rb.rewards, i, 0, whoop::reward_from_privileged_observation(reward, o, a, baseline, power, terminated));
    }
}

// ----- training statistics from the replay buffer -----------------------------------------------------------------------
struct Window{ double episode_length = 0, episode_return = 0, episodes = 0; };

template <typename RUNNER>
Window window(const RUNNER& runner, TI n){
    // length and return of the episodes that ended within the last n transitions (the first, partial one left out)
    Window w;
    const auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    using BUFFER = std::decay_t<decltype(rb)>;
    TI count = std::min(n, filled(rb));
    double steps = 0, rewards = 0, lengths = 0, returns = 0;
    bool started = false;
    for(TI k = count; k-- > 0;){  // oldest first
        TI i = (rb.position + BUFFER::CAPACITY - 1 - k) % BUFFER::CAPACITY;
        steps += 1;
        rewards += rlt::get(rb.rewards, i, 0);
        if(rlt::get(rb.truncated, i, 0) > T(0.5)){
            if(started){ lengths += steps; returns += rewards; w.episodes += 1; }
            started = true;
            steps = 0;
            rewards = 0;
        }
    }
    if(w.episodes > 0){
        w.episode_length = lengths / w.episodes;
        w.episode_return = returns / w.episodes;
    }
    return w;
}

// ----- check that the RLtools runner uses the whoop samplers ------------------------------------------------------------
template <typename RUNNER>
std::string runner_checks(const RUNNER& runner){
    // the runner reaches whoop::sample_initial_state and whoop::sample_initial_parameters through argument-dependent
    // lookup only; a mistake there would still compile and fall back silently to L2F's own samplers. The episode starts
    // in the replay buffer must show attitudes beyond 57 deg (the RLtools sampler draws at most 1 rad), and the
    // velocity task's references speeds of 1-4 m/s (the L2F Lissajous default stays below about 1.4 m/s)
    const auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    TI n = filled(rb), starts = 0, tilted = 0, cut_steps = 0, cut_episodes = 0;
    bool cut_in_episode = false;
    double max_reference_speed = 0, max_reference_vz = 0;
    const double cos_57 = std::cos(57.3 / 180 * 3.14159265358979323846);
    // the most recent command of the action history comes first in its block at the end of the actor observation; a
    // cut step writes exactly -1 for all four motors there, which the squashed policy and the uniform warmup never do
    constexpr TI LAST_ACTION = ENVIRONMENT::Observation::DIM - FACTORY::ACTION_HISTORY_LENGTH * ENVIRONMENT::ACTION_DIM;
    for(TI i = 0; i < n; i++){
        bool start = i == 0 || rlt::get(rb.truncated, i - 1, 0) > T(0.5);
        if(start){
            starts++;
            tilted += rlt::get(rb.observations_privileged, i, 11) < cos_57;  // R_zz = cos(tilt)
            cut_in_episode = false;
        }
        bool cut = true;
        for(TI j = 0; j < ENVIRONMENT::ACTION_DIM; j++) cut &= rlt::get(rb.observations, i, LAST_ACTION + j) == T(-1);
        if(cut && !start){
            cut_steps++;
            if(!cut_in_episode) cut_episodes++;
            cut_in_episode = true;
        }
        if constexpr(WHOOP_TASK_VELOCITY){
            double v_ref[3];
            for(TI j = 0; j < 3; j++) v_ref[j] = rlt::get(rb.observations_privileged, i, 18 + j) - rlt::get(rb.observations_privileged, i, 12 + j);
            max_reference_speed = std::max(max_reference_speed, std::sqrt(v_ref[0] * v_ref[0] + v_ref[1] * v_ref[1] + v_ref[2] * v_ref[2]));
            max_reference_vz = std::max(max_reference_vz, std::abs(v_ref[2]));
        }
    }
    double tilted_fraction = starts > 0 ? double(tilted) / starts : 0;
    const double speed_max = base_parameters.velocity_task.speed_max;
    bool ok = tilted_fraction > 0.1 && (!WHOOP_TASK_VELOCITY || (max_reference_speed > 2 && max_reference_speed < speed_max + 0.01 && max_reference_vz < 1e-3));
    std::ostringstream json;
    json << "{\n  \"episode_starts\": " << starts << ",\n  \"starts_tilted_above_57_deg\": " << tilted_fraction
         << ",\n  \"max_reference_speed\": " << max_reference_speed << ",\n  \"max_reference_vertical_speed\": " << max_reference_vz
         << ",\n  \"episodes_with_motor_cut\": " << cut_episodes << ",\n  \"motor_cut_steps\": " << cut_steps
         << ",\n  \"whoop_samplers_used\": " << (ok ? "true" : "false") << "\n}\n";
    return json.str();
}

// ----- evaluation on fixed test flights ---------------------------------------------------------------------------------
constexpr TI N_TESTS = 100;
constexpr TI N_RECORDED = 10;  // test flights written to flights.csv

struct TestSet{
    std::vector<ENVIRONMENT::Parameters> parameters;  // the velocity task's reference is part of the parameters
    std::vector<ENVIRONMENT::State> starts;
};

// note: whoop::sample_initial_parameters, whoop::sample_initial_state and whoop::reward are called by name here. The
// RLtools runner finds them by argument-dependent lookup, but a call qualified with rlt:: would bypass them
TestSet test_set(DEVICE& device, TI seed, bool randomization){
    // sampled from the training distribution with a fixed seed, so that every run is tested on the same flights. No
    // motor cuts; the sampler draws random numbers for the randomization only when it is on, so the nominal test set
    // has the same starts and references whatever the training options
    TestSet set;
    set.parameters.resize(N_TESTS);
    set.starts.resize(N_TESTS);
    ENVIRONMENT env;
    init_environment(device, env);
    env.parameters.randomization.enabled = randomization;
    env.parameters.motor_cut.probability = 0;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    for(TI k = 0; k < N_TESTS; k++){
        whoop::sample_initial_parameters(device, env, set.parameters[k], rng);
        whoop::sample_initial_state(device, env, set.parameters[k], set.starts[k], rng);
    }
    rlt::free(device, rng);
    return set;
}

struct Summary{
    double success_rate = 0, position_rmse = 0, position_rmse_last_half = 0, velocity_rmse = 0, velocity_rmse_last_half = 0;
    double mean_return = 0, mean_length = 0, plate_force_to_weight = 0, max_plate_force_to_weight = 0, mean_action_change = 0;
    // over every step flown (whoop/metrics.h): the world-z plate force and the total rotor thrust, over the weight
    double plate_lift_to_weight = 0, rotor_thrust_to_weight = 0;
};

double norm3(const T* v){ return std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]); }

const double DEG = 180 / 3.14159265358979323846;

struct Angles{ double roll, pitch, yaw; };  // ZYX, degrees
Angles euler_zyx(const T* q){
    Angles a;
    a.roll = std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * DEG;
    a.pitch = std::asin(std::clamp(2 * (q[0] * q[2] - q[3] * q[1]), T(-1), T(1))) * DEG;
    a.yaw = std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])) * DEG;
    return a;
}

double wrap180(double a){
    while(a > 180) a -= 360;
    while(a <= -180) a += 360;
    return a;
}

double yaw_to_path(const T* q, const T* reference_velocity){
    // yaw of the body x axis relative to the direction of the reference velocity (the path), in (-180, 180] deg; 0 when
    // the reference is at rest. 0 or 180: the plate edge-on to the path, +-90: face-on
    double heading = std::atan2(double(reference_velocity[1]), double(reference_velocity[0])) * DEG;
    return wrap180(euler_zyx(q).yaw - heading);
}

double crab_angle(double yaw_to_path_deg){
    // the same folded into [0, 90] deg: the plate is symmetric front to back
    double c = std::abs(wrap180(yaw_to_path_deg));
    return c > 90 ? 180 - c : c;
}

// a template only so that the velocity-task branch is not compiled for hover
template <typename PARAMETERS, typename STATE>
void desired_state(DEVICE& device, ENVIRONMENT& env, PARAMETERS& p, const STATE& state, STATE& reference, RNG& rng){
    // L2F's reference, except after its last step: L2F's trajectory then runs backwards (its velocity changes sign).
    // An episode ends there, so only the state after the last step sees it, but that one step adds an error of twice
    // the speed; for the errors and the plots the reference holds the end of the line instead
    rlt::get_desired_state(device, env, p, state, reference, rng);
    if constexpr(WHOOP_TASK_VELOCITY){
        using TRAJECTORY = decltype(p.trajectory);
        if(state.trajectory_step >= TRAJECTORY::LENGTH){
            const auto& last = p.trajectory.steps[TRAJECTORY::LENGTH - 1];
            T ahead = (state.trajectory_step - (TRAJECTORY::LENGTH - 1)) * p.integration.dt;
            for(TI i = 0; i < 3; i++){
                reference.position[i] = last.position[i] + last.linear_velocity[i] * ahead;
                reference.linear_velocity[i] = last.linear_velocity[i];
            }
        }
    }
}

// flights.csv: one row per simulation step, the state after the step (the scenario adds a motors_cut column)
// the plate force is in the body frame; plate_lift_to_weight, rotor_thrust_to_weight: whoop/metrics.h; yaw_to_path_deg:
// body x relative to the reference velocity (0 when the reference is at rest)
const char* FLIGHTS_HEADER = "flight,t,x,y,z,vx,vy,vz,roll_deg,pitch_deg,yaw_deg,reference_x,reference_y,reference_z,reference_vx,reference_vy,reference_vz,"
                             "command_1,command_2,command_3,command_4,rotor_1,rotor_2,rotor_3,rotor_4,plate_fx,plate_fy,plate_fz,plate_alpha_deg,airspeed,terminated,"
                             "plate_lift_to_weight,rotor_thrust_to_weight,yaw_to_path_deg";

void write_flight_row(std::ostream& out, TI flight, double t, const ENVIRONMENT::State& next, const ENVIRONMENT::State& reference, const T command[4],
                      const whoop::FlightMetrics<T>& metrics, bool terminated){
    // command: the motor command in [0, 1] that was applied during the step
    Angles a = euler_zyx(next.orientation);
    const auto& aero = metrics.aero;
    out << flight << "," << t;
    for(TI i = 0; i < 3; i++) out << "," << next.position[i];
    for(TI i = 0; i < 3; i++) out << "," << next.linear_velocity[i];
    out << "," << a.roll << "," << a.pitch << "," << a.yaw;
    for(TI i = 0; i < 3; i++) out << "," << reference.position[i];
    for(TI i = 0; i < 3; i++) out << "," << reference.linear_velocity[i];
    for(TI i = 0; i < 4; i++) out << "," << command[i];
    for(TI i = 0; i < 4; i++) out << "," << next.rpm[i];
    for(TI i = 0; i < 3; i++) out << "," << aero.force[i];
    out << "," << aero.alpha * DEG << "," << aero.airspeed << "," << terminated << "," << metrics.plate_lift_to_weight << ","
        << metrics.rotor_thrust_to_weight << "," << yaw_to_path(next.orientation, reference.linear_velocity);
}

template <typename ACTOR>
Summary evaluate(DEVICE& device, const ACTOR& actor, TestSet& tests, RNG& rng, const std::filesystem::path& flights_csv = {}){
    // the deterministic actor (mean action) from every test start for a full episode, with the sensor noise of
    // training. A flight succeeds if it is not terminated (|p - p_ref| within the threshold on every axis).
    // RMS errors over the completed flights, over the whole flight and over its second half
    using EVAL_ACTOR = typename ACTOR::template CHANGE_BATCH_SIZE<TI, N_TESTS>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    typename EVAL_ACTOR::template State<true> actor_state;
    typename EVAL_ACTOR::template Buffer<true> actor_buffer;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_TESTS, ENVIRONMENT::Observation::DIM>> observations;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_TESTS, ENVIRONMENT::ACTION_DIM>> actions;
    rlt::malloc(device, actor_state);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, observations);
    rlt::malloc(device, actions);
    rlt::reset(device, actor, actor_state, rng);

    ENVIRONMENT env;
    init_environment(device, env);
    std::vector<ENVIRONMENT::State> states = tests.starts;
    std::vector<bool> done(N_TESTS, false), failed(N_TESTS, false);
    std::vector<double> returns(N_TESTS, 0), change(N_TESTS, 0);
    std::vector<std::vector<std::array<double, 2>>> errors(N_TESTS);  // position and velocity error per step
    double plate_force_sum = 0, plate_force_max = 0, plate_samples = 0;
    double lift_sum = 0, thrust_sum = 0, samples = 0;
    std::vector<std::array<T, 4>> previous(N_TESTS);
    for(TI k = 0; k < N_TESTS; k++){
        for(TI i = 0; i < 4; i++) previous[k][i] = whoop::action_baseline(tests.parameters[k].dynamics.hovering_throttle_relative);
    }
    std::ofstream flights;
    if(!flights_csv.empty()){
        flights.open(flights_csv);
        flights << FLIGHTS_HEADER << "\n";
    }
    for(TI step = 0; step < ENVIRONMENT::EPISODE_STEP_LIMIT; step++){
        for(TI k = 0; k < N_TESTS; k++){
            auto row = rlt::view(device, observations, rlt::matrix::ViewSpec<1, ENVIRONMENT::Observation::DIM>{}, k, 0);
            rlt::observe(device, env, tests.parameters[k], states[k], typename ENVIRONMENT::Observation{}, row, rng);
        }
        auto input = rlt::to_tensor(device, observations);
        auto output = rlt::to_tensor(device, actions);
        rlt::evaluate_step(device, actor, input, actor_state, output, actor_buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for(TI k = 0; k < N_TESTS; k++){
            if(done[k]) continue;
            auto& p = tests.parameters[k];
            auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, ENVIRONMENT::ACTION_DIM>{}, k, 0);
            for(TI i = 0; i < 4; i++) rlt::set(action, 0, i, std::clamp(rlt::get(action, 0, i), T(-1), T(1)));
            ENVIRONMENT::State next;
            whoop::step(device, env, p, states[k], action, next, rng);
            returns[k] += whoop::reward(device, env, p, states[k], action, next, rng);
            bool terminated = rlt::terminated(device, env, p, next, rng);
            ENVIRONMENT::State reference;
            desired_state(device, env, p, next, reference, rng);
            T e_p[3], e_v[3];
            for(TI i = 0; i < 3; i++){
                e_p[i] = next.position[i] - reference.position[i];
                e_v[i] = next.linear_velocity[i] - reference.linear_velocity[i];
            }
            errors[k].push_back({norm3(e_p), norm3(e_v)});
            double c = 0;
            for(TI i = 0; i < 4; i++){ c += std::abs(rlt::get(action, 0, i) - previous[k][i]); previous[k][i] = rlt::get(action, 0, i); }
            change[k] += c / 4;
            whoop::FlightMetrics<T> metrics = whoop::flight_metrics(device, p, next);
            if(p.plate.area > 0){
                double ratio = norm3(metrics.aero.force) / (p.dynamics.mass * 9.81);
                plate_force_sum += ratio;
                plate_force_max = std::max(plate_force_max, ratio);
                plate_samples += 1;
            }
            lift_sum += metrics.plate_lift_to_weight;
            thrust_sum += metrics.rotor_thrust_to_weight;
            samples += 1;
            if(k < N_RECORDED && flights.is_open()){
                T command[4];
                for(TI i = 0; i < 4; i++) command[i] = (rlt::get(action, 0, i) + 1) / 2;  // command in [0, 1]
                write_flight_row(flights, k, (step + 1) * p.integration.dt, next, reference, command, metrics, terminated);
                flights << "\n";
            }
            states[k] = next;
            if(terminated){ done[k] = true; failed[k] = true; }
        }
    }
    Summary s;
    int complete = 0;
    for(TI k = 0; k < N_TESTS; k++){
        s.mean_return += returns[k] / N_TESTS;
        s.mean_length += double(errors[k].size()) / N_TESTS;
        if(failed[k]) continue;
        complete++;
        size_t n = errors[k].size(), half = n / 2;
        double sp = 0, sv = 0, sp2 = 0, sv2 = 0;
        for(size_t t = 0; t < n; t++){
            sp += errors[k][t][0] * errors[k][t][0];
            sv += errors[k][t][1] * errors[k][t][1];
            if(t >= half){ sp2 += errors[k][t][0] * errors[k][t][0]; sv2 += errors[k][t][1] * errors[k][t][1]; }
        }
        s.position_rmse += std::sqrt(sp / n);
        s.velocity_rmse += std::sqrt(sv / n);
        s.position_rmse_last_half += std::sqrt(sp2 / (n - half));
        s.velocity_rmse_last_half += std::sqrt(sv2 / (n - half));
        s.mean_action_change += change[k] / n;
    }
    s.success_rate = complete / double(N_TESTS);
    double c = std::max(complete, 1);
    s.position_rmse /= c;
    s.velocity_rmse /= c;
    s.position_rmse_last_half /= c;
    s.velocity_rmse_last_half /= c;
    s.mean_action_change /= c;
    if(complete == 0){
        s.position_rmse = s.velocity_rmse = s.position_rmse_last_half = s.velocity_rmse_last_half = std::numeric_limits<double>::quiet_NaN();
    }
    s.plate_force_to_weight = plate_samples > 0 ? plate_force_sum / plate_samples : 0;
    s.max_plate_force_to_weight = plate_force_max;
    s.plate_lift_to_weight = samples > 0 ? lift_sum / samples : 0;
    s.rotor_thrust_to_weight = samples > 0 ? thrust_sum / samples : 0;
    rlt::free(device, actor_state);
    rlt::free(device, actor_buffer);
    rlt::free(device, observations);
    rlt::free(device, actions);
    return s;
}

std::string number(double v){
    std::ostringstream o;
    o << std::setprecision(8);
    if(std::isfinite(v)) o << v; else o << "null";
    return o.str();
}

std::string to_json(const Summary& s, const std::string& sweep = ""){
    // sweep: the speed sweep's JSON object, added as "speed_sweep" when given
    std::ostringstream json;
    json << "{\n  \"test_flights\": " << N_TESTS << ",\n  \"test_seed\": 12345,\n  \"episode_steps\": " << ENVIRONMENT::EPISODE_STEP_LIMIT
         << ",\n  \"success_rate\": " << number(s.success_rate)
         << ",\n  \"position_rmse\": " << number(s.position_rmse) << ",\n  \"position_rmse_last_half\": " << number(s.position_rmse_last_half)
         << ",\n  \"velocity_rmse\": " << number(s.velocity_rmse) << ",\n  \"velocity_rmse_last_half\": " << number(s.velocity_rmse_last_half)
         << ",\n  \"mean_return\": " << number(s.mean_return) << ",\n  \"mean_episode_length\": " << number(s.mean_length)
         << ",\n  \"mean_action_change\": " << number(s.mean_action_change)
         << ",\n  \"mean_plate_force_to_weight\": " << number(s.plate_force_to_weight)
         << ",\n  \"max_plate_force_to_weight\": " << number(s.max_plate_force_to_weight)
         << ",\n  \"mean_plate_lift_to_weight\": " << number(s.plate_lift_to_weight)
         << ",\n  \"mean_rotor_thrust_to_weight\": " << number(s.rotor_thrust_to_weight);
    if(!sweep.empty()) json << ",\n  \"speed_sweep\": " << sweep;
    json << "\n}\n";
    return json.str();
}

// ----- speed sweep: the same references for every policy, to compare thrust and plate lift at given speeds ------------
const std::vector<T> SWEEP_SPEEDS = {2, 2.5, 3, 3.5, 4};
constexpr TI SWEEP_HEADINGS = 8;      // 0, 45, ..., 315 deg
constexpr double SWEEP_STEADY_FROM = 2;  // [s] averages over t = 2-5 s: after the 1 s ramp and the transient

template <typename ACTOR>
void save_actor(DEVICE& device, const ACTOR& actor, TestSet& tests, const std::filesystem::path& path, const std::string& name){
    // as the L2F checkpoints (learning-to-fly src/steps/checkpoint.h): the actor with batch size 1 as
    // rl_tools::checkpoint::actor, and an observation with the action the actor gives for it, to check a deployment
    using FORWARD = typename ACTOR::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    FORWARD forward;
    typename FORWARD::template State<true> state;
    typename FORWARD::template Buffer<true> buffer;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(device, forward);
    rlt::malloc(device, state);
    rlt::malloc(device, buffer);
    rlt::malloc(device, observation);
    rlt::malloc(device, action);
    rlt::copy(device, device, actor, forward);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    ENVIRONMENT env;
    init_environment(device, env);
    rlt::observe(device, env, tests.parameters[0], tests.starts[0], typename ENVIRONMENT::Observation{}, observation, rng);
    rlt::reset(device, forward, state, rng);
    auto input = rlt::to_tensor(device, observation);
    auto output = rlt::to_tensor(device, action);
    rlt::evaluate_step(device, forward, input, state, output, buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    std::ofstream out(path);
    out << rlt::save_code(device, forward, "rl_tools::checkpoint::actor", true);
    out << "\n" << rlt::save_code(device, observation, "rl_tools::checkpoint::observation", true);
    out << "\n" << rlt::save_code(device, action, "rl_tools::checkpoint::action", true);
    out << "\nnamespace rl_tools::checkpoint::meta{\n    char name[] = \"" << name << "\";\n}\n";
    rlt::free(device, forward);
    rlt::free(device, state);
    rlt::free(device, buffer);
    rlt::free(device, observation);
    rlt::free(device, action);
    rlt::free(device, rng);
}

// ----- the actor as a tar file (weights only), read back by the scenario mode ---------------------------------------------
template <typename ACTOR>
void save_actor_tar(DEVICE& device, const ACTOR& actor, const std::filesystem::path& path){
    ACTOR_FORWARD forward;
    rlt::malloc(device, forward);
    rlt::copy(device, device, actor, forward);
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root{"", &writer};
    auto group = rlt::create_group(device, root, "actor");
    rlt::save(device, forward, group);
    rlt::persist::backends::tar::finalize(device, writer);
    std::ofstream out(path, std::ios::binary);
    out.write(writer.buffer.data(), writer.buffer.size());
    rlt::free(device, forward);
}

void load_actor_tar(DEVICE& device, ACTOR_FORWARD& actor, const std::filesystem::path& path){
    std::ifstream file(path, std::ios::binary);
    if(!file) throw std::runtime_error("cannot read " + path.string());
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader;
    reader.data.data = data.data();
    reader.data.size = data.size();
    auto group = rlt::get_group(device, reader, "actor");
    if(!rlt::load(device, actor, group)) throw std::runtime_error("could not load the actor from " + path.string() + " (another variant?)");
}

// the deterministic action (mean of the policy) for one observation
struct Policy{
    ACTOR_FORWARD::State<true> state;
    ACTOR_FORWARD::Buffer<true> buffer;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    Policy(DEVICE& device, ACTOR_FORWARD& actor, RNG& rng){
        rlt::malloc(device, state);
        rlt::malloc(device, buffer);
        rlt::malloc(device, observation);
        rlt::malloc(device, action);
        rlt::reset(device, actor, state, rng);
    }
    void act(DEVICE& device, ACTOR_FORWARD& actor, RNG& rng){
        auto input = rlt::to_tensor(device, observation);
        auto output = rlt::to_tensor(device, action);
        rlt::evaluate_step(device, actor, input, state, output, buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for(TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++) rlt::set(action, 0, i, std::clamp(rlt::get(action, 0, i), T(-1), T(1)));
    }
    void free(DEVICE& device){
        rlt::free(device, state);
        rlt::free(device, buffer);
        rlt::free(device, observation);
        rlt::free(device, action);
    }
};

void level_start(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& p, ENVIRONMENT::State& s, RNG& rng, bool random_force){
    // L2F's sampler for the parts of the state it owns, then level at rest at the origin, with the rotors and the
    // action history at the hover command; without random_force the constant random force and torque are zero
    whoop::sample_initial_state(device, env, p, s, rng);
    for(TI i = 0; i < 3; i++){ s.position[i] = 0; s.linear_velocity[i] = 0; s.angular_velocity[i] = 0; }
    s.orientation[0] = 1; s.orientation[1] = s.orientation[2] = s.orientation[3] = 0;
    T hover = p.dynamics.hovering_throttle_relative;
    for(TI i = 0; i < 4; i++) s.rpm[i] = hover;
    for(TI h = 0; h < FACTORY::ACTION_HISTORY_LENGTH; h++) for(TI i = 0; i < 4; i++) s.action_history[h][i] = whoop::action_baseline(hover);
    if(!random_force){
        for(TI i = 0; i < 3; i++){ s.force[i] = 0; s.torque[i] = 0; }
    }
}

std::string speed_sweep(DEVICE& device, ACTOR_FORWARD& actor, const std::filesystem::path& flights_csv){
    // every speed of SWEEP_SPEEDS in SWEEP_HEADINGS headings: level start at rest at the origin, the velocity-task
    // reference (1 s ramp), nominal aerodynamics of this run, no random force or torque (so that the thrust measures the
    // aerodynamics alone, not a disturbance), sensor noise as in training. Averages over t >= SWEEP_STEADY_FROM of the
    // flights that are not terminated. Written step by step to flights_csv (flight = speed index * 8 + heading index)
    if constexpr(!WHOOP_TASK_VELOCITY){
        return "";
    }
    else{
        std::ofstream flights(flights_csv);
        flights << FLIGHTS_HEADER << ",speed,heading_deg\n";
        std::ostringstream per_flight, per_speed;
        per_flight << std::setprecision(5);
        per_speed << std::setprecision(5);
        const double dt = base_parameters.integration.dt;
        for(TI si = 0; si < SWEEP_SPEEDS.size(); si++){
            double n_ok = 0, thrust = 0, lift = 0, force = 0, crab = 0, roll = 0, pitch = 0, alpha = 0, e_p = 0, e_v = 0, thrust_newton = 0;
            for(TI hi = 0; hi < SWEEP_HEADINGS; hi++){
                const TI flight = si * SWEEP_HEADINGS + hi;
                const T heading = T(2 * 3.14159265358979323846 * hi / SWEEP_HEADINGS);
                RNG rng;
                rlt::malloc(device, rng);
                rlt::init(device, rng, 1000 + flight);
                ENVIRONMENT env;
                init_environment(device, env);
                ENVIRONMENT::Parameters p = env.parameters;
                p.randomization.enabled = false;
                p.motor_cut.probability = 0;
                p.motor_cut.start_step = p.motor_cut.end_step = 0;
                whoop::set_velocity_reference(device, p, heading, SWEEP_SPEEDS[si]);
                ENVIRONMENT::State s;
                level_start(device, env, p, s, rng, false);
                Policy policy(device, actor, rng);
                bool terminated = false;
                double n = 0, f_thrust = 0, f_lift = 0, f_force = 0, f_crab = 0, f_roll = 0, f_pitch = 0, f_alpha = 0, f_ep = 0, f_ev = 0;
                for(TI step = 0; step < ENVIRONMENT::EPISODE_STEP_LIMIT && !terminated; step++){
                    rlt::observe(device, env, p, s, typename ENVIRONMENT::Observation{}, policy.observation, rng);
                    policy.act(device, actor, rng);
                    ENVIRONMENT::State next, reference;
                    whoop::step(device, env, p, s, policy.action, next, rng);
                    terminated = rlt::terminated(device, env, p, next, rng);
                    desired_state(device, env, p, next, reference, rng);
                    whoop::FlightMetrics<T> m = whoop::flight_metrics(device, p, next);
                    T command[4];
                    for(TI i = 0; i < 4; i++) command[i] = (rlt::get(policy.action, 0, i) + 1) / 2;
                    double t = (step + 1) * dt;
                    write_flight_row(flights, flight, t, next, reference, command, m, terminated);
                    flights << "," << SWEEP_SPEEDS[si] << "," << heading * DEG << "\n";
                    if(t >= SWEEP_STEADY_FROM - 1e-9){
                        Angles a = euler_zyx(next.orientation);
                        T e1[3], e2[3];
                        for(TI i = 0; i < 3; i++){ e1[i] = next.position[i] - reference.position[i]; e2[i] = next.linear_velocity[i] - reference.linear_velocity[i]; }
                        n += 1;
                        f_thrust += m.rotor_thrust_to_weight;
                        f_lift += m.plate_lift_to_weight;
                        f_force += m.plate_force_to_weight;
                        f_crab += crab_angle(yaw_to_path(next.orientation, reference.linear_velocity));
                        f_roll += std::abs(a.roll);
                        f_pitch += std::abs(a.pitch);
                        f_alpha += m.aero.alpha * DEG;
                        f_ep += norm3(e1) * norm3(e1);
                        f_ev += norm3(e2) * norm3(e2);
                    }
                    s = next;
                }
                policy.free(device);
                rlt::free(device, rng);
                n = std::max(n, 1.0);
                const double weight = p.dynamics.mass * 9.81;
                per_flight << (flight == 0 ? "" : ",") << "\n      {\"flight\": " << flight << ", \"speed\": " << SWEEP_SPEEDS[si] << ", \"heading_deg\": " << heading * DEG
                           << ", \"success\": " << (terminated ? "false" : "true") << ", \"rotor_thrust_to_weight\": " << f_thrust / n
                           << ", \"plate_lift_to_weight\": " << f_lift / n << ", \"plate_force_to_weight\": " << f_force / n << ", \"crab_angle_deg\": " << f_crab / n
                           << ", \"abs_roll_deg\": " << f_roll / n << ", \"abs_pitch_deg\": " << f_pitch / n << ", \"plate_alpha_deg\": " << f_alpha / n
                           << ", \"position_rmse\": " << std::sqrt(f_ep / n) << ", \"velocity_rmse\": " << std::sqrt(f_ev / n) << "}";
                if(!terminated){
                    n_ok += 1;
                    thrust += f_thrust / n; lift += f_lift / n; force += f_force / n; crab += f_crab / n; roll += f_roll / n; pitch += f_pitch / n;
                    alpha += f_alpha / n; e_p += std::sqrt(f_ep / n); e_v += std::sqrt(f_ev / n); thrust_newton += f_thrust / n * weight;
                }
            }
            double c = std::max(n_ok, 1.0);
            auto value = [&](double v){ return n_ok > 0 ? number(v / c) : std::string("null"); };
            per_speed << (si == 0 ? "" : ",") << "\n      {\"speed\": " << SWEEP_SPEEDS[si] << ", \"success\": " << n_ok / SWEEP_HEADINGS
                      << ", \"rotor_thrust_to_weight\": " << value(thrust) << ", \"rotor_thrust_newton\": " << value(thrust_newton)
                      << ", \"plate_lift_to_weight\": " << value(lift) << ", \"plate_force_to_weight\": " << value(force)
                      << ", \"crab_angle_deg\": " << value(crab) << ", \"abs_roll_deg\": " << value(roll) << ", \"abs_pitch_deg\": " << value(pitch)
                      << ", \"plate_alpha_deg\": " << value(alpha) << ", \"position_rmse\": " << value(e_p) << ", \"velocity_rmse\": " << value(e_v) << "}";
        }
        std::ostringstream json;
        json << "{\n    \"headings\": " << SWEEP_HEADINGS << ", \"steady_from_s\": " << SWEEP_STEADY_FROM << ", \"random_force\": false,"
             << "\n    \"per_speed\": [" << per_speed.str() << "\n    ],\n    \"flights\": [" << per_flight.str() << "\n    ]\n  }";
        return json.str();
    }
}

// ----- motor-cut scenario -----------------------------------------------------------------------------------------------
struct ScenarioOptions{
    TI seeds = 5;
    T speed = 3;            // [m/s] reference along world x, reached after the 1 s ramp of the velocity task
    T cut_time = 1.5;       // [s]
    T cut_duration = 0.4;   // [s]
    T duration = 5;         // [s]
};

std::string motor_cut_scenario(DEVICE& device, ACTOR_FORWARD& actor, const ScenarioOptions& o, const std::filesystem::path& flights_csv){
    // fly the velocity-task policy forward, cut every motor for a while, then give control back and measure what is
    // lost. There is no ground in L2F: the altitude lost is relative to the height at the cut (the reference is z = 0)
    if constexpr(!WHOOP_TASK_VELOCITY){
        throw std::runtime_error("the motor-cut scenario needs a velocity-task policy");
    }
    else{
        const double dt = FACTORY::nominal_parameters.integration.dt;
        const TI n_steps = (TI)std::lround(o.duration / dt);
        if(n_steps < 1 || n_steps > ENVIRONMENT::EPISODE_STEP_LIMIT) throw std::runtime_error("--duration must be within the 5 s episode");
        const TI cut_start = (TI)std::lround(o.cut_time / dt), cut_end = (TI)std::lround((o.cut_time + o.cut_duration) / dt);
        if(cut_end > n_steps) throw std::runtime_error("the cut must end before --duration");
        std::ofstream flights(flights_csv);
        flights << FLIGHTS_HEADER << ",motors_cut\n";
        std::ostringstream json;
        json << std::setprecision(5) << "{\n  \"speed\": " << o.speed << ",\n  \"cut_time\": " << o.cut_time << ",\n  \"cut_duration\": " << o.cut_duration
             << ",\n  \"duration\": " << o.duration << ",\n  \"plate_area\": " << base_parameters.plate.area
             << ",\n  \"plate_centroid_z\": " << base_parameters.plate.centroid[2] << ",\n  \"flights\": [";
        double lost_sum = 0;
        int recovered_count = 0;
        for(TI k = 0; k < o.seeds; k++){
            RNG rng;
            rlt::malloc(device, rng);
            rlt::init(device, rng, k);
            ENVIRONMENT env;
            init_environment(device, env);
            // nominal aerodynamics and no random cut: this flight's cut is set here
            ENVIRONMENT::Parameters p = env.parameters;
            p.randomization.enabled = false;
            p.motor_cut.probability = 0;
            whoop::set_velocity_reference(device, p, T(0), o.speed);
            p.motor_cut.start_step = cut_start;
            p.motor_cut.end_step = cut_end;
            // L2F's sampler gives the random force and torque of the seed; then level at rest at the origin, with the
            // rotors and the action history at the hover command
            ENVIRONMENT::State s;
            whoop::sample_initial_state(device, env, p, s, rng);
            for(TI i = 0; i < 3; i++){ s.position[i] = 0; s.linear_velocity[i] = 0; s.angular_velocity[i] = 0; }
            s.orientation[0] = 1; s.orientation[1] = s.orientation[2] = s.orientation[3] = 0;
            T hover = p.dynamics.hovering_throttle_relative;
            for(TI i = 0; i < 4; i++) s.rpm[i] = hover;
            for(TI h = 0; h < FACTORY::ACTION_HISTORY_LENGTH; h++) for(TI i = 0; i < 4; i++) s.action_history[h][i] = whoop::action_baseline(hover);
            Policy policy(device, actor, rng);
            const double weight = p.dynamics.mass * 9.81;
            double z_cut = 0, speed_cut = 0, z_min = 1e9, t_min = 0, t_arrest = -1, max_error = 0, max_tilt = 0;
            double plate_lift_cut = 0, plate_force_cut = 0, cut_samples = 0;
            double error_end = 0, speed_error_end = 0, max_error_last_second = 0;
            bool falling = false;
            for(TI step = 0; step < n_steps; step++){
                if(step == cut_start){
                    z_cut = s.position[2];
                    speed_cut = norm3(s.linear_velocity);
                }
                rlt::observe(device, env, p, s, typename ENVIRONMENT::Observation{}, policy.observation, rng);
                policy.act(device, actor, rng);
                bool cut = whoop::motors_cut(p, s);
                ENVIRONMENT::State next;
                whoop::step(device, env, p, s, policy.action, next, rng);
                ENVIRONMENT::State reference;
                desired_state(device, env, p, next, reference, rng);
                bool terminated = rlt::terminated(device, env, p, next, rng);
                whoop::FlightMetrics<T> metrics = whoop::flight_metrics(device, p, next);
                const auto& aero = metrics.aero;
                T command[4];
                for(TI i = 0; i < 4; i++) command[i] = cut ? T(0) : (rlt::get(policy.action, 0, i) + 1) / 2;
                double t = (step + 1) * dt;
                write_flight_row(flights, k, t, next, reference, command, metrics, terminated);
                flights << "," << cut << "\n";
                T e[3], e_v[3];
                for(TI i = 0; i < 3; i++){ e[i] = next.position[i] - reference.position[i]; e_v[i] = next.linear_velocity[i] - reference.linear_velocity[i]; }
                // tilt of the body z axis from the vertical: cos(tilt) = 1 - 2 (q_x^2 + q_y^2)
                double cos_tilt = 1 - 2 * (double(next.orientation[1]) * next.orientation[1] + double(next.orientation[2]) * next.orientation[2]);
                double tilt = std::acos(std::clamp(cos_tilt, -1.0, 1.0)) * 180 / 3.14159265358979323846;
                if(step >= cut_start){
                    if(next.position[2] < z_min){ z_min = next.position[2]; t_min = t; }
                    max_error = std::max(max_error, norm3(e));
                    max_tilt = std::max(max_tilt, tilt);
                    if(next.linear_velocity[2] < 0) falling = true;
                    // fall arrested: the first step after the cut at which the whoop stops going down
                    if(step >= cut_end && t_arrest < 0 && falling && next.linear_velocity[2] >= 0) t_arrest = t - (o.cut_time + o.cut_duration);
                }
                if(cut && p.plate.area > 0){
                    // the plate force in world axes: its vertical part is what holds the whoop up during the cut
                    T f_world[3];
                    rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(next.orientation, aero.force, f_world);
                    plate_lift_cut += f_world[2] / weight;
                    plate_force_cut += norm3(aero.force) / weight;
                    cut_samples += 1;
                }
                if(t > o.duration - 1 + 1e-9) max_error_last_second = std::max(max_error_last_second, norm3(e));
                error_end = norm3(e);
                speed_error_end = norm3(e_v);
                s = next;
            }
            // recovered: back within 0.5 m of the reference over the last second, and at its velocity (within
            // 0.5 m/s) at the end; the policies track the line to about 0.2-0.3 m RMS without a cut
            bool recovered = max_error_last_second < 0.5 && speed_error_end < 0.5;
            recovered_count += recovered;
            lost_sum += z_cut - z_min;
            json << (k == 0 ? "" : ",") << "\n    {\"seed\": " << k << ", \"z_at_cut\": " << z_cut << ", \"speed_at_cut\": " << speed_cut
                 << ", \"altitude_lost\": " << z_cut - z_min << ", \"lowest_z\": " << z_min << ", \"time_of_lowest_z\": " << t_min
                 << ", \"fall_arrested_after_cut_end\": " << t_arrest << ", \"max_position_error\": " << max_error << ", \"max_tilt_deg\": " << max_tilt
                 << ", \"mean_plate_lift_to_weight_during_cut\": " << (cut_samples > 0 ? plate_lift_cut / cut_samples : 0)
                 << ", \"mean_plate_force_to_weight_during_cut\": " << (cut_samples > 0 ? plate_force_cut / cut_samples : 0)
                 << ", \"max_position_error_last_second\": " << max_error_last_second << ", \"final_position_error\": " << error_end
                 << ", \"final_velocity_error\": " << speed_error_end << ", \"recovered\": " << (recovered ? "true" : "false") << "}";
            policy.free(device);
            rlt::free(device, rng);
        }
        json << "\n  ],\n  \"mean_altitude_lost\": " << lost_sum / o.seeds << ",\n  \"recovered\": " << recovered_count << ",\n  \"flights_total\": " << o.seeds << "\n}\n";
        return json.str();
    }
}

// ----- command line -----------------------------------------------------------------------------------------------------
struct Options{
    std::filesystem::path out_dir;
    TI seed = 0;
    TI steps = 300000;  // L2F: 300k
    TI eval_interval = 50000;
    bool curriculum = true;
    bool randomization = true;
    T motor_cuts = 0;  // fraction of the training episodes with a motor cut
    std::string scenario_actor;
    ScenarioOptions scenario;
    std::string evaluate_actor;
    // parameters of the run (see the top of the file)
    std::string params_file;
    bool plate_z_set = false;
    double plate_z = 0;
    bool plate_tilt_set = false;
    double plate_tilt_deg = 0;
    bool downwash_set = false;
    double downwash = 0;
    T thrust_cost = 0;
    bool free_yaw = false;
    T speed_min = FACTORY::velocity_task.speed_min, speed_max = FACTORY::velocity_task.speed_max;
};

Options parse(int argc, char** argv){
    Options o;
    o.out_dir = argv[1];
    int positional = 0;
    auto value = [&](int& i, const std::string& option){
        if(i + 1 >= argc) throw std::runtime_error(option + " needs a value");
        return std::string(argv[++i]);
    };
    for(int i = 2; i < argc; i++){
        std::string a = argv[i];
        if(a.rfind("--", 0) != 0){
            // the seed, then the number of steps
            if(positional == 0) o.seed = std::stoul(a);
            else if(positional == 1) o.steps = std::stoul(a);
            else throw std::runtime_error("unexpected argument " + a);
            positional++;
        }
        else if(a == "--eval-interval") o.eval_interval = std::stoul(value(i, a));
        else if(a == "--no-curriculum") o.curriculum = false;
        else if(a == "--no-randomization") o.randomization = false;
        else if(a == "--motor-cuts") o.motor_cuts = std::stof(value(i, a));
        else if(a == "--scenario") o.scenario_actor = value(i, a);
        else if(a == "--scenario-seeds") o.scenario.seeds = std::stoul(value(i, a));
        else if(a == "--speed") o.scenario.speed = std::stof(value(i, a));
        else if(a == "--cut-time") o.scenario.cut_time = std::stof(value(i, a));
        else if(a == "--cut-duration") o.scenario.cut_duration = std::stof(value(i, a));
        else if(a == "--duration") o.scenario.duration = std::stof(value(i, a));
        else if(a == "--evaluate") o.evaluate_actor = value(i, a);
        else if(a == "--params") o.params_file = value(i, a);
        else if(a == "--plate-z"){ o.plate_z = std::stod(value(i, a)); o.plate_z_set = true; }
        else if(a == "--plate-tilt"){ o.plate_tilt_deg = std::stod(value(i, a)); o.plate_tilt_set = true; }
        else if(a == "--downwash"){ o.downwash = std::stod(value(i, a)); o.downwash_set = true; }
        else if(a == "--thrust-cost") o.thrust_cost = std::stof(value(i, a));
        else if(a == "--free-yaw") o.free_yaw = true;
        else if(a == "--speed-min") o.speed_min = std::stof(value(i, a));
        else if(a == "--speed-max") o.speed_max = std::stof(value(i, a));
        else throw std::runtime_error("unknown option " + a);
    }
    if(o.thrust_cost < 0) throw std::runtime_error("--thrust-cost must be >= 0");
    if(!(o.speed_min > 0 && o.speed_min <= o.speed_max)) throw std::runtime_error("need 0 < --speed-min <= --speed-max");
    if(!o.scenario_actor.empty() && !o.evaluate_actor.empty()) throw std::runtime_error("--scenario and --evaluate are separate modes");
    if(o.steps >= LOOP_CORE_PARAMETERS::STEP_LIMIT) throw std::runtime_error("steps above the compiled STEP_LIMIT");
    if(o.eval_interval == 0) throw std::runtime_error("--eval-interval must be positive");
    if(o.motor_cuts < 0 || o.motor_cuts > 1) throw std::runtime_error("--motor-cuts must be in [0, 1]");
    if(o.motor_cuts > 0 && !WHOOP_TASK_VELOCITY) throw std::runtime_error("--motor-cuts needs the velocity task (the cut is timed by its reference)");
    return o;
}

void set_base_parameters(const Options& o){
    // the options that change the parameters of every environment of the run; without them base_parameters stays the
    // compiled nominal set
    if(!o.params_file.empty() || o.plate_z_set || o.plate_tilt_set || o.downwash_set){
        whoop::runtime::WhoopParams w = o.params_file.empty() ? whoop::runtime::compiled_params() : whoop::runtime::params_from_json_file(o.params_file);
        if(o.plate_z_set) w.plate.centroid[2] = o.plate_z;
        if(o.plate_tilt_set) whoop::runtime::tilt_plate(w, o.plate_tilt_deg);
        if(o.downwash_set) w.plate.downwash_factor = o.downwash;
        whoop::runtime::apply<T, TI, WHOOP_PLATE>(w, base_parameters);
    }
    base_parameters.mdp.reward.thrust = o.thrust_cost;
    base_parameters.mdp.reward.tilt_only = o.free_yaw;
    base_parameters.velocity_task.speed_min = o.speed_min;
    base_parameters.velocity_task.speed_max = o.speed_max;
}

std::string parameters_json(const Options& o){
    // the parameters of the run, for config.json
    const auto& b = base_parameters;
    std::ostringstream j;
    j << std::setprecision(10) << "\"params_file\": \"" << std::filesystem::path(o.params_file).filename().string() << "\""
      << ",\n  \"mass\": " << b.dynamics.mass << ",\n  \"hovering_throttle\": " << b.dynamics.hovering_throttle_relative
      << ",\n  \"rotor_thrust_coefficients\": [" << b.dynamics.rotor_thrust_coefficients[0][0] << ", " << b.dynamics.rotor_thrust_coefficients[0][1] << ", "
      << b.dynamics.rotor_thrust_coefficients[0][2] << "]"
      << ",\n  \"rotor_time_constants\": [" << b.dynamics.rotor_time_constants_rising[0] << ", " << b.dynamics.rotor_time_constants_falling[0] << "]"
      << ",\n  \"J_diagonal\": [" << b.dynamics.J[0][0] << ", " << b.dynamics.J[1][1] << ", " << b.dynamics.J[2][2] << "]"
      << ",\n  \"rotor_torque_constant\": " << b.dynamics.rotor_torque_constants[0]
      << ",\n  \"plate_area\": " << b.plate.area << ",\n  \"plate_centroid\": [" << b.plate.centroid[0] << ", " << b.plate.centroid[1] << ", " << b.plate.centroid[2] << "]"
      << ",\n  \"plate_normal\": [" << b.plate.normal[0] << ", " << b.plate.normal[1] << ", " << b.plate.normal[2] << "]"
      << ",\n  \"plate_chord_direction\": [" << b.plate.chord_direction[0] << ", " << b.plate.chord_direction[1] << ", " << b.plate.chord_direction[2] << "]"
      << ",\n  \"plate_edges\": [" << b.plate.chord_x << ", " << b.plate.height_z << "]"
      << ",\n  \"plate_downwash_factor\": " << b.plate.downwash_factor << ",\n  \"rotor_disc_area\": " << b.plate.rotor_disc_area
      << ",\n  \"plate_coefficients\": {\"CN90\": " << b.plate.CN90 << ", \"Kp_x\": " << b.plate.Kp_x << ", \"Kp_z\": " << b.plate.Kp_z << ", \"CT\": " << b.plate.CT
      << ", \"e0\": " << b.plate.e0 << "}"
      << ",\n  \"body_drag\": [" << b.body_drag[0] << ", " << b.body_drag[1] << ", " << b.body_drag[2] << "]"
      << ",\n  \"thrust_cost\": " << b.mdp.reward.thrust << ",\n  \"free_yaw\": " << (b.mdp.reward.tilt_only ? "true" : "false")
      << ",\n  \"velocity_task_speed\": [" << b.velocity_task.speed_min << ", " << b.velocity_task.speed_max << "]";
    return j.str();
}

int evaluate_mode(DEVICE& device, const Options& o, const std::string& program){
    // a trained actor on the test flights and the speed sweep, with the parameters of this run (nominal aerodynamics)
    std::ofstream(o.out_dir / "config.json") << "{\n  \"program\": \"" << program << "\",\n  \"mode\": \"evaluate\",\n  \"actor\": \""
        << std::filesystem::path(o.evaluate_actor).generic_string() << "\",\n  " << parameters_json(o) << "\n}\n";
    ACTOR_FORWARD actor;
    rlt::malloc(device, actor);
    load_actor_tar(device, actor, o.evaluate_actor);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);
    TestSet tests = test_set(device, 12345, false);
    Summary s = evaluate(device, actor, tests, rng, o.out_dir / "flights.csv");
    std::string sweep = speed_sweep(device, actor, o.out_dir / "sweep_flights.csv");
    std::string json = to_json(s, sweep);
    std::ofstream(o.out_dir / "evaluation.json") << json;
    std::cout << "success " << 100 * s.success_rate << " %, position RMSE (2nd half) " << s.position_rmse_last_half << " m, velocity RMSE (2nd half) "
              << s.velocity_rmse_last_half << " m/s, rotor thrust " << s.rotor_thrust_to_weight << " of the weight, plate lift " << s.plate_lift_to_weight << std::endl;
    rlt::free(device, rng);
    rlt::free(device, actor);
    return 0;
}

int scenario(DEVICE& device, const Options& o, const std::string& program){
    std::ofstream(o.out_dir / "config.json") << std::setprecision(10) << "{\n  \"program\": \"" << program << "\",\n  \"mode\": \"motor_cut_scenario\",\n  \"actor\": \""
        << std::filesystem::path(o.scenario_actor).generic_string() << "\",\n  \"seeds\": " << o.scenario.seeds << ",\n  \"speed\": " << o.scenario.speed
        << ",\n  \"cut_time\": " << o.scenario.cut_time << ",\n  \"cut_duration\": " << o.scenario.cut_duration << ",\n  \"duration\": " << o.scenario.duration
        << ",\n  \"randomization\": false,\n  " << parameters_json(o) << "\n}\n";
    ACTOR_FORWARD actor;
    rlt::malloc(device, actor);
    load_actor_tar(device, actor, o.scenario_actor);
    std::string json = motor_cut_scenario(device, actor, o.scenario, o.out_dir / "flights.csv");
    std::ofstream(o.out_dir / "scenario.json") << json;
    std::cout << json;
    rlt::free(device, actor);
    return 0;
}

int run(int argc, char** argv){
    const std::string program = std::string("whoop_train_") + (WHOOP_TD3 ? "td3" : "sac") + (WHOOP_TASK_VELOCITY ? "_velocity" : "_hover") + (WHOOP_PLATE ? "_plate" : "_noplate");
    if(argc < 2){
        std::cerr << "usage: " << program << " <out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum] [--no-randomization] [--motor-cuts P]\n"
                  << "       " << program << " <out_dir> --scenario <actor.tar> [--scenario-seeds N] [--speed V] [--cut-time S] [--cut-duration S] [--duration S]\n"
                  << "       " << program << " <out_dir> --evaluate <actor.tar>\n"
                  << "  any mode: [--params FILE.json] [--plate-z Z] [--plate-tilt DEG] [--downwash K] [--thrust-cost W] [--free-yaw] [--speed-min V] [--speed-max V]" << std::endl;
        return 1;
    }
    Options options = parse(argc, argv);
    const std::filesystem::path& out_dir = options.out_dir;
    std::filesystem::create_directories(out_dir);
    DEVICE device;
    rlt::init(device);
    set_base_parameters(options);
    if(!options.scenario_actor.empty()){
        return scenario(device, options, program);
    }
    if(!options.evaluate_actor.empty()){
        return evaluate_mode(device, options, program);
    }
    const TI seed = options.seed, steps = options.steps, eval_interval = options.eval_interval;
    const bool curriculum = options.curriculum;

    const auto& nominal = base_parameters;
    std::ofstream(out_dir / "config.json") << std::setprecision(10) << "{\n  \"program\": \"" << program << "\",\n  \"seed\": " << seed
        << ",\n  \"steps\": " << steps << ",\n  \"algorithm\": \"" << (WHOOP_TD3 ? "td3" : "sac") << "\",\n  \"task\": \""
        << (WHOOP_TASK_VELOCITY ? "velocity" : "hover") << "\",\n  \"plate\": " << (WHOOP_PLATE ? "true" : "false")
        << ",\n  " << parameters_json(options)
        << ",\n  \"randomization\": " << (options.randomization ? "true" : "false")
        << ",\n  \"randomization_plate\": " << nominal.randomization.plate << ",\n  \"randomization_body_drag\": " << nominal.randomization.body_drag
        << ",\n  \"motor_cuts\": " << options.motor_cuts
        << ",\n  \"motor_cut_start\": [" << nominal.motor_cut.start_min << ", " << nominal.motor_cut.start_max << "]"
        << ",\n  \"motor_cut_duration\": [" << nominal.motor_cut.duration_min << ", " << nominal.motor_cut.duration_max << "]"
        << ",\n  \"observation_dim\": " << ENVIRONMENT::Observation::DIM << ",\n  \"observation_dim_privileged\": " << ENVIRONMENT::ObservationPrivileged::DIM
        << ",\n  \"action_history\": " << FACTORY::ACTION_HISTORY_LENGTH << ",\n  \"episode_steps\": " << ENVIRONMENT::EPISODE_STEP_LIMIT
        << ",\n  \"dt\": " << nominal.integration.dt << ",\n  \"actor_hidden\": [64, 64],\n  \"critic_hidden\": [64, 64],\n  \"activation\": \"fast_tanh\""
        << ",\n  \"batch\": " << BATCH << ",\n  \"critic_training_interval\": " << CRITIC_INTERVAL << ",\n  \"actor_training_interval\": " << ACTOR_INTERVAL
        << ",\n  \"warmup_critic\": " << WARMUP_CRITIC << ",\n  \"warmup_actor\": " << WARMUP_ACTOR << ",\n  \"learning_rate\": " << LEARNING_RATE
        << ",\n  \"weight_decay\": " << WEIGHT_DECAY << ",\n  \"gamma\": 0.99,\n  \"replay_buffer\": " << WHOOP_REPLAY_BUFFER
        << ",\n  \"curriculum\": " << (curriculum ? "true" : "false") << ",\n  \"curriculum_period\": " << whoop::Curriculum<T>::PERIOD
        << ",\n  \"termination_position\": " << nominal.mdp.termination.position_threshold
        << ",\n  \"velocity_task_ramp_time\": " << nominal.velocity_task.ramp_time
        << ",\n  \"eval_interval\": " << eval_interval << "\n}\n";

    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    auto& runner = ts.off_policy_runner;
    // the training options go into the environment's parameters, from which sample_initial_parameters starts every
    // episode (the first episode is sampled at the first step, after this)
    auto& runner_env = rlt::get(runner.envs, 0, 0);
    runner_env.parameters = base_parameters;
    runner_env.parameters.randomization.enabled = options.randomization;
    runner_env.parameters.motor_cut.probability = options.motor_cuts;
    // evaluation with its own random numbers: training is the same with or without it
    RNG eval_rng;
    rlt::malloc(device, eval_rng);
    rlt::init(device, eval_rng, seed + 1);
    // the fixed test flights use the nominal aerodynamics (the same 100 starts and references as without the
    // randomization); a second set with randomized aerodynamics is flown at the end when training randomized them
    TestSet tests = test_set(device, 12345, false);

    std::ofstream progress(out_dir / "progress.csv");
    progress << "time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,rollout/episodes,curriculum/position_weight,"
                "curriculum/linear_velocity_weight,curriculum/action_weight,time/time_elapsed\n";
    std::ofstream evaluation_progress(out_dir / "evaluation_progress.csv");
    evaluation_progress << "step,success_rate,position_rmse,velocity_rmse,position_rmse_last_half,velocity_rmse_last_half,mean_return,"
                           "mean_episode_length,mean_plate_force_to_weight,time_elapsed\n";
    whoop::Reward<T> reward = base_parameters.mdp.reward;  // the paper's initial weights, plus the thrust-cost and free-yaw options
    auto start = std::chrono::steady_clock::now();
    auto elapsed = [&](){ return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    for(TI step = 1; step <= steps; step++){
        rlt::step(device, ts);
        if(curriculum && step % whoop::Curriculum<T>::PERIOD == 0 && step < steps){
            reward = whoop::Curriculum<T>::next(reward);
            apply_reward(device, runner, reward);
        }
        if(step % 10000 == 0){
            Window w = window(runner, 50000);
            double t = elapsed();
            progress << step << "," << w.episode_length << "," << w.episode_return << "," << w.episodes << "," << reward.position << ","
                     << reward.linear_velocity << "," << reward.action << "," << t << std::endl;
            std::cout << "step " << step << "  episode length " << std::setprecision(4) << w.episode_length << "  return " << w.episode_return
                      << "  (" << w.episodes << " episodes)  " << std::setprecision(3) << step / t << " steps/s" << std::endl;
        }
        if(step % eval_interval == 0 || step == steps){
            Summary s = evaluate(device, rlt::get_actor(ts), tests, eval_rng);
            double t = elapsed();
            evaluation_progress << step << "," << s.success_rate << "," << s.position_rmse << "," << s.velocity_rmse << "," << s.position_rmse_last_half
                                << "," << s.velocity_rmse_last_half << "," << s.mean_return << "," << s.mean_length << "," << s.plate_force_to_weight
                                << "," << t << std::endl;
            std::cout << "  test flights: success " << std::setprecision(3) << 100 * s.success_rate << " %, position RMSE " << s.position_rmse
                      << " m, velocity RMSE " << s.velocity_rmse << " m/s, mean length " << s.mean_length << " steps" << std::endl;
        }
    }
    double seconds = elapsed();
    std::string checks = runner_checks(runner);
    std::ofstream(out_dir / "runner_checks.json") << checks;
    std::cout << "runner checks: " << checks;
    Summary s = evaluate(device, rlt::get_actor(ts), tests, eval_rng, out_dir / "flights.csv");
    std::string sweep;
    {
        ACTOR_FORWARD forward;
        rlt::malloc(device, forward);
        rlt::copy(device, device, rlt::get_actor(ts), forward);
        sweep = speed_sweep(device, forward, out_dir / "sweep_flights.csv");
        rlt::free(device, forward);
    }
    std::string json = to_json(s, sweep);
    std::ofstream(out_dir / "evaluation.json") << json;
    if(options.randomization){
        TestSet randomized = test_set(device, 12345, true);
        Summary r = evaluate(device, rlt::get_actor(ts), randomized, eval_rng);
        std::ofstream(out_dir / "evaluation_randomized.json") << to_json(r);
        std::cout << "randomized aerodynamics: success " << std::setprecision(3) << 100 * r.success_rate << " %, position RMSE (2nd half) "
                  << r.position_rmse_last_half << " m" << std::endl;
    }
    save_actor(device, rlt::get_actor(ts), tests, out_dir / "actor.h", program + "_seed" + std::to_string(seed) + "_" + std::to_string(steps));
    save_actor_tar(device, rlt::get_actor(ts), out_dir / "actor.tar");
    {
        // the tar file must give back the same actor: the same action for the first test start
        ACTOR_FORWARD loaded, original;
        rlt::malloc(device, loaded);
        rlt::malloc(device, original);
        rlt::copy(device, device, rlt::get_actor(ts), original);
        load_actor_tar(device, loaded, out_dir / "actor.tar");
        ENVIRONMENT env;
        init_environment(device, env);
        Policy a(device, original, eval_rng), b(device, loaded, eval_rng);
        rlt::observe(device, env, tests.parameters[0], tests.starts[0], typename ENVIRONMENT::Observation{}, a.observation, eval_rng);
        rlt::copy(device, device, a.observation, b.observation);
        a.act(device, original, eval_rng);
        b.act(device, loaded, eval_rng);
        double diff = 0;
        for(TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++) diff = std::max(diff, (double)std::abs(rlt::get(a.action, 0, i) - rlt::get(b.action, 0, i)));
        std::cout << "actor.tar read back: max action difference " << diff << std::endl;
        if(diff > 1e-6) throw std::runtime_error("actor.tar does not reproduce the actor");
        a.free(device);
        b.free(device);
        rlt::free(device, loaded);
        rlt::free(device, original);
    }
    std::cout << to_json(s) << "training seconds: " << seconds << std::endl;
    rlt::free(device, eval_rng);
    rlt::free(device, ts);
    return 0;
}

int main(int argc, char** argv){
    try{
        return run(argc, argv);
    }
    catch(const std::exception& e){
        std::cerr << "error: " << e.what() << std::endl;
        return 2;
    }
}
