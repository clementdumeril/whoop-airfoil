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
// whoop_train_<sac|td3>_<hover|velocity>_<noplate|plate> <out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum]
//   out_dir gets config.json; progress.csv (episode length and return of the last 50k steps, every 10k steps, with the
//   wall clock); evaluation_progress.csv (the fixed test flights every N steps, default 50000); evaluation.json and
//   flights.csv (the final actor on the fixed test flights: 100 starts sampled once with seed 12345, the first 10
//   written step by step); actor.h (the trained actor as C++ code, the checkpoint format of L2F); runner_checks.json
//   (evidence from the replay buffer that the RLtools runner used the whoop initial states and references)
#include <rl_tools/operations/cpu_mux.h>

#include <whoop/operations.h>

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
    auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    constexpr TI N = ENVIRONMENT::ObservationPrivileged::DIM;
    T o[N], a[ENVIRONMENT::ACTION_DIM];
    for(TI i = 0; i < filled(rb); i++){
        for(TI j = 0; j < N; j++) o[j] = rlt::get(rb.observations_privileged, i, j);
        for(TI j = 0; j < ENVIRONMENT::ACTION_DIM; j++) a[j] = rlt::get(rb.actions, i, j);
        bool terminated = rlt::get(rb.terminated, i, 0) > T(0.5);
        rlt::set(rb.rewards, i, 0, whoop::reward_from_privileged_observation(reward, o, a, baseline, terminated));
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
    TI n = filled(rb), starts = 0, tilted = 0;
    double max_reference_speed = 0, max_reference_vz = 0;
    const double cos_57 = std::cos(57.3 / 180 * 3.14159265358979323846);
    for(TI i = 0; i < n; i++){
        bool start = i == 0 || rlt::get(rb.truncated, i - 1, 0) > T(0.5);
        if(start){
            starts++;
            tilted += rlt::get(rb.observations_privileged, i, 11) < cos_57;  // R_zz = cos(tilt)
        }
        if constexpr(WHOOP_TASK_VELOCITY){
            double v_ref[3];
            for(TI j = 0; j < 3; j++) v_ref[j] = rlt::get(rb.observations_privileged, i, 18 + j) - rlt::get(rb.observations_privileged, i, 12 + j);
            max_reference_speed = std::max(max_reference_speed, std::sqrt(v_ref[0] * v_ref[0] + v_ref[1] * v_ref[1] + v_ref[2] * v_ref[2]));
            max_reference_vz = std::max(max_reference_vz, std::abs(v_ref[2]));
        }
    }
    double tilted_fraction = starts > 0 ? double(tilted) / starts : 0;
    bool ok = tilted_fraction > 0.1 && (!WHOOP_TASK_VELOCITY || (max_reference_speed > 2 && max_reference_speed < 4.01 && max_reference_vz < 1e-3));
    std::ostringstream json;
    json << "{\n  \"episode_starts\": " << starts << ",\n  \"starts_tilted_above_57_deg\": " << tilted_fraction
         << ",\n  \"max_reference_speed\": " << max_reference_speed << ",\n  \"max_reference_vertical_speed\": " << max_reference_vz
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
TestSet test_set(DEVICE& device, TI seed){
    // sampled from the training distribution with a fixed seed, so that every run is tested on the same flights
    TestSet set;
    set.parameters.resize(N_TESTS);
    set.starts.resize(N_TESTS);
    ENVIRONMENT env;
    rlt::init(device, env);
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
};

double norm3(const T* v){ return std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]); }

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
    rlt::init(device, env);
    std::vector<ENVIRONMENT::State> states = tests.starts;
    std::vector<bool> done(N_TESTS, false), failed(N_TESTS, false);
    std::vector<double> returns(N_TESTS, 0), change(N_TESTS, 0);
    std::vector<std::vector<std::array<double, 2>>> errors(N_TESTS);  // position and velocity error per step
    double plate_force_sum = 0, plate_force_max = 0, plate_samples = 0;
    std::vector<std::array<T, 4>> previous(N_TESTS);
    for(TI k = 0; k < N_TESTS; k++){
        for(TI i = 0; i < 4; i++) previous[k][i] = whoop::action_baseline(tests.parameters[k].dynamics.hovering_throttle_relative);
    }
    std::ofstream flights;
    if(!flights_csv.empty()){
        flights.open(flights_csv);
        flights << "flight,t,x,y,z,vx,vy,vz,roll_deg,pitch_deg,yaw_deg,reference_x,reference_y,reference_z,reference_vx,reference_vy,reference_vz,"
                   "command_1,command_2,command_3,command_4,rotor_1,rotor_2,rotor_3,rotor_4,plate_fx,plate_fy,plate_fz,plate_alpha_deg,airspeed,terminated\n";
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
            rlt::step(device, env, p, states[k], action, next, rng);
            returns[k] += whoop::reward(device, env, p, states[k], action, next, rng);
            bool terminated = rlt::terminated(device, env, p, next, rng);
            ENVIRONMENT::State reference;
            rlt::get_desired_state(device, env, p, next, reference, rng);
            T e_p[3], e_v[3];
            for(TI i = 0; i < 3; i++){
                e_p[i] = next.position[i] - reference.position[i];
                e_v[i] = next.linear_velocity[i] - reference.linear_velocity[i];
            }
            errors[k].push_back({norm3(e_p), norm3(e_v)});
            double c = 0;
            for(TI i = 0; i < 4; i++){ c += std::abs(rlt::get(action, 0, i) - previous[k][i]); previous[k][i] = rlt::get(action, 0, i); }
            change[k] += c / 4;
            whoop::PlateAerodynamics<T> aero{};
            if(p.plate.area > 0){
                whoop::plate_aerodynamics(device, p.plate, next.orientation, next.linear_velocity, next.angular_velocity, aero);
                double ratio = norm3(aero.force) / (p.dynamics.mass * 9.81);
                plate_force_sum += ratio;
                plate_force_max = std::max(plate_force_max, ratio);
                plate_samples += 1;
            }
            if(k < N_RECORDED && flights.is_open()){
                const T* q = next.orientation;
                const double deg = 180 / 3.14159265358979323846;
                double roll = std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * deg;
                double pitch = std::asin(std::clamp(2 * (q[0] * q[2] - q[3] * q[1]), T(-1), T(1))) * deg;
                double yaw = std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])) * deg;
                flights << k << "," << (step + 1) * p.integration.dt;
                for(TI i = 0; i < 3; i++) flights << "," << next.position[i];
                for(TI i = 0; i < 3; i++) flights << "," << next.linear_velocity[i];
                flights << "," << roll << "," << pitch << "," << yaw;
                for(TI i = 0; i < 3; i++) flights << "," << reference.position[i];
                for(TI i = 0; i < 3; i++) flights << "," << reference.linear_velocity[i];
                for(TI i = 0; i < 4; i++) flights << "," << (rlt::get(action, 0, i) + 1) / 2;  // command in [0, 1]
                for(TI i = 0; i < 4; i++) flights << "," << next.rpm[i];
                for(TI i = 0; i < 3; i++) flights << "," << aero.force[i];
                flights << "," << aero.alpha * deg << "," << aero.airspeed << "," << terminated << "\n";
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
    rlt::free(device, actor_state);
    rlt::free(device, actor_buffer);
    rlt::free(device, observations);
    rlt::free(device, actions);
    return s;
}

std::string to_json(const Summary& s){
    auto number = [](double v){ std::ostringstream o; o << std::setprecision(8); if(std::isfinite(v)) o << v; else o << "null"; return o.str(); };
    std::ostringstream json;
    json << "{\n  \"test_flights\": " << N_TESTS << ",\n  \"test_seed\": 12345,\n  \"episode_steps\": " << ENVIRONMENT::EPISODE_STEP_LIMIT
         << ",\n  \"success_rate\": " << number(s.success_rate)
         << ",\n  \"position_rmse\": " << number(s.position_rmse) << ",\n  \"position_rmse_last_half\": " << number(s.position_rmse_last_half)
         << ",\n  \"velocity_rmse\": " << number(s.velocity_rmse) << ",\n  \"velocity_rmse_last_half\": " << number(s.velocity_rmse_last_half)
         << ",\n  \"mean_return\": " << number(s.mean_return) << ",\n  \"mean_episode_length\": " << number(s.mean_length)
         << ",\n  \"mean_action_change\": " << number(s.mean_action_change)
         << ",\n  \"mean_plate_force_to_weight\": " << number(s.plate_force_to_weight)
         << ",\n  \"max_plate_force_to_weight\": " << number(s.max_plate_force_to_weight) << "\n}\n";
    return json.str();
}

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
    rlt::init(device, env);
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

int run(int argc, char** argv){
    const std::string program = std::string("whoop_train_") + (WHOOP_TD3 ? "td3" : "sac") + (WHOOP_TASK_VELOCITY ? "_velocity" : "_hover") + (WHOOP_PLATE ? "_plate" : "_noplate");
    if(argc < 2){
        std::cerr << "usage: " << program << " <out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum]" << std::endl;
        return 1;
    }
    std::filesystem::path out_dir = argv[1];
    std::filesystem::create_directories(out_dir);
    TI seed = argc > 2 ? std::stoul(argv[2]) : 0;
    TI steps = argc > 3 ? std::stoul(argv[3]) : 300000;  // L2F: 300k
    TI eval_interval = 50000;
    bool curriculum = true;
    for(int i = 4; i < argc; i++){
        std::string option = argv[i];
        if(option == "--eval-interval" && i + 1 < argc) eval_interval = std::stoul(argv[++i]);
        else if(option == "--no-curriculum") curriculum = false;
        else throw std::runtime_error("unknown option " + option);
    }
    if(steps >= LOOP_CORE_PARAMETERS::STEP_LIMIT) throw std::runtime_error("steps above the compiled STEP_LIMIT");
    if(eval_interval == 0) throw std::runtime_error("--eval-interval must be positive");

    const auto& nominal = FACTORY::nominal_parameters;
    std::ofstream(out_dir / "config.json") << std::setprecision(10) << "{\n  \"program\": \"" << program << "\",\n  \"seed\": " << seed
        << ",\n  \"steps\": " << steps << ",\n  \"algorithm\": \"" << (WHOOP_TD3 ? "td3" : "sac") << "\",\n  \"task\": \""
        << (WHOOP_TASK_VELOCITY ? "velocity" : "hover") << "\",\n  \"plate\": " << (WHOOP_PLATE ? "true" : "false")
        << ",\n  \"plate_area\": " << nominal.plate.area << ",\n  \"mass\": " << nominal.dynamics.mass
        << ",\n  \"hovering_throttle\": " << nominal.dynamics.hovering_throttle_relative
        << ",\n  \"observation_dim\": " << ENVIRONMENT::Observation::DIM << ",\n  \"observation_dim_privileged\": " << ENVIRONMENT::ObservationPrivileged::DIM
        << ",\n  \"action_history\": " << FACTORY::ACTION_HISTORY_LENGTH << ",\n  \"episode_steps\": " << ENVIRONMENT::EPISODE_STEP_LIMIT
        << ",\n  \"dt\": " << nominal.integration.dt << ",\n  \"actor_hidden\": [64, 64],\n  \"critic_hidden\": [64, 64],\n  \"activation\": \"fast_tanh\""
        << ",\n  \"batch\": " << BATCH << ",\n  \"critic_training_interval\": " << CRITIC_INTERVAL << ",\n  \"actor_training_interval\": " << ACTOR_INTERVAL
        << ",\n  \"warmup_critic\": " << WARMUP_CRITIC << ",\n  \"warmup_actor\": " << WARMUP_ACTOR << ",\n  \"learning_rate\": " << LEARNING_RATE
        << ",\n  \"weight_decay\": " << WEIGHT_DECAY << ",\n  \"gamma\": 0.99,\n  \"replay_buffer\": " << WHOOP_REPLAY_BUFFER
        << ",\n  \"curriculum\": " << (curriculum ? "true" : "false") << ",\n  \"curriculum_period\": " << whoop::Curriculum<T>::PERIOD
        << ",\n  \"termination_position\": " << nominal.mdp.termination.position_threshold
        << ",\n  \"velocity_task_speed\": [" << nominal.velocity_task.speed_min << ", " << nominal.velocity_task.speed_max << "]"
        << ",\n  \"velocity_task_ramp_time\": " << nominal.velocity_task.ramp_time
        << ",\n  \"eval_interval\": " << eval_interval << "\n}\n";

    DEVICE device;
    rlt::init(device);
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    auto& runner = ts.off_policy_runner;
    // evaluation with its own random numbers: training is the same with or without it
    RNG eval_rng;
    rlt::malloc(device, eval_rng);
    rlt::init(device, eval_rng, seed + 1);
    TestSet tests = test_set(device, 12345);

    std::ofstream progress(out_dir / "progress.csv");
    progress << "time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,rollout/episodes,curriculum/position_weight,"
                "curriculum/linear_velocity_weight,curriculum/action_weight,time/time_elapsed\n";
    std::ofstream evaluation_progress(out_dir / "evaluation_progress.csv");
    evaluation_progress << "step,success_rate,position_rmse,velocity_rmse,position_rmse_last_half,velocity_rmse_last_half,mean_return,"
                           "mean_episode_length,mean_plate_force_to_weight,time_elapsed\n";
    whoop::Reward<T> reward = whoop::reward_initial<T>;
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
    std::string json = to_json(s);
    std::ofstream(out_dir / "evaluation.json") << json;
    save_actor(device, rlt::get_actor(ts), tests, out_dir / "actor.h", program + "_seed" + std::to_string(seed) + "_" + std::to_string(steps));
    std::cout << json << "training seconds: " << seconds << std::endl;
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
