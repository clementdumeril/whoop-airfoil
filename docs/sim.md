# Simulator and training (`sim/`)

The simulator is the L2F multirotor environment of RLtools ("Learning to Fly in Seconds", Eschmann, Albani, Loianno,
RA-L 2024) with the whoop parameters, plus the force and torque of the plate exactly as in `docs/plate_model.md`. The
policy is trained with the L2F training setup. Everything is C++ on top of RLtools (header-only, commit b32d9985), and
the RLtools checkout itself is not modified.

## Files

| file | what it does |
|---|---|
| `sim/tools/generate_params.py` | reads `params/whoop_nominal.json` and writes `whoop/params.h` into the build directory at build time. It checks the inputs and derives J⁻¹, the hover command and the mass and inertia with the plate |
| `sim/include/whoop/plate.h` | `plate_aerodynamics`: the equations of `docs/plate_model.md`, line by line |
| `sim/include/whoop/environment.h` | parameter, reward and task types, and `EnvironmentFactory<T, TI, PLATE, TASK>`: the L2F environment type with the whoop numbers, states and observations |
| `sim/include/whoop/operations.h` | the few L2F operations that change for the whoop (see "Design" below) |
| `sim/tests/test_whoop.cpp` | unit tests (`whoop_test`) |
| `sim/src/train.cpp` | training, evaluation and export (`whoop_train_*`) |
| `sim/CMakeLists.txt` | the build: 1 test target and 8 training targets |
| `sim/slurm/setup.sh`, `build.slurm`, `train.slurm` | Savio: configure, build and test, then a 40-run array |

## Design: how the plate enters L2F without changing RLtools

L2F builds its parameters as a chain of structs (`ParametersBase` → `ParametersDisturbances` → ...). Its operations
(dynamics, initial state, reward) are template overloads on these types, and L2F calls them **unqualified**: the
integrator calls `multirotor_dynamics`, and the off-policy runner calls `sample_initial_parameters`,
`sample_initial_state`, `reward` and `log_reward`. `environment.h` adds one more link on top of the chain,
`whoop::ParametersWhoop`, which holds the plate and task parameters. `operations.h` then defines overloads in namespace
`whoop` that take `ParametersWhoop<...>&` where L2F takes any `PARAMETERS&`. Argument-dependent lookup finds them, and
C++ picks them because they are more specialized. Each overload calls back into L2F's own code for everything it does
not change:

- `multirotor_dynamics` (for `StateBase`) runs L2F's rigid-body and rotor equations with the L2F part of the
  parameters, then adds `R F_plate / m` to dv/dt and `J⁻¹ τ_plate` to dω/dt when S > 0. L2F sums the rotor forces and
  torques in the body frame before `a = R F / m + g` and `dω/dt = J⁻¹(τ − ω × Jω)`. Both are linear in F and τ, so
  adding the plate afterwards is the same as adding it to L2F's sums, up to rounding. With S = 0 the code path is
  L2F's, and the tests check it bit for bit. RK4 evaluates the dynamics 4 times per step, so the plate is integrated
  like the rest.
- `sample_initial_parameters` runs L2F's, then copies the plate and task parameters. For the velocity task it then
  draws the episode's reference.
- `sample_initial_state` runs L2F's (rotors, action history, random force), then overwrites the rigid-body state with
  the sampler of the L2F paper. The RLtools sampler at this commit ignores `max_angle` and draws the attitude angle
  uniformly in [0, 1] rad. The paper draws it within 90°.
- `reward` / `log_reward` use the reward of the L2F paper. The current RLtools "Squared" reward is different: it uses
  norms instead of squares, and its orientation term only sees yaw.

Why not the alternatives:

- **Patching `60_dynamics.h`** is the smallest diff, but the checkout is shared with another project and must not
  change.
- **Vendoring a copy** of the L2F files would mean copying the whole dispatch chain (dynamics, initial state, the
  parameter structs and `operations_generic.h`, about 10 files), and the copy would drift from upstream.
- **A separate environment**, like the Fixed-Wing one, would lose L2F's observations, asymmetric critic, disturbances
  and rotor model, which are exactly what should be reproduced.

The overloads add about 200 lines, and every other part of the environment is L2F's own code.

The price of this design: an overload is only reached through unqualified calls. A call qualified with `rlt::` (for
example `rlt::sample_initial_state(...)`) silently uses L2F's version. `train.cpp` and the tests therefore call the
overridden operations as `whoop::...`. The reward overload is guaranteed: the L2F reward would not compile with the
whoop reward type. For the samplers, every training run writes `runner_checks.json`, which reads the replay buffer and
checks that the episode starts reach tilts above 57°. L2F's own sampler never does. For the velocity task it also
checks that the references reach 1–4 m/s. In the smoke runs about 37% of the starts were above 57°, and the reference
speeds reached 3.99 m/s.

## Observations and tasks

- **Does L2F observe the linear velocity? Yes.** The actor in the paper observes position, rotation matrix, linear
  velocity, angular velocity and a 32-step action history: 18 + 128 = 146 values (learning-to-fly
  `src/config/parameters.h`). In that observation the linear velocity is `state.linear_velocity`, the absolute world-frame
  velocity, with noise of std 0.002. The hover task uses exactly this observation (`observation::LinearVelocity`, at
  indices 12–14). In still air this is the airspeed the plate sees, expressed in world axes; the policy also has R.
  Note that the *default* observation of the current RLtools (`DefaultObservation`) has the velocity **error** to a
  reference instead. That equals the absolute velocity only for a fixed setpoint.
- **Critic (asymmetric actor-critic, as in the paper):** the same values without noise and without the action history,
  plus the random force and torque and the 4 rotor speeds (28 values).
- **Task `hover`** is L2F's position hold at the origin from random initial states. Its success condition is not to be
  terminated in 5 s, where termination means |p| > 0.6 m on any axis.
- **Task `velocity`** follows a horizontal straight line at a constant speed. The speed is U(1, 4) m/s and the heading
  random. The reference starts at the origin and accelerates uniformly from rest in 1 s. It uses L2F's trajectory
  reference (`ParametersTrajectory` / `StateTrajectory`), so the reward, the termination (|p − p_ref| > 1 m on any axis)
  and the tracking observations are L2F's code. The actor observes p − p_ref, R, v − v_ref, ω, **the absolute v**
  (indices 18–20) and the action history: 149 values. The critic gets the same without noise and without the history,
  plus the random force and rotor speeds (31 values). Another task only needs a new reference sampler in
  `sample_velocity_reference` (`operations.h`), or a new `Task` value.

## Training setup (`train.cpp`)

L2F paper settings, from the released learning-to-fly `src/config`:

- networks: actor and critics 64 × 64 with fast tanh
- batch 256; critics trained every 10 steps (and their targets updated), actor every 20
- Adam with learning rate 1e-3, weight decay 1e-4 on every layer, ε = 1e-8
- γ = 0.99; critic warmup 15k steps, actor warmup 30k
- 100 Hz, one RK4 step per action; episodes of 500 steps; 300k steps by default
- observation noise (position and orientation 0.001, velocities 0.002)
- random force and torque per episode: Gaussian with std m g/20 and m g/10000
- the paper's reward and curriculum: every 100k steps the position weight is multiplied by 1.2 (capped at 40), the
  velocity and action weights by 1.4 (capped at 1). All stored rewards are then recomputed from the critic
  observations.

**SAC** is the default, with RLtools' entropy defaults (initial coefficient 0.5, target entropy −4). The
`whoop_train_td3_*` targets use TD3 as in the paper: exploration noise 0.1, target noise 0.2 clipped at 0.5, and target
updates at the training intervals.

Each run writes these files to `<out_dir>`:

| file | content |
|---|---|
| `config.json` | the settings of the run |
| `progress.csv` | every 10k steps: mean episode length and return over the last 50k transitions, curriculum weights, wall clock |
| `evaluation_progress.csv` | every 50k steps: success rate, position and velocity RMSE, and mean plate force on the fixed test flights |
| `evaluation.json` | the final actor on 100 fixed test flights (training distribution, sampled once with seed 12345) |
| `flights.csv` | the first 10 test flights, step by step, including the plate force and angle of attack |
| `runner_checks.json` | the sampler check described in "Design" |
| `actor.h` | the actor as C++ code, in L2F's checkpoint format (`rl_tools::checkpoint::actor`, plus an observation and the action it gives, to check a deployment) |

## Build, test, run (Windows laptop)

Builds and runs live outside the repository:

```powershell
# from a "x64 Native Tools" prompt (MSVC), with cmake and ninja on PATH; builds and runs outside the repository
cmake -S sim -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_DIR=<rl_tools checkout at b32d9985>
#   optional MKL: -DWHOOP_MKL=ON -DMKL_DIR=<venv>/Library/lib/cmake/mkl -DMKL_ROOT=<venv>/Library, and <venv>/Library/bin on PATH
cmake --build <build>
<build>/whoop_test
<build>/whoop_train_sac_hover_plate <runs>/sac_hover_plate 0 300000
```

- Omit the `-DWHOOP_MKL...` options to build without MKL.
- The training targets are `whoop_train_{sac,td3}_{hover,velocity}_{noplate,plate}`; arguments:
  `<out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum]`.
- The plate is a compile-time switch (`WHOOP_PLATE`). It sets the nominal plate area (0 without the plate) and the
  vehicle's mass and inertia. The code is the same for both, and the area can be changed at run time in
  `parameters.plate.area`.

## Savio

All three scripts run from `sim/`:

1. `source slurm/setup.sh` configures the build on the login node. Set `RL_TOOLS_DIR` to an RLtools checkout at
   b32d9985; the default is the copy fetched by the Fixed-Wing build.
2. `sbatch slurm/build.slurm` builds on a compute node with gcc 11.4 (`-O3 -march=native`) and runs the tests.
3. `sbatch slurm/train.slurm` runs 4 variants × 10 seeds with SAC for 300k steps. Override with
   `--export=ALL,STEPS=...,ALGORITHM=td3`.

They use account `ic_piml`, partition `savio4_htc` and the Fixed-Wing venv.

## Test results (`whoop_test`, laptop, MSVC 19.44: 24/24 pass)

| check | result |
|---|---|
| pure wind of 3 m/s on the plate normal | F_y = −0.04832 N = the formula to 1e-12 (16% of the weight). τ = r_c × F |
| oblique wind with the whoop moving, rotated and rolling | F and τ equal an independent transcription of `docs/plate_model.md` to 7e-18 N |
| plate in the equations of motion | dv/dt changes by exactly R F / m (to 2e-16) and dω/dt by J⁻¹ τ; dp/dt and dq/dt do not change |
| α = 90° | the centre-of-pressure offset is exactly 0 |
| α = 0 (forward flight) | the CoP is e0·c_x = 3 cm forward of the centroid |
| α = 45° | the CoP shift is e0 cos α c_x |
| pure roll rate ±2 and 10 rad/s | the plate torque opposes the rate (τ·ω < 0), e.g. τ_x = −1.37e-6 N m at 2 rad/s |
| S = 0 against plain L2F | 2000 random steps × 4 configurations (float and double, both masses, hover and velocity task): **0 bits differ** |
| hover command from hover, 2 s, with and without the plate | drift 7e-15 m, no tilt; reward = the L2F constant 2 |
| absolute velocity in the actor observation | within the noise (max 0.006 m/s for hover, 0.009 for the velocity task) |
| reward recomputed from the critic observation | equals the environment reward to 1e-6 (both tasks) |
| initial states | tilts up to 89.9°, 64% of the rotation angles above 57°, 10.2% guidance starts at the origin |

## Smoke training (laptop, one seed, 300k steps each, SAC unless noted)

| variant | wall clock | test success | position RMSE (2nd half) | velocity RMSE (2nd half) | mean command change per step | mean (max) plate force / weight |
|---|---|---|---|---|---|---|
| hover, no plate | 59 s | 100% (0% at 50k, 50% at 100k, 93% at 150k) | 0.14 m | 0.04 m/s | 0.049 | 0 |
| hover, plate | 50 s | 96% (73% at 100k) | 0.12 m | 0.39 m/s | 0.049 | 0.4% (11%) |
| velocity, no plate | 84 s | 93% (8% at 100k, 94% at 150k) | 0.44 m | 0.42 m/s | 0.033 | 0 |
| velocity, plate | 87 s | 91% (0% at 100k, 52% at 150k) | 0.26 m | 0.32 m/s | 0.031 | 7.7% (50%) |
| hover, no plate, **TD3** | 67 s | 42% (98% at 200k, 97% at 250k) | 0.48 m | 0.66 m/s | **0.27** | 0 |

How to read the table:

- The mean training episode length rises from about 40 steps (random warmup) to about 500, out of 500.
- Hover position errors are mostly a steady offset of 5–25 cm. The constant random force is m g/20, and the policy has
  no integral term (neither does L2F).
- With the plate, the 300k-step hover policy stays within about 10 cm but keeps moving slowly (0.3–0.5 m/s, ±10° roll)
  instead of holding still.
- TD3 commands change about 5× more per step than SAC's: the bang-bang behaviour seen in the Fixed-Wing project.

These are single seeds, so `slurm/train.slurm` is the real comparison.

## Assumptions

All marked `[assumption]` in the code as well:

- **Parameters.** Every whoop and plate number comes from `params/whoop_nominal.json`, with the assumptions labelled
  there: thrust curve, torque constant, motor time constants, inertia and plate coefficients.
- **Plate mass and inertia.** `plate.added_mass` (0.7 g) is the mass of the sheet (3 mm × 0.0078 m² × about 30 kg/m³).
  With the plate on:
  - mass = 30.7 g;
  - J = J_whoop + the inertia of a thin c_x × c_z sheet at `plate.centroid`, about the combined centre of mass, plus
    the parallel-axis term of the whoop's own centre of mass shift. This gives J_xx +12%, J_yy +19%, J_zz +4%;
  - the hover command follows: 0.4754 instead of 0.4694.
  - The rotor positions are not shifted for the 0.9 mm move of the centre of mass. Their z offset does not change the
    torque of a z-axis thrust.
  - `sysid/` should use the same convention, or the JSON should give the measured totals.
- **Plate model.**
  - Quasi-steady, exactly `docs/plate_model.md` with ε = 1e-6.
  - No wind (`plate.wind` = 0, but the parameter exists).
  - No propeller downwash: k_w = 0. The generator refuses k_w ≠ 0, since v_i is not defined yet.
  - The plate's horizontal edge is body x. The generator checks that the normal has no x component.
- **No aerodynamic drag of the whoop itself.** L2F has none. At 1–4 m/s this matters, and `sysid/` plans to identify
  it.
- **Disturbances.** The random force and torque use L2F's ratios (m g/20, m g/10000) with the whoop's mass. The
  RLtools sampler divides the yaw-torque std by another 100.
- **Velocity task.**
  - Horizontal references only, with speed U(1, 4) m/s and a 1 s ramp from rest.
  - The termination threshold is 1 m instead of 0.6 m, for the lag during the ramp.
  - The reward weights and curriculum are those of hover.
- **Action.** A normalized motor command in [0, 1], as in the current RLtools Crazyflie model and the JSON. The paper
  used RPM.
- **Domain randomization.** None, of the dynamics or of the plate. L2F had none either.

## Differences from the L2F paper setup

- **SAC instead of TD3** (TD3 is available as an option).
- **Warmup.** RLtools' loop acts uniformly at random for the first 15k steps. L2F acted with the untrained actor plus
  noise.
- **Replay buffer.** 1M transitions (about 2.6 GB allocated in all). L2F kept all 300k; for runs up to 1M steps nothing
  is overwritten either.
- **No exploration-noise decay.** In L2F it starts at 500k steps, so it has no effect in a 300k run.
- **Evaluation.** 100 fixed flights at the end and every 50k steps, instead of L2F's 1000 random episodes.

## Open issues and what is not verified

- **GCC / Linux build.** It was not compiled here (no GCC or Clang on the laptop). The code avoids MSVC-only features
  and uses qualified calls only where two-phase lookup allows them, but `slurm/build.slurm` is the first real test.
- **Behaviour.**
  - The slow orbit of the plate hover policy should be checked on more seeds and steps.
  - The plate torques are small at hover speeds (about 3e-5 N m at 0.4 m/s), so this is probably convergence, not
    instability.
  - The velocity task was only trained for 300k steps (91–93% success).
- **Model.**
  - Only the parameters of the JSON were checked, not their physical values.
  - The plate model is quasi-steady. Unsteady effects and dynamic stall are not modelled.
- **No deployment.** The exported `actor.h` was checked to compile on its own and to reproduce its stored action
  (to 2.4e-7), but it was not run on hardware.
