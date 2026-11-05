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
| `sim/include/whoop/body_drag.h` | `body_drag_force`: the linear body drag of `docs/plate_model.md` |
| `sim/include/whoop/environment.h` | parameter, reward and task types, and `EnvironmentFactory<T, TI, PLATE, TASK>`: the L2F environment type with the whoop numbers, states and observations |
| `sim/include/whoop/operations.h` | the few L2F operations that change for the whoop (see "Design" below) |
| `sim/tests/test_whoop.cpp` | unit tests (`whoop_test`) |
| `sim/src/train.cpp` | training, evaluation and export (`whoop_train_*`), and the motor-cut scenario (`--scenario`) |
| `sim/tools/animate_flight.py` | a `flights.csv` as a GIF: trajectory, close-up of the whoop and plate, height against time |
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
  parameters, then adds `R (F_plate + F_drag) / m` to dv/dt and `J⁻¹ τ_plate` to dω/dt (the plate when S > 0, the body
  drag when d ≠ 0). L2F sums the rotor forces and torques in the body frame before `a = R F / m + g` and
  `dω/dt = J⁻¹(τ − ω × Jω)`. Both are linear in F and τ, so adding the aerodynamics afterwards is the same as adding
  them to L2F's sums, up to rounding. With S = 0 and d = 0 the code path is L2F's, and the tests check it bit for bit.
  RK4 evaluates the dynamics 4 times per step, so the plate and the drag are integrated like the rest.
- `sample_initial_parameters` runs L2F's, then copies the plate, drag and task parameters. For the velocity task it
  then draws the episode's reference. With randomization on, it multiplies C_N90, K_px, K_pz, C_T, e0 and each drag
  component by its own factor from U(1 − r, 1 + r) (r = 0.4 and 0.5, from the JSON). With motor cuts on, it draws
  whether the episode has a cut and when. It draws random numbers only for the options that are on, so with both off
  the episodes are the same as before.
- `step` runs L2F's step, except during the episode's motor cut: then all four commands are −1 (motor command 0),
  whatever the policy asks. The forced command goes into the action history, as it would on the drone (the flight
  controller knows the motors are off). The rotors spin down with L2F's falling time constant. The replay buffer and
  the reward keep the policy's own action.
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

The overloads add about 300 lines, and every other part of the environment is L2F's own code.

The price of this design: an overload is only reached through unqualified calls. A call qualified with `rlt::` (for
example `rlt::sample_initial_state(...)`) silently uses L2F's version. `train.cpp` and the tests therefore call the
overridden operations as `whoop::...`. The reward overload is guaranteed: the L2F reward would not compile with the
whoop reward type. For the samplers, every training run writes `runner_checks.json`, which reads the replay buffer and
checks that the episode starts reach tilts above 57°. L2F's own sampler never does. For the velocity task it also
checks that the references reach 1–4 m/s, and it counts the episodes with a motor cut (from the forced −1 commands in
the stored action histories). In the smoke runs about 37% of the starts were above 57°, and the reference
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

Added for the whoop:

- body drag d = (0.008, 0.008, 0.004) kg/s, with and without the plate
- randomization of the aerodynamics per episode (on by default, `--no-randomization` turns it off): factors
  U(0.6, 1.4) on C_N90, K_px, K_pz, C_T and e0, U(0.5, 1.5) on each drag component. L2F randomized nothing in the
  paper; here the plate coefficients are priors until the plate is identified
- optional motor cuts (`--motor-cuts P`, velocity task): in a fraction P of the episodes, all motor commands are
  forced to 0 for U(0.2, 0.5) s, starting at U(1, 4) s

**SAC** is the default, with RLtools' entropy defaults (initial coefficient 0.5, target entropy −4). The
`whoop_train_td3_*` targets use TD3 as in the paper: exploration noise 0.1, target noise 0.2 clipped at 0.5, and target
updates at the training intervals.

Each run writes these files to `<out_dir>`:

| file | content |
|---|---|
| `config.json` | the settings of the run |
| `progress.csv` | every 10k steps: mean episode length and return over the last 50k transitions, curriculum weights, wall clock |
| `evaluation_progress.csv` | every 50k steps: success rate, position and velocity RMSE, and mean plate force on the fixed test flights |
| `evaluation.json` | the final actor on 100 fixed test flights (training distribution of the starts and references, sampled once with seed 12345; nominal aerodynamics, so the flights are the same with or without randomization) |
| `evaluation_randomized.json` | the same, with the aerodynamics randomized as in training (only when they were) |
| `flights.csv` | the first 10 test flights, step by step, including the plate force and angle of attack |
| `runner_checks.json` | the sampler check described in "Design" |
| `actor.h` | the actor as C++ code, in L2F's checkpoint format (`rl_tools::checkpoint::actor`, plus an observation and the action it gives, to check a deployment) |
| `actor.tar` | the same weights in RLtools' tar checkpoint format, read back by `--scenario` (checked at the end of training to give the same action) |

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
  `<out_dir> [seed] [steps] [--eval-interval N] [--no-curriculum] [--no-randomization] [--motor-cuts P]`.
- The motor-cut scenario: `<build>/whoop_train_sac_velocity_plate <out> --scenario <run>/actor.tar` (options below).
- `python sim/tools/animate_flight.py <out>/flights.csv <out>/flight.gif` makes the animation.
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

## Test results (`whoop_test`, laptop, MSVC 19.44: 29/29 pass)

| check | result |
|---|---|
| pure wind of 3 m/s on the plate normal | F_y = −0.04832 N = the formula to 1e-12 (16% of the weight). τ = r_c × F |
| oblique wind with the whoop moving, rotated and rolling | F and τ equal an independent transcription of `docs/plate_model.md` to 7e-18 N |
| plate in the equations of motion (body drag on in both cases, same wind) | dv/dt changes by exactly R F / m (to 2e-16) and dω/dt by J⁻¹ τ; dp/dt and dq/dt do not change |
| body drag at 3 m/s forward, level | F_x = −0.024 N = −d_x v (8.0% of the weight) |
| body drag of a rotated, moving whoop in wind (plate off) | F = −diag(d) Rᵀ(v − wind) to 2e-18 N; dv/dt changes by R F / m (to 2e-16); dω/dt, dp/dt, dq/dt do not change |
| randomization, 4000 episodes | every factor within U(1 − r, 1 + r) and reaching its ends, means within 0.008 of 1; plate geometry unchanged |
| randomization off, 50 episodes | nominal coefficients exactly, and the same references as the sampler without randomization (no extra random numbers) |
| motor cut from step 10 to 49 | 40 steps cut, action history −1, rotor state = u₀ e^(−0.4/τ_falling) to 6e-8; every step outside the cut equals L2F's step bit for bit |
| α = 90° | the centre-of-pressure offset is exactly 0 |
| α = 0 (forward flight) | the CoP is e0·c_x = 3 cm forward of the centroid |
| α = 45° | the CoP shift is e0 cos α c_x |
| pure roll rate ±2 and 10 rad/s | the plate torque opposes the rate (τ·ω < 0), e.g. τ_x = −1.37e-6 N m at 2 rad/s |
| S = 0 and d = 0 against plain L2F, through the whoop `step` | 2000 random steps × 4 configurations (float and double, both masses, hover and velocity task): **0 bits differ** |
| hover command from hover, 2 s, with and without the plate | drift 7e-15 m, no tilt; reward = the L2F constant 2 |
| absolute velocity in the actor observation | within the noise (max 0.006 m/s for hover, 0.009 for the velocity task) |
| reward recomputed from the critic observation | equals the environment reward to 1e-6 (both tasks) |
| initial states | tilts up to 89.9°, 64% of the rotation angles above 57°, 10.2% guidance starts at the origin |

## Smoke training (laptop, one seed, 300k steps each, SAC unless noted)

With the body drag and the randomization of the aerodynamics (the current default). Test flights with the nominal
aerodynamics; in brackets, the same starts with randomized aerodynamics (`evaluation_randomized.json`):

| variant | wall clock | test success | position RMSE (2nd half) | velocity RMSE (2nd half) | mean command change per step | mean (max) plate force / weight |
|---|---|---|---|---|---|---|
| hover, no plate | 46 s | 99% [100%] (61% at 100k, 98% at 150k) | 0.12 m [0.13 m] | 0.04 m/s | 0.066 | 0 |
| hover, plate | 52 s | 98% [99%] (18% at 100k, 98% at 150k) | 0.35 m [0.35 m] | 0.04 m/s | 0.049 | 0.2% (11%) |
| velocity, no plate | 51 s | 87% [89%] (28% at 150k, 88% at 200k) | 0.40 m [0.45 m] | 0.08 m/s | 0.034 | 0 |
| velocity, plate | 57 s | 92% [94%] (31% at 150k, 97% at 250k) | 0.22 m [0.22 m] | 0.19 m/s | 0.027 | 7.8% (64%) |

The same runs before (no body drag, no randomization, the same 100 test starts and references):

| variant | wall clock | test success | position RMSE (2nd half) | velocity RMSE (2nd half) | mean command change per step | mean (max) plate force / weight |
|---|---|---|---|---|---|---|
| hover, no plate | 59 s | 100% (0% at 50k, 50% at 100k, 93% at 150k) | 0.14 m | 0.04 m/s | 0.049 | 0 |
| hover, plate | 50 s | 96% (73% at 100k) | 0.12 m | 0.39 m/s | 0.049 | 0.4% (11%) |
| velocity, no plate | 84 s | 93% (8% at 100k, 94% at 150k) | 0.44 m | 0.42 m/s* | 0.033 | 0 |
| velocity, plate | 87 s | 91% (0% at 100k, 52% at 150k) | 0.26 m | 0.32 m/s* | 0.031 | 7.7% (50%) |
| hover, no plate, **TD3** | 67 s | 42% (98% at 200k, 97% at 250k) | 0.48 m | 0.66 m/s | **0.27** | 0 |

\* These two included one wrong step: after the last step of an episode, L2F's trajectory reference turns back, so
the velocity error of that step was twice the speed. The evaluation now holds the end of the line there. On the new
runs this changes the velocity RMSE from 0.31 to 0.08 m/s (no plate) and from 0.39 to 0.19 m/s (plate); success and
position RMSE do not change.

Velocity task, 1M steps (the policies of the motor-cut scenario below; their velocity RMSE has the wrong last step):

| variant | wall clock | test success | position RMSE (2nd half) | mean (max) plate force / weight |
|---|---|---|---|---|
| no plate | 174 s | 100% [100%] | 0.20 m [0.21 m] | 0 |
| plate | 223 s | 97% [97%] | 0.21 m [0.20 m] | 7.3% (53%) |
| no plate, motor cuts in 30% of the episodes | 184 s | 95% [99%] | 0.17 m [0.17 m] | 0 |
| plate, motor cuts in 30% of the episodes | 175 s | 97% [94%] | 0.32 m [0.28 m] | 7.4% (45%) |

How to read the tables:

- The mean training episode length rises from about 40 steps (random warmup) to about 500, out of 500.
- Hover position errors are mostly a steady offset of 5–35 cm. The constant random force is m g/20, and the policy has
  no integral term (neither does L2F).
- With the body drag, the plate hover policy now holds still (0.04 m/s instead of 0.3–0.5 m/s), with a larger offset
  (0.35 m). One seed each, so this may not be the drag.
- The randomized test flights give the same numbers within a few percent: the policies do not depend on the exact
  plate coefficients or drag.
- In the motor-cut scenario at 3 m/s, the plate policies fly with a yaw offset to the path (11° at 1M steps, 38° at
  300k), so that the plate meets the air at 14–45° instead of edge-on.
- TD3 commands change about 5× more per step than SAC's: the bang-bang behaviour seen in the Fixed-Wing project.

These are single seeds, so `slurm/train.slurm` is the real comparison.

## Motor-cut scenario (`--scenario`)

`whoop_train_sac_velocity_<plate|noplate> <out> --scenario <run>/actor.tar [--cut-duration S] [--cut-time S]
[--speed V] [--duration S] [--scenario-seeds N]`: the whoop starts level at rest at the origin and follows the
velocity-task reference along world x at 3 m/s (1 s ramp). At t = 1.5 s all motor commands are forced to 0 for the
cut duration, then the policy flies again until t = 5 s (the episode length). Nominal aerodynamics; the seeds change the
random force and torque and the sensor noise. `flights.csv` has the training format plus `motors_cut`; `scenario.json`
gives per seed the altitude lost (height at the cut minus the lowest height after it), when the fall is arrested
(vertical speed back to ≥ 0), the largest tilt, the plate lift during the cut, and `recovered` (within 0.5 m of the
reference over the last second and within 0.5 m/s of its velocity at the end). There is no ground in L2F.

Physics first. Without motors the whoop falls: about 0.2 m in 0.2 s and 0.8 m in 0.4 s, plus what it takes the
motors (rise time 0.04 s) to stop the fall. The plate gives no lift that matters here: its vertical force during the
cut averages 0.2–5% of the weight. Worse, it makes the fall unstable in roll. The air now comes from below, and the
plate hangs 4 cm under the centre of mass, upstream of it: any roll turns the plate into that flow, and its normal
force rolls the whoop further, with nothing to oppose it while the motors are off. In `sysid/model.py` (the same plate
model, written separately) a level whoop at 3 m/s with 1° of roll is at 37° after 0.4 s without thrust; yawed 11° to
the path, as the plate policy flies, it passes 100° after 0.3 s. Without the plate the attitude does not change.

Results, 5 seeds per cell, 1M-step policies (altitude lost; flips = tilted past 150°, after which none came back to the
line; back = `recovered`):

| policy | cut 0.2 s | cut 0.3 s | cut 0.4 s |
|---|---|---|---|
| no plate | 0.26–0.31 m, back 5/5 | 0.59–0.71 m, back 5/5 | 1.12–1.58 m, back 5/5 (fall stopped 0.25–0.49 s after the motors return) |
| plate | 0.34–0.40 m, back 5/5 | flips 5/5 | flips 5/5 |
| no plate, trained with cuts | back 4/5, flips 1/5 | back 1/5, flips 3/5 | flips 5/5 |
| plate, trained with cuts | back 3/5, flips 2/5 | back 1/5, flips 4/5 | flips 5/5 |
| no plate, trained with cuts, no curriculum | 0.21–0.26 m, back 5/5 | 0.45–0.60 m, back 4/5 | 0.89–1.10 m, fall stopped 5/5, back 2/5 (0.4–0.6 m off at the end) |
| plate, trained with cuts, no curriculum | 0.43–0.62 m, back 4/5 | flips 4/5 | 3/5 turn to about 123° and recover, lose 1.7–3.3 m and climb back (0.75–1.95 m off at 5 s); flips 2/5 |

What this shows:

- Without the plate, the policy trained without cuts recovers from a 0.4 s cut every time, losing 1.1–1.6 m.
- With the plate, a cut longer than about 0.25 s rolls the whoop over before the motors return, and the policies
  trained without cuts never saw such attitudes (L2F's starts stop at 90°).
- Training with cuts (`--motor-cuts 0.3`) and the usual curriculum made recovery worse, with or without the plate: after
  a cut the policy tilts past 90° even without the plate. A likely cause is the reward: by the end of the curriculum
  the position weight is 26, so the reward is negative beyond about 0.4 m of error, and with no termination penalty
  ending the episode (1 m) beats fighting back. Without the curriculum (`--no-curriculum`, weight 5, reward positive up
  to about 0.9 m), training with cuts helps: 15–25% less altitude lost without the plate, and the plate policy
  recovers from the roll-over in 3 of 5 flights. Its tracking is looser without the curriculum (test RMSE 0.27 m
  without the plate, 0.34 m with it, against 0.20 and 0.21 m).
- In none of these does the plate hold the whoop up. When it recovers, it is on the motors.

The GIFs are made with `sim/tools/animate_flight.py <out>/flights.csv <gif> --flight K`: a red "MOTORS OFF" label, red
rotor discs and a red trajectory segment during the cut, a camera that follows the whoop, and the height against time.

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
- **Body drag.** Linear, at the centre of mass, d = (0.008, 0.008, 0.004) kg/s: the truth of the synthetic flights of
  `sysid/`, not a measurement. L2F has none. It sees the same wind as the plate (`plate.wind`, zero).
- **Disturbances.** The random force and torque use L2F's ratios (m g/20, m g/10000) with the whoop's mass. The
  RLtools sampler divides the yaw-torque std by another 100.
- **Velocity task.**
  - Horizontal references only, with speed U(1, 4) m/s and a 1 s ramp from rest.
  - The termination threshold is 1 m instead of 0.6 m, for the lag during the ramp.
  - The reward weights and curriculum are those of hover.
- **Action.** A normalized motor command in [0, 1], as in the current RLtools Crazyflie model and the JSON. The paper
  used RPM.
- **Domain randomization.** Only the aerodynamics (plate coefficients ±40%, body drag ±50%, uniform). The mass,
  inertia, thrust curve and motor constants are not randomized (L2F did not randomize anything).
- **Motor cuts.** All four commands at 0 at once; the rotors spin down with the falling time constant (0.08 s) and
  keep no thrust of their own. There is no ground in L2F, so a cut never ends in a crash.

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
  - The plate hover policy moved slowly without the body drag and holds still with it, with a 0.35 m offset; both are
    single seeds.
  - The velocity task reaches 97–100% at 1M steps (one seed each).
  - Motor cuts: 5 starts per case and one training seed per policy. Whether a termination penalty, or starts beyond
    90° of tilt, would let the plate policy recover from the roll-over every time was not tried.
- **Training reference at the end of an episode.** The state after the last step of an episode sees L2F's reference
  turned back (velocity of the opposite sign). The evaluation now corrects for it, but the critic still bootstraps
  from that last observation in training. It affects one transition per 500.
- **Model.**
  - Only the parameters of the JSON were checked, not their physical values.
  - The plate model is quasi-steady. Unsteady effects and dynamic stall are not modelled. The roll-over during a motor
    cut comes from this model at 3–5 m/s with the air from below; windmilling propellers, which would add damping, are
    not modelled either.
- **No deployment.** The exported `actor.h` was checked to compile on its own and to reproduce its stored action
  (to 2.4e-7), but it was not run on hardware.
