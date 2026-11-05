# sysid: whoop and plate identification, validated on synthetic flights

Identification chain for the 30 g whoop with and without the vertical EPP plate. The real drone has not
flown yet, so everything here is validated on synthetic flights generated from the known model in
`params/whoop_nominal.json` and `docs/plate_model.md`, with realistic logging (Betaflight-like blackbox +
mocap). The whoop part reproduces Eschmann, Albani, Loianno, *Data-driven system identification of
quadrotors subject to motor delays*, IROS 2024 (arXiv 2404.07837), as the official notebook
(`sysid.ipynb`, commit 2d267dd) does it; the plate part is new.

## How to run

Python with numpy, scipy, matplotlib (tested with Anaconda Python: numpy 2.4,
scipy 1.18, matplotlib 3.11; the `fixed-wing` venv has no scipy). From `sysid/`:

```
python synthetic.py          # smoke test: simulates one excitation and one plate flight, prints stats
python validate.py --quick   # nominal noise, medium duration: ~3 min (after the cache exists)
python validate.py           # 3 noise levels x 3 durations + Monte Carlo + prediction: ~10 min
```

The noise-free truth flights are simulated once (~6 min in total, 500 Hz RK4 in numpy) and cached as npz
in `~/Work/whoop-airfoil-sysid/cache` (override with `WHOOP_SYSID_CACHE`). Sensor noise is
added at load time, so changing noise levels does not re-simulate. Tables are written to
`~/Work/whoop-airfoil-sysid/results.md` (and `results.json`), figures to `sysid/figures/`.

## Files

| file | what it does |
|---|---|
| `model.py` | Whoop rigid body (L2F frame: x forward, y left, z up; q = (w,x,y,z) body to world), first-order motor lag with rising/falling time constants, quadratic thrust in normalized command, yaw torque k_tau * thrust, linear body drag, plate force/torque exactly as `docs/plate_model.md`. Vectorized so the simulator and the identification share it. |
| `synthetic.py` | Stabilizing controller (position PD + geometric attitude PD + yaw-last mixer), manoeuvre schedules, simulation, sensor model, cache. |
| `identify.py` | The chain: pre-processing, motor time constant(s) and thrust curve, inertia, k_tau, body drag, plate coefficients, e0. |
| `validate.py` | Runs the chain on synthetic data, prints the recovery tables, the per-manoeuvre identifiability, the held-out prediction RMSE, and makes the figures. |
| `figures/` | `finding_tau.png` (RMSE vs T_m with the minimum, and the 2-D rising/falling map), `thrust_fit.png` (predicted vs measured thrust, without/with motor delay), `inertia_fit.png`, `plate_cn_alpha.png` (C_N vs alpha, true vs identified, with per-sample measured C_N), `prediction_rmse.png`. |

## Model choices and assumptions

- **Motor lag.** The JSON has both `rotor_time_constant_rising` (0.04 s) and `_falling` (0.08 s). As in
  L2F (`rl_tools/.../l2f/operations_generic/70_post_integration.h`, closed-form branch, verified in the
  local checkout) the lag acts on the normalized motor state u_m, with T = rising if command >= u_m else
  falling, exact exponential update per step, and thrust = c0 + c1 u_m + c2 u_m^2. The JSON comment says
  "response of the thrust"; we follow the L2F code because the field names are L2F's and `sim/` builds on
  L2F. **To confirm with the `sim/` owner.**
- **Truth body drag** [assumption]: F = -diag(0.008, 0.008, 0.004) kg/s * v_body (0.024 N, 8 % of the
  weight, at 3 m/s horizontally). Acts at the CoM, no moment.
- **Plate mass.** `added_mass` = 0.7 g is read as the physical mass of the sheet; with the plate the
  vehicle is 30.7 g and its inertia is J + thin-plate inertia about the CoM (parallel axis). Rotor
  positions and the plate centroid are taken relative to the CoM with the plate (0.9 mm shift neglected).
  If `added_mass` was meant as aerodynamic added mass, this must change in both `sim/` and here.
- Downwash factor is 0 in the JSON and is not implemented (asserted).
- Simulation: physics and control at 500 Hz (= log rate), IMU at the CoM, white noise, constant
  biases, perfect mocap/blackbox time sync, still air, no ground effect. The controller uses the true
  aerodynamic force as feed-forward (it only has to fly the manoeuvres; the identification never sees it).

## Synthetic data

- Blackbox frame at 500 Hz: gyro, accelerometer, motor commands quantized to 1/2000. Mocap position and
  attitude at 125 Hz. A 3 s static segment (level on the ground, motors off) before each flight for bias
  calibration.
- Noise levels (1 sigma per sample):

  | level | gyro [rad/s] | acc [m/s2] | gyro bias | acc bias | mocap pos | mocap att |
  |---|---|---|---|---|---|---|
  | low | 0.005 | 0.05 | 0.005 | 0.05 | 0.2 mm | 0.1 deg |
  | nominal | 0.02 | 0.3 | 0.02 | 0.2 | 0.5 mm | 0.3 deg |
  | high | 0.08 | 1.5 | 0.05 | 0.5 | 2 mm | 1 deg |

- **(a) Excitation flight**, whoop without plate, like the paper's data, in place: thirds of
  [thrust: collective multisine 0.3-6 Hz of +-75 % hover thrust | roll/pitch: torque multisines 1-10 Hz,
  tilts up to ~50 deg | yaw: heading multisine +-70 deg plus yaw-torque multisine]. 30, 60, 120 s.
- **(b) Plate flights** at fixed heading (yaw = 0), flown with the plate and, identically, without it:
  fixed-heading circles at 1, 2, 3, 4 m/s in both directions (centripetal acceleration <= 6 m/s2, radius
  2.7 m at 4 m/s), forward and sideways shuttles at 2 and 3.5 m/s (+-2 m), diagonal shuttles at 3 m/s,
  sideways climbs (45 deg in the y-z plane, 2.5 m/s) and vertical shuttles (1.5 m/s). Every segment is a
  whole number of laps/cycles with smooth speed ramps. Nominal 60/120/240 s, real lengths 86/116/233 s
  (rounded to whole laps).

## Pipeline (identify.py) and deviations from the notebook

Followed as in the notebook: EMA motor model on the normalized commands initialized with the first
command; T_m by 1-D grid (0.005-0.2 s, 100 steps, the sysid.tools default) with, for each candidate, one
lumped thrust curve (exponents 0, 1, 2) by linear LS on all three accelerometer rows b = m * acc and the
RMSE as objective; K_f of the minimum; I_xx, I_yy per axis by scalar LS between angular acceleration and
geometric rotor torque; I_zz = 1.832 (I_xx + I_yy)/2; K_tau = <u, dw_z I_zz>/<u, u>; hover check.

Deviations (each is a switch; the notebook-faithful result is also reported):
1. **Asymmetric motor model** (paper section V, future work): 2-D grid over (T_rise, T_fall), coarse 24 x
   24 then 11 x 11 refinement, same LS per candidate, L2F switching rule. A single T_m cannot represent
   0.04/0.08 s.
2. **Vertical drag column** in the thrust LS (-v_body_z from mocap). Velocity lags thrust, so an unmodelled
   drag shifts the time constants (rise/fall came out 5 % low without it).
3. **Inertia**: reciprocal fit 1/I = <tau, dw>/<tau, tau> (the paper says it worked better; not in the
   notebook) because the notebook's <dw, tau>/<dw, dw> puts the noisy angular acceleration in the
   regressor and is biased low (errors in variables: -3 % at nominal noise, about -50 % at high noise);
   both sides zero-phase low-passed at 20 Hz; **gyroscopic term** w x J w included by fixed-point
   iteration (the notebook drops it; it caused +2.5 % with the yaw motion present in roll/pitch data).
4. K_tau with the same filtering and gyroscopic term. I_zz still from the 1.832 ratio (truth ratio is
   1.833 by construction of the JSON, so this step is not really tested).
5. IMU biases from the static pre-flight segment (the paper's PX4 data is bias-corrected by the EKF).
6. EMA is causal (state at frame k uses the command of frame k-1); the notebook uses frame k (2 ms lead).
   The filter runs over the whole flight and phases are cut afterwards (no re-initialization transient).

New steps:
- **Body drag** from the no-plate fixed-heading flights: x/y accelerometer rows contain no rotor thrust, so
  m a_x = -d_x v_x is exact; d_z from the z row minus the identified thrust.
- **Plate force**: F_meas = m a_imu - F_rotor - F_drag in the body frame; nonlinear LS (scipy
  `least_squares`, start (1, 1, 1, 0.05), not the truth) for (C_N90, K_px, K_pz, C_T) on the x and y rows.
  The z row is left out: it carries the rotor thrust, whose model errors dwarf the in-plane friction.
  Body velocity from Savitzky-Golay-differentiated mocap rotated by the (smoothed) mocap attitude.
- **e0**: the plate moment is r_c x F + e0 (cos(a) c(psi) v_t / V_t) x F, linear in e0 once F uses the
  identified coefficients; LS on the moment residual J dw + w x J w - tau_rotors - r_c x F (all axes,
  low-passed), with J = identified whoop inertia + measured plate inertia.

## Results (output of `python validate.py`, copied from results.md)

Short/medium/long = excitation flight of 30/60/120 s plus a plate flight and a no-plate flight of the listed length. One truth trajectory per flight; the Monte Carlo varies only the sensor-noise seed.

### Recovery: nominal noise

| parameter | true | short (30 s exc. + 2 x 86 s plate/no plate) | medium (60 s exc. + 2 x 116 s plate/no plate) | long (120 s exc. + 2 x 233 s plate/no plate) |
|---|---|---|---|---|
| tau_rise [s] | 0.04 | 0.03974 (-0.7 %) | 0.03974 (-0.7 %) | 0.03974 (-0.7 %) |
| tau_fall [s] | 0.08 | 0.08104 (+1.3 %) | 0.08104 (+1.3 %) | 0.07939 (-0.8 %) |
| kf0 [N] | 0 | -0.0009589 (-9.6e-04 abs) | -0.0002905 (-2.9e-04 abs) | 0.0001372 (+1.4e-04 abs) |
| kf1 [N] | 0.03 | 0.03322 (+10.7 %) | 0.02971 (-1.0 %) | 0.03016 (+0.5 %) |
| kf2 [N] | 0.27 | 0.2666 (-1.2 %) | 0.271 (+0.4 %) | 0.2691 (-0.3 %) |
| thrust(u=0.5) [N] | 0.0825 | 0.08231 (-0.2 %) | 0.08232 (-0.2 %) | 0.08249 (-0.0 %) |
| thrust(u=1) [N] | 0.3 | 0.2989 (-0.4 %) | 0.3004 (+0.1 %) | 0.2994 (-0.2 %) |
| I_xx [kg m2] | 1.2e-05 | 1.198e-05 (-0.2 %) | 1.201e-05 (+0.1 %) | 1.201e-05 (+0.1 %) |
| I_yy [kg m2] | 1.2e-05 | 1.198e-05 (-0.2 %) | 1.201e-05 (+0.1 %) | 1.201e-05 (+0.1 %) |
| I_zz [kg m2] (ratio) | 2.2e-05 | 2.195e-05 (-0.2 %) | 2.2e-05 (+0.0 %) | 2.2e-05 (+0.0 %) |
| k_tau [m] | 0.005 | 0.004997 (-0.1 %) | 0.004988 (-0.2 %) | 0.005006 (+0.1 %) |
| d_x [kg/s] | 0.008 | 0.007979 (-0.3 %) | 0.007996 (-0.1 %) | 0.008008 (+0.1 %) |
| d_y [kg/s] | 0.008 | 0.008024 (+0.3 %) | 0.007999 (-0.0 %) | 0.007997 (-0.0 %) |
| d_z [kg/s] | 0.004 | 0.003762 (-6.0 %) | 0.003919 (-2.0 %) | 0.003979 (-0.5 %) |
| d_z thrust-fit [kg/s] | 0.004 | 0.003924 (-1.9 %) | 0.004473 (+11.8 %) | 0.003969 (-0.8 %) |
| C_N90 | 1.15 | 1.14 (-0.9 %) | 1.148 (-0.1 %) | 1.148 (-0.1 %) |
| K_px | 0.84 | 0.8299 (-1.2 %) | 0.8409 (+0.1 %) | 0.8388 (-0.1 %) |
| K_pz | 2.46 | 2.523 (+2.5 %) | 2.475 (+0.6 %) | 2.46 (-0.0 %) |
| C_T | 0.02 | 0.02597 (+29.8 %) | 0.02238 (+11.9 %) | 0.01978 (-1.1 %) |
| e0 | 0.25 | 0.2502 (+0.1 %) | 0.2492 (-0.3 %) | 0.25 (-0.0 %) |

### Relative error for every noise level and duration

| parameter | low/short | low/medium | low/long | nominal/short | nominal/medium | nominal/long | high/short | high/medium | high/long |
|---|---|---|---|---|---|---|---|---|---|
| tau_rise [s] | -0.7 % | -0.7 % | -0.7 % | -0.7 % | -0.7 % | -0.7 % | -4.8 % | -4.8 % | -0.7 % |
| tau_fall [s] | +1.3 % | +1.3 % | -0.8 % | +1.3 % | +1.3 % | -0.8 % | +5.4 % | +3.4 % | -0.8 % |
| kf0 [N] | -4.7e-04 abs | -4.7e-04 abs | +1.2e-04 abs | -9.6e-04 abs | -2.9e-04 abs | +1.4e-04 abs | -4.8e-03 abs | -6.5e-04 abs | +1.9e-04 abs |
| kf1 [N] | +4.1 % | +4.1 % | -0.0 % | +10.7 % | -1.0 % | +0.5 % | +56.3 % | -9.9 % | +3.7 % |
| kf2 [N] | -0.4 % | -0.3 % | -0.2 % | -1.2 % | +0.4 % | -0.3 % | -6.8 % | +1.9 % | -1.2 % |
| thrust(u=0.5) [N] | -0.1 % | -0.1 % | -0.0 % | -0.2 % | -0.2 % | -0.0 % | -1.2 % | -1.0 % | -0.1 % |
| thrust(u=1) [N] | -0.1 % | -0.1 % | -0.1 % | -0.4 % | +0.1 % | -0.2 % | -2.1 % | +0.5 % | -0.6 % |
| I_xx [kg m2] | -0.1 % | -0.1 % | +0.2 % | -0.2 % | +0.1 % | +0.1 % | -0.8 % | +0.7 % | -0.3 % |
| I_yy [kg m2] | -0.1 % | -0.0 % | +0.2 % | -0.2 % | +0.1 % | +0.1 % | -0.8 % | +0.9 % | -0.3 % |
| I_zz [kg m2] (ratio) | -0.1 % | -0.1 % | +0.1 % | -0.2 % | +0.0 % | +0.0 % | -0.9 % | +0.7 % | -0.4 % |
| k_tau [m] | -0.1 % | -0.1 % | +0.1 % | -0.1 % | -0.2 % | +0.1 % | +0.0 % | -0.4 % | +0.3 % |
| d_x [kg/s] | -0.0 % | -0.0 % | +0.0 % | -0.3 % | -0.1 % | +0.1 % | -1.3 % | -0.3 % | +0.4 % |
| d_y [kg/s] | +0.1 % | -0.0 % | -0.0 % | +0.3 % | -0.0 % | -0.0 % | +1.4 % | -0.1 % | -0.3 % |
| d_z [kg/s] | -2.7 % | -0.6 % | -1.4 % | -6.0 % | -2.0 % | -0.5 % | -32.6 % | -14.4 % | +2.4 % |
| d_z thrust-fit [kg/s] | +4.9 % | +5.2 % | -4.5 % | -1.9 % | +11.8 % | -0.8 % | -23.2 % | +37.1 % | +15.3 % |
| C_N90 | -0.1 % | -0.0 % | -0.0 % | -0.9 % | -0.1 % | -0.1 % | -4.6 % | -0.5 % | -0.7 % |
| K_px | -0.2 % | +0.0 % | -0.0 % | -1.2 % | +0.1 % | -0.1 % | -6.7 % | +0.4 % | -0.6 % |
| K_pz | +0.4 % | +0.1 % | +0.0 % | +2.5 % | +0.6 % | -0.0 % | +14.8 % | +0.6 % | -0.8 % |
| C_T | +4.8 % | +1.9 % | -0.1 % | +29.8 % | +11.9 % | -1.1 % | +149.2 % | +59.9 % | -6.5 % |
| e0 | -0.1 % | -0.2 % | -0.0 % | +0.1 % | -0.3 % | -0.0 % | +0.9 % | -0.7 % | +0.0 % |

### Notebook-faithful whoop chain (one T_m, I from <dw,tau>/<dw,dw>, no gyroscopic term, no drag)

| parameter | true | short | medium | long |
|---|---|---|---|---|
| T_m [s] (truth: rise 0.04 / fall 0.08) | - | 0.05424 | 0.05424 | 0.05424 |
| thrust(u=0.5) [N] | 0.0825 | 0.0856 (+3.8 %) | 0.08488 (+2.9 %) | 0.08496 (+3.0 %) |
| thrust(u=1) [N] | 0.3 | 0.3052 (+1.7 %) | 0.3122 (+4.1 %) | 0.3084 (+2.8 %) |
| kf0 [N] | 0 | 0.009078 (+9.1e-03 abs) | 0.01313 (+1.3e-02 abs) | 0.01068 (+1.1e-02 abs) |
| I_xx [kg m2] | 1.2e-05 | 1.168e-05 (-2.7 %) | 1.168e-05 (-2.7 %) | 1.156e-05 (-3.6 %) |
| I_yy [kg m2] | 1.2e-05 | 1.181e-05 (-1.6 %) | 1.167e-05 (-2.7 %) | 1.169e-05 (-2.6 %) |
| I_zz [kg m2] (ratio) | 2.2e-05 | 2.152e-05 (-2.2 %) | 2.139e-05 (-2.8 %) | 2.13e-05 (-3.2 %) |
| k_tau [m] | 0.005 | 0.004859 (-2.8 %) | 0.00483 (-3.4 %) | 0.00482 (-3.6 %) |
| C_N90 (plate fit on top, no body drag) | 1.15 | 1.723 (+49.8 %) | 1.696 (+47.5 %) | 1.672 (+45.4 %) |
| C_T (plate fit on top, no body drag) | 0.02 | 0.5029 (+2414.5 %) | 0.5127 (+2463.4 %) | 0.5036 (+2418.1 %) |

Other diagnostics (nominal, medium): symmetric T_m with our asymmetric-model chain = 0.0542 s; rise/fall without the drag column = 0.0381 / 0.0777 s; reciprocal inertia without gyroscopic term = 1.2301e-05 / 1.2242e-05; hover command median 0.470 vs predicted 0.470 (true 0.469); e0 from yaw only = 0.2497.

### Monte Carlo, nominal noise, medium duration, 5 noise seeds: mean +- std of the error

| parameter | true | mean | error mean +- std | formal std (plate fit) |
|---|---|---|---|---|
| tau_rise [s] | 0.04 | 0.03974 | -0.7 +- 0.0 % |  |
| tau_fall [s] | 0.08 | 0.08005 | +0.1 +- 1.1 % |  |
| kf0 [N] | 0 | 0.000309 | +3.1e-04 +- 1.4e-03 abs |  |
| kf1 [N] | 0.03 | 0.0287 | -4.3 +- 20.1 % |  |
| kf2 [N] | 0.27 | 0.2713 | +0.5 +- 2.4 % |  |
| thrust(u=0.5) [N] | 0.0825 | 0.08248 | -0.0 +- 0.1 % |  |
| thrust(u=1) [N] | 0.3 | 0.3003 | +0.1 +- 0.6 % |  |
| I_xx [kg m2] | 1.2e-05 | 1.201e-05 | +0.1 +- 0.2 % |  |
| I_yy [kg m2] | 1.2e-05 | 1.201e-05 | +0.1 +- 0.1 % |  |
| I_zz [kg m2] (ratio) | 2.2e-05 | 2.201e-05 | +0.0 +- 0.1 % |  |
| k_tau [m] | 0.005 | 0.005 | -0.0 +- 0.1 % |  |
| d_x [kg/s] | 0.008 | 0.007997 | -0.0 +- 0.2 % |  |
| d_y [kg/s] | 0.008 | 0.008014 | +0.2 +- 0.3 % |  |
| d_z [kg/s] | 0.004 | 0.003994 | -0.1 +- 2.8 % |  |
| d_z thrust-fit [kg/s] | 0.004 | 0.003855 | -3.6 +- 10.2 % |  |
| C_N90 | 1.15 | 1.147 | -0.2 +- 0.2 % | 0.3 % |
| K_px | 0.84 | 0.8372 | -0.3 +- 0.3 % | 0.5 % |
| K_pz | 2.46 | 2.485 | +1.0 +- 0.6 % | 1.6 % |
| C_T | 0.02 | 0.02095 | +4.8 +- 11.2 % | 8.7 % |
| e0 | 0.25 | 0.2499 | -0.0 +- 0.3 % | 0.0 % |

### Plate identifiability per manoeuvre (nominal noise, medium plate flight): formal relative std

| manoeuvre | time [s] | C_N90 | K_px | K_pz | C_T | e0 |
|---|---|---|---|---|---|---|
| circles | 59 | 0.5 % | 0.5 % | 2.3 % | 8.8 % | 0.013 % (fit 0.249) |
| forward passes | 17 | 3.2e+02 % | 13 % | 43 % | 18 % | 0.21 % (fit 0.248) |
| sideways passes | 17 | 0.63 % | 7.5 % | 6.9 % | 3.3e+02 % | 0.053 % (fit 0.246) |
| diagonal passes | 6 | 24 % | 6.7 % | 20 % | 55 % | 0.049 % (fit 0.249) |
| sideways climbs + vertical | 18 | 0.77 % | 11 % | 2.5 % | 4.3e+02 % | 0.063 % (fit 0.250) |
| all | 116 | 0.3 % | 0.47 % | 1.5 % | 7.8 % | 0.012 % (fit 0.249) |

### Prediction on held-out flights (identified model from nominal noise, medium duration)

Held-out excitation (no plate): one-step = specific force and angular acceleration predicted at the measured state vs noise-free truth; open-loop rollouts from the mocap/gyro state with logged commands, windows every 0.5 s (57 windows).

| model | one-step sf [m/s2] | one-step dw [rad/s2] | pos/att @ 0.02 s | pos/att @ 0.10 s | pos/att @ 0.25 s | pos/att @ 0.50 s |
|---|---|---|---|---|---|---|
| truth | 0.006 | 0.1 | 0.07 cm / 0.23 deg | 0.26 cm / 0.29 deg | 0.64 cm / 0.52 deg | 1.37 cm / 0.98 deg |
| identified | 0.019 | 0.2 | 0.07 cm / 0.23 deg | 0.26 cm / 0.29 deg | 0.64 cm / 0.55 deg | 1.39 cm / 1.05 deg |
| notebook-faithful | 0.195 | 3.6 | 0.07 cm / 0.23 deg | 0.28 cm / 1.12 deg | 1.02 cm / 5.51 deg | 7.89 cm / 15.85 deg |

Held-out plate flight: one-step = specific force and angular acceleration predicted at the measured state vs noise-free truth; open-loop rollouts from the mocap/gyro state with logged commands, windows every 0.5 s (170 windows).

| model | one-step sf [m/s2] | one-step dw [rad/s2] | pos/att @ 0.02 s | pos/att @ 0.10 s | pos/att @ 0.25 s | pos/att @ 0.50 s |
|---|---|---|---|---|---|---|
| truth | 0.012 | 0.7 | 0.06 cm / 0.23 deg | 0.25 cm / 0.33 deg | 0.64 cm / 1.18 deg | 2.41 cm / 7.74 deg |
| identified | 0.023 | 0.7 | 0.06 cm / 0.23 deg | 0.25 cm / 0.34 deg | 0.65 cm / 1.25 deg | 2.52 cm / 8.27 deg |
| identified, plate model removed | 0.708 | 59.2 | 0.06 cm / 0.71 deg | 0.28 cm / 17.10 deg | 6.65 cm / 72.13 deg | 51.42 cm / 84.59 deg |
| notebook-faithful, no drag/plate | 1.128 | 60.0 | 0.07 cm / 0.72 deg | 0.41 cm / 17.29 deg | 6.34 cm / 73.01 deg | 52.66 cm / 84.67 deg |

### Reading the results

Verified on synthetic data (the chain recovers the truth; on noise-free data every parameter comes back
exactly, which checks that `identify.py` and `model.py` are consistent):
- **Motor lag**: the 2-D grid recovers rising/falling within one grid step (fine grid resolution 1.7 ms,
  so the -0.7 % on T_rise is the grid). The notebook's single T_m lands at 0.054 s, between the two, and
  then biases the thrust curve (+3 % thrust, kf0 = +0.01 N instead of 0) and the inertias (-3 %).
- **Thrust curve**: thrust at u = 0.5 and u = 1 within 0.5 % (2 % at high noise). The individual
  coefficients trade off (kf1 is +-20 % across noise seeds) because the excitation only covers u ~
  0.25-0.6 per motor (see `thrust_fit.png`); compare curves, not coefficients. **u = 1 is an
  extrapolation.**
- **Inertia, k_tau**: within 1 % at all noise levels with the reciprocal fit + gyroscopic term. The
  notebook formula is biased low (-3 % nominal, about -50 % at high noise in a separate 60 s run) because
  of the noisy regressor. I_zz relies on the 1.832 ratio, which the JSON satisfies by construction
  (2.2/1.2 = 1.833): **I_zz is not identified from data**, only I_zz/k_tau is.
- **Body drag**: d_x, d_y within 0.5 % (they are exact regressions on rows without rotor thrust); d_z is
  weaker (2-6 % nominal, up to 30 % at high noise) because it sits on the thrust row.
- **Plate**: C_N90, K_px within 1 %, K_pz within 1-2.5 %, e0 within 0.3 % (nominal noise, >= 2 min).
  **C_T is weakly identifiable**: +-11 % std over noise seeds at nominal noise (medium), +30 % for the
  short flight, +150 % at high noise. The friction force is only ~3 % of the body drag at the same speed,
  so C_T is mostly a residual of the drag fit. Not doing the drag separation is catastrophic: on top of the
  notebook-faithful chain (no drag) C_N90 comes out +47 % and C_T x25.
- The formal standard errors are honest for the force coefficients (close to the Monte Carlo spread)
  but too optimistic for e0 (the 20 Hz low-pass makes the residual samples correlated, roughly a factor
  3.5 on the std); the Monte Carlo std (0.3 %) is the number to trust.
- **Prediction**: the identified model predicts the held-out flights as well as the true model (one-step
  specific force 0.02 vs 0.01 m/s2; position after 0.5 s open loop 1.4 vs 1.4 cm without plate, 2.5 vs
  2.4 cm with plate; the residual is the initial-state error from the noisy mocap/gyro). Removing the
  plate model, or using the notebook model, gives 50 cm and 85 deg after 0.5 s on the plate flight.

### Which manoeuvre identifies what

| parameter | from | notes |
|---|---|---|
| T_rise, T_fall, thrust curve | thrust phase of the excitation flight (collective multisine) | needs mocap for the d_z column; cover the full throttle range |
| I_xx, I_yy | roll/pitch phase (torque multisines 1-10 Hz) | sensitive to the time constants (I scales with the identified torque amplitude) |
| I_zz | not identifiable (ratio 1.832) | measure (bifilar pendulum) if it matters, especially with the plate |
| k_tau | yaw phase | given I_zz |
| d_x, d_y | no-plate fixed-heading circles and passes | exact regression on the x/y accelerometer rows |
| d_z | thrust phase (with the drag column) or vertical shuttles | weak |
| C_N90, K_px | fixed-heading circles alone give 0.5 % | circles sweep alpha 0-90 deg with the in-plane wind along x |
| K_pz | sideways passes and sideways climbs/vertical shuttles (2.5 %); circles help (2.3 %) through tilt | in-plane wind along z only happens with roll tilt in sideways flight or with vertical speed; no data below alpha ~ 40 deg with c^2 = 0 (see `plate_cn_alpha.png`), so the small-alpha slope K_pz is the least certain normal-force parameter |
| C_T | circles + forward passes, long flights | weak, confounded with body drag; forward (edge-on) passes alone give 18 % |
| e0 | every manoeuvre with a normal force; circles best | from the yaw and roll moment residual; depends on k_tau and I_zz |

## Recommended real flights

Order: weigh and balance; fly (1)-(2) without the plate, mount the plate, re-weigh, then (3).
1. **Paper excitation, no plate, mocap on**: >= 60 s in total (>= 20 s per phase; 30 s total already
   gives < 1 % on the whoop parameters with nominal noise, but real data is worse than our white noise).
   Hover-in-place. Thrust phase: fast throttle pumps from near idle to near full throttle (our synthetic
   data only reached u ~ 0.6 per motor, so on the real drone include short punches to full throttle).
   Roll/pitch: sharp stick doublets / stick stirring at 1-10 Hz, tilts up to ~45 deg. Yaw: fast yaw
   reversals. Log 3 s on level ground before arming.
2. **No plate, fixed heading (yaw held at 0)**: circles at 1, 2, 3, 4 m/s, CW and CCW (radius ~1 m at
   1-2 m/s, 1.5 m at 3 m/s, 2.7 m at 4 m/s for 6 m/s2 centripetal), forward and sideways shuttles over
   +-2 m at 2-3.5 m/s, a diagonal shuttle, sideways climbs/descents (45 deg in the y-z plane, ~2.5 m/s) and
   vertical shuttles at ~1.5 m/s: about 2 minutes (the "medium" case). This gives the body drag.
3. **With the plate, the same schedule as (2)**, >= 2 min, 4 min if the battery allows (C_T goes from
   +12 % to 1 %). Repeat the whole set with yaw offset by 90 deg if the room is not isotropic (walls,
   ventilation). Circles at fixed heading are the workhorse; sideways passes and sideways climbs are
   what pins down K_pz; do not skip them.
- Limits seen in simulation: at 4 m/s the plate's yaw moment (~1e-3 N m at alpha ~ 45 deg) is about half
  of the yaw authority, the heading drifted by up to 22 deg and tilts reached ~50 deg; the circle at 4
  m/s needs a ~5.5 m wide volume. 3-3.5 m/s may be the practical maximum.
- Log at >= 500 Hz (blackbox: gyro, accelerometer, motor outputs, battery voltage; bidirectional DShot
  eRPM if available, which would make the motor lag directly observable), mocap >= 100 Hz with markers off
  the plate, and a sharp yaw spike at the start and end for time synchronization.

## Caveats and open issues

- Everything is validated against the same model it identifies (no structural mismatch except the
  motor model in the notebook variant). Real data will add unmodelled effects: propeller downwash on the
  plate (k_w), rotor drag that is not linear, ground effect, battery sag (thrust curve depends on voltage;
  not modelled), unsteady plate aerodynamics (reduced frequency ~0.14 at 2 Hz, 3 m/s), plate flutter.
- Sensor model is optimistic: white noise, constant biases, IMU at the CoM, no onboard gyro/acc low-pass
  delay, perfect time sync, no mocap dropouts, still air. Vibration noise on a whoop accelerometer is
  coloured and large; check the real spectrum and filter both sides of every regression identically.
- Interpretation of the JSON to confirm with the `sim/` owner: lag on the motor state (L2F) rather than on
  thrust; `added_mass` as physical mass added to the vehicle with its thin-plate inertia.
- I_zz is never identified; with the plate the 1.832 ratio is not guaranteed. e0 and k_tau both depend on
  it.
- The plate's c^2 = 0 branch is only observed at alpha >= 40 deg; K_pz (the small-alpha slope with the
  wind along z) is extrapolated below that.
- Real-data loader not written yet: Betaflight blackbox fields (gyroADC, accSmooth, motor[0..3]) need the
  motor output range normalized to [0, 1] (including the idle offset, which ends up in kf0), the axis
  signs and the motor order mapped to the L2F frame and to `rotor_positions` (Betaflight numbers Quad-X
  motors differently; verify on the bench by spinning each motor), and the clocks aligned with the mocap.
- Simulation speed: pure numpy, roughly 2x faster than real time; the truth flights are cached.

