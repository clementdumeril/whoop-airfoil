# whoop-airfoil

A 30 g quadrotor ("whoop", 75 mm) with a vertical 3 mm EPP foam plate under it, flown by a policy trained in
simulation that learns to use the plate's aerodynamic force. UC Berkeley MEng capstone, ARPL lab (Giuseppe Loianno,
Jonas Eschmann).

The plan follows two ARPL papers: identify the drone from a few flights (Eschmann, Albani, Loianno, "Data-driven
system identification of quadrotors subject to motor delays", IROS 2024), then train with the recipe of "Learning to
Fly in Seconds" (RA-L 2024) and run the policy on board in Betaflight.

## Status

- [x] Simulator: the L2F multirotor environment of RLtools with the whoop, its body drag and the plate force and
  torque, unit tests (29/29), hover and velocity tasks, training with SAC and the L2F update intervals
  ([`docs/sim.md`](docs/sim.md))
- [x] Randomization of the plate coefficients and the body drag in training; optional motor cuts in training and a
  motor-cut scenario
- [x] Identification tested on synthetic flights, with and without the plate ([`sysid/README.md`](sysid/README.md))
- [ ] 10-seed trainings on Savio (`sim/slurm/`)
- [ ] Real flights in the motion-capture room, identification of the real drone
- [ ] Policy on board (rl-tools/betaflight)

All parameters in [`params/whoop_nominal.json`](params/whoop_nominal.json) are assumptions until the drone is weighed
and identified.

## First results (simulation only)

One seed per variant, 300k steps on a laptop core, 100 test flights each, with the body drag and randomized
aerodynamics ([`docs/sim.md`](docs/sim.md)):

| Task | Without the plate | With the plate |
| --- | --- | --- |
| Hover from random starts | 99 %, 0.12 m, 46 s | 98 %, 0.35 m, 52 s |
| Fly at 1–4 m/s | 87 %, 0.40 m, 51 s | 92 %, 0.22 m, 57 s |

Success rate, position RMSE over the second half of the flights, training time. In the velocity task the plate force
averages 8 % of the weight, up to 64 %. At 1M steps the velocity task reaches 100 % without the plate and 97 % with it.

Motor cuts at 3 m/s: without the plate the policy recovers from a 0.4 s cut, losing 1.1–1.6 m. With the plate it
recovers from 0.2 s, but a longer cut rolls the whoop over before the motors return: the plate, under the centre of
mass, makes the powerless fall unstable in roll. Trained with cuts and without the curriculum, the plate policy comes
back from the roll-over in 3 of 5 flights, losing 1.7–3.3 m. The plate never holds the whoop up.

On synthetic flights of 2 min, the identification recovers the thrust curve, motor time constants, roll and pitch
inertia, torque constant and the plate's normal-force coefficients within about 1–2.5 % of the truth; the plate's
friction coefficient needs about 4 min ([`sysid/README.md`](sysid/README.md)).

`sim/tools/animate_flight.py` turns a `flights.csv` into an animation of the whoop and its plate.

## Layout

| Folder | Content |
| --- | --- |
| `params/` | whoop and plate parameters, shared by the simulator and the identification |
| `docs/` | the plate model ([`plate_model.md`](docs/plate_model.md)) and the simulator details ([`sim.md`](docs/sim.md)) |
| `sim/` | C++ simulator, tests and training on top of RLtools (commit b32d9985, not modified) |
| `sysid/` | Python identification and its validation on synthetic flights |

## Credits

The simulator reuses the L2F environment of [RLtools](https://github.com/rl-tools/rl-tools) and the identification
follows the notebook of [data-driven-system-identification](https://github.com/arplaboratory/data-driven-system-identification).
