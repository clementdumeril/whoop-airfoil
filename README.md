# whoop-airfoil

A 30 g quadrotor ("whoop", 75 mm) with a vertical 3 mm EPP foam plate under it, flown by a policy trained in
simulation that learns to use the plate's aerodynamic force. UC Berkeley MEng capstone, ARPL lab (Giuseppe Loianno,
Jonas Eschmann).

The plan follows two ARPL papers: identify the drone from a few flights (Eschmann, Albani, Loianno, "Data-driven
system identification of quadrotors subject to motor delays", IROS 2024), then train with the recipe of "Learning to
Fly in Seconds" (RA-L 2024) and run the policy on board in Betaflight.

## Status

- [x] Simulator: the L2F multirotor environment of RLtools with the whoop and the plate force and torque, unit tests
  (24/24), hover and velocity tasks, training with SAC and the L2F update intervals ([`docs/sim.md`](docs/sim.md))
- [x] Identification tested on synthetic flights, with and without the plate ([`sysid/README.md`](sysid/README.md))
- [ ] Body drag of the whoop in the training simulator (the identification already has it)
- [ ] 10-seed trainings on Savio (`sim/slurm/`)
- [ ] Real flights in the motion-capture room, identification of the real drone
- [ ] Policy on board (rl-tools/betaflight)

All parameters in [`params/whoop_nominal.json`](params/whoop_nominal.json) are assumptions until the drone is weighed
and identified.

## First results (simulation only)

One seed per variant, 300k steps on a laptop core, 100 test flights each ([`docs/sim.md`](docs/sim.md)):

| Task | Without the plate | With the plate |
| --- | --- | --- |
| Hover from random starts | 100 %, 0.14 m, 59 s | 96 %, 0.12 m, 50 s |
| Fly at 1–4 m/s | 93 %, 0.44 m, 84 s | 91 %, 0.26 m, 87 s |

Success rate, position RMSE over the second half of the flights, training time. In the velocity task the plate force
averages 8 % of the weight, up to 50 %.

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
