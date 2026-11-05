"""Identify a whoop from synthetic flights and write the result as a parameter file for the simulator.

    python export_params.py --truth ../params/whoop_truth_example.json --out <dir>/whoop_identified.json
                            [--duration short|medium|long] [--noise low|nominal|high]

The sim-to-sim test of the whole pipeline:
  (a) synthetic.py flies the "true" drone of --truth (the same three flights as validate.py: the excitation flight
      without the plate, then the fixed-heading schedule without and with the plate), with blackbox and mocap noise;
  (b) identify.py identifies it, knowing only what is measured on the bench: the mass, the geometry and the plate size
      from params/whoop_nominal.json (never the true thrust curve, time constants, drag or plate coefficients);
  (c) the identified values go into a JSON in the format of params/whoop_nominal.json, which the simulator reads with
      --params (sim/src/train.cpp) for training.
Next to --out, <stem>_errors.json lists, per parameter, the truth, the identified value, the nominal value and the
relative errors of both.

What the identification does not estimate and the file keeps from params/whoop_nominal.json: the mass, the rotor
geometry, the plate size and position, the air density, the randomization ranges. I_zz comes from the ratio
1.832 (I_xx + I_yy) / 2 of the paper, as in identify.py.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import identify as I
import model as M
import synthetic as S

NOMINAL = M.DEFAULT_PARAMS
DURATIONS = {"short": (30, 60), "medium": (60, 120), "long": (120, 240)}  # as validate.py
SEEDS = dict(exc=1, noplate=2, plate=3)
# attitude gains of the manoeuvre controller for the plate flight (synthetic.Controller defaults: kR 120, 120, 60; kw 16, 16, 12)
GAINS_PLATE = dict(kR=(200.0, 200.0, 100.0), kw=(28.0, 28.0, 18.0))


def identify(truth, duration, noise):
    de, dp = DURATIONS[duration]
    Le = S.truth_flight("excitation", de, False, SEEDS["exc"], params=truth)
    Ln = S.truth_flight("plate", dp, False, SEEDS["noplate"], params=truth)
    # stiffer attitude control for the plate flight: with the default gains the example truth (C_N90 +20 %, motors 25 %
    # slower) tilts past 80 deg in the sideways climbs and diverges (56 deg at most with these). Only the manoeuvre
    # controller changes, not what is identified
    Lp = S.truth_flight("plate", dp, True, SEEDS["plate"], params=truth, gains=GAINS_PLATE)
    # what is measured on the bench: mass, geometry, plate size (the same in the truth file by construction)
    Pw, Pp = M.load_params(NOMINAL, plate=False), M.load_params(NOMINAL, plate=True)
    w = I.identify_whoop(S.measure(Le, noise, SEEDS["exc"]), Pw)
    dr = I.identify_body_drag(S.measure(Ln, noise, SEEDS["noplate"]), I.whoop_model(Pw, w))
    pr = I.identify_plate(S.measure(Lp, noise, SEEDS["plate"]), Pp, w, dr["body_drag"])
    th = pr["fit"]["theta"]
    flights = dict(excitation_s=float(Le["t"][-1]), no_plate_s=float(Ln["t"][-1]), plate_s=float(Lp["t"][-1]))
    return dict(kf=[float(x) for x in w["kf"]], k_tau=float(w["k_tau"]), tau_rise=float(w["tau_rise"]), tau_fall=float(w["tau_fall"]),
                J=[float(w["J"][i, i]) for i in range(3)], body_drag=[float(x) for x in dr["body_drag"]],
                CN90=float(th["CN90"]), Kp_x=float(th["Kpx"]), Kp_z=float(th["Kpz"]), CT=float(th["CT"]), e0=float(pr["e0"]["e0"]), flights=flights)


def as_params_file(identified, truth_path, duration, noise):
    """the identified values in the format of params/whoop_nominal.json (everything else copied from it)"""
    p = json.loads(Path(NOMINAL).read_text())
    out = copy.deepcopy(p)
    out["_comment"] = (f"Identified by sysid/export_params.py from synthetic flights of {Path(truth_path).name} "
                       f"({duration} flights, {noise} sensor noise): thrust curve, motor time constants, inertia "
                       "(I_zz from the 1.832 ratio), torque constant, body drag and plate coefficients. Mass, geometry, plate "
                       "size and randomization ranges are those of whoop_nominal.json (measured, not identified).")
    w, pl = out["whoop"], out["plate"]
    w["rotor_thrust_coefficients"] = identified["kf"]
    w["rotor_torque_constant"] = identified["k_tau"]
    w["rotor_time_constant_rising"] = identified["tau_rise"]
    w["rotor_time_constant_falling"] = identified["tau_fall"]
    J = identified["J"]
    w["J"] = [[J[0], 0, 0], [0, J[1], 0], [0, 0, J[2]]]
    w["body_drag"] = identified["body_drag"]
    for k in ("CN90", "Kp_x", "Kp_z", "CT", "e0"):
        pl[k] = identified[k]
    return out


def flat(p):
    """the parameters compared in the error table, from a parameter file"""
    w, pl = p["whoop"], p["plate"]
    kf = w["rotor_thrust_coefficients"]
    thrust = lambda u: kf[0] + kf[1] * u + kf[2] * u * u
    return {"thrust at u = 0.5 [N]": thrust(0.5), "thrust at u = 1 [N]": thrust(1.0),
            "tau_rise [s]": w["rotor_time_constant_rising"], "tau_fall [s]": w["rotor_time_constant_falling"],
            "I_xx [kg m2]": w["J"][0][0], "I_yy [kg m2]": w["J"][1][1], "k_tau [m]": w["rotor_torque_constant"],
            "d_x [kg/s]": w["body_drag"][0], "d_y [kg/s]": w["body_drag"][1], "d_z [kg/s]": w["body_drag"][2],
            "C_N90": pl["CN90"], "K_px": pl["Kp_x"], "K_pz": pl["Kp_z"], "C_T": pl["CT"], "e0": pl["e0"]}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--truth", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--duration", default="medium", choices=list(DURATIONS))
    a.add_argument("--noise", default="nominal", choices=list(S.NOISE))
    args = a.parse_args()
    identified = identify(args.truth, args.duration, args.noise)
    out = as_params_file(identified, args.truth, args.duration, args.noise)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    truth, nominal, ident = (flat(json.loads(Path(f).read_text())) for f in (args.truth, NOMINAL, out_path))
    rel = lambda x, ref: 100 * (x - ref) / ref
    rows = {k: dict(truth=truth[k], identified=ident[k], nominal=nominal[k], identified_error_pct=rel(ident[k], truth[k]),
                    nominal_error_pct=rel(nominal[k], truth[k])) for k in truth}
    errors_path = out_path.with_name(out_path.stem + "_errors.json")
    errors_path.write_text(json.dumps(dict(flights=identified["flights"], noise=args.noise, parameters=rows), indent=1))
    print(f"flights: {identified['flights']}")
    print(f"{'parameter':24s} {'truth':>11s} {'identified':>11s} {'error':>8s} {'nominal':>11s} {'error':>8s}")
    for k, r in rows.items():
        print(f"{k:24s} {r['truth']:11.4g} {r['identified']:11.4g} {r['identified_error_pct']:+7.1f}% {r['nominal']:11.4g} {r['nominal_error_pct']:+7.1f}%")
    print("wrote", out_path, "and", errors_path)


if __name__ == "__main__":
    main()
