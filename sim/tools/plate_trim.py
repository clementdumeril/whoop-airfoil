"""Steady straight flight with the plate: the least rotor thrust physics allows (a bound for the trained policies).

python plate_trim.py [--params params/whoop_nominal.json] [--plate-z Z] [--speeds 1,2,3,4] [--json out.json]

For a horizontal velocity V (still air, no rotation), the script searches the attitude (yaw relative to the path, roll,
pitch) and the total rotor thrust T that hold the whoop in unaccelerated flight with the least T, using the plate and
body-drag model of docs/plate_model.md (sysid/model.py, the same equations as the simulator). It then splits T between the
four rotors so that they also cancel the plate's moment, and checks that every rotor stays within its thrust range.

Why the plate can only save a little thrust here: the rotor thrust is along body z and the plate's normal force along
body y, so the two are perpendicular. In steady flight they must together cancel W = weight + body drag, which gives
T^2 + F_n^2 = |W|^2 (neglecting the small in-plane friction). Any normal force F_n lowers T, and the vertical part of
the plate force is then F_n^2 / |W|; the largest F_n is the plate face-on to the air, C_N90 q S.
[assumption: quasi-steady plate model, no propeller downwash on the plate, rotors give pure thrust along body z]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sysid"))
import model as M  # noqa: E402


def rotation_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def residual_force(P, V, angles):
    """force the rotors must give (body frame) for steady flight at V along world x with this attitude"""
    R = rotation_zyx(*angles)
    v_body = R.T @ np.array([V, 0.0, 0.0])
    F_aero, tau_aero = M.aero_wrench(P, v_body, np.zeros(3))
    need = R.T @ (-P["mass"] * P["g"]) - F_aero
    return need, tau_aero, v_body


def trim(P, V):
    """least-thrust attitude: the rotor force must be along body z (x and y components zero)"""
    best = None
    for yaw0 in np.radians(np.arange(-90, 91, 15)):
        for roll0 in np.radians([-20, 0, 20]):
            x0 = np.array([yaw0, -0.05, roll0])
            cons = {"type": "eq", "fun": lambda a: residual_force(P, V, a)[0][:2] * 100}
            # upright only (roll and pitch within 80 deg), so that the thrust stays positive
            bounds = [(-np.pi, np.pi), (-1.4, 1.4), (-1.4, 1.4)]
            r = optimize.minimize(lambda a: residual_force(P, V, a)[0][2], x0, constraints=[cons], method="SLSQP",
                                  bounds=bounds, options=dict(maxiter=300, ftol=1e-12))
            if not r.success or np.abs(residual_force(P, V, r.x)[0][:2]).max() > 1e-7 or r.fun <= 0:
                continue
            if best is None or r.fun < best.fun:
                best = r
    need, tau_aero, v_body = residual_force(P, V, best.x)
    T = need[2]
    R = rotation_zyx(*best.x)
    F_plate = np.zeros(3)
    alpha = 0.0
    if P["plate"] is not None:
        F_plate, _, aux = M.plate_wrench(P["plate"], v_body, np.zeros(3), aux=True)
        alpha = float(np.degrees(aux["alpha"]))
    weight = P["mass"] * -P["g"][2]
    # rotor thrusts that give T and cancel the aerodynamic moment
    A = np.vstack([P["rotor_thrust_dir"][:, 2], P["torque_per_thrust"].T])
    f = np.linalg.solve(A, np.r_[T, -tau_aero])
    f_hover = weight / 4
    yaw, pitch, roll = np.degrees(best.x)
    return dict(speed=V, thrust_to_weight=T / weight, plate_lift_to_weight=(R @ F_plate)[2] / weight,
                plate_force_to_weight=np.linalg.norm(F_plate) / weight, yaw_to_path_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                alpha_deg=alpha, rotor_thrusts=f.tolist(), rotors_within_range=bool(np.all(f >= M.rotor_thrust(P, 0.0)) and np.all(f <= M.rotor_thrust(P, 1.0))),
                power_to_hover=float(np.mean((np.maximum(f, 0) / f_hover) ** 1.5)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--params", default=str(REPO / "params" / "whoop_nominal.json"))
    a.add_argument("--plate-z", type=float, default=None, help="plate centroid height relative to the centre of mass [m]")
    a.add_argument("--speeds", default="1,1.5,2,2.5,3,3.5,4")
    a.add_argument("--json", default=None)
    args = a.parse_args()
    raw = json.loads(Path(args.params).read_text())
    drag = raw["whoop"]["body_drag"]
    out = {}
    for name, plate in (("noplate", False), ("plate", True)):
        P = M.load_params(args.params, plate=plate, body_drag=drag)
        if plate and args.plate_z is not None:
            P["plate"]["rc"] = np.array([P["plate"]["rc"][0], P["plate"]["rc"][1], args.plate_z])
        rows = [trim(P, float(v)) for v in args.speeds.split(",")]
        out[name] = rows
        for r in rows:
            print(f"{name:8s} {r['speed']:.1f} m/s: thrust {100 * r['thrust_to_weight']:.1f} % of its weight, plate lift "
                  f"{100 * r['plate_lift_to_weight']:.1f} %, plate force {100 * r['plate_force_to_weight']:.1f} %, yaw to path "
                  f"{r['yaw_to_path_deg']:.0f} deg, roll {r['roll_deg']:.0f}, pitch {r['pitch_deg']:.0f}, alpha {r['alpha_deg']:.0f} deg, "
                  f"power {100 * r['power_to_hover']:.1f} % of hover, rotors in range {r['rotors_within_range']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
