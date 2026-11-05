"""Steady straight flight with the plate: the least rotor thrust physics allows (a bound for the trained policies).

python plate_trim.py [--params params/whoop_nominal.json] [--plate-z Z] [--tilts 0,30,45,60,90] [--downwash 0,1]
                     [--speeds 2,2.5,3,3.5,4] [--json out.json] [--png out.png]

For a horizontal velocity V (still air, no rotation), the script searches the attitude (yaw relative to the path, roll,
pitch) and the total rotor thrust T that hold the whoop in unaccelerated flight with the least T, using the plate and
body-drag model of docs/plate_model.md (sysid/model.py, the same equations as the simulator). It then splits T between the
four rotors so that they also cancel the plate's moment, and checks that every rotor stays within its thrust range.
This is done per plate tilt theta about body x (normal (0, cos theta, sin theta), chord along x: 0 = the nominal
vertical plate, 90 = horizontal) and per downwash factor k_w (0: no downwash; 1: the whole plate in the rotors' induced
velocity, the crude bound of docs/plate_model.md). With k_w > 0 the plate force depends on T, so T is solved with the
attitude.

What the force balance says (neglecting the small in-plane friction and the body drag's own direction):
  - vertical plate (theta = 0): the rotor thrust is along body z and the plate's normal force along body y, so the two
    are perpendicular. In steady flight they must together cancel W = weight + body drag: T^2 + F_n^2 = |W|^2. The
    saving is second order, F_n^2 / (2 |W|).
  - horizontal plate (theta = 90): the normal force is along body z, the same axis as the thrust. To fly forward the
    thrust axis must lean into the path, so the air meets the plate from above, and the normal force pushes down: in
    steady level flight a horizontal plate parallel to the rotor discs can only cost thrust.
  - tilted plate: a wing set at 90 deg - theta to the rotor discs. Flying toward body -y, the air meets it from below
    and its normal force has an upward part: lift L and drag D_p at first order, T^2 = (W - L)^2 + (D + D_p)^2.
[assumption: quasi-steady plate model, rotors give pure thrust along body z]
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


def tilt_plate(P, theta_deg):
    """the plate tilted by theta about body x, as train.cpp --plate-tilt (whoop/params_runtime.h tilt_plate)"""
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s
    P["plate"]["n"] = np.array([0.0, c, s])
    P["plate"]["e1"] = np.array([1.0, 0.0, 0.0])


def residual_force(P, V, x):
    """force the rotors must give (body frame) for steady flight at V along world x with this attitude, minus the
    thrust T along body z (x = yaw, pitch, roll, T): zero in trim"""
    R = rotation_zyx(*x[:3])
    v_body = R.T @ np.array([V, 0.0, 0.0])
    F_aero, tau_aero = M.aero_wrench(P, v_body, np.zeros(3), thrust=x[3])
    need = R.T @ (-P["mass"] * P["g"]) - F_aero
    return need - np.array([0.0, 0.0, x[3]]), tau_aero, v_body


def trim(P, V):
    """least-thrust attitude: the rotor force must be along body z (x and y components zero) and equal to T"""
    best = None
    weight = P["mass"] * -P["g"][2]
    for yaw0 in np.radians(np.arange(-180, 180, 45)):
        for roll0 in np.radians([-20, 0, 20]):
            for pitch0 in (-0.15, 0.1):
                x0 = np.array([yaw0, pitch0, roll0, weight])
                cons = {"type": "eq", "fun": lambda x: residual_force(P, V, x)[0] * 100}
                # upright only (roll and pitch within 80 deg), so that the thrust stays positive
                bounds = [(-np.pi - 0.5, np.pi + 0.5), (-1.4, 1.4), (-1.4, 1.4), (0.0, 3 * weight)]
                r = optimize.minimize(lambda x: x[3], x0, constraints=[cons], method="SLSQP", bounds=bounds,
                                      options=dict(maxiter=500, ftol=1e-13))
                if not r.success or np.abs(residual_force(P, V, r.x)[0]).max() > 1e-7 or r.x[3] <= 0:
                    continue
                if best is None or r.x[3] < best.x[3]:
                    best = r
    x = best.x
    _, tau_aero, v_body = residual_force(P, V, x)
    T = x[3]
    R = rotation_zyx(*x[:3])
    F_plate = np.zeros(3)
    alpha = 0.0
    downwash = 0.0
    if P["plate"] is not None:
        F_plate, _, aux = M.plate_wrench(P["plate"], v_body, np.zeros(3), aux=True, thrust=T)
        alpha = float(np.degrees(aux["alpha"]))
        downwash = float(aux["v"][2] - v_body[2])
    # rotor thrusts that give T and cancel the aerodynamic moment
    A = np.vstack([P["rotor_thrust_dir"][:, 2], P["torque_per_thrust"].T])
    f = np.linalg.solve(A, np.r_[T, -tau_aero])
    f_hover = weight / 4
    yaw, pitch, roll = np.degrees(x[:3])
    yaw = (yaw + 180) % 360 - 180
    return dict(speed=V, thrust_to_weight=T / weight, thrust_newton=T, plate_lift_to_weight=(R @ F_plate)[2] / weight,
                plate_force_to_weight=np.linalg.norm(F_plate) / weight, yaw_to_path_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                alpha_deg=alpha, downwash=downwash, rotor_thrusts=f.tolist(),
                rotors_within_range=bool(np.all(f >= M.rotor_thrust(P, 0.0)) and np.all(f <= M.rotor_thrust(P, 1.0))),
                power_to_hover=float(np.mean((np.maximum(f, 0) / f_hover) ** 1.5)))


def plot(out, png, speeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY = "#012367"
    colors = {0: "#012367", 30: "#2A9D8F", 45: "#EF9100", 60: "#8E44AD", 90: "#D62828"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    base = {r["speed"]: r["thrust_newton"] for r in out["noplate"]}
    for key, rows in out.items():
        if key == "noplate":
            continue
        theta, kw = rows[0]["tilt_deg"], rows[0]["downwash_factor"]
        c = colors.get(int(round(theta)), "grey")
        ls = "-" if kw == 0 else "--"
        v = [r["speed"] for r in rows]
        label = f"θ = {theta:.0f}°" + ("" if kw == 0 else f", k_w = {kw:g}")
        axes[0].plot(v, [100 * r["thrust_newton"] / base[r["speed"]] for r in rows], ls, color=c, marker="o", ms=3, label=label)
        axes[1].plot(v, [100 * r["plate_lift_to_weight"] for r in rows], ls, color=c, marker="o", ms=3, label=label)
    axes[0].axhline(100, color="black", lw=0.8, ls=":")
    axes[0].set_ylabel("least rotor thrust (% of the bare whoop's)")
    axes[1].axhline(0, color="black", lw=0.8, ls=":")
    axes[1].set_ylabel("plate lift (% of the weight)")
    for ax in axes:
        ax.set_xlabel("speed (m/s)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Steady straight flight, least rotor thrust per plate tilt θ about body x (solid: no downwash; dashed: k_w = 1)",
                 color=NAVY, fontsize=11)
    fig.text(0.01, 0.01, "k_w = 1: the whole plate in the rotors' induced velocity (crude). For θ = 0-30° the term lets the plate turn the\n"
             "wake sideways at no cost in rotor thrust, so those dashed curves are not a bound (docs/sim.md).", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(png, dpi=120)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--params", default=str(REPO / "params" / "whoop_nominal.json"))
    a.add_argument("--plate-z", type=float, default=None, help="plate centroid height relative to the centre of mass [m]")
    a.add_argument("--tilts", default="0", help="plate tilts about body x [deg], comma-separated")
    a.add_argument("--downwash", default="0", help="downwash factors k_w, comma-separated")
    a.add_argument("--speeds", default="1,1.5,2,2.5,3,3.5,4")
    a.add_argument("--json", default=None)
    a.add_argument("--png", default=None)
    a.add_argument("--plot-from", default=None, help="only plot a JSON written by --json to --png")
    args = a.parse_args()
    if args.plot_from:
        plot(json.loads(Path(args.plot_from).read_text()), args.png, None)
        return
    raw = json.loads(Path(args.params).read_text())
    drag = raw["whoop"]["body_drag"]
    speeds = [float(v) for v in args.speeds.split(",")]
    out = {}
    cases = [("noplate", None, 0.0)] + [(None, float(t), float(k)) for k in args.downwash.split(",") for t in args.tilts.split(",")]
    for name, theta, kw in cases:
        P = M.load_params(args.params, plate=name is None, body_drag=drag)
        if name is None:
            if args.plate_z is not None:
                P["plate"]["rc"] = np.array([P["plate"]["rc"][0], P["plate"]["rc"][1], args.plate_z])
            tilt_plate(P, theta)
            P["plate"]["kw"] = kw
            if kw > 0:
                assert P["plate"]["A"] > 0 and P["plate"]["rc"][2] < np.min(P["rotor_pos"][:, 2]), "downwash needs the plate below the rotors"
            name = f"tilt{theta:g}_kw{kw:g}"
        rows = [trim(P, v) for v in speeds]
        for r in rows:
            r["tilt_deg"], r["downwash_factor"] = theta, kw
        out[name] = rows
        for r in rows:
            print(f"{name:14s} {r['speed']:.1f} m/s: thrust {100 * r['thrust_to_weight']:.1f} % of its weight ({1000 * r['thrust_newton']:.1f} mN), "
                  f"plate lift {100 * r['plate_lift_to_weight']:.1f} %, plate force {100 * r['plate_force_to_weight']:.1f} %, yaw to path "
                  f"{r['yaw_to_path_deg']:.0f} deg, roll {r['roll_deg']:.0f}, pitch {r['pitch_deg']:.0f}, alpha {r['alpha_deg']:.0f} deg, "
                  f"downwash {r['downwash']:.2f} m/s, power {100 * r['power_to_hover']:.1f} % of hover, rotors in range {r['rotors_within_range']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    if args.png:
        plot(out, args.png, speeds)


if __name__ == "__main__":
    main()
