"""Animate a test flight of a trained policy from flights.csv (written by whoop_train_*): the whole trajectory with
the reference on the left, and a close-up that follows the whoop on the right, drawn to scale with its plate and the
plate force.

python animate_flight.py <flights.csv> <out.gif> [--flight K] [--title TEXT]
--flight -1 (default) picks the recorded flight with the largest mean plate force.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

PARAMS = Path(__file__).resolve().parents[2] / "params" / "whoop_nominal.json"
NAVY, ORANGE, GREY, TEAL = "#012367", "#EF9100", "#9A9A9A", "#2A9D8F"


def rotation(roll, pitch, yaw):
    """body to world, from the ZYX angles written by train.cpp (degrees)"""
    r, p, y = np.radians([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("flights")
    a.add_argument("out")
    a.add_argument("--flight", type=int, default=-1)
    a.add_argument("--title", default="")
    a.add_argument("--every", type=int, default=4, help="draw one simulation step out of N")
    args = a.parse_args()
    with open(args.flights) as f:
        rows = list(csv.DictReader(f))
    params = json.loads(PARAMS.read_text())
    w, plate = params["whoop"], params["plate"]
    weight = (w["mass"] + plate["added_mass"]) * 9.81
    flights = sorted({int(r["flight"]) for r in rows})
    col = lambda f, c: np.array([float(r[c]) for r in rows if int(r["flight"]) == f])
    if args.flight < 0:
        args.flight = max(flights, key=lambda f: np.mean(np.sqrt(col(f, "plate_fx") ** 2 + col(f, "plate_fy") ** 2
                                                                  + col(f, "plate_fz") ** 2)))
    g = lambda c: col(args.flight, c)
    t, pos, ref = g("t"), np.c_[g("x"), g("y"), g("z")], np.c_[g("reference_x"), g("reference_y"), g("reference_z")]
    angles = np.c_[g("roll_deg"), g("pitch_deg"), g("yaw_deg")]
    force = np.c_[g("plate_fx"), g("plate_fy"), g("plate_fz")]
    speed = np.linalg.norm(np.c_[g("vx"), g("vy"), g("vz")], axis=1)
    alpha = g("plate_alpha_deg")

    # the whoop in its body frame: arms to the rotors, rotor discs, and the plate rectangle in the x-z plane
    rotors = np.array(w["rotor_positions"])
    disc = 0.02 * np.c_[np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24)), np.zeros(24)]
    cx, cz, c = plate["chord_x"], plate["height_z"], np.array(plate["centroid"])
    plate_corners = c + np.array([[-cx / 2, 0, -cz / 2], [cx / 2, 0, -cz / 2], [cx / 2, 0, cz / 2],
                                  [-cx / 2, 0, cz / 2], [-cx / 2, 0, -cz / 2]])
    has_plate = plate["area"] > 0 and np.any(np.abs(force) > 0)

    fig = plt.figure(figsize=(11, 5.2))
    world = fig.add_subplot(1, 2, 1, projection="3d")
    close = fig.add_subplot(1, 2, 2, projection="3d")
    fig.suptitle(args.title or f"Test flight {args.flight + 1}", color=NAVY, fontsize=13)
    lo, hi = np.minimum(pos.min(0), ref.min(0)) - 0.2, np.maximum(pos.max(0), ref.max(0)) + 0.2
    span = (hi - lo).max() / 2
    mid = (hi + lo) / 2
    frames = range(0, len(t), args.every)

    def draw(i):
        for ax in (world, close):
            ax.cla()
        world.plot(*ref.T, color=ORANGE, ls="--", lw=1.5, label="reference")
        world.plot(*pos[:i + 1].T, color=NAVY, lw=2, label="whoop")
        world.scatter(*pos[i], color=NAVY, s=30)
        world.scatter(*ref[i], color=ORANGE, s=30)
        for k, ax_set in enumerate((world.set_xlim, world.set_ylim, world.set_zlim)):
            ax_set(mid[k] - span, mid[k] + span)
        world.set_xlabel("x (m)", fontsize=9)
        world.set_ylabel("y (m)", fontsize=9)
        world.set_zlabel("z (m)", fontsize=9)
        world.tick_params(labelsize=7)
        world.legend(loc="upper left", fontsize=8, frameon=False)
        world.set_title(f"t = {t[i]:.2f} s, {speed[i]:.1f} m/s", fontsize=11)

        R, p = rotation(*angles[i]), pos[i]
        body = lambda pts: (R @ pts.T).T + p
        for r in rotors:
            close.plot(*body(np.array([[0, 0, 0], r])).T, color=GREY, lw=2)
            close.plot(*body(disc + r).T, color=NAVY, lw=1.5)
        if has_plate:
            corners = body(plate_corners)
            close.plot(*corners.T, color=TEAL, lw=2.5)
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            close.add_collection3d(Poly3DCollection([corners[:4]], color=TEAL, alpha=0.35))
            f_world = R @ force[i]
            start = body(c[None])[0]
            close.quiver(*start, *(0.25 * f_world / weight), color=ORANGE, lw=2)  # 0.25 m = one weight
            close.set_title(f"plate force {np.linalg.norm(force[i]) / weight * 100:.0f} % of the weight, "
                            f"α = {alpha[i]:.0f}°", fontsize=11)
        else:
            close.set_title("no plate", fontsize=11)
        h = 0.12
        close.set_xlim(p[0] - h, p[0] + h)
        close.set_ylim(p[1] - h, p[1] + h)
        close.set_zlim(p[2] - h, p[2] + h)
        close.set_xticks([])
        close.set_yticks([])
        close.set_zticks([])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    fps = max(1, int(round(1 / ((t[1] - t[0]) * args.every))))  # real time
    anim.save(args.out, writer=PillowWriter(fps=fps), dpi=90)
    print("wrote", args.out, "flight", args.flight, "frames", len(frames), "fps", fps)


if __name__ == "__main__":
    main()
