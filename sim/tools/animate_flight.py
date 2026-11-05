"""Animate a flight from flights.csv (written by whoop_train_*, in training or in the motor-cut scenario): the
trajectory with the reference on the left, a close-up that follows the whoop on the right, drawn to scale with its plate
and the plate force, and the height against time below.

python animate_flight.py <flights.csv> <out.gif> [--flight K] [--title TEXT] [--every N] [--window M] [--frame T --png P]
--flight -1 (default) picks the recorded flight with the largest mean plate force, or flight 0 of a motor-cut scenario.
When the flight covers more than the window (default 1.5 m on each side), the left camera follows the whoop.
If the CSV has a motors_cut column, the cut is shown in red: a "MOTORS OFF" label, red rotor discs, the trajectory
segment flown without motors and a band on the height plot.
--frame T --png P writes the frame at time T [s] as a PNG instead of the animation (to check how it reads).
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PARAMS = Path(__file__).resolve().parents[2] / "params" / "whoop_nominal.json"
NAVY, ORANGE, GREY, TEAL, RED = "#012367", "#EF9100", "#9A9A9A", "#2A9D8F", "#D62828"


def rotation(roll, pitch, yaw):
    """body to world, from the ZYX angles written by train.cpp (degrees)"""
    r, p, y = np.radians([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def segments(mask):
    """(start, end) index pairs of the runs where mask is true, end exclusive"""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("flights")
    a.add_argument("out")
    a.add_argument("--flight", type=int, default=-1)
    a.add_argument("--title", default="")
    a.add_argument("--every", type=int, default=4, help="draw one simulation step out of N")
    a.add_argument("--window", type=float, default=1.5, help="half-width of the following camera [m]")
    a.add_argument("--frame", type=float, default=None, help="write only the frame at this time [s] to --png")
    a.add_argument("--png", default=None)
    args = a.parse_args()
    with open(args.flights) as f:
        rows = list(csv.DictReader(f))
    params = json.loads(PARAMS.read_text())
    w, plate = params["whoop"], params["plate"]
    flights = sorted({int(r["flight"]) for r in rows})
    col = lambda f, c: np.array([float(r[c]) for r in rows if int(r["flight"]) == f])
    has_cut = "motors_cut" in rows[0]
    if args.flight < 0:
        if has_cut:
            args.flight = flights[0]
        else:
            args.flight = max(flights, key=lambda f: np.mean(np.sqrt(col(f, "plate_fx") ** 2 + col(f, "plate_fy") ** 2
                                                                      + col(f, "plate_fz") ** 2)))
    g = lambda c: col(args.flight, c)
    t, pos, ref = g("t"), np.c_[g("x"), g("y"), g("z")], np.c_[g("reference_x"), g("reference_y"), g("reference_z")]
    angles = np.c_[g("roll_deg"), g("pitch_deg"), g("yaw_deg")]
    force = np.c_[g("plate_fx"), g("plate_fy"), g("plate_fz")]
    speed = np.linalg.norm(np.c_[g("vx"), g("vy"), g("vz")], axis=1)
    alpha = g("plate_alpha_deg")
    cut = g("motors_cut") > 0.5 if has_cut else np.zeros(len(t), bool)
    cut_runs = segments(cut)
    has_plate = plate["area"] > 0 and np.any(np.abs(force) > 0)
    weight = (w["mass"] + (plate["added_mass"] if has_plate else 0)) * 9.81

    # the whoop in its body frame: arms to the rotors, rotor discs, and the plate rectangle in the x-z plane
    rotors = np.array(w["rotor_positions"])
    disc = 0.02 * np.c_[np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24)), np.zeros(24)]
    cx, cz, c = plate["chord_x"], plate["height_z"], np.array(plate["centroid"])
    plate_corners = c + np.array([[-cx / 2, 0, -cz / 2], [cx / 2, 0, -cz / 2], [cx / 2, 0, cz / 2],
                                  [-cx / 2, 0, cz / 2], [-cx / 2, 0, -cz / 2]])

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.18, wspace=0.02, left=0.06, right=0.97, top=0.9, bottom=0.08)
    world = fig.add_subplot(grid[0, 0], projection="3d")
    close = fig.add_subplot(grid[0, 1], projection="3d")
    height = fig.add_subplot(grid[1, :])
    fig.suptitle(args.title or f"Flight {args.flight + 1}", color=NAVY, fontsize=13)
    # the whole flight fits in a fixed view, or the camera follows the whoop (forward flight)
    lo, hi = np.minimum(pos.min(0), ref.min(0)) - 0.2, np.maximum(pos.max(0), ref.max(0)) + 0.2
    follow = (hi - lo).max() > 2 * args.window
    span = args.window if follow else (hi - lo).max() / 2
    mid_fixed = (hi + lo) / 2
    every = args.every
    frames = list(range(0, len(t), every))
    if args.frame is not None:
        frames = [int(np.argmin(np.abs(t - args.frame)))]

    # height plot, drawn once; a cursor moves along it
    height.plot(t, ref[:, 2], color=ORANGE, ls="--", lw=1.5, label="reference")
    height.plot(t, pos[:, 2], color=NAVY, lw=2, label="whoop")
    for s, e in cut_runs:
        height.axvspan(t[s] - (t[1] - t[0]), t[e - 1], color=RED, alpha=0.15, lw=0, label="motors off" if s == cut_runs[0][0] else None)
    height.set_xlim(0, t[-1])
    # a whoop that falls for good would squash the scale: show at most 3 m below the reference, and say where it ends
    z_low = min(pos[:, 2].min(), ref[:, 2].min())
    if z_low < -3:
        height.set_ylim(-3, max(pos[:, 2].max(), ref[:, 2].max()) + 0.3)
        height.text(0.99, 0.08, f"lowest point z = {z_low:.0f} m (below the plot)", transform=height.transAxes, ha="right", fontsize=9,
                    color=RED)
    height.set_xlabel("t (s)", fontsize=9)
    height.set_ylabel("height z (m)", fontsize=9)
    height.tick_params(labelsize=8)
    height.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    height.grid(alpha=0.3)
    cursor = height.axvline(0, color=GREY, lw=1)
    marker, = height.plot([], [], "o", color=NAVY, ms=5)

    def draw(i):
        for ax in (world, close):
            ax.cla()
        off = cut[i]
        world.plot(*ref.T, color=ORANGE, ls="--", lw=1.5, label="reference")
        world.plot(*pos[:i + 1].T, color=NAVY, lw=2, label="whoop")
        for s, e in cut_runs:
            if s <= i:
                seg = pos[max(s - 1, 0):min(e, i + 1)]
                world.plot(*seg.T, color=RED, lw=3, label="motors off" if s == cut_runs[0][0] else None)
        world.scatter(*pos[i], color=RED if off else NAVY, s=30)
        world.scatter(*ref[i], color=ORANGE, s=30)
        mid = pos[i] if follow else mid_fixed
        for k, ax_set in enumerate((world.set_xlim, world.set_ylim, world.set_zlim)):
            ax_set(mid[k] - span, mid[k] + span)
        world.set_box_aspect((1, 1, 1))
        world.set_xlabel("x (m)", fontsize=9)
        world.set_ylabel("y (m)", fontsize=9)
        world.set_zlabel("z (m)", fontsize=9)
        world.tick_params(labelsize=7)
        world.legend(loc="upper left", fontsize=8, frameon=False)
        world.set_title(f"t = {t[i]:.2f} s, {speed[i]:.1f} m/s, height {pos[i, 2] - ref[i, 2]:+.2f} m from the reference", fontsize=10)

        R, p = rotation(*angles[i]), pos[i]
        body = lambda pts: (R @ pts.T).T + p
        for r in rotors:
            close.plot(*body(np.array([[0, 0, 0], r])).T, color=GREY, lw=2)
            close.plot(*body(disc + r).T, color=RED if off else NAVY, lw=2.5 if off else 1.5, ls="--" if off else "-")
        if has_plate:
            corners = body(plate_corners)
            close.plot(*corners.T, color=TEAL, lw=2.5)
            close.add_collection3d(Poly3DCollection([corners[:4]], color=TEAL, alpha=0.35))
            f_world = R @ force[i]
            start = body(c[None])[0]
            close.quiver(*start, *(0.25 * f_world / weight), color=ORANGE, lw=2)  # 0.25 m = one weight
            close.set_title(f"plate force {np.linalg.norm(force[i]) / weight * 100:.0f} % of the weight, "
                            f"α = {alpha[i]:.0f}°\nroll {angles[i, 0]:.0f}°, pitch {angles[i, 1]:.0f}°", fontsize=10)
        else:
            close.set_title(f"no plate\nroll {angles[i, 0]:.0f}°, pitch {angles[i, 1]:.0f}°", fontsize=10)
        if off:
            close.text2D(0.5, 0.9, "MOTORS OFF", transform=close.transAxes, ha="center", va="center", fontsize=18,
                         fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec=RED))
        h = 0.12
        close.set_xlim(p[0] - h, p[0] + h)
        close.set_ylim(p[1] - h, p[1] + h)
        close.set_zlim(p[2] - h, p[2] + h)
        close.set_box_aspect((1, 1, 1))
        close.set_xticks([])
        close.set_yticks([])
        close.set_zticks([])
        cursor.set_xdata([t[i], t[i]])
        marker.set_data([t[i]], [pos[i, 2]])
        marker.set_color(RED if off else NAVY)
        return []

    if args.frame is not None:
        draw(frames[0])
        fig.savefig(args.png or Path(args.out).with_suffix(".png"), dpi=90)
        print("wrote", args.png, "flight", args.flight, "t", t[frames[0]])
        return
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    fps = max(1, int(round(1 / ((t[1] - t[0]) * every))))  # real time
    anim.save(args.out, writer=PillowWriter(fps=fps), dpi=80)
    print("wrote", args.out, "flight", args.flight, "frames", len(frames), "fps", fps)


if __name__ == "__main__":
    main()
