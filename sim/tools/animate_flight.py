"""Animate a flight from flights.csv (written by whoop_train_*, in training, in the motor-cut scenario or in the speed
sweep): the trajectory with the reference on the left, a close-up that follows the whoop in the middle, drawn to scale
with its plate and the plate force, two gauges on the right (rotor thrust and plate lift in % of the weight, with the
plate's angle of attack), and below the height and a strip chart of the same two forces against time.

python animate_flight.py <flights.csv> <out.gif> [--flight K] [--pick force|lift] [--title TEXT] [--every N] [--window M]
                         [--frame T --png P]
--flight -1 (default) picks the recorded flight with the largest mean plate force (--pick lift: the largest mean plate
lift), or flight 0 of a motor-cut scenario. When the flight covers more than the window (default 1.5 m on each side),
the left camera follows the whoop. If the CSV has a motors_cut column, the cut is shown in red: a "MOTORS OFF" label,
red rotor discs, the trajectory segment flown without motors and a band on the height plot.
--frame T --png P writes the frame at time T [s] as a PNG instead of the animation (to check how it reads).

The vehicle (mass, plate size, position and orientation) comes from config.json next to the CSV when it is there (so a
run with --plate-z, --plate-tilt or --params is drawn as it flew), else from params/whoop_nominal.json.
Plate lift = world-z component of the plate force; rotor thrust = sum of the four rotor thrusts; both over the weight of
the vehicle that flew. Older CSVs without these columns get them recomputed from the rotor states and the plate force.
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


def vehicle(csv_path):
    """mass, rotor positions, thrust curve, plate size and centroid of the vehicle that flew"""
    params = json.loads(PARAMS.read_text())
    w, plate = params["whoop"], params["plate"]
    v = dict(rotors=np.array(w["rotor_positions"]), kf=np.array(w["rotor_thrust_coefficients"]), cx=plate["chord_x"],
             cz=plate["height_z"], centroid=np.array(plate["centroid"], float), mass=None, plate_area=None,
             normal=np.array(plate["normal"], float), e1=np.array(plate.get("chord_direction", [1, 0, 0]), float), downwash=0.0)
    config = Path(csv_path).with_name("config.json")
    if config.exists():
        c = json.loads(config.read_text())
        v["mass"] = c.get("mass")
        v["plate_area"] = c.get("plate_area")
        if "plate_centroid" in c:
            v["centroid"] = np.array(c["plate_centroid"], float)
        if "rotor_thrust_coefficients" in c:
            v["kf"] = np.array(c["rotor_thrust_coefficients"], float)
        if "plate_normal" in c:
            v["normal"] = np.array(c["plate_normal"], float)
            v["e1"] = np.array(c["plate_chord_direction"], float)
            v["cx"], v["cz"] = c["plate_edges"]
            v["downwash"] = c.get("plate_downwash_factor", 0.0)
    v["nominal_mass"] = w["mass"]
    v["plate_mass"] = plate["added_mass"]
    return v


def main():
    a = argparse.ArgumentParser()
    a.add_argument("flights")
    a.add_argument("out")
    a.add_argument("--flight", type=int, default=-1)
    a.add_argument("--pick", choices=["force", "lift"], default="force", help="which flight --flight -1 picks")
    a.add_argument("--title", default="")
    a.add_argument("--every", type=int, default=4, help="draw one simulation step out of N")
    a.add_argument("--window", type=float, default=1.5, help="half-width of the following camera [m]")
    a.add_argument("--frame", type=float, default=None, help="write only the frame at this time [s] to --png")
    a.add_argument("--png", default=None)
    args = a.parse_args()
    with open(args.flights) as f:
        rows = list(csv.DictReader(f))
    V = vehicle(args.flights)
    flights = sorted({int(r["flight"]) for r in rows})
    col = lambda f, c: np.array([float(r[c]) for r in rows if int(r["flight"]) == f])
    has_cut = "motors_cut" in rows[0]
    has_lift = "plate_lift_to_weight" in rows[0]
    if args.flight < 0:
        if has_cut:
            args.flight = flights[0]
        elif args.pick == "lift" and has_lift:
            args.flight = max(flights, key=lambda f: np.mean(col(f, "plate_lift_to_weight")))
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
    has_plate = (V["plate_area"] > 0) if V["plate_area"] is not None else bool(np.any(np.abs(force) > 0))
    mass = V["mass"] if V["mass"] is not None else V["nominal_mass"] + (V["plate_mass"] if has_plate else 0)
    weight = mass * 9.81
    R_all = [rotation(*ang) for ang in angles]
    if has_lift:
        lift, thrust = g("plate_lift_to_weight"), g("rotor_thrust_to_weight")
    else:
        u = np.c_[g("rotor_1"), g("rotor_2"), g("rotor_3"), g("rotor_4")]
        thrust = (V["kf"][0] + V["kf"][1] * u + V["kf"][2] * u * u).sum(axis=1) / weight
        lift = np.array([(R @ f)[2] for R, f in zip(R_all, force)]) / weight
    yaw_to_path = g("yaw_to_path_deg") if "yaw_to_path_deg" in rows[0] else None

    # the whoop in its body frame: arms to the rotors, rotor discs, and the plate rectangle: edge c_x along e_1, edge c_z
    # along e_2 = e_1 x n (the body x-z plane for the vertical plate; docs/plate_model.md)
    rotors = V["rotors"]
    disc = 0.02 * np.c_[np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24)), np.zeros(24)]
    cx, cz, c = V["cx"], V["cz"], V["centroid"]
    e1, normal = V["e1"], V["normal"]
    e2 = np.cross(e1, normal)
    plate_corners = c + np.array([sx * cx / 2 * e1 + sz * cz / 2 * e2 for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    tilt_deg = float(np.degrees(np.arctan2(normal[2], normal[1])))  # about body x, for the title

    fig = plt.figure(figsize=(13, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.15], width_ratios=[1, 1, 0.34], hspace=0.2, wspace=0.05, left=0.05, right=0.96,
                            top=0.9, bottom=0.08)
    world = fig.add_subplot(grid[0, 0], projection="3d")
    close = fig.add_subplot(grid[0, 1], projection="3d")
    gauge = fig.add_subplot(grid[0, 2])
    bottom = grid[1, :].subgridspec(1, 2, wspace=0.28, width_ratios=[1, 1.2])
    height = fig.add_subplot(bottom[0, 0])
    strip = fig.add_subplot(bottom[0, 1])
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
        height.text(0.99, 0.88, f"lowest point z = {z_low:.0f} m (below the plot)", transform=height.transAxes, ha="right", fontsize=9,
                    color=RED)
    height.set_xlabel("t (s)", fontsize=9)
    height.set_ylabel("height z (m)", fontsize=9)
    height.tick_params(labelsize=8)
    height.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    height.grid(alpha=0.3)
    cursor = height.axvline(0, color=GREY, lw=1)
    marker, = height.plot([], [], "o", color=NAVY, ms=5)

    # strip chart: rotor thrust (left axis) and plate lift (right axis), in % of the weight
    strip.plot(t, 100 * thrust, color=NAVY, lw=1.5, label="rotor thrust")
    strip.axhline(100, color=NAVY, lw=0.8, ls=":")
    strip.set_xlim(0, t[-1])
    thr_lo, thr_hi = np.percentile(100 * thrust, [2, 98])
    strip.set_ylim(min(80, thr_lo - 5), max(120, thr_hi + 5))
    strip.set_ylabel("rotor thrust (% weight)", fontsize=9, color=NAVY)
    strip.tick_params(labelsize=8)
    strip.set_xlabel("t (s)", fontsize=9)
    strip.grid(alpha=0.3)
    strip2 = strip.twinx()
    strip2.plot(t, 100 * lift, color=TEAL, lw=1.5, label="plate lift")
    strip2.axhline(0, color=TEAL, lw=0.8, ls=":")
    strip2.set_ylim(-10, max(20, np.percentile(100 * lift, 98) + 3))
    strip2.set_ylabel("plate lift (% weight)", fontsize=9, color=TEAL)
    strip2.tick_params(labelsize=8)
    strip.legend(handles=strip.get_lines()[:1] + strip2.get_lines()[:1], loc="upper right", fontsize=8, frameon=False, ncol=2)
    strip_cursor = strip.axvline(0, color=GREY, lw=1)
    for s, e in cut_runs:
        strip.axvspan(t[s] - (t[1] - t[0]), t[e - 1], color=RED, alpha=0.15, lw=0)

    def draw_gauges(i):
        # two vertical bars on their own scales: rotor thrust 0-150 % and plate lift -10-30 %; the weight (100 %) and 0
        # are marked. The angle of attack and the heading to the path are written below
        gauge.cla()
        gauge.set_xlim(0, 2)
        gauge.set_ylim(0, 1)
        gauge.axis("off")
        bars = [("rotor\nthrust", 100 * thrust[i], 0, 150, 100, RED if cut[i] else NAVY),
                ("plate\nlift", 100 * lift[i], -10, 30, 0, TEAL)]
        for k, (name, value, vmin, vmax, mark, color) in enumerate(bars):
            x0, bottom, top = 0.25 + k, 0.2, 0.92
            scale = lambda v: bottom + (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin) * (top - bottom)
            gauge.add_patch(plt.Rectangle((x0, bottom), 0.5, top - bottom, fc="#F1F1F1", ec=GREY, lw=1))
            base = scale(max(vmin, 0) if vmin < 0 else vmin)
            gauge.add_patch(plt.Rectangle((x0, min(base, scale(value))), 0.5, abs(scale(value) - base), fc=color, ec="none"))
            gauge.plot([x0 - 0.06, x0 + 0.56], [scale(mark)] * 2, color="black", lw=1.2)
            gauge.text(x0 + 0.62 if k == 0 else x0 - 0.08, scale(mark), "weight" if k == 0 else "0", fontsize=7,
                       ha="left" if k == 0 else "right", va="center")
            gauge.text(x0 + 0.25, top + 0.02, f"{value:.0f} %", ha="center", va="bottom", fontsize=12, color=color, fontweight="bold")
            gauge.text(x0 + 0.25, bottom - 0.02, name, ha="center", va="top", fontsize=9)
        info = f"α = {alpha[i]:.0f}°" if has_plate else "no plate"
        if yaw_to_path is not None:
            info += f"\nyaw to path {yaw_to_path[i]:+.0f}°"
        gauge.text(1.0, 0.02, info, ha="center", va="bottom", fontsize=10, color=NAVY)

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

        R, p = R_all[i], pos[i]
        body = lambda pts: (R @ pts.T).T + p
        for r in rotors:
            close.plot(*body(np.array([[0, 0, 0], r])).T, color=GREY, lw=2)
            close.plot(*body(disc + r).T, color=RED if off else NAVY, lw=2.5 if off else 1.5, ls="--" if off else "-")
        # the flight direction, through the centre of mass
        v = (pos[min(i + 1, len(t) - 1)] - pos[max(i - 1, 0)])
        if np.linalg.norm(v) > 1e-6:
            v = 0.09 * v / np.linalg.norm(v)
            close.quiver(*p, *v, color=ORANGE, lw=1.2, ls="--", arrow_length_ratio=0.25)
        if has_plate:
            corners = body(plate_corners)
            close.plot(*corners.T, color=TEAL, lw=2.5)
            close.add_collection3d(Poly3DCollection([corners[:4]], color=TEAL, alpha=0.35))
            f_world = R @ force[i]
            start = body(c[None])[0]
            close.quiver(*start, *(0.4 * f_world / weight), color=ORANGE, lw=2)  # 0.4 m = one weight
            # its vertical part, the lift, on the same scale (teal, upward when the plate carries weight)
            if abs(f_world[2]) > 1e-4 * weight:
                close.quiver(*start, 0, 0, 0.4 * f_world[2] / weight, color=TEAL, lw=3, arrow_length_ratio=0.3)
            if abs(tilt_deg) < 0.5:
                plate_name = "vertical plate"
            elif abs(abs(tilt_deg) - 90) < 0.5:
                plate_name = "horizontal plate"
            else:
                plate_name = f"plate tilted {tilt_deg:.0f}° about body x"
            if V["downwash"] > 0:
                plate_name += f", downwash k_w = {V['downwash']:g}"
            close.set_title(f"{plate_name}: force {np.linalg.norm(force[i]) / weight * 100:.0f} % of the weight (orange), "
                            f"its vertical part (teal)\nroll {angles[i, 0]:.0f}°, pitch {angles[i, 1]:.0f}°", fontsize=10)
        else:
            close.set_title(f"no plate\nroll {angles[i, 0]:.0f}°, pitch {angles[i, 1]:.0f}°", fontsize=10)
        if off:
            close.text2D(0.5, 0.9, "MOTORS OFF", transform=close.transAxes, ha="center", va="center", fontsize=18,
                         fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec=RED))
        # the close-up camera follows the body whatever the heading (the default view sees a crabbed plate edge-on).
        # Vertical plate: 60 deg off its normal (body y) and from slightly above. Tilted plate: nearly along the chord
        # (body x, 18 deg off), low, so that the plate shows as a line at its tilt to the rotor discs
        if abs(tilt_deg) > 0.5:
            chord_world = R @ e1
            close.view_init(elev=8, azim=np.degrees(np.arctan2(chord_world[1], chord_world[0])) + 180 + 18)
        else:
            side_world = R @ np.array([0.0, 1.0, 0.0])
            close.view_init(elev=22, azim=np.degrees(np.arctan2(side_world[1], side_world[0])) + 60)
        h = 0.12
        close.set_xlim(p[0] - h, p[0] + h)
        close.set_ylim(p[1] - h, p[1] + h)
        close.set_zlim(p[2] - h, p[2] + h)
        close.set_box_aspect((1, 1, 1))
        close.set_xticks([])
        close.set_yticks([])
        close.set_zticks([])
        draw_gauges(i)
        cursor.set_xdata([t[i], t[i]])
        strip_cursor.set_xdata([t[i], t[i]])
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
