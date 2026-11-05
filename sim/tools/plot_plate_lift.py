"""Rotor thrust and plate lift against speed, from the speed sweep of several runs (evaluation.json, "speed_sweep").

python plot_plate_lift.py <out.png> <label>=<run_dir> [<label>=<run_dir> ...] [--trim trim.json] [--title TEXT]

Three panels: the total rotor thrust in % of the vehicle's own weight; the same thrust in % of the weight of the whoop
without the plate (30 g), which is what the motors push in absolute terms (the plate itself weighs 0.7 g); and the
vertical (world z) plate force in % of the weight. Every point is the mean over the 8 headings of the sweep and over
t = 2-5 s of each flight (whoop_train_* evaluation). A run whose label contains "no plate" is drawn dashed.
--trim adds the least-thrust steady flight of sim/tools/plate_trim.py (--json output) as a black line: what the model
allows at best, for a whoop that may choose its attitude freely.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOPLATE_MASS = 0.030
COLORS = ["#012367", "#2A9D8F", "#EF9100", "#D62828", "#6A4C93", "#9A9A9A"]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("runs", nargs="+")
    a.add_argument("--trim", default=None)
    a.add_argument("--title", default="")
    args = a.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, item in enumerate(args.runs):
        label, run = item.split("=", 1)
        ev = json.loads((Path(run) / "evaluation.json").read_text())
        cfg = json.loads((Path(run) / "config.json").read_text())
        rows = [r for r in ev["speed_sweep"]["per_speed"] if r["rotor_thrust_to_weight"] is not None]
        v = [r["speed"] for r in rows]
        style = dict(color=COLORS[k % len(COLORS)], marker="o", ms=4, lw=1.8, ls="--" if "no plate" in label else "-", label=label)
        ax[0].plot(v, [100 * r["rotor_thrust_to_weight"] for r in rows], **style)
        ax[1].plot(v, [100 * r["rotor_thrust_to_weight"] * cfg["mass"] / NOPLATE_MASS for r in rows], **style)
        ax[2].plot(v, [100 * r["plate_lift_to_weight"] for r in rows], **style)
    if args.trim:
        trim = json.loads(Path(args.trim).read_text())
        for name, ls in (("plate", "-"), ("noplate", ":")):
            rows = trim[name]
            v = [r["speed"] for r in rows]
            mass_ratio = (NOPLATE_MASS + (0.0007 if name == "plate" else 0)) / NOPLATE_MASS  # [assumption: nominal masses]
            lab = "least possible, " + ("plate" if name == "plate" else "no plate") + " (steady trim)"
            ax[0].plot(v, [100 * r["thrust_to_weight"] for r in rows], color="black", lw=1, ls=ls, label=lab)
            ax[1].plot(v, [100 * r["thrust_to_weight"] * mass_ratio for r in rows], color="black", lw=1, ls=ls, label=lab)
            if name == "plate":
                ax[2].plot(v, [100 * r["plate_lift_to_weight"] for r in rows], color="black", lw=1, ls=ls, label=lab)
    ax[0].set_ylabel("rotor thrust (% of own weight)")
    ax[1].set_ylabel("rotor thrust (% of the 30 g whoop's weight)")
    ax[2].set_ylabel("plate lift, world z (% of weight)")
    for x in ax:
        x.set_xlabel("reference speed (m/s)")
        x.grid(alpha=0.3)
    ax[0].axhline(100, color="0.5", lw=0.8)
    ax[1].axhline(100, color="0.5", lw=0.8)
    ax[2].axhline(0, color="0.5", lw=0.8)
    ax[2].legend(fontsize=7, loc="upper left")
    fig.suptitle(args.title or "Speed sweep: mean over 8 headings, t = 2-5 s", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
