"""Run the whole identification chain on synthetic flights and check it recovers the true parameters.

    python validate.py            # 3 noise levels x 3 durations, Monte Carlo, prediction, figures (~10 min
                                  # the first time, the truth flights are then cached)
    python validate.py --quick    # nominal noise, medium duration only

Prints (and writes to ~/Work/whoop-airfoil-sysid/results.md):
  1. recovery table: true value, identified value, relative error, per parameter, per noise level and
     flight duration;
  2. notebook-faithful variant (symmetric T_m, notebook inertia and K_tau, no drag column);
  3. Monte Carlo spread over sensor-noise seeds (nominal noise, medium duration);
  4. which plate parameter each manoeuvre identifies (formal standard errors per manoeuvre);
  5. one-step and short-horizon prediction RMSE on held-out flights.
Figures go to sysid/figures/.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import identify as I  # noqa: E402
import model as M  # noqa: E402
import synthetic as S  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
OUT = Path.home() / "Work" / "whoop-airfoil-sysid"
# (excitation flight, plate flights) nominal durations [s]. Plate flights need >= 60 s (4 s per segment)
# and are rounded up to whole laps, so their real length is printed with the results.
DURATIONS = {"short": (30, 60), "medium": (60, 120), "long": (120, 240)}
SEEDS = dict(exc=1, noplate=2, plate=3, exc_test=101, plate_test=103)
LEVELS = ("low", "nominal", "high")


# ----------------------------------------------------------------------------------------------------
# Chain
# ----------------------------------------------------------------------------------------------------
def run_chain(level, dur_key, noise_seed=0, motor="asym"):
    de, dp = DURATIONS[dur_key]
    Le = S.truth_flight("excitation", de, False, SEEDS["exc"])
    Ln = S.truth_flight("plate", dp, False, SEEDS["noplate"])
    Lp = S.truth_flight("plate", dp, True, SEEDS["plate"])
    Pw, Pp = M.load_params(plate=False), M.load_params(plate=True)
    ns = 1000 * noise_seed
    w = I.identify_whoop(S.measure(Le, level, SEEDS["exc"] + ns), Pw, motor=motor)
    Pwi = I.whoop_model(Pw, w)
    dr = I.identify_body_drag(S.measure(Ln, level, SEEDS["noplate"] + ns), Pwi)
    drag = dr["body_drag"] if motor == "asym" else np.zeros(3)
    pr = I.identify_plate(S.measure(Lp, level, SEEDS["plate"] + ns), Pp, w, drag)
    durs = (Le["t"][-1], Ln["t"][-1], Lp["t"][-1])
    return dict(w=w, drag=dr, plate=pr, dur=durs, level=level, dur_key=dur_key)


def truth_values():
    Pw = M.load_params(plate=False)
    pl = M.load_params(plate=True)["plate"]
    kf = Pw["kf"]
    return {
        "tau_rise [s]": Pw["tau_rise"], "tau_fall [s]": Pw["tau_fall"],
        "kf0 [N]": kf[0], "kf1 [N]": kf[1], "kf2 [N]": kf[2],
        "thrust(u=0.5) [N]": I.poly(kf, 0.5), "thrust(u=1) [N]": I.poly(kf, 1.0),
        "I_xx [kg m2]": Pw["J"][0, 0], "I_yy [kg m2]": Pw["J"][1, 1], "I_zz [kg m2] (ratio)": Pw["J"][2, 2],
        "k_tau [m]": Pw["k_tau"],
        "d_x [kg/s]": M.TRUE_BODY_DRAG[0], "d_y [kg/s]": M.TRUE_BODY_DRAG[1], "d_z [kg/s]": M.TRUE_BODY_DRAG[2],
        "d_z thrust-fit [kg/s]": M.TRUE_BODY_DRAG[2],
        "C_N90": pl["CN90"], "K_px": pl["Kpx"], "K_pz": pl["Kpz"], "C_T": pl["CT"], "e0": pl["e0"],
    }


ABS_ERR = {"kf0 [N]"}          # true value is 0: report the absolute error


def estimates(res):
    w, dr, pr = res["w"], res["drag"], res["plate"]
    th = pr["fit"]["theta"]
    kf = w["kf"]
    return {
        "tau_rise [s]": w["tau_rise"], "tau_fall [s]": w["tau_fall"],
        "kf0 [N]": kf[0], "kf1 [N]": kf[1], "kf2 [N]": kf[2],
        "thrust(u=0.5) [N]": I.poly(kf, 0.5), "thrust(u=1) [N]": I.poly(kf, 1.0),
        "I_xx [kg m2]": w["J"][0, 0], "I_yy [kg m2]": w["J"][1, 1], "I_zz [kg m2] (ratio)": w["J"][2, 2],
        "k_tau [m]": w["k_tau"],
        "d_x [kg/s]": dr["body_drag"][0], "d_y [kg/s]": dr["body_drag"][1], "d_z [kg/s]": dr["body_drag"][2],
        "d_z thrust-fit [kg/s]": w["asym"]["d_z"] if w["asym"]["d_z"] is not None else np.nan,
        "C_N90": th["CN90"], "K_px": th["Kpx"], "K_pz": th["Kpz"], "C_T": th["CT"], "e0": pr["e0"]["e0"],
    }


def err(name, true, est):
    if name in ABS_ERR:
        return f"{est - true:+.1e} abs"
    return f"{100 * (est - true) / true:+.1f} %"


def fmt(x):
    return f"{x:.4g}"


# ----------------------------------------------------------------------------------------------------
# Prediction on held-out data
# ----------------------------------------------------------------------------------------------------
def accel_pred(P, d, u):
    """Model specific force and angular acceleration at the measured state (one-step prediction)."""
    f = M.rotor_thrust(P, u)
    Fr, tr = M.rotor_wrench(P, f)
    Fa, ta = M.aero_wrench(P, d["v_body"], d["gyro"])
    w = d["gyro"]
    wdot = (tr + ta - M.cross(w, w @ P["J"].T)) @ P["J_inv"].T
    return (Fr + Fa) / P["mass"], wdot


def truth_accels(P, L):
    sf = L["sf"]
    wdot = np.array([M.derivative(P, x, u)[0][10:13] for x, u in zip(L["x"], L["u_m"])])
    return sf, wdot


def rollout_errors(P, d, L, horizons=(0.02, 0.1, 0.25, 0.5), stride=0.5):
    dt = d["dt_mean"]
    H = [int(round(h / dt)) for h in horizons]
    u_all = I.ema(d["t"], d["cmd"], P["tau_rise"], P["tau_fall"])
    starts = np.arange(int(1.0 / dt), len(d["t"]) - H[-1] - 1, int(stride / dt))
    e_p = np.zeros((len(starts), len(H)))
    e_v = np.zeros_like(e_p)
    e_a = np.zeros_like(e_p)
    for i, s in enumerate(starts):
        x = np.concatenate([d["p"][s], d["v_world"][s], d["q"][s], d["gyro"][s]])
        u = u_all[s].copy()
        j = 0
        for k in range(H[-1]):
            x, _ = M.rk4_step(P, x, u, dt)
            u = M.motor_update(P, u, d["cmd"][s + k], dt)
            if k + 1 == H[j]:
                xt = L["x"][s + k + 1]
                e_p[i, j] = np.linalg.norm(x[:3] - xt[:3])
                e_v[i, j] = np.linalg.norm(x[3:6] - xt[3:6])
                dq = M.quat_mul(np.array([xt[6], -xt[7], -xt[8], -xt[9]]), x[6:10])
                e_a[i, j] = np.degrees(2 * np.arccos(min(1.0, abs(dq[0]))))
                j += 1
    rms = lambda e: np.sqrt(np.mean(e ** 2, axis=0))
    return dict(horizons=horizons, pos=rms(e_p), vel=rms(e_v), att=rms(e_a), n=len(starts))


def prediction_study(res_ref, res_nb):
    """Held-out flights (different seeds): one excitation flight without plate, one plate flight."""
    out = {}
    Pw_true, Pp_true = M.load_params(plate=False), M.load_params(plate=True)
    w, drag, plate = res_ref["w"], res_ref["drag"]["body_drag"], res_ref["plate"]
    P_id_plate = plate["model"]
    P_id_noplate_model = M.with_updates(P_id_plate)
    P_id_noplate_model["plate"] = None
    P_id_noplate_model = M.finalize(P_id_noplate_model)
    wn = res_nb["w"]
    P_nb_plate = I.whoop_model(M.with_updates(Pp_true), wn, body_drag=np.zeros(3))
    P_nb_plate["plate"] = None
    P_nb_plate = M.finalize(P_nb_plate)
    P_nb = I.whoop_model(Pw_true, wn, body_drag=np.zeros(3))
    P_id_whoop = I.whoop_model(Pw_true, w, body_drag=drag)
    cases = {
        "excitation (no plate)": (S.truth_flight("excitation", 30, False, SEEDS["exc_test"]), Pw_true,
                                  {"truth": Pw_true, "identified": P_id_whoop, "notebook-faithful": P_nb}),
        "plate flight": (S.truth_flight("plate", 60, True, SEEDS["plate_test"]), Pp_true,
                         {"truth": Pp_true, "identified": P_id_plate,
                          "identified, plate model removed": P_id_noplate_model,
                          "notebook-faithful, no drag/plate": P_nb_plate}),
    }
    for cname, (L, Ptrue, models) in cases.items():
        log = S.measure(L, "nominal", 777)
        d = I.prepare(log)
        sf_t, wd_t = truth_accels(Ptrue, L)
        out[cname] = {}
        for mname, P in models.items():
            u = I.ema(d["t"], d["cmd"], P["tau_rise"], P["tau_fall"])
            sf, wd = accel_pred(P, d, u)
            one = dict(sf=float(np.sqrt(np.mean(np.sum((sf - sf_t) ** 2, axis=1)))),
                       wdot=float(np.sqrt(np.mean(np.sum((wd - wd_t) ** 2, axis=1)))))
            ro = rollout_errors(P, d, L)
            out[cname][mname] = dict(one_step=one, rollout=ro)
            print(f"  {cname:22s} {mname:34s} one-step |sf| {one['sf']:.3f} m/s2 |wdot| {one['wdot']:.1f} rad/s2 | "
                  + " ".join(f"{h:.2f}s: {p * 100:.2f}cm {a:.2f}deg" for h, p, a in zip(ro['horizons'], ro['pos'], ro['att'])),
                  flush=True)
    return out


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def figures(res, pred):
    FIG.mkdir(exist_ok=True)
    w = res["w"]
    true = truth_values()
    # 1. RMSE vs T_m (notebook sweep) and the 2-D rising/falling map.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    s = w["sym"]
    ax[0].plot(s["grid"], s["rmse"], color="C0")
    ax[0].axvline(s["T_m"], color="C3", ls="--", label=f"minimum T_m = {s['T_m']:.3f} s")
    ax[0].axvline(true["tau_rise [s]"], color="0.5", ls=":", label="true rising / falling")
    ax[0].axvline(true["tau_fall [s]"], color="0.5", ls=":")
    ax[0].set_xlabel("first-order time constant T_m [s]")
    ax[0].set_ylabel("RMSE of m*acc fit [N]")
    ax[0].set_title("Notebook sweep (one T_m)")
    ax[0].legend(fontsize=8)
    c = w["asym"]["coarse"]
    im = ax[1].contourf(c["g"], c["g"], np.log10(c["rmse"].T), 30, cmap="viridis")
    fig.colorbar(im, ax=ax[1], label="log10 RMSE [N]")
    ax[1].plot(true["tau_rise [s]"], true["tau_fall [s]"], "w*", ms=12, label="truth")
    ax[1].plot(w["tau_rise"], w["tau_fall"], "r+", ms=12, mew=2, label="identified")
    ax[1].set_xlabel("T_rise [s]")
    ax[1].set_ylabel("T_fall [s]")
    ax[1].set_title("Extension: rising/falling grid")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "finding_tau.png", dpi=130)
    plt.close(fig)

    # 2. Thrust: without and with the motor model, as in the notebook figure.
    d = w["prep"]
    thr = w["fit_signals"]["thr"]
    m = M.load_params(plate=False)["mass"]
    meas = m * d["acc"][thr, 2] + w["asym"]["d_z"] * d["v_body"][thr, 2]
    pred_sp = I.poly(w["kf"], d["cmd"][thr]).sum(axis=1)
    pred_m = w["f"][thr].sum(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for a, p, title in [(ax[0], pred_sp, "commands, no motor delay"), (ax[1], pred_m, "with identified motor delay")]:
        a.scatter(p, meas, s=0.3, alpha=0.3)
        lo, hi = meas.min(), meas.max()
        a.plot([lo, hi], [lo, hi], "r", lw=1.5, label="identity")
        a.set_xlabel("predicted total thrust [N]")
        a.set_title(title)
        rmse = np.sqrt(np.mean((p - meas) ** 2))
        a.text(0.05, 0.9, f"RMSE {rmse * 1000:.1f} mN", transform=a.transAxes)
    ax[0].set_ylabel("m a_z + d_z v_z (accelerometer) [N]")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "thrust_fit.png", dpi=130)
    plt.close(fig)

    # 3. Inertia fits (reciprocal, filtered).
    fs = w["fit_signals"]
    rp = fs["rp"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    for i, nm in enumerate("xy"):
        Ii = w["J"][i, i]
        ax[i].scatter(fs["tq_f"][rp, i], fs["dw_f"][rp, i], s=0.3, alpha=0.3)
        tt = np.linspace(fs["tq_f"][rp, i].min(), fs["tq_f"][rp, i].max(), 10)
        ax[i].plot(tt, tt / Ii, "r", label=f"I_{nm}{nm} = {Ii:.3e} (true {true[f'I_{nm}{nm} [kg m2]']:.2e})")
        ax[i].set_xlabel(f"rotor torque {nm} [N m]")
        ax[i].set_ylabel(f"angular acceleration {nm} [rad/s2]")
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "inertia_fit.png", dpi=130)
    plt.close(fig)

    # 4. Plate C_N(alpha): truth vs identified, with the per-sample "measured" C_N from the y row.
    pr = res["plate"]
    dp, Pm = pr["prep"], pr["model"]
    pl_true = M.load_params(plate=True)["plate"]
    F, _, aux = M.plate_wrench(Pm["plate"], dp["v_body"], dp["gyro"], aux=True)
    # The friction acts in the plate plane (v_t has no y component), so the measured y force is the
    # normal force alone and gives a per-sample C_N.
    Fy = pr["fit"]["F_meas"][:, 1]
    qS = 0.5 * Pm["plate"]["rho"] * Pm["plate"]["S"]
    V = np.sqrt(aux["V2"])
    sel = (V > 2.0) & (np.abs(aux["vn"]) > 0.05)
    CN_meas = -Fy[sel] / (qS * aux["V2"][sel] * np.sign(aux["vn"][sel]))
    al = np.linspace(0, np.pi / 2, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sc = ax.scatter(np.degrees(aux["alpha"][sel]), CN_meas, c=aux["c2"][sel], s=0.5, alpha=0.4, cmap="coolwarm",
                    vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="c^2 (1: in-plane wind along x, 0: along z)")
    for c2, col in [(1.0, "C3"), (0.0, "C0")]:
        for pl, ls, lab in [(pl_true, "-", "true"), (Pm["plate"], "--", "identified")]:
            Kp = pl["Kpx"] * c2 + pl["Kpz"] * (1 - c2)
            ax.plot(np.degrees(al), pl["CN90"] * np.tanh(Kp * al / pl["CN90"]), ls, color=col, lw=2,
                    label=f"{lab}, c^2 = {c2:.0f}")
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel("C_N")
    ax.set_ylim(-0.5, 2.0)
    ax.set_title("Plate normal-force coefficient (samples with V > 2 m/s)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "plate_cn_alpha.png", dpi=130)
    plt.close(fig)

    # 5. Prediction error vs horizon.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, cname in zip(ax, pred):
        for mname, r in pred[cname].items():
            ro = r["rollout"]
            a.plot(ro["horizons"], np.array(ro["pos"]) * 100, "o-", label=mname)
        a.set_xlabel("prediction horizon [s]")
        a.set_ylabel("position RMSE [cm]")
        a.set_yscale("log")
        a.set_title(f"held-out {cname}")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "prediction_rmse.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mc", type=int, default=5, help="noise seeds for the Monte Carlo (nominal, medium)")
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    true = truth_values()
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    levels = ("nominal",) if args.quick else LEVELS
    durs = ("medium",) if args.quick else tuple(DURATIONS)
    results = {}
    for lv in levels:
        for dk in durs:
            t0 = time.time()
            results[(lv, dk)] = run_chain(lv, dk)
            print(f"chain {lv:8s} {dk:6s} done in {time.time() - t0:.0f} s", flush=True)
    ref = results[("nominal", "medium")]

    # 1. Recovery table (nominal noise, all durations), then relative errors for all combinations.
    emit("## Recovery: nominal noise")
    emit("")
    hdr = "| parameter | true | " + " | ".join(
        f"{dk} ({results[('nominal', dk)]['dur'][0]:.0f} s exc. + 2 x {results[('nominal', dk)]['dur'][2]:.0f} s plate/no plate)"
        for dk in durs) + " |"
    emit(hdr)
    emit("|" + "---|" * (2 + len(durs)))
    for k, tv in true.items():
        cells = []
        for dk in durs:
            e = estimates(results[("nominal", dk)])[k]
            cells.append(f"{fmt(e)} ({err(k, tv, e)})")
        emit(f"| {k} | {fmt(tv)} | " + " | ".join(cells) + " |")
    if not args.quick:
        emit("")
        emit("## Relative error for every noise level and duration")
        emit("")
        cols = [(lv, dk) for lv in levels for dk in durs]
        emit("| parameter | " + " | ".join(f"{lv}/{dk}" for lv, dk in cols) + " |")
        emit("|" + "---|" * (1 + len(cols)))
        for k, tv in true.items():
            emit(f"| {k} | " + " | ".join(err(k, tv, estimates(results[c])[k]) for c in cols) + " |")

    # 2. Notebook-faithful variant.
    emit("")
    emit("## Notebook-faithful whoop chain (one T_m, I from <dw,tau>/<dw,dw>, no gyroscopic term, no drag)")
    emit("")
    emit("| parameter | true | " + " | ".join(durs) + " |")
    emit("|" + "---|" * (2 + len(durs)))
    nb = {dk: run_chain("nominal", dk, motor="sym") for dk in durs}
    rows = [("T_m [s] (truth: rise 0.04 / fall 0.08)", None, lambda r: r["w"]["sym"]["T_m"]),
            ("thrust(u=0.5) [N]", true["thrust(u=0.5) [N]"], lambda r: I.poly(r["w"]["kf"], 0.5)),
            ("thrust(u=1) [N]", true["thrust(u=1) [N]"], lambda r: I.poly(r["w"]["kf"], 1.0)),
            ("kf0 [N]", true["kf0 [N]"], lambda r: r["w"]["kf"][0]),
            ("I_xx [kg m2]", true["I_xx [kg m2]"], lambda r: r["w"]["J"][0, 0]),
            ("I_yy [kg m2]", true["I_yy [kg m2]"], lambda r: r["w"]["J"][1, 1]),
            ("I_zz [kg m2] (ratio)", true["I_zz [kg m2] (ratio)"], lambda r: r["w"]["J"][2, 2]),
            ("k_tau [m]", true["k_tau [m]"], lambda r: r["w"]["k_tau"]),
            ("C_N90 (plate fit on top, no body drag)", true["C_N90"], lambda r: r["plate"]["fit"]["theta"]["CN90"]),
            ("C_T (plate fit on top, no body drag)", true["C_T"], lambda r: r["plate"]["fit"]["theta"]["CT"])]
    for name, tv, f in rows:
        cells = []
        for dk in durs:
            e = f(nb[dk])
            cells.append(fmt(e) if tv is None else f"{fmt(e)} ({err(name if name in ABS_ERR else 'x', tv, e)})")
        emit(f"| {name} | {'-' if tv is None else fmt(tv)} | " + " | ".join(cells) + " |")
    emit("")
    emit("Other diagnostics (nominal, medium): symmetric T_m with our asymmetric-model chain = "
         f"{ref['w']['sym']['T_m']:.4f} s; rise/fall without the drag column = "
         f"{ref['w']['asym_nodrag']['tau_rise']:.4f} / {ref['w']['asym_nodrag']['tau_fall']:.4f} s; "
         f"reciprocal inertia without gyroscopic term = {ref['w']['I_recip_nogyro'][0]:.4e} / "
         f"{ref['w']['I_recip_nogyro'][1]:.4e}; hover command median {ref['w']['hover_cmd_median']:.3f} vs "
         f"predicted {ref['w']['hover_cmd_pred']:.3f} (true {M.hover_command(M.load_params()):.3f}); "
         f"e0 from yaw only = {ref['plate']['e0_yaw']['e0']:.4f}.")

    # 3. Monte Carlo over sensor-noise seeds.
    if args.mc > 1:
        emit("")
        emit(f"## Monte Carlo, nominal noise, medium duration, {args.mc} noise seeds: mean +- std of the error")
        emit("")
        mc = [estimates(ref)] + [estimates(run_chain("nominal", "medium", noise_seed=s)) for s in range(1, args.mc)]
        emit("| parameter | true | mean | error mean +- std | formal std (plate fit) |")
        emit("|---|---|---|---|---|")
        se = ref["plate"]["fit"]["se"]
        se_map = {"C_N90": se["CN90"], "K_px": se["Kpx"], "K_pz": se["Kpz"], "C_T": se["CT"], "e0": ref["plate"]["e0"]["se"]}
        for k, tv in true.items():
            v = np.array([m_[k] for m_ in mc])
            if k in ABS_ERR:
                e_ = v - tv
                s_ = f"{e_.mean():+.1e} +- {e_.std(ddof=1):.1e} abs"
            else:
                e_ = 100 * (v - tv) / tv
                s_ = f"{e_.mean():+.1f} +- {e_.std(ddof=1):.1f} %"
            fse = f"{100 * se_map[k] / tv:.1f} %" if k in se_map else ""
            emit(f"| {k} | {fmt(tv)} | {fmt(v.mean())} | {s_} | {fse} |")

    # 4. Identifiability per manoeuvre (plate parameters).
    emit("")
    emit("## Plate identifiability per manoeuvre (nominal noise, medium plate flight): formal relative std")
    emit("")
    pr = ref["plate"]
    d, Pm = pr["prep"], pr["model"]
    th = pr["fit"]["theta"]
    subsets = {"circles": [10], "forward passes": [11], "sideways passes": [12], "diagonal passes": [13],
               "sideways climbs + vertical": [14, 15], "all": [10, 11, 12, 13, 14, 15]}
    emit("| manoeuvre | time [s] | C_N90 | K_px | K_pz | C_T | e0 |")
    emit("|---|---|---|---|---|---|---|")
    for nm, labs in subsets.items():
        mask = np.isin(d["seg"], labs)
        se = I.plate_fisher_se(d, Pm, th, mask, pr["fit"]["rms"])
        try:
            e0 = I.fit_e0(d, Pm, mask=mask)
            e0s = f"{100 * e0['se'] / Pm['plate']['e0']:.2g} % (fit {e0['e0']:.3f})"
        except Exception:  # noqa: BLE001
            e0s = "-"
        cells = [f"{100 * se[k] / th[k]:.2g} %" if np.isfinite(se[k]) else "n/i" for k in I.PLATE_KEYS]
        emit(f"| {nm} | {mask.sum() * d['dt_mean']:.0f} | " + " | ".join(cells) + f" | {e0s} |")

    # 5. Prediction.
    emit("")
    emit("## Prediction on held-out flights (identified model from nominal noise, medium duration)")
    emit("")
    print("prediction study ...", flush=True)
    pred = prediction_study(ref, nb["medium"] if "medium" in nb else list(nb.values())[0])
    for cname, models in pred.items():
        emit(f"Held-out {cname}: one-step = specific force and angular acceleration predicted at the measured "
             "state vs noise-free truth; open-loop rollouts from the mocap/gyro state with logged commands, "
             f"windows every 0.5 s ({list(models.values())[0]['rollout']['n']} windows).")
        emit("")
        hz = list(models.values())[0]["rollout"]["horizons"]
        emit("| model | one-step sf [m/s2] | one-step dw [rad/s2] | " +
             " | ".join(f"pos/att @ {h:.2f} s" for h in hz) + " |")
        emit("|" + "---|" * (3 + len(hz)))
        for mname, r in models.items():
            ro = r["rollout"]
            emit(f"| {mname} | {r['one_step']['sf']:.3f} | {r['one_step']['wdot']:.1f} | " +
                 " | ".join(f"{p * 100:.2f} cm / {a:.2f} deg" for p, a in zip(ro["pos"], ro["att"])) + " |")
        emit("")

    figures(ref, pred)
    (OUT / ("results_quick.md" if args.quick else "results.md")).write_text("\n".join(lines))
    js = {f"{lv}/{dk}": {k: float(v) for k, v in estimates(r).items()} for (lv, dk), r in results.items()}
    (OUT / "results.json").write_text(json.dumps(dict(true=true, estimates=js), indent=1, default=float))
    print(f"total {time.time() - t_start:.0f} s; tables in {OUT}, figures in {FIG}")


if __name__ == "__main__":
    main()
