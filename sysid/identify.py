"""Identification chain: whoop (Eschmann, Albani, Loianno, IROS 2024) then body drag, then the plate.

Inputs are logs shaped like synthetic.measure(): blackbox frame (t, gyro, acc, cmd) at one rate, mocap
(t_mocap, pos, quat) at its own rate, a static pre-flight segment and a segment label per frame.

Where we follow the official notebook (sysid.ipynb, commit 2d267dd) and where we deviate:
  Followed
    * Motor model: per-sample EMA of the normalized commands, alpha = exp(-dt/T_m), initialized with the
      first command (cell 13 `filter_ema`).
    * T_m: 1-D grid search; for each candidate a linear least-squares fit of one thrust curve shared by
      the 4 motors with features lumped per exponent (separate=False), objective = RMSE over all three
      accelerometer rows b = m * acc (cells 20-23). Grid 0.005-0.2 s, 100 steps (sysid.tools default).
    * Thrust curve: the K_f of the minimum-RMSE candidate, exponents (0, 1, 2).
    * I_xx, I_yy: per-axis scalar LS between angular acceleration and the geometric rotor torque
      I = <dw, tau>/<dw, dw> with dw = np.gradient(gyro) (cell 29), gyroscopic term ignored.
    * I_zz = 1.832 (I_xx + I_yy)/2 (cell 30); K_tau = <u, dw_z I_zz>/<u, u> (cell 31).
  Deviations (each one is a switch, the notebook-faithful variant is also reported)
    * Asymmetric motor model (paper section V, future work): 2-D grid over (T_rise, T_fall) with the
      L2F switch (rising when the command >= state). Our truth has 0.04/0.08 s, so a single T_m is a
      model error; the downstream steps use the asymmetric model by default.
    * The EMA runs over the whole flight and the phases are cut afterwards (the notebook slices first
      and re-initializes the filter on each slice; this only removes a start-up transient).
    * IMU biases are calibrated on the static pre-flight segment (PX4 data in the paper is already
      bias-corrected by the EKF; Betaflight is not).
    * Inertia: the notebook regresses the torque on the noisy angular acceleration, which biases I low
      (errors in variables). We also report the reciprocal fit 1/I = <tau, dw>/<tau, tau> (paper IV-B
      says the reciprocal worked better on the large drone, not in the published notebook) on zero-phase
      low-pass filtered signals, and use it downstream.
      The gyroscopic term w x J w is included (paper eq. (7); the notebook drops it): with the yaw motion
      present during roll/pitch excitation it biased I_xx, I_yy by +2-3 % in our synthetic data.
    * K_tau: same formula, on low-pass filtered signals and with the gyroscopic term (the noise is on the
      b side, so it is unbiased either way; the filter only reduces variance).
    * Thrust fit with mocap: an extra -v_body_z column in the z row fits the vertical body drag together
      with the thrust curve. Without it the drag (velocity lags thrust) biases the time constants low.
    * EMA: the state at frame k uses the command of frame k-1 (zero-order hold, causal). The notebook
      uses the command of frame k, a one-frame lead (2 ms here), negligible against T_m ~ 50 ms.
New steps (not in the paper)
    * Body drag d (diagonal, linear) from the no-plate flights: x/y accelerometer rows are free of rotor
      thrust, so m a_x = -d_x v_x exactly; d_z uses the z row minus the identified rotor thrust.
    * Plate: F_plate = m a_imu - F_rotor - F_drag in the body frame, nonlinear LS on the x and y rows for
      (C_N90, K_px, K_pz, C_T), body velocity from mocap. e0 by linear LS on the moment residual.
"""
import numpy as np
from scipy import optimize, signal

import model as M

G = 9.81
INERTIA_RATIO = 1.832          # notebook cell 2 (the web tool defaults to 1.879)
LPF_HZ = 20.0                  # zero-phase low-pass used for angular accelerations and torques


# ----------------------------------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------------------------------
def _quat_continuous(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def prepare(log):
    """Bias calibration, mocap velocity and attitude on the blackbox time base, body-frame velocity."""
    st = log["static"]
    t = log["t"]
    # Static segment: gyro reads its bias; the accelerometer reads R^T (0, 0, g) + bias. The attitude on
    # the ground comes from the mocap (mean of the static samples).
    tm = log["t_mocap"]
    qm = _quat_continuous(log["quat"])
    st_m = tm <= t[st][-1]
    q0 = qm[st_m].mean(axis=0)
    q0 /= np.linalg.norm(q0)
    bg = log["gyro"][st].mean(axis=0)
    ba = log["acc"][st].mean(axis=0) - M.quat_to_rot(q0).T @ np.array([0, 0, G])
    fl = ~st
    # Mocap velocity: Savitzky-Golay derivative (cubic, 11 samples = 88 ms at 125 Hz), then interpolation.
    # Only the flight part is differentiated (the synthetic static segment sits on the ground, so a filter
    # window across the take-off would see a jump).
    mf = tm > t[st][-1]
    tm, qm = tm[mf], qm[mf]
    dtm = np.median(np.diff(tm))
    v_m = signal.savgol_filter(log["pos"][mf], 11, 3, deriv=1, delta=dtm, axis=0)
    p_m = signal.savgol_filter(log["pos"][mf], 11, 3, axis=0)
    # The same filter on the quaternion components reduces the attitude noise that enters v_body.
    qm = signal.savgol_filter(qm, 11, 3, axis=0)
    tf = t[fl]
    interp = lambda y: np.stack([np.interp(tf, tm, y[:, i]) for i in range(y.shape[1])], axis=1)
    q = interp(qm)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = M.quat_to_rot(q)
    v_w = interp(v_m)
    gyro = log["gyro"][fl] - bg
    dt = np.diff(tf)
    return dict(t=tf, gyro=gyro, acc=log["acc"][fl] - ba, cmd=log["cmd"][fl], seg=log["seg"][fl],
                p=interp(p_m), v_world=v_w, q=q, R=R, v_body=np.einsum("nji,nj->ni", R, v_w),
                domega_raw=np.gradient(gyro, tf, axis=0), dt_mean=float(np.mean(dt)),
                bias=dict(gyro=bg, acc=ba))


def lowpass(x, fs, fc=LPF_HZ):
    b, a = signal.butter(4, fc / (fs / 2))
    return signal.filtfilt(b, a, x, axis=0)


# ----------------------------------------------------------------------------------------------------
# Motor model
# ----------------------------------------------------------------------------------------------------
def ema(t, cmd, tau_r, tau_f):
    """Motor state for one (tau_rise, tau_fall) pair; tau_r == tau_f is the paper's symmetric EMA."""
    u = np.empty_like(cmd)
    u[0] = cmd[0]                                    # notebook: start from the first setpoint
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        c = cmd[k - 1]                               # command held over [t_{k-1}, t_k] (zero-order hold)
        tau = np.where(c >= u[k - 1], tau_r, tau_f)
        a = np.exp(-dt / tau)
        u[k] = a * u[k - 1] + (1 - a) * c
    return u


def thrust_normal_equations(t, cmd, acc, mask, mass, tau_r, tau_f, extra=None):
    """For K candidate (tau_r, tau_f) pairs, accumulate the normal equations of the lumped thrust fit.

    Equivalent to building the notebook's A (3 rows per sample) and b = m*acc, but without storing K
    copies of the filtered motor states: the time loop runs once, vectorized over the candidates.
    extra: optional additional regressor of the z row (we pass -v_body_z to fit the vertical body drag
    together with the thrust curve; the notebook has no such column).
    """
    tau_r = np.asarray(tau_r, float)[:, None]
    tau_f = np.asarray(tau_f, float)[:, None]
    K = tau_r.shape[0]
    nf = 3 if extra is None else 4
    u = np.repeat(cmd[:1], K, axis=0)
    AtA = np.zeros((K, nf, nf))
    Atb = np.zeros((K, nf))
    b = mass * acc
    last = int(np.flatnonzero(mask)[-1]) + 1          # no need to filter past the last used sample
    for k in range(last):
        if k > 0:
            dt = t[k] - t[k - 1]
            c = cmd[k - 1]
            tau = np.where(c[None, :] >= u, tau_r, tau_f)
            a = np.exp(-dt / tau)
            u = a * u + (1 - a) * c[None, :]
        if mask[k]:
            cols = [np.full(K, 4.0), u.sum(axis=1), (u * u).sum(axis=1)]
            if extra is not None:
                cols.append(np.full(K, extra[k]))
            phi = np.stack(cols, axis=1)
            AtA += phi[:, :, None] * phi[:, None, :]
            Atb += phi * b[k, 2]
    bb = np.sum(b[mask] ** 2)         # all three rows, as in the notebook RMSE
    return AtA, Atb, bb, 3 * int(mask.sum())


def solve_thrust(AtA, Atb, bb, nrows):
    kf = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    sse = bb - 2 * np.einsum("ki,ki->k", kf, Atb) + np.einsum("ki,kij,kj->k", kf, AtA, kf)
    return kf, np.sqrt(np.maximum(sse, 0) / nrows)


def find_tm_symmetric(d, mass, mask, grid=np.linspace(0.005, 0.2, 100)):
    nm = thrust_normal_equations(d["t"], d["cmd"], d["acc"], mask, mass, grid, grid)
    kf, rmse = solve_thrust(*nm)
    i = int(np.argmin(rmse))
    return dict(T_m=grid[i], kf=kf[i], grid=grid, rmse=rmse)


def find_tm_asymmetric(d, mass, mask, lo=0.01, hi=0.2, n=24, drag_z=True):
    """Coarse 2-D grid, then a finer grid around the minimum. drag_z adds the -v_body_z column."""
    ex = -d["v_body"][:, 2] if drag_z else None
    g = np.linspace(lo, hi, n)
    TR, TF = np.meshgrid(g, g, indexing="ij")
    kf, rmse = solve_thrust(*thrust_normal_equations(d["t"], d["cmd"], d["acc"], mask, mass, TR.ravel(), TF.ravel(), ex))
    i = int(np.argmin(rmse))
    coarse = dict(g=g, rmse=rmse.reshape(TR.shape))
    step = g[1] - g[0]
    gr = np.linspace(max(lo / 2, TR.ravel()[i] - step), TR.ravel()[i] + step, 11)
    gf = np.linspace(max(lo / 2, TF.ravel()[i] - step), TF.ravel()[i] + step, 11)
    TR2, TF2 = np.meshgrid(gr, gf, indexing="ij")
    kf2, rmse2 = solve_thrust(*thrust_normal_equations(d["t"], d["cmd"], d["acc"], mask, mass, TR2.ravel(), TF2.ravel(), ex))
    j = int(np.argmin(rmse2))
    return dict(tau_rise=TR2.ravel()[j], tau_fall=TF2.ravel()[j], kf=kf2[j][:3], rmse=rmse2[j], coarse=coarse,
                d_z=float(kf2[j][3]) if drag_z else None)


def poly(kf, u):
    return kf[0] + kf[1] * u + kf[2] * u * u


# ----------------------------------------------------------------------------------------------------
# Whoop chain
# ----------------------------------------------------------------------------------------------------
def identify_whoop(log, P_known, motor="asym"):
    """P_known: measured quantities (mass, rotor geometry and directions). Returns identified values and
    the intermediate signals used by the plots."""
    d = prepare(log)
    m = P_known["mass"]
    fs = 1.0 / d["dt_mean"]
    seg = d["seg"]
    thr, rp, yw = seg == 1, seg == 2, seg == 3
    out = dict(prep=d)

    sym = find_tm_symmetric(d, m, thr)
    asy = find_tm_asymmetric(d, m, thr, drag_z=True)
    out["sym"], out["asym"] = sym, asy
    out["asym_nodrag"] = find_tm_asymmetric(d, m, thr, drag_z=False)
    if motor == "asym":
        tr, tf, kf = asy["tau_rise"], asy["tau_fall"], asy["kf"]
    else:
        tr = tf = sym["T_m"]
        kf = sym["kf"]
    u = ema(d["t"], d["cmd"], tr, tf)
    f = poly(kf, u)                                                   # (N, 4) per-rotor thrust
    geo = np.cross(P_known["rotor_pos"], P_known["rotor_thrust_dir"])  # (4, 3)
    tau_geo = f @ geo
    out.update(u=u, f=f, tau_rise=tr, tau_fall=tf, kf=kf)

    # Hover check of the notebook (cell 25): median command of the 5 % samples closest to a_z = g.
    az = np.abs(d["acc"][thr, 2] - G)
    sel = az < np.percentile(az, 5)
    out["hover_cmd_median"] = float(np.median(u[thr][sel]))
    out["hover_cmd_pred"] = float((-kf[1] + np.sqrt(kf[1] ** 2 - 4 * kf[2] * (kf[0] - G * m / 4))) / (2 * kf[2]))

    # Roll/pitch inertia.
    dw_raw = d["domega_raw"]
    I_nb = [np.dot(dw_raw[rp, i], tau_geo[rp, i]) / np.dot(dw_raw[rp, i], dw_raw[rp, i]) for i in (0, 1)]
    dw_f = lowpass(dw_raw, fs)
    tq_f = lowpass(tau_geo, fs)
    w_f = lowpass(d["gyro"], fs)
    # Reciprocal fit dw = tau / I, with the gyroscopic term -(w x J w) added to the torque. The term needs
    # J, which is what we estimate, so iterate (it converges in 2-3 passes because the term is small).
    Ixx, Iyy = [np.dot(tq_f[rp, i], tq_f[rp, i]) / np.dot(tq_f[rp, i], dw_f[rp, i]) for i in (0, 1)]
    out["I_recip_nogyro"] = [Ixx, Iyy]
    for _ in range(5):
        Jc = np.diag([Ixx, Iyy, INERTIA_RATIO * (Ixx + Iyy) / 2])
        tq_g = tq_f - M.cross(w_f, w_f @ Jc.T)
        Ixx, Iyy = [np.dot(tq_g[rp, i], tq_g[rp, i]) / np.dot(tq_g[rp, i], dw_f[rp, i]) for i in (0, 1)]
    I_rec = [Ixx, Iyy]
    out["I_notebook"], out["I_recip"] = I_nb, I_rec
    Ixx, Iyy = I_rec if motor == "asym" else I_nb
    Izz = INERTIA_RATIO * (Ixx + Iyy) / 2
    out["J"] = np.diag([Ixx, Iyy, Izz])

    # Yaw torque constant.
    uz = f @ P_known["rotor_torque_dir"][:, 2]
    k_nb = np.dot(uz[yw], dw_raw[yw, 2] * Izz) / np.dot(uz[yw], uz[yw])
    uz_f = lowpass(uz, fs)
    # b = I_zz dw_z + (w x J w)_z - geometric z torque (zero for a flat quad); A = sum of r_tau,z f_i.
    bz = dw_f[:, 2] * Izz + M.cross(w_f, w_f @ out["J"].T)[:, 2] - lowpass(tau_geo[:, 2], fs)
    k_f = np.dot(uz_f[yw], bz[yw]) / np.dot(uz_f[yw], uz_f[yw])
    out["k_tau_notebook"], out["k_tau_filtered"] = k_nb, k_f
    out["k_tau"] = k_f if motor == "asym" else k_nb
    out["fit_signals"] = dict(thr=thr, rp=rp, yw=yw, dw_f=dw_f, tq_f=tq_f, uz_f=uz_f)
    return out


def whoop_model(P_known, wid, body_drag=None):
    """Model parameter dict with the identified whoop values (keeps measured mass, geometry, plate)."""
    J = wid["J"].copy()
    if P_known["plate"] is not None:
        pl = P_known["plate"]
        J = J + M.plate_inertia(pl["mass"], pl["cx"], pl["cz"], pl["rc"])
    upd = dict(kf=np.array(wid["kf"]), k_tau=float(wid["k_tau"]), tau_rise=float(wid["tau_rise"]),
               tau_fall=float(wid["tau_fall"]), J=J)
    if body_drag is not None:
        upd["body_drag"] = np.array(body_drag)
    return M.with_updates(P_known, **upd)


# ----------------------------------------------------------------------------------------------------
# Body drag (no-plate flights)
# ----------------------------------------------------------------------------------------------------
def identify_body_drag(log, Pw):
    d = prepare(log)
    m = Pw["mass"]
    u = ema(d["t"], d["cmd"], Pw["tau_rise"], Pw["tau_fall"])
    Fz_rot = M.rotor_thrust(Pw, u).sum(axis=1)
    vb = d["v_body"]
    b = m * d["acc"]
    b[:, 2] -= Fz_rot
    dd = np.array([-np.dot(vb[:, i], b[:, i]) / np.dot(vb[:, i], vb[:, i]) for i in range(3)])
    return dict(body_drag=dd, prep=d)


# ----------------------------------------------------------------------------------------------------
# Plate
# ----------------------------------------------------------------------------------------------------
PLATE_KEYS = ("CN90", "Kpx", "Kpz", "CT")


def plate_force_meas(d, Pm):
    """Measured plate force in the body frame: m a_imu - rotor thrust - body drag."""
    u = ema(d["t"], d["cmd"], Pm["tau_rise"], Pm["tau_fall"])
    F = Pm["mass"] * d["acc"] - M.body_drag_force(Pm, d["v_body"])
    F[:, 2] -= M.rotor_thrust(Pm, u).sum(axis=1)
    return F, u


def _plate_with(pl, th):
    q = dict(pl)
    q.update(dict(zip(PLATE_KEYS, th)))
    return q


def fit_plate(d, Pm, mask=None, rows=(0, 1), x0=(1.0, 1.0, 1.0, 0.05), min_speed=0.3):
    """Nonlinear LS for (C_N90, K_px, K_pz, C_T). The z row is left out by default: it carries the rotor
    thrust, whose model errors (lag, thrust curve) are much larger than the plate's in-plane friction."""
    F_meas, _ = plate_force_meas(d, Pm)
    sel = np.linalg.norm(d["v_body"], axis=1) > min_speed
    if mask is not None:
        sel &= mask
    vb, w, Fm = d["v_body"][sel], d["gyro"][sel], F_meas[sel][:, list(rows)]
    pl = Pm["plate"]

    def res(th):
        F, _ = M.plate_wrench(_plate_with(pl, th), vb, w)
        return (F[:, list(rows)] - Fm).ravel()

    sol = optimize.least_squares(res, np.array(x0, float), bounds=([0.05, 0.01, 0.01, 0.0], [5, 20, 20, 1]),
                                 x_scale=[1, 1, 1, 0.02], method="trf")
    Jm = sol.jac
    dof = max(1, len(sol.fun) - len(sol.x))
    s2 = np.sum(sol.fun ** 2) / dof
    try:
        cov = s2 * np.linalg.inv(Jm.T @ Jm)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(4, np.inf)
    return dict(theta=dict(zip(PLATE_KEYS, sol.x)), se=dict(zip(PLATE_KEYS, se)), n=int(sel.sum()),
                rms=float(np.sqrt(s2)), sel=sel, F_meas=F_meas)


def plate_fisher_se(d, Pm, theta, mask, sigma, rows=(0, 1), min_speed=0.3):
    """Formal standard errors of the plate parameters for a subset of the data, evaluated at `theta`
    with a white residual of std `sigma` [N]. Used to say which manoeuvre identifies which parameter."""
    sel = mask & (np.linalg.norm(d["v_body"], axis=1) > min_speed)
    if sel.sum() < 10:
        return {k: np.inf for k in PLATE_KEYS}
    vb, w = d["v_body"][sel], d["gyro"][sel]
    pl = Pm["plate"]
    th0 = np.array([theta[k] for k in PLATE_KEYS])
    cols = []
    for i in range(4):
        h = 1e-4 * max(abs(th0[i]), 1e-2)
        tp, tm = th0.copy(), th0.copy()
        tp[i] += h
        tm[i] -= h
        Fp, _ = M.plate_wrench(_plate_with(pl, tp), vb, w)
        Fn, _ = M.plate_wrench(_plate_with(pl, tm), vb, w)
        cols.append(((Fp - Fn)[:, list(rows)] / (2 * h)).ravel())
    Jm = np.stack(cols, axis=1)
    info = Jm.T @ Jm / sigma ** 2
    try:
        se = np.sqrt(np.diag(np.linalg.inv(info)))
    except np.linalg.LinAlgError:
        se = np.full(4, np.inf)
    return dict(zip(PLATE_KEYS, se))


def fit_e0(d, Pm, axes=(0, 1, 2), mask=None):
    """e0 from the moment residual. tau_plate = r_c x F + e0 * (cos(a) c(psi) v_t/V_t) x F is linear in e0
    once F is evaluated with the identified force coefficients."""
    fs = 1.0 / d["dt_mean"]
    u = ema(d["t"], d["cmd"], Pm["tau_rise"], Pm["tau_fall"])
    f = M.rotor_thrust(Pm, u)
    _, tau_rot = M.rotor_wrench(Pm, f)
    w = d["gyro"]
    J = Pm["J"]
    dw = lowpass(d["domega_raw"], fs)
    w_f = lowpass(w, fs)
    tau_need = dw @ J.T + M.cross(w_f, w_f @ J.T)          # J dw + w x J w
    pl = Pm["plate"]
    F, _, aux = M.plate_wrench(dict(pl, e0=0.0), d["v_body"], w, aux=True)
    known = M.cross(pl["rc"], F)
    lever = np.cos(aux["alpha"]) * aux["chord"] / aux["Vt"]
    g = M.cross(lever[:, None] * aux["vt"], F)
    y = tau_need - lowpass(tau_rot, fs) - lowpass(known, fs)
    g = lowpass(g, fs)
    sel = np.ones(len(y), bool) if mask is None else mask
    ax = list(axes)
    gy, yy = g[sel][:, ax].ravel(), y[sel][:, ax].ravel()
    e0 = float(np.dot(gy, yy) / np.dot(gy, gy))
    r = yy - e0 * gy
    se = float(np.sqrt(np.sum(r ** 2) / (len(r) - 1) / np.dot(gy, gy)))
    return dict(e0=e0, se=se)


def identify_plate(log_plate, P_plate_known, wid, drag, mask_fn=None):
    """P_plate_known: measured quantities of the vehicle with the plate (mass, plate geometry, rho)."""
    Pm = whoop_model(P_plate_known, wid, body_drag=drag)
    d = prepare(log_plate)
    fit = fit_plate(d, Pm)
    Pm = M.with_updates(Pm, plate=fit["theta"])
    e0_all = fit_e0(d, Pm)
    e0_yaw = fit_e0(d, Pm, axes=(2,))
    Pm = M.with_updates(Pm, plate=dict(e0=e0_all["e0"]))
    return dict(fit=fit, e0=e0_all, e0_yaw=e0_yaw, model=Pm, prep=d)
