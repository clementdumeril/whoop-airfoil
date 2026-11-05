"""Synthetic whoop flights with realistic logging, generated from the known model in model.py.

Two kinds of flights:
  "excitation": the paper's identification data (thrust, roll/pitch and yaw excitation, flown more or less
                in place), whoop without the plate. Thirds of the flight: [thrust | roll/pitch | yaw].
  "plate":      constant-heading circles at 1-4 m/s (both directions), straight forward, sideways and
                diagonal passes, sideways climbs and vertical shuttles (research/flat_plate_aero.md 5.2).
                The same schedule is flown with and without the plate so the body drag can be separated.

The simulator runs the physics (RK4) and the controller at 500 Hz, the blackbox rate, to keep a 4-minute
flight under a minute of wall time. Logs mimic what we will have on the real drone:
  * Betaflight blackbox: gyro, accelerometer and motor commands in one frame at 500 Hz, motor command
    quantized to 1/2000 (Betaflight motor output resolution). On the real drone the PID loop runs faster
    than the log, so the logged command is a sub-sample of what the motors got [simplification].
  * Mocap: position and attitude at 125 Hz (every 4th blackbox frame), with white noise.
  * A 3 s static segment before take-off (drone level on the ground, motors off) where the gyro and
    accelerometer biases can be calibrated, as Betaflight does at arming.
The noise-free truth is simulated once and cached; sensor noise is added afterwards for each noise level.

Simplifications [assumption]: IMU at the CoM, white sensor noise (no vibration spectrum, no onboard
low-pass filter delay), constant biases, perfect clock sync between mocap and blackbox, still air, no
ground effect, the controller knows the true state and uses the true aerodynamic force as feed-forward
(it only has to make the reference manoeuvres happen; the identification never sees the controller).
"""
import os
from pathlib import Path

import numpy as np

import model as M

CACHE = Path(os.environ.get("WHOOP_SYSID_CACHE", Path.home() / "Work" / "whoop-airfoil-sysid" / "cache"))
DT = 2e-3          # physics and control step [s]
LOG_EVERY = 1      # blackbox at 500 Hz
MOCAP_EVERY = 4    # mocap at 125 Hz (in blackbox frames)
STATIC_S = 3.0     # static calibration segment before the flight [s]
Z0 = 1.5           # flight altitude [m]

# Sensor noise levels (1 sigma per sample). "nominal" is our guess for a filtered whoop IMU in flight and a
# Vicon/OptiTrack room; "high" mimics strong prop vibration on the accelerometer [assumption].
NOISE = {
    "low":     dict(gyro=0.005, acc=0.05, gyro_bias=0.005, acc_bias=0.05, pos=0.2e-3, att_deg=0.1),
    "nominal": dict(gyro=0.02,  acc=0.3,  gyro_bias=0.02,  acc_bias=0.2,  pos=0.5e-3, att_deg=0.3),
    "high":    dict(gyro=0.08,  acc=1.5,  gyro_bias=0.05,  acc_bias=0.5,  pos=2.0e-3, att_deg=1.0),
}

SEG_NAMES = {0: "hover", 1: "thrust exc.", 2: "roll/pitch exc.", 3: "yaw exc.",
             10: "circles", 11: "forward passes", 12: "sideways passes", 13: "diagonal passes",
             14: "sideways climbs", 15: "vertical shuttles"}


# ----------------------------------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------------------------------
def multisine(t, f_lo, f_hi, n, rng):
    """Sum of n sines with log-spaced frequencies and random phases, scaled to unit peak (Schroeder-like)."""
    f = np.geomspace(f_lo, f_hi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sin(2 * np.pi * f[None, :] * t[:, None] + ph[None, :]).sum(axis=1)
    return s / np.max(np.abs(s))


def window(t, t0, t1, ramp):
    """C2 (quintic smootherstep) window: 0 outside [t0, t1], 1 inside, ramps of length `ramp`."""
    def ss(x):
        x = np.clip(x, 0, 1)
        return x ** 3 * (10 - 15 * x + 6 * x * x)
    return ss((t - t0) / ramp) * ss((t1 - t) / ramp)


def derivatives(p, dt):
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return v, a


# ----------------------------------------------------------------------------------------------------
# Manoeuvre schedules: reference position/yaw, torque and collective perturbations, segment labels
# ----------------------------------------------------------------------------------------------------
def excitation_schedule(duration, P, rng):
    """Hover with thrust, then roll/pitch, then yaw excitation (paper section IV-A, about one minute)."""
    t = np.arange(0, duration, DT)
    n = len(t)
    T3 = duration / 3
    seg = np.where(t < T3, 1, np.where(t < 2 * T3, 2, 3))
    w1, w2, w3 = window(t, 0, T3, 0.5), window(t, T3, 2 * T3, 0.5), window(t, 2 * T3, duration, 0.5)
    p = np.zeros((n, 3))
    p[:, 2] = Z0
    # Thrust phase: slow altitude wandering plus a large collective multisine (0.3-6 Hz) so the motor
    # commands sweep a wide part of the thrust curve and the lag is excited.
    p[:, 2] += 0.3 * w1 * multisine(t, 0.1, 0.5, 4, rng)
    hover_T = P["mass"] * 9.81
    coll = 0.75 * hover_T * w1 * multisine(t, 0.3, 6.0, 12, rng)
    # Roll/pitch phase: lateral wandering (real tilts) plus torque multisines (1-10 Hz).
    p[:, 0] += 0.3 * w2 * multisine(t, 0.1, 0.6, 4, rng)
    p[:, 1] += 0.3 * w2 * multisine(t, 0.1, 0.6, 4, rng)
    tq = np.zeros((n, 3))
    tq[:, 0] = 1.0e-3 * w2 * multisine(t, 1.0, 10.0, 12, rng)
    tq[:, 1] = 1.0e-3 * w2 * multisine(t, 1.0, 10.0, 12, rng)
    # Yaw phase: large heading changes plus yaw torque multisine.
    yaw = 1.2 * w3 * multisine(t, 0.1, 0.8, 5, rng)
    tq[:, 2] = 4.0e-4 * w3 * multisine(t, 0.5, 6.0, 10, rng)
    # Small background torque dither everywhere (decorrelates the motors, as pilot inputs do).
    tq += 1.0e-4 * np.stack([multisine(t, 0.5, 8, 8, rng) for _ in range(3)], axis=1)
    v, a = derivatives(p, DT)
    return dict(t=t, p=p, v=v, a=a, yaw=yaw, torque=tq, coll=coll, seg=seg)


def plate_segments():
    """List of (label, kind, speed, extra). Same list with and without the plate."""
    segs = [(10, "circle", v, d) for v in (1.0, 2.0, 3.0, 4.0) for d in (+1, -1)]
    segs += [(11, "line", 2.0, (1, 0, 0)), (11, "line", 3.5, (1, 0, 0)),
             (12, "line", 2.0, (0, 1, 0)), (12, "line", 3.5, (0, 1, 0)),
             (13, "line", 3.0, (np.sqrt(0.5), np.sqrt(0.5), 0)),
             (14, "line", 2.5, (0, np.sqrt(0.5), np.sqrt(0.5))),
             (15, "line", 1.5, (0, 0, 1))]
    return segs


def plate_schedule(duration, P, rng):
    """Constant-heading (yaw = 0) circles and straight shuttles, about duration/15 seconds per segment.

    Every segment starts and ends at rest at the same point: an integer number of laps (circles) or of
    full shuttle cycles (lines) with a smooth speed ramp, so consecutive segments join without jumps. To keep
    the peak speed, the circle radius (centripetal acceleration 6 m/s2) and the shuttle amplitude (room
    size) exact, each segment's length is rounded to fit a whole number of laps; the real total duration
    is therefore close to, not exactly, `duration`. Needs duration >= 60 s (segments of ~4 s).
    """
    segs = plate_segments()
    Ts0 = duration / len(segs)
    assert Ts0 >= 4.0, "plate flights need >= 4 s per segment (>= 60 s in total)"
    plan = []
    for lab, kind, v, extra in segs:
        ramp = max(1.0, 0.6 * v)                 # tangential acceleration during the ramp <= ~3 m/s2
        if kind == "circle":
            size = max(1.0, v * v / 6.0)         # radius: centripetal acceleration <= 6 m/s2 (r = 2.7 m at 4 m/s)
            per = 2 * np.pi * size               # path length of one lap
        else:
            d = np.array(extra, float)
            size = 0.6 if d[2] > 0.99 else (1.0 if d[2] > 0 else 2.0)   # amplitude: room is lower than wide
            per = 2 * np.pi * size               # "path length" of one full shuttle cycle (0, +A, 0, -A, 0)
        k = max(1, int(round(v * (Ts0 - ramp) / per)))
        W = per * k / v                          # integral of the speed profile that gives peak speed v
        plan.append((lab, kind, v, extra, ramp, size, k, W + ramp))
    total = sum(x[-1] for x in plan)
    t = np.arange(0, total, DT)
    n = len(t)
    p = np.zeros((n, 3))
    seg = np.zeros(n, int)
    t0 = 0.0
    for lab, kind, v, extra, ramp, size, k, Ts in plan:
        t1 = t0 + Ts
        m = (t >= t0) & (t < t1)
        seg[m] = lab
        w = window(t[m], t0, t1, ramp)
        S = np.cumsum(w) / w.sum()               # 0 -> 1 progress along the path
        if kind == "circle":
            ph = 2 * np.pi * k * S
            raw = np.stack([size * (np.cos(ph) - 1), extra * size * np.sin(ph), 0 * ph], axis=1)
        else:
            raw = size * np.sin(2 * np.pi * k * S)[:, None] * np.array(extra, float)[None, :]
        p[m] = raw
        t0 = t1
    p[:, 2] += Z0
    tq = 2.0e-4 * np.stack([multisine(t, 0.5, 8, 8, rng) for _ in range(3)], axis=1)
    v_, a_ = derivatives(p, DT)
    return dict(t=t, p=p, v=v_, a=a_, yaw=np.zeros(n), torque=tq, coll=np.zeros(n), seg=seg)


# ----------------------------------------------------------------------------------------------------
# Controller: position PD with feed-forward -> geometric attitude PD -> mixer -> inverse thrust curve
# ----------------------------------------------------------------------------------------------------
class Controller:
    def __init__(self, P, kp=8.0, kv=5.0, kR=(120.0, 120.0, 60.0), kw=(16.0, 16.0, 12.0), max_tilt=0.9):
        self.P = P
        self.kp, self.kv = kp, kv
        self.kR, self.kw = np.array(kR), np.array(kw)
        self.max_tilt = max_tilt
        A = np.vstack([P["rotor_thrust_dir"][:, 2], P["torque_per_thrust"].T])   # 4x4 allocation
        self.A_inv = np.linalg.inv(A)
        self.f_min = float(M.rotor_thrust(P, 0.0))
        self.f_max = float(M.rotor_thrust(P, 1.0))

    def __call__(self, x, p_ref, v_ref, a_ref, yaw_ref, tq_exc, coll_exc):
        P = self.P
        p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        R = M.quat_to_rot(q)
        Fa, ta = M.aero_wrench(P, R.T @ v, w)               # true aero feed-forward (simulation only)
        a_des = a_ref + self.kp * (p_ref - p) + self.kv * (v_ref - v)
        F_des = P["mass"] * (a_des - P["g"]) - R @ Fa
        # Limit the tilt so the excitation never flips the vehicle.
        horiz = np.linalg.norm(F_des[:2])
        if F_des[2] < 1e-3:
            F_des[2] = 1e-3
        if horiz > np.tan(self.max_tilt) * F_des[2]:
            F_des[:2] *= np.tan(self.max_tilt) * F_des[2] / horiz
        zb = F_des / np.linalg.norm(F_des)
        xc = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        Rd = np.stack([xb, yb, zb], axis=1)
        E = Rd.T @ R - R.T @ Rd
        eR = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        J = P["J"]
        tau = J @ (-self.kR * eR - self.kw * w) + np.cross(w, J @ w) - ta + tq_exc
        T = float(F_des @ R[:, 2]) + coll_exc
        # Yaw has the lowest priority (as in Betaflight): thrust and roll/pitch first, then as much of the
        # yaw torque as fits within the motor limits.
        f0 = self.A_inv @ np.array([T, tau[0], tau[1], 0.0])
        fy = self.A_inv @ np.array([0.0, 0.0, 0.0, tau[2]])
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = np.where(fy > 0, (self.f_max - f0) / fy, np.where(fy < 0, (self.f_min - f0) / fy, np.inf))
        s = float(np.clip(np.min(lim), 0.0, 1.0))
        f = np.clip(f0 + s * fy, self.f_min, self.f_max)
        return M.thrust_to_command(P, f)


# ----------------------------------------------------------------------------------------------------
# Simulation (noise-free truth) and sensor model
# ----------------------------------------------------------------------------------------------------
def simulate(P, sched, ctrl_gains=None):
    """ctrl_gains: keyword arguments of Controller (gains) instead of its defaults"""
    t = sched["t"]
    n = len(t)
    x = np.zeros(13)
    x[0:3] = sched["p"][0]
    x[6] = 1.0
    u_m = np.full(4, M.hover_command(P))
    ctrl = Controller(P, **(ctrl_gains or {}))
    nl = (n + LOG_EVERY - 1) // LOG_EVERY
    L = dict(t=np.zeros(nl), x=np.zeros((nl, 13)), u_cmd=np.zeros((nl, 4)), u_m=np.zeros((nl, 4)),
             sf=np.zeros((nl, 3)), seg=np.zeros(nl, int))
    j = 0
    for k in range(n):
        u_cmd = ctrl(x, sched["p"][k], sched["v"][k], sched["a"][k], sched["yaw"][k],
                     sched["torque"][k], sched["coll"][k])
        x_next, sf = M.rk4_step(P, x, u_m, DT)
        if k % LOG_EVERY == 0:
            L["t"][j], L["x"][j], L["u_cmd"][j], L["u_m"][j], L["sf"][j] = t[k], x, u_cmd, u_m, sf
            L["seg"][j] = sched["seg"][k]
            j += 1
        x = x_next
        u_m = M.motor_update(P, u_m, u_cmd, DT)
        if not np.all(np.isfinite(x)) or x[2] < 0.0:
            raise RuntimeError(f"simulation diverged at t={t[k]:.2f}s")
    return L


def truth_flight(kind, duration, plate, seed, body_drag=M.TRUE_BODY_DRAG, params=None, gains=None):
    """Noise-free truth log, cached as npz under CACHE.

    params: another parameter file in the format of params/whoop_nominal.json to simulate instead of the nominal
    whoop (a "true" drone for the sim-to-sim test of sysid/export_params.py). Its own body_drag is then the truth, and
    the cache file name carries the file's stem so that the nominal cache is never reused for it.
    gains: attitude gains of the controller, dict(kR=(...), kw=(...)), instead of its defaults (kR 120, 120, 60 and
    kw 16, 16, 12); they go into the cache name. The controller only flies the manoeuvres (the identification never
    sees it); a drone with a stronger plate and slower motors needs stiffer attitude control in the sideways climbs,
    where the plate moment is largest.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "" if params is None else "_" + Path(params).stem
    if gains is not None:
        tag += "".join(f"_{k}" + "-".join(f"{g:g}" for g in v) for k, v in sorted(gains.items()))
    fn = CACHE / f"truth{tag}_{kind}_{int(duration)}s_plate{int(plate)}_seed{seed}.npz"
    if fn.exists():
        d = np.load(fn)
        return {k: d[k] for k in d.files}
    if params is None:
        P = M.load_params(plate=plate, body_drag=body_drag)
    else:
        import json
        drag = json.loads(Path(params).read_text())["whoop"]["body_drag"]
        P = M.load_params(params, plate=plate, body_drag=drag)
    rng = np.random.default_rng(seed)
    sched = (excitation_schedule if kind == "excitation" else plate_schedule)(duration, P, rng)
    L = simulate(P, sched, gains)
    np.savez_compressed(fn, **L)
    return L


def measure(L, level, seed):
    """Apply the sensor model to a truth log; returns blackbox-like and mocap-like arrays."""
    nz = NOISE[level]
    rng = np.random.default_rng(10_000 + seed)
    dt_log = DT * LOG_EVERY
    ns = int(round(STATIC_S / dt_log))
    t = np.concatenate([L["t"][0] - dt_log * np.arange(ns, 0, -1), L["t"]])
    # Static segment: level on the ground under the take-off point, motors off.
    # The static pose is level with yaw 0, so the specific force is +g along body z.
    x0 = L["x"][0]
    sf_static = np.tile([0.0, 0.0, 9.81], (ns, 1))
    gyro_true = np.concatenate([np.zeros((ns, 3)), L["x"][:, 10:13]])
    sf_true = np.concatenate([sf_static, L["sf"]])
    cmd = np.concatenate([np.zeros((ns, 4)), L["u_cmd"]])
    pos_true = np.concatenate([np.tile([x0[0], x0[1], 0.02], (ns, 1)), L["x"][:, 0:3]])
    q_true = np.concatenate([np.tile([1.0, 0, 0, 0], (ns, 1)), L["x"][:, 6:10]])
    bg = rng.normal(0, nz["gyro_bias"], 3)
    ba = rng.normal(0, nz["acc_bias"], 3)
    gyro = gyro_true + bg + rng.normal(0, nz["gyro"], gyro_true.shape)
    acc = sf_true + ba + rng.normal(0, nz["acc"], sf_true.shape)
    cmd = np.round(cmd * 2000) / 2000
    im = np.arange(0, len(t), MOCAP_EVERY)
    pos = pos_true[im] + rng.normal(0, nz["pos"], (len(im), 3))
    dq = M.quat_from_rotvec(rng.normal(0, np.deg2rad(nz["att_deg"]), (len(im), 3)))
    quat = M.quat_mul(q_true[im], dq)
    static = np.zeros(len(t), bool)
    static[:ns] = True
    seg = np.concatenate([np.full(ns, -1), L["seg"]])
    return dict(t=t, gyro=gyro, acc=acc, cmd=cmd, static=static, seg=seg,
                t_mocap=t[im], pos=pos, quat=quat,
                truth=dict(bg=bg, ba=ba, x=np.concatenate([np.tile(L["x"][:1], (ns, 1)), L["x"]]),
                           u_m=np.concatenate([np.zeros((ns, 4)), L["u_m"]]), sf=sf_true))


if __name__ == "__main__":
    import time
    for kind, dur, plate in [("excitation", 30, False), ("plate", 60, True)]:
        t0 = time.time()
        L = truth_flight(kind, dur, plate, seed=0)
        sp = np.linalg.norm(L["x"][:, 3:6], axis=1)
        print(f"{kind} {dur}s plate={plate}: {time.time() - t0:.1f}s wall, max speed {sp.max():.2f} m/s, "
              f"cmd range {L['u_cmd'].min():.2f}-{L['u_cmd'].max():.2f}, "
              f"sat frac {(L['u_cmd'] >= 0.999).mean():.3f}/{(L['u_cmd'] <= 0.001).mean():.3f}, "
              f"max |w| {np.abs(L['x'][:, 10:13]).max(axis=0).round(1)}, "
              f"max tilt {np.degrees(np.arccos(np.clip(M.quat_to_rot(L['x'][:, 6:10])[:, 2, 2], -1, 1))).max():.0f} deg")
