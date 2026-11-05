"""Whoop rigid-body model: motor lag, rotor thrust and torque, body drag and the vertical plate.

Frames and conventions follow L2F (RLtools): body x forward, y left, z up, thrust along +z, gravity
(0, 0, -9.81), quaternion q = (w, x, y, z) rotating body vectors into the world frame, normalized motor
command u in [0, 1].

Everything is read from params/whoop_nominal.json and docs/plate_model.md. The functions work on single
vectors and on (N, 3) arrays so the same code is used by the simulator and by the identification.

How the motor lag is handled (asymmetric rising/falling):
  The JSON has both rotor_time_constant_rising and rotor_time_constant_falling. As in L2F
  (rl_tools .../l2f/operations_generic/70_post_integration.h, closed-form branch) the lag acts on the
  normalized motor state u_m (not on the thrust): over one step of length dt with the command held,
      tau   = tau_rising if u_cmd >= u_m else tau_falling
      u_m  <- exp(-dt/tau) u_m + (1 - exp(-dt/tau)) u_cmd
  and the thrust is the quadratic curve evaluated at u_m. The JSON comment says "response of the thrust";
  we follow the L2F implementation because the JSON field names are the L2F ones and the simulator in sim/
  is built on L2F. [assumption, to confirm with the sim/ owner]
"""
import copy
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO / "params" / "whoop_nominal.json"
EPS = 1e-6  # docs/plate_model.md: regularizes V and V_t at zero airspeed

# Body drag of the bare whoop used as ground truth in the synthetic flights [assumption].
# Linear drag F = -diag(d) v_body. 0.008 kg/s gives 0.024 N (8 % of the weight) at 3 m/s horizontally,
# a typical order for a ducted 75 mm whoop; vertical drag is taken smaller (ducts are open along z).
TRUE_BODY_DRAG = np.array([0.008, 0.008, 0.004])


# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
def plate_inertia(m_p, cx, cz, rc):
    """Inertia of a thin rectangular plate in the body x-z plane about the CoM (parallel-axis theorem).

    The plate centroid rc only has a z component in the nominal file, so the products of inertia vanish;
    we keep only the diagonal to stay consistent with the diagonal J of the whoop.
    """
    x, y, z = rc
    return m_p * np.diag([cz**2 / 12 + y**2 + z**2,
                          (cx**2 + cz**2) / 12 + x**2 + z**2,
                          cx**2 / 12 + x**2 + y**2])


def load_params(path=DEFAULT_PARAMS, plate=False, body_drag=TRUE_BODY_DRAG):
    """Return a plain dict of numpy parameters.

    plate=True adds the plate aerodynamics and its mass and inertia. "added_mass" in the JSON is read as
    the physical mass of the EPP sheet (3 mm EPP of 78 cm2 weighs about 0.5-0.7 g) [assumption]; rotor
    positions and the plate centroid are taken relative to the CoM of the vehicle with the plate (the
    0.9 mm CoM shift is neglected) [assumption].
    """
    raw = json.loads(Path(path).read_text())
    w, p = raw["whoop"], raw["plate"]
    P = {
        "mass": float(w["mass"]),
        "g": np.array(w["gravity"], float),
        "rotor_pos": np.array(w["rotor_positions"], float),
        "rotor_thrust_dir": np.array(w["rotor_thrust_directions"], float),
        "rotor_torque_dir": np.array(w["rotor_torque_directions"], float),
        "kf": np.array(w["rotor_thrust_coefficients"], float),  # thrust = kf0 + kf1 u + kf2 u^2 [N]
        "k_tau": float(w["rotor_torque_constant"]),             # yaw torque = k_tau * thrust [m]
        "tau_rise": float(w["rotor_time_constant_rising"]),
        "tau_fall": float(w["rotor_time_constant_falling"]),
        "J": np.array(w["J"], float),
        "body_drag": np.array(body_drag, float),
        "plate": None,
    }
    if plate:
        pl = {
            "S": float(p["area"]), "cx": float(p["chord_x"]), "cz": float(p["height_z"]),
            "n": np.array(p["normal"], float), "rc": np.array(p["centroid"], float),
            "CN90": float(p["CN90"]), "Kpx": float(p["Kp_x"]), "Kpz": float(p["Kp_z"]),
            "CT": float(p["CT"]), "e0": float(p["e0"]), "rho": float(p["air_density"]),
            "mass": float(p["added_mass"]),
        }
        # The downwash term k_w v_i z needs the induced velocity; it is 0 in the file, so it is not implemented.
        assert float(p["downwash_factor"]) == 0.0, "downwash term not implemented"
        P["plate"] = pl
        P["mass"] += pl["mass"]
        P["J"] = P["J"] + plate_inertia(pl["mass"], pl["cx"], pl["cz"], pl["rc"])
    finalize(P)
    return P


def finalize(P):
    """Precompute derived quantities (call again after editing a parameter dict)."""
    P["J_inv"] = np.linalg.inv(P["J"])
    # Column i of the allocation: [thrust; torque] produced by 1 N on rotor i.
    geo = np.cross(P["rotor_pos"], P["rotor_thrust_dir"])                 # (4, 3) r x r_f
    yaw = P["k_tau"] * P["rotor_torque_dir"]                              # (4, 3)
    P["torque_per_thrust"] = geo + yaw                                    # (4, 3)
    return P


def with_updates(P, **kw):
    """Copy of a parameter dict with some entries replaced (plate entries via plate={...})."""
    Q = copy.deepcopy(P)
    plate_kw = kw.pop("plate", None)
    Q.update(kw)
    if plate_kw is not None and Q["plate"] is not None:
        Q["plate"].update(plate_kw)
    return finalize(Q)


# ----------------------------------------------------------------------------------------------------
# Small math helpers (vectorized over a leading axis)
# ----------------------------------------------------------------------------------------------------
def cross(a, b):
    # np.cross is slow on single 3-vectors; this explicit form is ~5x faster and broadcasts the same way.
    a = np.asarray(a)
    b = np.asarray(b)
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def quat_to_rot(q):
    """Rotation matrix (body -> world) from q = (w, x, y, z); q may be (4,) or (N, 4)."""
    q = np.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_from_rotvec(r):
    r = np.asarray(r, float)
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * ang
    # sin(half)/ang -> 0.5 as ang -> 0
    k = np.where(ang > 1e-12, np.sin(half) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([np.cos(half), k * r], axis=-1)


def rot_to_quat(R):
    """Single rotation matrix -> quaternion (w, x, y, z), Shepperd's method."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = [(R[k, j] - R[j, k]) / s, *v]
    q = np.array(q)
    return q / np.linalg.norm(q)


# ----------------------------------------------------------------------------------------------------
# Rotors
# ----------------------------------------------------------------------------------------------------
def rotor_thrust(P, u):
    """Per-rotor thrust [N] from the motor state u (..., 4)."""
    k0, k1, k2 = P["kf"]
    return k0 + k1 * u + k2 * u * u


def thrust_to_command(P, f):
    """Inverse of the thrust curve (used only by the controller)."""
    k0, k1, k2 = P["kf"]
    disc = np.maximum(k1 * k1 + 4 * k2 * (f - k0), 0.0)
    return np.clip((-k1 + np.sqrt(disc)) / (2 * k2), 0.0, 1.0)


def motor_update(P, u_m, u_cmd, dt):
    """Closed-form first-order lag with rising/falling time constants (L2F closed-form branch)."""
    tau = np.where(u_cmd >= u_m, P["tau_rise"], P["tau_fall"])
    a = np.exp(-dt / tau)
    return a * u_m + (1 - a) * u_cmd


def rotor_wrench(P, f):
    """Body force and torque of the rotors for per-rotor thrusts f (..., 4)."""
    F = f @ P["rotor_thrust_dir"]            # (..., 3)
    tau = f @ P["torque_per_thrust"]          # (..., 3) eq. (9) of the paper: (r_p x r_f) f + r_tau K_tau f
    return F, tau


# ----------------------------------------------------------------------------------------------------
# Aerodynamics
# ----------------------------------------------------------------------------------------------------
def body_drag_force(P, v_body):
    return -P["body_drag"] * v_body


def plate_wrench(pl, v_body, omega, aux=False):
    """Plate force and torque in the body frame, exactly docs/plate_model.md.

    v_body: velocity of the CoM relative to the air in the body frame, (3,) or (N, 3).
    omega: body angular velocity. pl: plate dict (CN90, Kpx, Kpz, CT, e0 may be overridden).
    """
    n, rc = pl["n"], pl["rc"]
    v = v_body + cross(omega, rc)                    # local velocity at the plate centroid
    vn = v @ n
    vt = v - vn[..., None] * n
    V2 = np.sum(v * v, axis=-1) + EPS
    Vt = np.sqrt(np.sum(vt * vt, axis=-1) + EPS)
    alpha = np.arctan2(np.abs(vn), Vt)
    c2 = vt[..., 0] ** 2 / Vt ** 2
    Kp = pl["Kpx"] * c2 + pl["Kpz"] * (1 - c2)
    CN = pl["CN90"] * np.tanh(Kp * alpha / pl["CN90"])
    qS = 0.5 * pl["rho"] * pl["S"]
    F = (-qS * CN * V2 * np.sign(vn))[..., None] * n - (qS * pl["CT"] * np.sqrt(V2))[..., None] * vt
    chord = pl["cx"] * c2 + pl["cz"] * (1 - c2)
    lever = pl["e0"] * np.cos(alpha) * chord / Vt     # CoP shift toward the leading edge
    rcp = rc + lever[..., None] * vt
    tau = cross(rcp, F)
    if aux:
        return F, tau, dict(alpha=alpha, c2=c2, CN=CN, V2=V2, vn=vn, vt=vt, Vt=Vt, chord=chord, v=v)
    return F, tau


def aero_wrench(P, v_body, omega):
    """Body drag (acts at the CoM, no torque) plus plate."""
    F = body_drag_force(P, v_body)
    tau = np.zeros_like(F)
    if P["plate"] is not None:
        Fp, tp = plate_wrench(P["plate"], v_body, omega)
        F = F + Fp
        tau = tau + tp
    return F, tau


# ----------------------------------------------------------------------------------------------------
# Rigid body
# ----------------------------------------------------------------------------------------------------
def derivative(P, x, u_m, wind=None):
    """State derivative and specific force (what an accelerometer at the CoM reads, body frame).

    x = [p (3), v_world (3), q (4), omega_body (3)], u_m = motor states (4,).
    """
    v, q, w = x[3:6], x[6:10], x[10:13]
    R = quat_to_rot(q)
    v_air_body = R.T @ (v if wind is None else v - wind)
    f = rotor_thrust(P, u_m)
    Fr, tr = rotor_wrench(P, f)
    Fa, ta = aero_wrench(P, v_air_body, w)
    F = Fr + Fa
    acc = R @ F / P["mass"] + P["g"]
    J = P["J"]
    wdot = P["J_inv"] @ (tr + ta - cross(w, J @ w))
    qdot = 0.5 * quat_mul(q, np.array([0.0, w[0], w[1], w[2]]))
    return np.concatenate([v, acc, qdot, wdot]), F / P["mass"]


def rk4_step(P, x, u_m, dt, wind=None):
    """RK4 on the rigid body with the motor states held over the step (then updated by motor_update)."""
    k1, sf = derivative(P, x, u_m, wind)
    k2, _ = derivative(P, x + 0.5 * dt * k1, u_m, wind)
    k3, _ = derivative(P, x + 0.5 * dt * k2, u_m, wind)
    k4, _ = derivative(P, x + dt * k3, u_m, wind)
    xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[6:10] /= np.linalg.norm(xn[6:10])
    return xn, sf


def hover_command(P):
    return float(thrust_to_command(P, P["mass"] * -P["g"][2] / 4))
