# Plate model (shared by the simulator and the identification)

Body frame as L2F: x forward, y left, z up. The plate is a flat rectangle of area S with its centroid at r_c relative
to the centre of mass. Its frame is given by the unit normal n and an in-plane unit vector e1 (`chord_direction`) along
the edge c1 (`chord_x`); the other edge c2 (`height_z`) is along e2 = e1 × n. Parameters in `params/whoop_nominal.json`.

| plate | n | e1 | e2 |
| --- | --- | --- | --- |
| vertical (nominal) | (0, 1, 0) | x | z |
| tilted by θ about body x (`--plate-tilt θ`) | (0, cos θ, sin θ) | x | (0, −sin θ, cos θ) |
| horizontal (θ = 90°) | (0, 0, 1) | x | −y |

Velocity of the plate relative to the air, body frame: v = Rᵀ (v_world − wind) + ω × r_c + k_w v_i ẑ (propeller
downwash, below).

```
v_n = n·v,  v_t = v − v_n n,  V² = |v|² + ε,  V_t = sqrt(|v_t|² + ε)
α   = atan2(|v_n|, V_t)                                   (0 … 90°)
c²  = (v_t·e1)² / V_t²                                    (in-plane wind along e1: c² = 1, along e2: c² = 0)
K_p = K_px c² + K_pz (1 − c²)
C_N = C_N90 tanh(K_p α / C_N90)
F   = −½ ρ S C_N V² sign(v_n) n − ½ ρ S C_T V v_t         (body frame)
c(ψ) = c1 c² + c2 (1 − c²)                                (chord along the in-plane wind)
r_cp = r_c + e0 cos(α) c(ψ) v_t / V_t                    (centre of pressure toward the leading edge)
τ   = r_cp × F
```

F and τ are added to the rotor force and torque in the body frame. With S = 0 the whoop has no plate. ε = 1e-6.
K_px and K_pz are the slopes for the in-plane wind along e1 and e2. With the nominal edges (c1 = 12 cm along e1,
c2 = 6.5 cm) a wind along e1 sees a span of 6.5 cm (aspect ratio 0.54, K_px = 0.84), a wind along e2 a span of 12 cm
(aspect ratio 1.85, K_pz = 2.46), whatever the plate's orientation in the body. For the vertical plate, c² = v_t,x² / V_t²
as before the plate frame was generalised, and the code gives the same bits (test `vertical plate: ... bit for bit`).

Mass and inertia of the sheet (thin, no thickness along n): J_sheet = m_p/12 (c1² (I − e1 e1ᵀ) + c2² (I − e2 e2ᵀ)),
moved to the centre of mass of the vehicle with the parallel-axis terms (`sim/tools/generate_params.py`). For the
vertical plate this is m_p/12 diag(c2², c1² + c2², c1²); a tilted plate has a product of inertia J_yz.

## Propeller downwash (crude, off by default)

k_w = `downwash_factor` (0 nominal). With k_w > 0 the plate's relative velocity gets k_w v_i along body +z (the wake moves
down at v_i, so the plate moves up relative to it), with v_i the induced velocity of the rotors from momentum theory with
Glauert's forward-flight correction and the axial flow neglected:

```
v_h² = T / (2 ρ A),   v_i² (v_i² + V_par²) = v_h⁴,   A = 4 π R²   (R = whoop.rotor_radius, 20 mm)
```

T is the total rotor thrust (from the rotor states in the simulator, solved with the trim in `plate_trim.py`), V_par the
speed of the centre of mass in the disc plane (body x, y). At hover v_i = 5.0 m/s. The term needs the plate centroid
below the rotor discs; the readers refuse k_w > 0 otherwise.

What this term is and is not: it puts the whole plate in a uniform stream of the disc's induced velocity (k_w = 1) or
outside the wake (k_w = 0). It ignores the wake's contraction and acceleration below the disc, its skew in forward
flight, the parts of the plate outside the four slipstreams, and the change of the rotor thrust when the plate blocks or
turns the wake. The rotor thrust is kept as if the plate were not there, so a plate that turns the wake sideways (θ near 0,
wake along the plate) gets a side force without the matching loss of rotor thrust: the momentum of the wake is not
conserved. k_w = 0 and 1 are two ends of a crude range, not measured bounds.

## Body drag of the whoop

With or without the plate, the whoop's own body (frame, ducts, battery) has a linear drag at the centre of mass, the
model of `sysid/model.py` (`body_drag_force`):

```
F_drag = −diag(d) v_cm,    v_cm = Rᵀ (v_world − wind)      (body frame, no torque)
```

v_cm is the velocity of the centre of mass relative to the air, without the ω × r_c term of the plate. d =
`whoop.body_drag` = (0.008, 0.008, 0.004) kg/s, the truth of the synthetic flights: 0.024 N, 8 % of the weight, at
3 m/s forward. The simulator adds F_drag to the rotor and plate forces (`sim/include/whoop/body_drag.h`).

In training, C_N90, K_px, K_pz, C_T, e0 and each component of d are multiplied per episode by independent factors
from U(1 − r, 1 + r), with r = 0.4 for the plate and 0.5 for the drag (`randomization` in the JSON). k_w is not
randomized.

Identification: the parameters to recover are C_N90, K_px, K_pz, C_T (and e0 from the moments); S, c1, c2, n, e1, r_c,
the added mass and inertia are measured. The whoop parameters (thrust curve, motor time constants, inertia, torque
constant) are identified first without the plate, with the method of Eschmann, Albani, Loianno, IROS 2024. The
identification in `sysid/` was only run on the vertical plate.
