# Plate model (shared by the simulator and the identification)

Body frame as L2F: x forward, y left, z up. The plate lies in the body x-z plane, normal n = (0, 1, 0), area S,
horizontal edge c_x, vertical edge c_z, centroid r_c relative to the centre of mass. Parameters in
`params/whoop_nominal.json`.

Velocity of the plate relative to the air, body frame: v = Rᵀ (v_world − wind) + ω × r_c (+ k_w v_i ẑ for the
propeller downwash, k_w = 0 for now).

```
v_n = n·v,  v_t = v − v_n n,  V² = |v|² + ε,  V_t = sqrt(|v_t|² + ε)
α   = atan2(|v_n|, V_t)                                   (0 … 90°)
c²  = v_t,x² / V_t²                                       (wind along x: c² = 1, along z: c² = 0)
K_p = K_px c² + K_pz (1 − c²)
C_N = C_N90 tanh(K_p α / C_N90)
F   = −½ ρ S C_N V² sign(v_n) n − ½ ρ S C_T V v_t         (body frame)
c(ψ) = c_x c² + c_z (1 − c²)                              (chord along the in-plane wind)
r_cp = r_c + e0 cos(α) c(ψ) v_t / V_t                    (centre of pressure toward the leading edge)
τ   = r_cp × F
```

F and τ are added to the rotor force and torque in the body frame. With S = 0 the whoop has no plate. ε = 1e-6.

Identification: the parameters to recover are C_N90, K_px, K_pz, C_T (and e0 from the moments); S, c_x, c_z, r_c,
the added mass and inertia are measured. The whoop parameters (thrust curve, motor time constants, inertia, torque
constant) are identified first without the plate, with the method of Eschmann, Albani, Loianno, IROS 2024.
