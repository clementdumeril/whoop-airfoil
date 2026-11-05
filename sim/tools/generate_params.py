"""Generate the C++ parameter header of the simulator from params/whoop_nominal.json.

The JSON is shared with the identification (sysid/), so the simulator reads its numbers from there instead of keeping
its own copy. CMake runs this script at build time (sim/CMakeLists.txt); the header goes to the build directory.

    python generate_params.py <whoop_nominal.json> <output.h>

Besides copying the values, the script derives what L2F needs and the JSON does not give directly:
  - J_inv, the inverse of the inertia matrix (L2F stores both);
  - the hover command u_h with c0 + c1 u_h + c2 u_h^2 = m g / 4 (L2F's hovering_throttle_relative, used by the reward);
  - the mass and inertia with the plate on [assumption]: plate.added_mass is taken as the mass of the EPP sheet
    (3 mm x 0.0078 m^2 x about 30 kg/m^3 = 0.7 g), the sheet as a thin rectangle with its edge c_1 (chord_x) along
    plate.chord_direction e_1 and its edge c_2 (height_z) along e_2 = e_1 x n, with its centroid at plate.centroid
    relative to the centre of mass of the whole vehicle, and the whoop's own J taken about its own centre of mass
    (parallel-axis terms for both parts). The nominal plate is vertical (n = y, e_1 = x, e_2 = z);
  - the total rotor disc area A = 4 pi R^2 for the crude downwash term (only used when plate.downwash_factor > 0).
It also checks the plate frame (unit n and e_1, n . e_1 = 0), the downwash option (k_w >= 0; k_w > 0 needs
whoop.rotor_radius and the plate centroid below the rotor discs), the body drag (non-negative) and the randomization
ranges (r in [0, 1), so that every factor stays positive).
plate.chord_direction and whoop.rotor_radius are optional: without them e_1 = x and no downwash is possible.
"""
import json
import math
import sys


def fail(message):
    sys.exit("generate_params.py: " + message)


def inverse3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-30:
        fail("singular inertia matrix")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def hover_command(coefficients, mass, gravity):
    # per-rotor thrust c0 + c1 u + c2 u^2 equal to a quarter of the weight, positive root
    c0, c1, c2 = coefficients
    target = mass * abs(gravity) / 4
    if c2 == 0:
        return (target - c0) / c1
    disc = c1 * c1 - 4 * c2 * (c0 - target)
    if disc < 0:
        fail("the rotors cannot lift the vehicle")
    u = (-c1 + math.sqrt(disc)) / (2 * c2)
    if not 0 < u < 1:
        fail("hover command %g outside (0, 1)" % u)
    return u


def parallel_axis(mass, r):
    # inertia of a point mass at r about the origin: m (|r|^2 I - r r^T)
    r2 = sum(x * x for x in r)
    return [[mass * ((r2 if i == j else 0) - r[i] * r[j]) for j in range(3)] for i in range(3)]


def sheet_inertia(mass, c1, c2, e1, e2):
    # thin rectangle about its own centroid: m / 12 (c_1^2 (I - e_1 e_1^T) + c_2^2 (I - e_2 e_2^T)); for e_1 = x and
    # e_2 = z this is m / 12 diag(c_2^2, c_1^2 + c_2^2, c_1^2), with exact zeros off the diagonal
    return [[mass * (c1 ** 2 * ((1 if i == j else 0) - e1[i] * e1[j]) + c2 ** 2 * ((1 if i == j else 0) - e2[i] * e2[j])) / 12
             for j in range(3)] for i in range(3)]


def cross3(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def add3(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def cpp(value):
    if isinstance(value, list):
        return "{" + ", ".join(cpp(v) for v in value) + "}"
    return repr(float(value))


def main():
    if len(sys.argv) != 3:
        fail("usage: generate_params.py <whoop_nominal.json> <output.h>")
    with open(sys.argv[1]) as f:
        p = json.load(f)
    w, plate = p["whoop"], p["plate"]

    n = plate["normal"]
    e1 = plate.get("chord_direction", [1, 0, 0])
    if abs(math.sqrt(sum(x * x for x in n)) - 1) > 1e-9:
        fail("plate normal is not a unit vector")
    if abs(math.sqrt(sum(x * x for x in e1)) - 1) > 1e-9:
        fail("plate chord_direction is not a unit vector")
    if abs(sum(a * b for a, b in zip(n, e1))) > 1e-9:
        # docs/plate_model.md: e_1 lies in the plate, the in-plane wind direction is measured along it
        fail("plate chord_direction must lie in the plate (perpendicular to the normal)")
    if any(len(r) != 3 for r in w["rotor_positions"]) or len(w["rotor_positions"]) != 4:
        fail("expected 4 rotors")
    k_w = plate["downwash_factor"]
    rotor_radius = w.get("rotor_radius", 0)
    if k_w < 0:
        fail("downwash_factor must be >= 0")
    if k_w > 0:
        if not rotor_radius > 0:
            fail("downwash_factor > 0 needs whoop.rotor_radius")
        if not plate["centroid"][2] < min(r[2] for r in w["rotor_positions"]):
            # the term puts the plate in the rotors' wake: it must hang below the discs
            fail("downwash_factor > 0 needs the plate centroid below the rotor discs")
    disc_area = 4 * math.pi * rotor_radius ** 2
    body_drag = w["body_drag"]
    if len(body_drag) != 3 or any(d < 0 for d in body_drag):
        fail("body_drag must be 3 non-negative values (kg/s, body x, y, z)")
    randomization = p["randomization"]
    for key in ("plate", "body_drag"):
        # a factor U(1 - r, 1 + r) must stay positive, or a coefficient would change sign
        if not 0 <= randomization[key] < 1:
            fail("randomization.%s must be in [0, 1)" % key)

    gravity = w["gravity"][2]
    J = w["J"]
    m_whoop = w["mass"]
    m_plate = plate["added_mass"]
    m_total = m_whoop + m_plate
    c_x, c_z = plate["chord_x"], plate["height_z"]
    r_c = plate["centroid"]
    # thin sheet about its own centroid (no thickness along its normal), then moved to the vehicle centre of mass;
    # the whoop's own centre of mass sits at -m_plate / m_whoop * r_c from the combined one
    e2 = cross3(e1, n)
    J_sheet = sheet_inertia(m_plate, c_x, c_z, e1, e2)
    r_whoop = [-m_plate / m_whoop * x for x in r_c]
    J_total = add3(add3(J, parallel_axis(m_whoop, r_whoop)), add3(J_sheet, parallel_axis(m_plate, r_c)))

    coefficients = w["rotor_thrust_coefficients"]
    out = []
    out.append("// Generated by sim/tools/generate_params.py from params/whoop_nominal.json: edit the JSON, not this file.")
    out.append("#pragma once")
    out.append("namespace whoop::params{")
    out.append("    // whoop (L2F conventions: body x forward, y left, z up; normalized motor command u in [0, 1])")
    out.append("    constexpr double mass = %s;" % cpp(m_whoop))
    out.append("    constexpr double gravity[3] = %s;" % cpp(w["gravity"]))
    out.append("    constexpr double rotor_positions[4][3] = %s;" % cpp(w["rotor_positions"]))
    out.append("    constexpr double rotor_thrust_directions[4][3] = %s;" % cpp(w["rotor_thrust_directions"]))
    out.append("    constexpr double rotor_torque_directions[4][3] = %s;" % cpp(w["rotor_torque_directions"]))
    out.append("    constexpr double rotor_thrust_coefficients[3] = %s;  // per rotor, N: c0 + c1 u + c2 u^2" % cpp(coefficients))
    out.append("    constexpr double rotor_torque_constant = %s;" % cpp(w["rotor_torque_constant"]))
    out.append("    constexpr double rotor_time_constant_rising = %s;" % cpp(w["rotor_time_constant_rising"]))
    out.append("    constexpr double rotor_time_constant_falling = %s;" % cpp(w["rotor_time_constant_falling"]))
    out.append("    constexpr double J[3][3] = %s;" % cpp(J))
    out.append("    constexpr double J_inv[3][3] = %s;" % cpp(inverse3(J)))
    out.append("    constexpr double hovering_throttle = %s;" % cpp(hover_command(coefficients, m_whoop, gravity)))
    out.append("    constexpr double body_drag[3] = %s;  // kg/s: F = -diag(d) v_rel, body frame" % cpp(body_drag))
    out.append("    namespace randomization{")
    out.append("        constexpr double plate = %s;  // factors U(1 - r, 1 + r)" % cpp(randomization["plate"]))
    out.append("        constexpr double body_drag = %s;" % cpp(randomization["body_drag"]))
    out.append("    }")
    out.append("    namespace plate{")
    out.append("        constexpr double area = %s;" % cpp(plate["area"]))
    out.append("        constexpr double chord_x = %s;" % cpp(c_x))
    out.append("        constexpr double height_z = %s;" % cpp(c_z))
    out.append("        constexpr double normal[3] = %s;" % cpp(n))
    out.append("        constexpr double chord_direction[3] = %s;  // e_1, along the edge chord_x" % cpp(e1))
    out.append("        constexpr double centroid[3] = %s;" % cpp(r_c))
    out.append("        constexpr double CN90 = %s;" % cpp(plate["CN90"]))
    out.append("        constexpr double Kp_x = %s;" % cpp(plate["Kp_x"]))
    out.append("        constexpr double Kp_z = %s;" % cpp(plate["Kp_z"]))
    out.append("        constexpr double CT = %s;" % cpp(plate["CT"]))
    out.append("        constexpr double e0 = %s;" % cpp(plate["e0"]))
    out.append("        constexpr double downwash_factor = %s;" % cpp(k_w))
    out.append("        constexpr double rotor_radius = %s;  // whoop.rotor_radius, 0 if not given" % cpp(rotor_radius))
    out.append("        constexpr double rotor_disc_area = %s;  // 4 pi R^2" % cpp(disc_area))
    out.append("        constexpr double air_density = %s;" % cpp(plate["air_density"]))
    out.append("        constexpr double added_mass = %s;" % cpp(m_plate))
    out.append("        // the whole vehicle with the plate on [assumption, see the script]")
    out.append("        constexpr double mass_total = %s;" % cpp(m_total))
    out.append("        constexpr double J_total[3][3] = %s;" % cpp(J_total))
    out.append("        constexpr double J_total_inv[3][3] = %s;" % cpp(inverse3(J_total)))
    out.append("        constexpr double hovering_throttle_total = %s;" % cpp(hover_command(coefficients, m_total, gravity)))
    out.append("    }")
    out.append("}")
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
