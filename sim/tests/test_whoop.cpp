// Unit tests of the whoop environment: the plate model against docs/plate_model.md, the body drag, their place in the
// dynamics, the randomization of the aerodynamics, the motor cut, and that the environment is L2F's when the plate and
// the drag are off.
//
// whoop_test   (exit code 0 if every check passes; prints each check with its numbers)
#include <rl_tools/operations/cpu_mux.h>

#include <whoop/operations.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;

const double PI = 3.14159265358979323846;
int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail = ""){
    std::cout << (ok ? "PASS  " : "FAIL  ") << name << (detail.empty() ? "" : "  (" + detail + ")") << std::endl;
    failures += ok ? 0 : 1;
}

std::string numbers(std::initializer_list<double> values){
    std::ostringstream s;
    s << std::setprecision(6);
    bool first = true;
    for(double v: values){ s << (first ? "" : ", ") << v; first = false; }
    return s.str();
}

// ----- an independent transcription of docs/plate_model.md (plain doubles, explicit rotation matrix) ----------------
struct Expected{ double F[3], tau[3], r_cp[3]; };

Expected expected_plate(const whoop::Plate<double>& p, const double q[4], const double v_world[3], const double omega[3]){
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    double rel[3], v[3];
    for(int i = 0; i < 3; i++) rel[i] = v_world[i] - p.wind[i];
    for(int i = 0; i < 3; i++) v[i] = R[0][i] * rel[0] + R[1][i] * rel[1] + R[2][i] * rel[2];  // R^T
    const double* r = p.centroid;
    v[0] += omega[1] * r[2] - omega[2] * r[1];
    v[1] += omega[2] * r[0] - omega[0] * r[2];
    v[2] += omega[0] * r[1] - omega[1] * r[0];
    const double eps = 1e-6;
    double vn = p.normal[0] * v[0] + p.normal[1] * v[1] + p.normal[2] * v[2];
    double vt[3] = {v[0] - vn * p.normal[0], v[1] - vn * p.normal[1], v[2] - vn * p.normal[2]};
    double V2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + eps;
    double Vt = std::sqrt(vt[0] * vt[0] + vt[1] * vt[1] + vt[2] * vt[2] + eps);
    double alpha = std::atan2(std::fabs(vn), Vt);
    double c2 = vt[0] * vt[0] / (Vt * Vt);
    double Kp = p.Kp_x * c2 + p.Kp_z * (1 - c2);
    double CN = p.CN90 * std::tanh(Kp * alpha / p.CN90);
    double sgn = vn > 0 ? 1 : (vn < 0 ? -1 : 0);
    Expected e;
    for(int i = 0; i < 3; i++){
        e.F[i] = -0.5 * p.air_density * p.area * CN * V2 * sgn * p.normal[i] - 0.5 * p.air_density * p.area * p.CT * std::sqrt(V2) * vt[i];
    }
    double chord = p.chord_x * c2 + p.height_z * (1 - c2);
    for(int i = 0; i < 3; i++) e.r_cp[i] = r[i] + p.e0 * std::cos(alpha) * chord * vt[i] / Vt;
    e.tau[0] = e.r_cp[1] * e.F[2] - e.r_cp[2] * e.F[1];
    e.tau[1] = e.r_cp[2] * e.F[0] - e.r_cp[0] * e.F[2];
    e.tau[2] = e.r_cp[0] * e.F[1] - e.r_cp[1] * e.F[0];
    return e;
}

double max_abs_diff(const double* a, const double* b, int n){
    double m = 0;
    for(int i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
double norm3(const double* a){ return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

using F_HOVER_PLATE = whoop::EnvironmentFactory<double, TI, true, whoop::Task::HOVER>;

void test_pure_wind(DEVICE& device){
    whoop::Plate<double> plate = F_HOVER_PLATE::nominal_parameters.plate;
    const double identity[4] = {1, 0, 0, 0}, zero[3] = {0, 0, 0};

    // 1. wind of 3 m/s toward -y on a whoop at rest and level: the plate sees the air come at 90 deg
    plate.wind[0] = 0; plate.wind[1] = -3; plate.wind[2] = 0;
    whoop::PlateAerodynamics<double> a;
    whoop::plate_aerodynamics(device, plate, identity, zero, zero, a);
    double alpha = std::atan2(3.0, std::sqrt(1e-6));
    double CN = plate.CN90 * std::tanh(plate.Kp_z * alpha / plate.CN90);  // no in-plane wind: c^2 = 0, K_p = K_pz
    double Fy = -0.5 * plate.air_density * plate.area * CN * (9 + 1e-6);
    check(std::fabs(a.force[1] - Fy) < 1e-12 && a.force[0] == 0 && a.force[2] == 0,
          "normal wind: F = -1/2 rho S C_N V^2 n", "F_y " + numbers({a.force[1], Fy}) + " N, " +
          numbers({a.force[1] / (F_HOVER_PLATE::nominal_parameters.dynamics.mass * 9.81)}) + " of the weight");
    check(std::fabs(a.alpha - alpha) < 1e-15, "normal wind: alpha = 90 deg (up to eps)", numbers({a.alpha * 180 / PI}) + " deg");
    double expected_tau_x = -plate.centroid[2] * Fy;  // r_c = (0, 0, z_c): tau_x = -z_c F_y
    check(std::fabs(a.torque[0] - expected_tau_x) < 1e-15 && a.torque[1] == 0 && a.torque[2] == 0,
          "normal wind: tau = r_c x F", "tau_x " + numbers({a.torque[0], expected_tau_x}) + " N m");

    // 2. oblique wind on a whoop that is moving, rotated and rolling: every term of the model at once
    plate.wind[0] = -2.0; plate.wind[1] = 1.5; plate.wind[2] = 0.7;
    double angle = 0.7, axis[3] = {0.3, -0.5, 0.81};
    double axis_norm = norm3(axis);
    double q[4] = {std::cos(angle / 2), 0, 0, 0};
    for(int i = 0; i < 3; i++) q[i + 1] = std::sin(angle / 2) * axis[i] / axis_norm;
    double v_world[3] = {1.2, -0.4, 0.3}, omega[3] = {0.8, -1.1, 0.5};
    whoop::plate_aerodynamics(device, plate, q, v_world, omega, a);
    Expected e = expected_plate(plate, q, v_world, omega);
    double dF = max_abs_diff(a.force, e.F, 3), dtau = max_abs_diff(a.torque, e.tau, 3);
    check(dF < 1e-12 && dtau < 1e-13 && norm3(e.F) > 1e-3, "oblique wind: force and torque as docs/plate_model.md",
          "|F| " + numbers({norm3(a.force)}) + " N, max diff F " + numbers({dF}) + ", tau " + numbers({dtau}) + ", alpha " + numbers({a.alpha * 180 / PI}) + " deg");

    // 3. the plate in the equations of motion: with and without it, the derivatives differ by R F / m and J^-1 tau
    using ENV = F_HOVER_PLATE::ENVIRONMENT;
    using STATE_BASE = rlt::rl::environments::l2f::StateBase<rlt::rl::environments::l2f::StateSpecification<double, TI>>;
    // the body drag is on in both and sees the same wind (plate.wind), so the difference is the plate alone
    ENV::Parameters with = F_HOVER_PLATE::nominal_parameters;
    with.plate = plate;
    ENV::Parameters without = with;
    without.plate.area = 0;
    STATE_BASE s{}, d_with{}, d_without{};
    for(int i = 0; i < 4; i++) s.orientation[i] = q[i];
    for(int i = 0; i < 3; i++){ s.position[i] = 0.1 * i; s.linear_velocity[i] = v_world[i]; s.angular_velocity[i] = omega[i]; }
    double rpm[4] = {0.4, 0.5, 0.45, 0.55};
    whoop::multirotor_dynamics(device, with, s, rpm, d_with);
    whoop::multirotor_dynamics(device, without, s, rpm, d_without);
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    double da_expected[3], dw_expected[3], da[3], dw[3];
    for(int i = 0; i < 3; i++){
        da_expected[i] = (R[i][0] * e.F[0] + R[i][1] * e.F[1] + R[i][2] * e.F[2]) / with.dynamics.mass;
        dw_expected[i] = with.dynamics.J_inv[i][0] * e.tau[0] + with.dynamics.J_inv[i][1] * e.tau[1] + with.dynamics.J_inv[i][2] * e.tau[2];
        da[i] = d_with.linear_velocity[i] - d_without.linear_velocity[i];
        dw[i] = d_with.angular_velocity[i] - d_without.angular_velocity[i];
    }
    double dpos = max_abs_diff(d_with.position, d_without.position, 3), dq = max_abs_diff(d_with.orientation, d_without.orientation, 4);
    check(max_abs_diff(da, da_expected, 3) < 1e-10 && max_abs_diff(dw, dw_expected, 3) < 1e-8 && dpos == 0 && dq == 0,
          "dynamics: plate adds R F / m to dv/dt and J^-1 tau to domega/dt",
          "dv/dt diff " + numbers({norm3(da), max_abs_diff(da, da_expected, 3)}) + ", domega/dt diff " + numbers({norm3(dw), max_abs_diff(dw, dw_expected, 3)}));
}

void test_center_of_pressure(DEVICE& device){
    whoop::Plate<double> plate = F_HOVER_PLATE::nominal_parameters.plate;
    const double identity[4] = {1, 0, 0, 0}, zero[3] = {0, 0, 0};
    whoop::PlateAerodynamics<double> a;
    // alpha = 90 deg: no in-plane wind, the centre of pressure is the centroid exactly
    double v_side[3] = {0, 2.5, 0};
    whoop::plate_aerodynamics(device, plate, identity, v_side, zero, a);
    double offset = std::sqrt(std::pow(a.center_of_pressure[0] - plate.centroid[0], 2) + std::pow(a.center_of_pressure[1] - plate.centroid[1], 2) + std::pow(a.center_of_pressure[2] - plate.centroid[2], 2));
    check(offset == 0, "alpha = 90 deg: zero centre-of-pressure offset", "offset " + numbers({offset}) + " m, alpha " + numbers({a.alpha * 180 / PI}) + " deg");
    // alpha = 0, the whoop flying forward: centre of pressure e0 c_x ahead of the centroid (toward the leading edge)
    double v_forward[3] = {3, 0, 0};
    whoop::plate_aerodynamics(device, plate, identity, v_forward, zero, a);
    double shift = a.center_of_pressure[0] - plate.centroid[0];
    // c^2 = v_t,x^2 / V_t^2 = 9 / (9 + eps): the chord is c_x up to eps
    double c2 = 9 / (9 + 1e-6);
    double expected = plate.e0 * (plate.chord_x * c2 + plate.height_z * (1 - c2)) * 3 / std::sqrt(9 + 1e-6);
    check(a.alpha == 0 && std::fabs(shift - expected) < 1e-12 * expected && shift > 0, "alpha = 0, forward flight: CoP e0 c_x forward",
          "shift " + numbers({shift, expected}) + " m");
    // intermediate angle: the shift follows cos(alpha)
    double v_mid[3] = {2, 2, 0};
    whoop::plate_aerodynamics(device, plate, identity, v_mid, zero, a);
    shift = a.center_of_pressure[0] - plate.centroid[0];
    c2 = 4 / (4 + 1e-6);
    expected = plate.e0 * std::cos(a.alpha) * (plate.chord_x * c2 + plate.height_z * (1 - c2)) * 2 / std::sqrt(4 + 1e-6);
    check(std::fabs(shift - expected) < 1e-12 * expected && std::fabs(a.alpha - PI / 4) < 1e-6, "alpha = 45 deg: CoP shift e0 cos(alpha) c_x",
          "shift " + numbers({shift, expected}) + " m");
}

void test_roll_damping(DEVICE& device){
    whoop::Plate<double> plate = F_HOVER_PLATE::nominal_parameters.plate;
    const double identity[4] = {1, 0, 0, 0}, zero[3] = {0, 0, 0};
    // a pure roll rate moves the plate, below the centre of mass, sideways: its normal force must oppose the roll
    for(double p: {2.0, -2.0, 10.0}){
        double omega[3] = {p, 0, 0};
        whoop::PlateAerodynamics<double> a;
        whoop::plate_aerodynamics(device, plate, identity, zero, omega, a);
        double power = a.torque[0] * omega[0] + a.torque[1] * omega[1] + a.torque[2] * omega[2];
        Expected e = expected_plate(plate, identity, zero, omega);
        check(a.torque[0] * p < 0 && power < 0 && std::fabs(a.torque[1]) < 1e-18 && max_abs_diff(a.torque, e.tau, 3) < 1e-15,
              "roll rate " + numbers({p}) + " rad/s: damping torque", "tau_x " + numbers({a.torque[0]}) + " N m, tau . omega " + numbers({power}) + " W");
    }
}

void test_body_drag(DEVICE& device){
    // the body drag F = -diag(d) R^T (v - wind) of sysid/model.py: as a function, then in the equations of motion
    using ENV = F_HOVER_PLATE::ENVIRONMENT;
    using STATE_BASE = rlt::rl::environments::l2f::StateBase<rlt::rl::environments::l2f::StateSpecification<double, TI>>;
    ENV::Parameters with = F_HOVER_PLATE::nominal_parameters;
    const double* d = with.body_drag;
    // 1. level whoop at 3 m/s forward in still air: F = (-3 d_x, 0, 0), 8 % of the weight for the nominal d_x
    const double identity[4] = {1, 0, 0, 0}, forward[3] = {3, 0, 0}, still[3] = {0, 0, 0};
    double F[3];
    whoop::body_drag_force(device, d, identity, forward, still, F);
    check(d[0] == 0.008 && d[1] == 0.008 && d[2] == 0.004 && std::fabs(F[0] + 3 * d[0]) < 1e-15 && F[1] == 0 && F[2] == 0,
          "body drag at 3 m/s forward: F = -d_x v", "F_x " + numbers({F[0]}) + " N, " + numbers({-F[0] / (with.dynamics.mass * 9.81)}) +
          " of the weight, d = " + numbers({d[0], d[1], d[2]}) + " kg/s");
    // 2. rotated, moving whoop in wind, plate off: the derivatives change by R F / m exactly, and nothing else
    with.plate.area = 0;
    with.plate.wind[0] = -1.0; with.plate.wind[1] = 0.6; with.plate.wind[2] = 0.3;
    ENV::Parameters without = with;
    for(int i = 0; i < 3; i++) without.body_drag[i] = 0;
    double angle = 0.9, axis[3] = {-0.4, 0.7, 0.2};
    double q[4] = {std::cos(angle / 2), 0, 0, 0};
    for(int i = 0; i < 3; i++) q[i + 1] = std::sin(angle / 2) * axis[i] / norm3(axis);
    double v_world[3] = {2.1, -1.3, 0.8}, omega[3] = {0.5, 1.5, -0.7};
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    double rel[3], v_body[3], F_expected[3];
    for(int i = 0; i < 3; i++) rel[i] = v_world[i] - with.plate.wind[i];
    for(int i = 0; i < 3; i++) v_body[i] = R[0][i] * rel[0] + R[1][i] * rel[1] + R[2][i] * rel[2];  // R^T
    for(int i = 0; i < 3; i++) F_expected[i] = -d[i] * v_body[i];
    whoop::body_drag_force(device, d, q, v_world, with.plate.wind, F);
    double dF = max_abs_diff(F, F_expected, 3);
    STATE_BASE s{}, d_with{}, d_without{};
    for(int i = 0; i < 4; i++) s.orientation[i] = q[i];
    for(int i = 0; i < 3; i++){ s.position[i] = 0.2 * i; s.linear_velocity[i] = v_world[i]; s.angular_velocity[i] = omega[i]; }
    double rpm[4] = {0.5, 0.42, 0.47, 0.51};
    whoop::multirotor_dynamics(device, with, s, rpm, d_with);
    whoop::multirotor_dynamics(device, without, s, rpm, d_without);
    double da[3], da_expected[3];
    for(int i = 0; i < 3; i++){
        da_expected[i] = (R[i][0] * F_expected[0] + R[i][1] * F_expected[1] + R[i][2] * F_expected[2]) / with.dynamics.mass;
        da[i] = d_with.linear_velocity[i] - d_without.linear_velocity[i];
    }
    double dw = max_abs_diff(d_with.angular_velocity, d_without.angular_velocity, 3);
    double dpos = max_abs_diff(d_with.position, d_without.position, 3), dq = max_abs_diff(d_with.orientation, d_without.orientation, 4);
    check(dF < 1e-17 && max_abs_diff(da, da_expected, 3) < 1e-14 && norm3(da) > 0.1 && dw == 0 && dpos == 0 && dq == 0,
          "body drag in the dynamics: R (-diag(d) R^T (v - wind)) / m, no torque",
          "|F| " + numbers({norm3(F)}) + " N, max diff F " + numbers({dF}) + ", dv/dt diff " + numbers({norm3(da), max_abs_diff(da, da_expected, 3)}) +
          ", domega/dt diff " + numbers({dw}));
}

void test_randomization(DEVICE& device){
    // per-episode factors U(1 - r, 1 + r) on C_N90, K_px, K_pz, C_T, e0 and each drag component; off gives the nominal
    // values, and the plate geometry is never changed
    using FACTORY = whoop::EnvironmentFactory<double, TI, true, whoop::Task::VELOCITY>;
    using ENV = typename FACTORY::ENVIRONMENT;
    ENV env;
    rlt::init(device, env);
    const auto& nominal = FACTORY::nominal_parameters;
    typename ENV::Parameters parameters = nominal;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 21);
    const int N = 4000;
    const double r_plate = nominal.randomization.plate, r_drag = nominal.randomization.body_drag;
    double lo[8], hi[8], mean[8];
    for(int j = 0; j < 8; j++){ lo[j] = 1e9; hi[j] = -1e9; mean[j] = 0; }
    bool geometry_same = true;
    for(int k = 0; k < N; k++){
        whoop::sample_initial_parameters(device, env, parameters, rng);
        const auto& p = parameters.plate;
        double f[8] = {p.CN90 / nominal.plate.CN90, p.Kp_x / nominal.plate.Kp_x, p.Kp_z / nominal.plate.Kp_z, p.CT / nominal.plate.CT,
                       p.e0 / nominal.plate.e0, parameters.body_drag[0] / nominal.body_drag[0], parameters.body_drag[1] / nominal.body_drag[1],
                       parameters.body_drag[2] / nominal.body_drag[2]};
        for(int j = 0; j < 8; j++){ lo[j] = std::min(lo[j], f[j]); hi[j] = std::max(hi[j], f[j]); mean[j] += f[j] / N; }
        geometry_same &= p.area == nominal.plate.area && p.chord_x == nominal.plate.chord_x && p.centroid[2] == nominal.plate.centroid[2];
    }
    bool in_range = true, mean_one = true, spread = true;
    double worst_mean = 0;
    for(int j = 0; j < 8; j++){
        double r = j < 5 ? r_plate : r_drag;
        in_range &= lo[j] >= 1 - r && hi[j] <= 1 + r;
        spread &= lo[j] < 1 - 0.95 * r && hi[j] > 1 + 0.95 * r;
        // standard error of the mean of U(1 - r, 1 + r) over N draws: r / sqrt(3 N), about 0.004 for r = 0.5
        mean_one &= std::fabs(mean[j] - 1) < 4 * r / std::sqrt(3.0 * N);
        worst_mean = std::max(worst_mean, std::fabs(mean[j] - 1));
    }
    check(in_range && spread && mean_one && geometry_same && r_plate == 0.4 && r_drag == 0.5, "randomization: factors in U(1 - r, 1 + r), mean 1",
          numbers({double(N)}) + " episodes, r = " + numbers({r_plate, r_drag}) + ", C_N90 factor in [" + numbers({lo[0], hi[0]}) + "], d_x factor in [" +
          numbers({lo[5], hi[5]}) + "], max |mean - 1| " + numbers({worst_mean}));
    // disabled: the nominal values exactly, and the same episodes (references) as with no randomization at all
    env.parameters.randomization.enabled = false;
    RNG rng_a, rng_b;
    rlt::malloc(device, rng_a);
    rlt::malloc(device, rng_b);
    rlt::init(device, rng_a, 22);
    rlt::init(device, rng_b, 22);
    bool nominal_values = true, same_reference = true;
    for(int k = 0; k < 50; k++){
        whoop::sample_initial_parameters(device, env, parameters, rng_a);
        const auto& p = parameters.plate;
        nominal_values &= p.CN90 == nominal.plate.CN90 && p.Kp_x == nominal.plate.Kp_x && p.Kp_z == nominal.plate.Kp_z && p.CT == nominal.plate.CT &&
                          p.e0 == nominal.plate.e0 && parameters.body_drag[0] == nominal.body_drag[0] && parameters.body_drag[1] == nominal.body_drag[1] &&
                          parameters.body_drag[2] == nominal.body_drag[2];
        typename ENV::Parameters reference = nominal;
        rlt::sample_initial_parameters(device, env, static_cast<typename ENV::Parameters::NEXT_COMPONENT&>(reference), rng_b);
        whoop::sample_velocity_reference(device, reference, rng_b);
        same_reference &= std::memcmp(&reference.trajectory, &parameters.trajectory, sizeof(reference.trajectory)) == 0;
    }
    check(nominal_values && same_reference, "randomization off: nominal coefficients, no extra random numbers",
          "50 episodes, references identical to the sampler without randomization");
    rlt::free(device, rng);
    rlt::free(device, rng_a);
    rlt::free(device, rng_b);
}

void test_motor_cut(DEVICE& device){
    // a motor cut forces every command to 0 between its start and end steps: the rotors spin down with the falling
    // time constant, the action history records the forced command, and outside the cut the step is L2F's
    using FACTORY = whoop::EnvironmentFactory<double, TI, true, whoop::Task::VELOCITY>;
    using ENV = typename FACTORY::ENVIRONMENT;
    ENV env;
    rlt::init(device, env);
    typename ENV::Parameters parameters = FACTORY::nominal_parameters;
    parameters.randomization.enabled = false;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 9);
    whoop::set_velocity_reference(device, parameters, 0.0, 3.0);
    parameters.motor_cut.start_step = 10;
    parameters.motor_cut.end_step = 50;
    typename ENV::State s;
    whoop::sample_initial_state(device, env, parameters, s, rng);
    for(int i = 0; i < 4; i++) s.rpm[i] = parameters.dynamics.hovering_throttle_relative;
    rlt::Matrix<rlt::matrix::Specification<double, TI, 1, 4>> action;
    rlt::malloc(device, action);
    for(TI i = 0; i < 4; i++) rlt::set(action, 0, i, 0.5);  // the policy asks for more than hover all along
    int cut_steps = 0, history_forced = 0, same_outside = 0, outside = 0;
    double rpm_at_start = 0, rpm_at_end = 0, rpm_expected = 0;
    for(int step = 0; step < 80; step++){
        bool cut = whoop::motors_cut(parameters, s);
        if(step == 10) rpm_at_start = s.rpm[0];
        cut_steps += cut;
        typename ENV::State next, next_l2f;
        auto rng_copy = rng;
        whoop::step(device, env, parameters, s, action, next, rng);
        rlt::step(device, env, parameters, s, action, next_l2f, rng_copy);  // L2F's own step, never cut
        TI last = (next.rotor_history_step + FACTORY::ACTION_HISTORY_LENGTH - 1) % FACTORY::ACTION_HISTORY_LENGTH;
        if(cut){
            history_forced += next.action_history[last][0] == -1;
        }
        else{
            outside++;
            same_outside += std::memcmp(&next, &next_l2f, sizeof(next)) == 0;
        }
        if(step == 49){
            // first-order lag toward 0 during 40 steps of 10 ms: rpm = rpm_0 exp(-0.4 / tau_falling) (RK4: to 1e-6)
            rpm_at_end = next.rpm[0];
            rpm_expected = rpm_at_start * std::exp(-0.4 / parameters.dynamics.rotor_time_constants_falling[0]);
        }
        s = next;
    }
    check(cut_steps == 40 && history_forced == 40 && same_outside == outside && rpm_at_start > 0.5 && std::fabs(rpm_at_end - rpm_expected) < 1e-5,
          "motor cut: commands 0 from step 10 to 49, L2F's step outside",
          numbers({double(cut_steps)}) + " cut steps, rotor state at the end " + numbers({rpm_at_end, rpm_expected}) + ", " +
          numbers({double(same_outside), double(outside)}) + " steps identical to L2F outside the cut");
    rlt::free(device, action);
    rlt::free(device, rng);
}

template <typename T, bool PLATE, whoop::Task TASK>
void test_same_as_l2f(DEVICE& device, const std::string& name){
    // with S = 0 and no body drag, a step of the whoop environment equals a step of the plain L2F environment bit for
    // bit (same parameter values, no whoop overloads on the L2F side)
    using FACTORY = whoop::EnvironmentFactory<T, TI, PLATE, TASK>;
    using ENV = typename FACTORY::ENVIRONMENT;
    using ENV_L2F = typename FACTORY::ENVIRONMENT_L2F;
    static_assert(sizeof(typename ENV::State) == sizeof(typename ENV_L2F::State));
    ENV env;
    ENV_L2F env_l2f;
    rlt::init(device, env);
    rlt::init(device, env_l2f);
    typename ENV::Parameters parameters = FACTORY::nominal_parameters;
    parameters.plate.area = 0;
    for(int i = 0; i < 3; i++) parameters.body_drag[i] = 0;
    typename ENV_L2F::Parameters parameters_l2f = FACTORY::l2f_nominal_parameters;
    RNG rng, rng_a, rng_b;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 7);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4>> action;
    rlt::malloc(device, action);
    int different = 0, n = 0;
    double max_v = 0;
    for(int episode = 0; episode < 20; episode++){
        if constexpr(TASK == whoop::Task::VELOCITY){
            // the same reference on both sides
            whoop::sample_velocity_reference(device, parameters, rng);
            parameters_l2f.trajectory = parameters.trajectory;
        }
        typename ENV::State s;
        whoop::sample_initial_state(device, env, parameters, s, rng);
        typename ENV_L2F::State s_l2f;
        std::memcpy(&s_l2f, &s, sizeof(s));
        for(int step = 0; step < 100; step++){
            for(TI i = 0; i < 4; i++) rlt::set(action, 0, i, rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng));
            rng_a = rng;  // the step draws action noise (std 0): the same random numbers on both sides
            rng_b = rng;
            typename ENV::State next;
            typename ENV_L2F::State next_l2f;
            whoop::step(device, env, parameters, s, action, next, rng_a);  // the whoop step (no motor cut)
            rlt::step(device, env_l2f, parameters_l2f, s_l2f, action, next_l2f, rng_b);
            different += std::memcmp(&next, &next_l2f, sizeof(next)) != 0;
            n++;
            max_v = std::max(max_v, (double)std::sqrt(next.linear_velocity[0] * next.linear_velocity[0] + next.linear_velocity[1] * next.linear_velocity[1] + next.linear_velocity[2] * next.linear_velocity[2]));
            s = next;
            std::memcpy(&s_l2f, &next_l2f, sizeof(s));
            rng = rng_a;
        }
    }
    check(different == 0, "S = 0 and no body drag equals plain L2F, " + name, numbers({double(n)}) + " random steps, " + numbers({double(different)}) + " different, max speed " + numbers({max_v}) + " m/s");
    rlt::free(device, action);
    rlt::free(device, rng);
}

template <bool PLATE>
void test_hover(DEVICE& device, const std::string& name){
    // hover state and hover command: the whoop stays where it is (the plate sees no air, so no force either)
    using FACTORY = whoop::EnvironmentFactory<double, TI, PLATE, whoop::Task::HOVER>;
    using ENV = typename FACTORY::ENVIRONMENT;
    ENV env;
    rlt::init(device, env);
    typename ENV::Parameters parameters = FACTORY::nominal_parameters;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 3);
    typename ENV::State s;
    whoop::sample_initial_state(device, env, parameters, s, rng);
    double hover = parameters.dynamics.hovering_throttle_relative;
    for(int i = 0; i < 3; i++){ s.position[i] = 0; s.linear_velocity[i] = 0; s.angular_velocity[i] = 0; s.force[i] = 0; s.torque[i] = 0; }
    s.orientation[0] = 1; s.orientation[1] = s.orientation[2] = s.orientation[3] = 0;
    for(int i = 0; i < 4; i++) s.rpm[i] = hover;
    rlt::Matrix<rlt::matrix::Specification<double, TI, 1, 4>> action;
    rlt::malloc(device, action);
    for(TI i = 0; i < 4; i++) rlt::set(action, 0, i, whoop::action_baseline(hover));
    double reward_sum = 0, max_drift = 0;
    for(int step = 0; step < 200; step++){
        typename ENV::State next;
        rlt::step(device, env, parameters, s, action, next, rng);
        reward_sum += whoop::reward(device, env, parameters, s, action, next, rng);
        max_drift = std::max(max_drift, norm3(next.position));
        s = next;
    }
    double tilt = 2 * std::acos(std::min(1.0, std::fabs(s.orientation[0]))) * 180 / PI;
    check(max_drift < 1e-6 && tilt < 1e-6 && norm3(s.linear_velocity) < 1e-6, "hover command keeps hover for 2 s, " + name,
          "drift " + numbers({max_drift}) + " m, tilt " + numbers({tilt}) + " deg, hover command " + numbers({hover}) + ", mass " + numbers({parameters.dynamics.mass}) + " kg");
    check(std::fabs(reward_sum / 200 - parameters.mdp.reward.constant) < 1e-9, "reward at hover = L2F constant, " + name, "mean reward " + numbers({reward_sum / 200}));
    rlt::free(device, action);
    rlt::free(device, rng);
}

template <whoop::Task TASK>
void test_observation_and_reward(DEVICE& device, const std::string& name){
    // the actor sees the absolute linear velocity, and the reward recomputed from the critic's observation (used after a
    // curriculum step) equals the reward of the environment
    using T = float;
    using FACTORY = whoop::EnvironmentFactory<T, TI, true, TASK>;
    using ENV = typename FACTORY::ENVIRONMENT;
    ENV env;
    rlt::init(device, env);
    typename ENV::Parameters parameters = FACTORY::nominal_parameters;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 11);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::Observation::DIM>> observation;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ObservationPrivileged::DIM>> privileged;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4>> action;
    rlt::malloc(device, observation);
    rlt::malloc(device, privileged);
    rlt::malloc(device, action);
    double max_velocity_error = 0, max_reward_error = 0;
    int n = 0, terminated = 0;
    for(int episode = 0; episode < 20; episode++){
        whoop::sample_initial_parameters(device, env, parameters, rng);
        typename ENV::State s;
        whoop::sample_initial_state(device, env, parameters, s, rng);
        for(int step = 0; step < 100; step++){
            rlt::observe(device, env, parameters, s, typename ENV::Observation{}, observation, rng);
            rlt::observe(device, env, parameters, s, typename ENV::ObservationPrivileged{}, privileged, rng);
            for(TI i = 0; i < 3; i++){
                max_velocity_error = std::max(max_velocity_error, (double)std::fabs(rlt::get(observation, 0, FACTORY::OBSERVATION_LINEAR_VELOCITY_INDEX + i) - s.linear_velocity[i]));
            }
            for(TI i = 0; i < 4; i++) rlt::set(action, 0, i, rlt::random::uniform_real_distribution(device.random, (T)-0.3, (T)0.3, rng));
            typename ENV::State next;
            rlt::step(device, env, parameters, s, action, next, rng);
            T r = whoop::reward(device, env, parameters, s, action, next, rng);
            bool done = rlt::terminated(device, env, parameters, next, rng);
            T a[4];
            for(TI i = 0; i < 4; i++) a[i] = rlt::get(action, 0, i);
            T o[ENV::ObservationPrivileged::DIM];
            for(TI i = 0; i < ENV::ObservationPrivileged::DIM; i++) o[i] = rlt::get(privileged, 0, i);
            T recomputed = whoop::reward_from_privileged_observation(parameters.mdp.reward, o, a,
                                                                       whoop::action_baseline(parameters.dynamics.hovering_throttle_relative), done);
            max_reward_error = std::max(max_reward_error, (double)std::fabs(r - recomputed));
            n++;
            terminated += done;
            if(done) break;
            s = next;
        }
    }
    check(max_velocity_error < 0.02, "actor observes the absolute linear velocity, " + name,
          "index " + numbers({double(FACTORY::OBSERVATION_LINEAR_VELOCITY_INDEX)}) + ", max |obs - v| " + numbers({max_velocity_error}) + " m/s (noise std 0.002)");
    check(max_reward_error < 1e-4, "reward from the critic observation = environment reward, " + name,
          numbers({double(n)}) + " steps, " + numbers({double(terminated)}) + " terminations, max diff " + numbers({max_reward_error}));
    rlt::free(device, observation);
    rlt::free(device, privileged);
    rlt::free(device, action);
    rlt::free(device, rng);
}

void test_initial_states(DEVICE& device){
    // the L2F paper's initial attitudes (up to 90 deg), not the [0, 1] rad of the current RLtools sampler
    using FACTORY = whoop::EnvironmentFactory<double, TI, false, whoop::Task::HOVER>;
    using ENV = typename FACTORY::ENVIRONMENT;
    ENV env;
    rlt::init(device, env);
    typename ENV::Parameters parameters = FACTORY::nominal_parameters;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 5);
    double max_angle = 0, max_position = 0;
    int above_57 = 0, guidance = 0;
    const int N = 2000;
    for(int k = 0; k < N; k++){
        typename ENV::State s;
        whoop::sample_initial_state(device, env, parameters, s, rng);
        double angle = 2 * std::acos(std::min(1.0, s.orientation[0])) * 180 / PI;
        max_angle = std::max(max_angle, angle);
        above_57 += angle > 57.3;
        guidance += (s.position[0] == 0 && s.orientation[0] == 1);
        for(int i = 0; i < 3; i++) max_position = std::max(max_position, std::fabs(s.position[i]));
    }
    check(max_angle <= 90 && above_57 > N / 10 && max_position <= 0.2 && guidance > N / 20 && guidance < N / 6,
          "initial states of the L2F paper", "max tilt " + numbers({max_angle}) + " deg, " + numbers({100.0 * above_57 / N}) + " % above 57 deg, " +
          numbers({100.0 * guidance / N}) + " % at the origin, max |p_i| " + numbers({max_position}) + " m");
    rlt::free(device, rng);
}

int main(){
    DEVICE device;
    rlt::init(device);
    std::cout << std::setprecision(6);
    test_pure_wind(device);
    test_center_of_pressure(device);
    test_roll_damping(device);
    test_body_drag(device);
    test_randomization(device);
    test_motor_cut(device);
    test_same_as_l2f<float, false, whoop::Task::HOVER>(device, "float, hover, no-plate mass");
    test_same_as_l2f<float, true, whoop::Task::HOVER>(device, "float, hover, plate mass");
    test_same_as_l2f<double, true, whoop::Task::HOVER>(device, "double, hover, plate mass");
    test_same_as_l2f<float, true, whoop::Task::VELOCITY>(device, "float, velocity task, plate mass");
    test_hover<false>(device, "no plate");
    test_hover<true>(device, "plate");
    test_observation_and_reward<whoop::Task::HOVER>(device, "hover");
    test_observation_and_reward<whoop::Task::VELOCITY>(device, "velocity task");
    test_initial_states(device);
    std::cout << (failures == 0 ? "all checks passed" : std::to_string(failures) + " check(s) failed") << std::endl;
    return failures == 0 ? 0 : 1;
}
