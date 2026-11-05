// Whoop and plate parameters read from a JSON file at run time (train.cpp --params, --plate-z).
//
// The default build takes its numbers from whoop/params.h, which tools/generate_params.py writes from
// params/whoop_nominal.json at build time. To fly or train another vehicle (for example the "true" drone of a
// sim-to-sim test, or the parameters that sysid/ identified) without a rebuild, this file reads a JSON in the same
// format and fills the same fields. The derived values (J^-1, hover command, mass and inertia with the plate) are
// computed exactly as generate_params.py does, operation by operation, so that the nominal JSON loaded here gives the
// compiled parameters bit for bit (checked in tests/test_whoop.cpp).
//
// Only the standard library is used: a small JSON reader (objects, arrays, numbers, strings, true/false/null) is
// enough for the parameter files, and no JSON library is vendored.
#pragma once

#include <whoop/environment.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace whoop::runtime{
    // ----- a minimal JSON reader -----------------------------------------------------------------------------------------
    struct Json{
        enum class Type{NUL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT};
        Type type = Type::NUL;
        bool boolean = false;
        double number = 0;
        std::string string;
        std::vector<Json> array;
        std::map<std::string, Json> object;

        const Json& operator[](const std::string& key) const{
            if(type != Type::OBJECT) throw std::runtime_error("JSON: '" + key + "' looked up in a value that is not an object");
            auto it = object.find(key);
            if(it == object.end()) throw std::runtime_error("JSON: missing key '" + key + "'");
            return it->second;
        }
        const Json& operator[](size_t i) const{
            if(type != Type::ARRAY || i >= array.size()) throw std::runtime_error("JSON: array index out of range");
            return array[i];
        }
        double num() const{
            if(type != Type::NUMBER) throw std::runtime_error("JSON: expected a number");
            return number;
        }
        size_t size() const{ return type == Type::ARRAY ? array.size() : 0; }
    };

    class JsonParser{
    public:
        explicit JsonParser(const std::string& text): s(text){}
        Json parse(){
            Json v = value();
            space();
            if(i != s.size()) fail("trailing characters");
            return v;
        }
    private:
        const std::string& s;
        size_t i = 0;
        [[noreturn]] void fail(const std::string& what){
            throw std::runtime_error("JSON: " + what + " at character " + std::to_string(i));
        }
        void space(){ while(i < s.size() && std::isspace((unsigned char)s[i])) i++; }
        bool literal(const char* word){
            size_t n = std::char_traits<char>::length(word);
            if(s.compare(i, n, word) == 0){ i += n; return true; }
            return false;
        }
        std::string string_value(){
            // the parameter files only use plain ASCII; escapes are kept as the escaped character
            if(s[i] != '"') fail("expected a string");
            i++;
            std::string out;
            while(i < s.size() && s[i] != '"'){
                if(s[i] == '\\' && i + 1 < s.size()) i++;
                out += s[i++];
            }
            if(i >= s.size()) fail("unterminated string");
            i++;
            return out;
        }
        Json value(){
            space();
            if(i >= s.size()) fail("unexpected end");
            Json v;
            char c = s[i];
            if(c == '{'){
                v.type = Json::Type::OBJECT;
                i++;
                space();
                if(s[i] == '}'){ i++; return v; }
                while(true){
                    space();
                    std::string key = string_value();
                    space();
                    if(s[i] != ':') fail("expected ':'");
                    i++;
                    v.object[key] = value();
                    space();
                    if(s[i] == ','){ i++; continue; }
                    if(s[i] == '}'){ i++; return v; }
                    fail("expected ',' or '}'");
                }
            }
            if(c == '['){
                v.type = Json::Type::ARRAY;
                i++;
                space();
                if(s[i] == ']'){ i++; return v; }
                while(true){
                    v.array.push_back(value());
                    space();
                    if(s[i] == ','){ i++; continue; }
                    if(s[i] == ']'){ i++; return v; }
                    fail("expected ',' or ']'");
                }
            }
            if(c == '"'){ v.type = Json::Type::STRING; v.string = string_value(); return v; }
            if(literal("true")){ v.type = Json::Type::BOOLEAN; v.boolean = true; return v; }
            if(literal("false")){ v.type = Json::Type::BOOLEAN; return v; }
            if(literal("null")) return v;
            // strtod reads the decimal text to the nearest double, as Python's float() does: the same JSON gives the
            // same doubles here as in generate_params.py
            const char* start = s.c_str() + i;
            char* end = nullptr;
            v.number = std::strtod(start, &end);
            if(end == start) fail("unexpected character");
            v.type = Json::Type::NUMBER;
            i += end - start;
            return v;
        }
    };

    inline Json read_json_file(const std::string& path){
        std::ifstream f(path);
        if(!f) throw std::runtime_error("cannot read " + path);
        std::stringstream text;
        text << f.rdbuf();
        std::string content = text.str();
        return JsonParser(content).parse();
    }

    // ----- the parameters of the JSON, plain doubles --------------------------------------------------------------------
    struct WhoopParams{
        double mass, gravity[3], rotor_positions[4][3], rotor_thrust_directions[4][3], rotor_torque_directions[4][3];
        double rotor_thrust_coefficients[3], rotor_torque_constant, rotor_time_constant_rising, rotor_time_constant_falling;
        double J[3][3], body_drag[3];
        double randomization_plate, randomization_body_drag;
        struct{
            double area, chord_x, height_z, normal[3], centroid[3], CN90, Kp_x, Kp_z, CT, e0, downwash_factor, air_density, added_mass;
        } plate;
    };

    // the compiled defaults (whoop/params.h, from params/whoop_nominal.json at build time)
    inline WhoopParams compiled_params(){
        WhoopParams p{};
        p.mass = params::mass;
        for(int j = 0; j < 3; j++){
            p.gravity[j] = params::gravity[j];
            p.rotor_thrust_coefficients[j] = params::rotor_thrust_coefficients[j];
            p.body_drag[j] = params::body_drag[j];
            p.plate.normal[j] = params::plate::normal[j];
            p.plate.centroid[j] = params::plate::centroid[j];
            for(int k = 0; k < 3; k++) p.J[j][k] = params::J[j][k];
        }
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 3; j++){
                p.rotor_positions[i][j] = params::rotor_positions[i][j];
                p.rotor_thrust_directions[i][j] = params::rotor_thrust_directions[i][j];
                p.rotor_torque_directions[i][j] = params::rotor_torque_directions[i][j];
            }
        }
        p.rotor_torque_constant = params::rotor_torque_constant;
        p.rotor_time_constant_rising = params::rotor_time_constant_rising;
        p.rotor_time_constant_falling = params::rotor_time_constant_falling;
        p.randomization_plate = params::randomization::plate;
        p.randomization_body_drag = params::randomization::body_drag;
        p.plate.area = params::plate::area;
        p.plate.chord_x = params::plate::chord_x;
        p.plate.height_z = params::plate::height_z;
        p.plate.CN90 = params::plate::CN90;
        p.plate.Kp_x = params::plate::Kp_x;
        p.plate.Kp_z = params::plate::Kp_z;
        p.plate.CT = params::plate::CT;
        p.plate.e0 = params::plate::e0;
        p.plate.downwash_factor = params::plate::downwash_factor;
        p.plate.air_density = params::plate::air_density;
        p.plate.added_mass = params::plate::added_mass;
        return p;
    }

    inline void check(bool ok, const std::string& what){
        if(!ok) throw std::runtime_error("parameter file: " + what);
    }

    // the checks of generate_params.py, so that a file refused at build time is refused here too
    inline void validate(const WhoopParams& p){
        const auto& n = p.plate.normal;
        check(std::fabs(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) - 1) <= 1e-9, "plate normal is not a unit vector");
        check(std::fabs(n[0]) <= 1e-9, "the plate must contain the body x axis (normal with zero x component)");
        check(p.plate.downwash_factor == 0, "downwash_factor != 0 is not implemented (docs/plate_model.md: k_w = 0 for now)");
        for(double d: p.body_drag) check(d >= 0, "body_drag must be non-negative");
        check(p.randomization_plate >= 0 && p.randomization_plate < 1, "randomization.plate must be in [0, 1)");
        check(p.randomization_body_drag >= 0 && p.randomization_body_drag < 1, "randomization.body_drag must be in [0, 1)");
        check(p.mass > 0 && p.plate.added_mass >= 0, "masses must be positive");
    }

    inline WhoopParams params_from_json(const Json& root){
        const Json& w = root["whoop"];
        const Json& pl = root["plate"];
        const Json& r = root["randomization"];
        WhoopParams p{};
        auto vec3 = [](const Json& a, double out[3]){
            check(a.size() == 3, "expected 3 values");
            for(int j = 0; j < 3; j++) out[j] = a[j].num();
        };
        p.mass = w["mass"].num();
        vec3(w["gravity"], p.gravity);
        check(w["rotor_positions"].size() == 4 && w["rotor_thrust_directions"].size() == 4 && w["rotor_torque_directions"].size() == 4, "expected 4 rotors");
        for(int i = 0; i < 4; i++){
            vec3(w["rotor_positions"][i], p.rotor_positions[i]);
            vec3(w["rotor_thrust_directions"][i], p.rotor_thrust_directions[i]);
            vec3(w["rotor_torque_directions"][i], p.rotor_torque_directions[i]);
        }
        vec3(w["rotor_thrust_coefficients"], p.rotor_thrust_coefficients);
        p.rotor_torque_constant = w["rotor_torque_constant"].num();
        p.rotor_time_constant_rising = w["rotor_time_constant_rising"].num();
        p.rotor_time_constant_falling = w["rotor_time_constant_falling"].num();
        check(w["J"].size() == 3, "J must be 3 x 3");
        for(int j = 0; j < 3; j++) vec3(w["J"][j], p.J[j]);
        vec3(w["body_drag"], p.body_drag);
        p.randomization_plate = r["plate"].num();
        p.randomization_body_drag = r["body_drag"].num();
        p.plate.added_mass = pl["added_mass"].num();
        p.plate.chord_x = pl["chord_x"].num();
        p.plate.height_z = pl["height_z"].num();
        p.plate.area = pl["area"].num();
        vec3(pl["normal"], p.plate.normal);
        vec3(pl["centroid"], p.plate.centroid);
        p.plate.CN90 = pl["CN90"].num();
        p.plate.Kp_x = pl["Kp_x"].num();
        p.plate.Kp_z = pl["Kp_z"].num();
        p.plate.CT = pl["CT"].num();
        p.plate.e0 = pl["e0"].num();
        p.plate.downwash_factor = pl["downwash_factor"].num();
        p.plate.air_density = pl["air_density"].num();
        validate(p);
        return p;
    }

    inline WhoopParams params_from_json_file(const std::string& path){ return params_from_json(read_json_file(path)); }

    // ----- derived values, transcribed from generate_params.py in the same order of operations ------------------------
    struct Mat3{ double m[3][3]; };

    inline Mat3 inverse3(const double M[3][3]){
        double a = M[0][0], b = M[0][1], c = M[0][2], d = M[1][0], e = M[1][1], f = M[1][2], g = M[2][0], h = M[2][1], i = M[2][2];
        double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        check(std::fabs(det) >= 1e-30, "singular inertia matrix");
        return Mat3{{{(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det},
                     {(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det},
                     {(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det}}};
    }

    inline double hover_command(const double coefficients[3], double mass, double gravity){
        // per-rotor thrust c0 + c1 u + c2 u^2 equal to a quarter of the weight, positive root
        double c0 = coefficients[0], c1 = coefficients[1], c2 = coefficients[2];
        double target = mass * std::fabs(gravity) / 4;
        if(c2 == 0) return (target - c0) / c1;
        double disc = c1 * c1 - 4 * c2 * (c0 - target);
        check(disc >= 0, "the rotors cannot lift the vehicle");
        double u = (-c1 + std::sqrt(disc)) / (2 * c2);
        check(u > 0 && u < 1, "hover command outside (0, 1)");
        return u;
    }

    inline Mat3 parallel_axis(double mass, const double r[3]){
        // inertia of a point mass at r about the origin: m (|r|^2 I - r r^T)
        double r2 = ((0 + r[0] * r[0]) + r[1] * r[1]) + r[2] * r[2];  // Python's sum() adds from 0, left to right
        Mat3 out;
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) out.m[i][j] = mass * ((i == j ? r2 : 0) - r[i] * r[j]);
        return out;
    }

    inline Mat3 add3(const Mat3& a, const Mat3& b){
        Mat3 out;
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) out.m[i][j] = a.m[i][j] + b.m[i][j];
        return out;
    }

    struct Derived{
        Mat3 J_inv, J_total, J_total_inv;
        double hovering_throttle, mass_total, hovering_throttle_total;
    };

    inline Derived derive(const WhoopParams& p){
        // as generate_params.py: the sheet is a thin c_x x c_z rectangle in the body x-z plane with its centroid at
        // plate.centroid relative to the vehicle's centre of mass; the whoop's own J is about its own centre of mass,
        // which sits at -m_plate / m_whoop r_c [assumption, see generate_params.py]
        Derived d;
        Mat3 J;
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) J.m[i][j] = p.J[i][j];
        double m_whoop = p.mass, m_plate = p.plate.added_mass;
        d.mass_total = m_whoop + m_plate;
        double c_x = p.plate.chord_x, c_z = p.plate.height_z;
        const double* r_c = p.plate.centroid;
        Mat3 J_sheet{{{m_plate * (c_z * c_z) / 12, 0, 0}, {0, m_plate * (c_x * c_x + c_z * c_z) / 12, 0}, {0, 0, m_plate * (c_x * c_x) / 12}}};
        double r_whoop[3];
        for(int j = 0; j < 3; j++) r_whoop[j] = -m_plate / m_whoop * r_c[j];
        d.J_total = add3(add3(J, parallel_axis(m_whoop, r_whoop)), add3(J_sheet, parallel_axis(m_plate, r_c)));
        d.J_inv = inverse3(J.m);
        d.J_total_inv = inverse3(d.J_total.m);
        d.hovering_throttle = hover_command(p.rotor_thrust_coefficients, m_whoop, p.gravity[2]);
        d.hovering_throttle_total = hover_command(p.rotor_thrust_coefficients, d.mass_total, p.gravity[2]);
        return d;
    }

    // ----- into the environment's parameters -----------------------------------------------------------------------------
    // Overwrites what whoop_dynamics, whoop_plate and the factory take from params.h (environment.h): the dynamics,
    // the random force and torque (they scale with the mass), the plate, the body drag and the randomization ranges.
    // Everything else (reward, task, termination, initial states, noise, options) is left as it is.
    template <typename T, typename TI, bool PLATE, typename PARAMETERS>
    void apply(const WhoopParams& w, PARAMETERS& p){
        validate(w);
        Derived d = derive(w);
        auto& dyn = p.dynamics;
        for(TI i = 0; i < 4; i++){
            for(TI j = 0; j < 3; j++){
                dyn.rotor_positions[i][j] = (T)w.rotor_positions[i][j];
                dyn.rotor_thrust_directions[i][j] = (T)w.rotor_thrust_directions[i][j];
                dyn.rotor_torque_directions[i][j] = (T)w.rotor_torque_directions[i][j];
                dyn.rotor_thrust_coefficients[i][j] = (T)w.rotor_thrust_coefficients[j];
            }
            dyn.rotor_torque_constants[i] = (T)w.rotor_torque_constant;
            dyn.rotor_time_constants_rising[i] = (T)w.rotor_time_constant_rising;
            dyn.rotor_time_constants_falling[i] = (T)w.rotor_time_constant_falling;
        }
        dyn.mass = (T)(PLATE ? d.mass_total : w.mass);
        for(TI j = 0; j < 3; j++){
            dyn.gravity[j] = (T)w.gravity[j];
            for(TI k = 0; k < 3; k++){
                dyn.J[j][k] = (T)(PLATE ? d.J_total.m[j][k] : w.J[j][k]);
                dyn.J_inv[j][k] = (T)(PLATE ? d.J_total_inv.m[j][k] : d.J_inv.m[j][k]);
            }
        }
        dyn.hovering_throttle_relative = (T)(PLATE ? d.hovering_throttle_total : d.hovering_throttle);
        // same expressions as make_l2f_parameters (environment.h)
        T weight = dyn.mass * (T)9.81;
        p.disturbances.random_force = {0, weight / 20};
        p.disturbances.random_torque = {0, weight / 10000};
        auto& plate = p.plate;
        plate.area = PLATE ? (T)w.plate.area : (T)0;
        plate.chord_x = (T)w.plate.chord_x;
        plate.height_z = (T)w.plate.height_z;
        for(int i = 0; i < 3; i++){
            plate.normal[i] = (T)w.plate.normal[i];
            plate.centroid[i] = (T)w.plate.centroid[i];
        }
        plate.CN90 = (T)w.plate.CN90;
        plate.Kp_x = (T)w.plate.Kp_x;
        plate.Kp_z = (T)w.plate.Kp_z;
        plate.CT = (T)w.plate.CT;
        plate.e0 = (T)w.plate.e0;
        plate.air_density = (T)w.plate.air_density;
        for(int i = 0; i < 3; i++) p.body_drag[i] = (T)w.body_drag[i];
        p.randomization.plate = (T)w.randomization_plate;
        p.randomization.body_drag = (T)w.randomization_body_drag;
    }
}
