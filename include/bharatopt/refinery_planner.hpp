#pragma once
// include/bharatopt/refinery_planner.hpp
//
// Translation layer: refinery JSON  ->  generic PoolingModel.
// This file does NOT touch the solver. It only builds the solver's input.
//
// ASSUMED PoolingModel API — adjust these calls in refinery_planner.cpp to
// match your actual class. The shape assumed here is the minimum a pooling
// solver needs: continuous variables with bounds, linear equality/inequality
// rows, and named bilinear terms (quality * volume) for the spatial B&B /
// McCormick layer to relax.
//
//   class PoolingModel {
//    public:
//     int add_variable(const std::string& name, double lb, double ub);
//     void add_linear_constraint(const std::string& name,
//                                 const std::vector<std::pair<int,double>>& coeffs,
//                                 char sense,           // '=' | '<' | '>'
//                                 double rhs);
//     void add_bilinear_term(const std::string& name,
//                             int quality_var, int volume_var,
//                             double lo_q, double hi_q,
//                             double lo_v, double hi_v);
//     void set_objective_coeff(int var, double coeff);   // maximize sense
//   };

#include <optional>
#include <string>
#include <vector>

#include "bharatopt/refinery_model_types.hpp"
// #include "bharatopt/pooling_model.hpp"   // your existing solver's model class

namespace bharatopt::refinery {

struct ValidationIssue {
    enum class Severity { Warning, Error };
    Severity severity;
    std::string code;       // short machine-readable tag, e.g. "UNKNOWN_STREAM_REF"
    std::string message;    // human-readable explanation
};

struct ValidationResult {
    bool valid = true;
    std::vector<ValidationIssue> errors;
    std::vector<ValidationIssue> warnings;

    void add(ValidationIssue::Severity sev, std::string code, std::string message) {
        ValidationIssue issue{sev, std::move(code), std::move(message)};
        if (sev == ValidationIssue::Severity::Error) {
            valid = false;
            errors.push_back(std::move(issue));
        } else {
            warnings.push_back(std::move(issue));
        }
    }

    // Serializes to the {"valid":..., "errors":[...], "warnings":[...]} shape
    // the CLI's `--validate` flag is expected to print.
    std::string to_json() const;
};

class RefineryPlanner {
public:
    // Loads and parses the JSON file. Throws std::runtime_error with a
    // clear message on malformed JSON or a missing required top-level key.
    explicit RefineryPlanner(const std::string& json_path);

    // Structural + domain validation. Safe to call before build(); build()
    // internally re-checks the subset of conditions it cannot proceed
    // without, but validate() is the complete, user-facing check list.
    ValidationResult validate() const;

    // Builds the generic pooling model. Call only after validate().valid
    // is true — behavior on an invalid model is intentionally undefined
    // rather than silently producing a wrong model.
    // PoolingModel build() const;

    const RefineryModel& model() const { return model_; }

private:
    RefineryModel model_;

    // --- validation helpers, each independently unit-testable ---
    void check_duplicate_names(ValidationResult&) const;
    void check_unknown_references(ValidationResult&) const;
    void check_bounds(ValidationResult&) const;          // negative availability, min > max, etc.
    void check_yield_sums(ValidationResult&) const;       // sum of a unit's yields per feed <= 1.0
    void check_graph_cycles(ValidationResult&) const;     // DFS over feed_streams -> streams -> pools -> products
    void check_orphans(ValidationResult&) const;          // stream with no consumer, pool with no product
};

}  // namespace bharatopt::refinery
