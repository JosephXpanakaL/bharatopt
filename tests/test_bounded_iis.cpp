#include "bharatopt/bounded_iis.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace bharatopt;

namespace {

LPModel make_conflict_model() {
  LPModel m;
  m.name = "bounded-iis-test";
  m.var_names = {"flow"};
  m.objective = {0.0};
  m.lower = {0.0};
  m.upper = {100.0};

  m.rows = {
      {"minimum_demand", RowSense::GreaterEqual, 10.0},
      {"unit_capacity", RowSense::LessEqual, 5.0},
      {"safe_lower_bound", RowSense::GreaterEqual, 0.0},
  };

  m.A.rows = 3;
  m.A.cols = 1;
  m.A.row_ptr = {0, 1, 2, 3};
  m.A.col_index = {0, 0, 0};
  m.A.values = {1.0, 1.0, 1.0};
  m.row_lower = {10.0, -INF, 0.0};
  m.row_upper = {INF, 5.0, INF};

  m.integer = {0};
  return m;
}

} // namespace

int main() {
  const LPModel model = make_conflict_model();

  int oracle_calls = 0;

  FeasibilityOracle oracle =
      [&](const LPModel&,
          const std::vector<int>& active_rows,
          double remaining_sec) -> FeasibilityStatus {
        ++oracle_calls;
        assert(remaining_sec >= 0.0);

        bool has_demand = false;
        bool has_capacity = false;

        for (int row : active_rows) {
          if (row == 0) has_demand = true;
          if (row == 1) has_capacity = true;
        }

        if (has_demand && has_capacity) {
          return FeasibilityStatus::Infeasible;
        }

        return FeasibilityStatus::Feasible;
      };

  BoundedIISOptions options;
  options.time_limit_sec = 100.0; // implementation must hard-cap to 2.5 s
  options.max_iterations = 100;   // implementation must hard-cap to 20 passes

  const auto result = computeBoundedIIS(model, oracle, options);

  assert(result.original_infeasible);
  assert(result.status == BoundedIISStatus::BOUNDED_IIS_FOUND);
  assert(result.irreducible_candidate);
  assert(!result.time_limit_hit);

  assert(result.conflicting_rows.size() == 2);
  assert(result.conflicting_rows[0] == 0);
  assert(result.conflicting_rows[1] == 1);

  assert(result.removed_rows.size() == 1);
  assert(result.removed_rows[0] == 2);

  assert(result.business_explanation.find("minimum_demand") !=
         std::string::npos);
  assert(result.business_explanation.find("unit_capacity") !=
         std::string::npos);
  assert(oracle_calls > 0);

  // A feasible model must not be presented as a conflict.
  auto feasible_oracle =
      [](const LPModel&, const std::vector<int>&, double) {
        return FeasibilityStatus::Feasible;
      };

  const auto feasible_result =
      computeBoundedIIS(model, feasible_oracle, {2.5, 20});

  assert(feasible_result.status ==
         BoundedIISStatus::ORIGINAL_MODEL_FEASIBLE);
  assert(!feasible_result.original_infeasible);
  assert(feasible_result.conflicting_rows.empty());

  std::cout << "Bounded IIS tests passed\n";
  return 0;
}
