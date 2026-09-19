#pragma once

#include "bharatopt/model.hpp"

#include <functional>
#include <string>
#include <vector>

namespace bharatopt {

enum class FeasibilityStatus {
  Feasible,
  Infeasible,
  Unknown
};

enum class BoundedIISStatus {
  ORIGINAL_MODEL_FEASIBLE,
  ORIGINAL_MODEL_UNKNOWN,
  BOUNDED_IIS_FOUND,
  TIME_LIMIT,
  ITERATION_LIMIT
};

using FeasibilityOracle =
    std::function<FeasibilityStatus(
        const LPModel& model,
        const std::vector<int>& active_rows,
        double time_limit_sec)>;

struct BoundedIISOptions {
  // The implementation hard-caps this at 2.5 seconds.
  double time_limit_sec{2.5};
  // The implementation hard-caps this at 20 passes.
  int max_iterations{20};
};

struct BoundedIISResult {
  BoundedIISStatus status{BoundedIISStatus::ORIGINAL_MODEL_UNKNOWN};
  bool original_infeasible{false};
  bool irreducible_candidate{false};
  bool time_limit_hit{false};
  int iterations{0};
  double elapsed_sec{0.0};
  std::vector<int> conflicting_rows;
  std::vector<int> removed_rows;
  std::string business_explanation;
};

BoundedIISResult computeBoundedIIS(
    const LPModel& model,
    const FeasibilityOracle& feasibility_oracle,
    const BoundedIISOptions& options = {});

} // namespace bharatopt
