#include "bharatopt/bounded_iis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace bharatopt {
namespace {

using Clock = std::chrono::steady_clock;

double remaining_seconds(const Clock::time_point& deadline) {
  const auto now = Clock::now();
  if (now >= deadline) return 0.0;
  return std::chrono::duration<double>(deadline - now).count();
}

std::string format_number(double value) {
  if (!std::isfinite(value)) {
    if (value > 0.0) return "+inf";
    if (value < 0.0) return "-inf";
    return "NaN";
  }
  std::ostringstream out;
  out << std::setprecision(6) << value;
  return out.str();
}

std::string row_label(const LPModel& model, int row) {
  if (row >= 0 &&
      static_cast<std::size_t>(row) < model.rows.size() &&
      !model.rows[row].name.empty()) {
    return model.rows[row].name;
  }
  return "row_" + std::to_string(row);
}

std::string linear_expression(const LPModel& model, int row) {
  if (row < 0 ||
      static_cast<std::size_t>(row) >= model.A.rows ||
      static_cast<std::size_t>(row + 1) >= model.A.row_ptr.size()) {
    return "invalid-row";
  }

  std::ostringstream out;
  bool first = true;
  const int begin = model.A.row_ptr[row];
  const int end = model.A.row_ptr[row + 1];

  for (int k = begin; k < end; ++k) {
    const int column = model.A.col_index[k];
    const double coefficient = model.A.values[k];
    if (coefficient == 0.0) continue;

    std::string variable;
    if (column >= 0 &&
        static_cast<std::size_t>(column) < model.var_names.size() &&
        !model.var_names[column].empty()) {
      variable = model.var_names[column];
    } else {
      variable = "x_" + std::to_string(column);
    }

    if (!first) {
      out << (coefficient >= 0.0 ? " + " : " - ");
    } else if (coefficient < 0.0) {
      out << "-";
    }

    const double magnitude = std::abs(coefficient);
    if (std::abs(magnitude - 1.0) > 1e-12) {
      out << format_number(magnitude) << " * ";
    }
    out << variable;
    first = false;
  }

  return first ? "0" : out.str();
}

std::string describe_constraint(const LPModel& model, int row) {
  if (row < 0 || static_cast<std::size_t>(row) >= model.rows.size()) {
    return "Invalid constraint row.";
  }

  const std::string label = row_label(model, row);
  const std::string lhs = linear_expression(model, row);
  const double lower = model.row_lower[row];
  const double upper = model.row_upper[row];

  std::ostringstream out;
  out << "\"" << label << "\" requires ";

  const bool has_lower = std::isfinite(lower);
  const bool has_upper = std::isfinite(upper);

  if (has_lower && has_upper && std::abs(lower - upper) <= 1e-12) {
    out << lhs << " to equal " << format_number(lower) << ".";
  } else if (has_lower && has_upper) {
    out << lhs << " to stay between " << format_number(lower)
        << " and " << format_number(upper) << ".";
  } else if (has_upper) {
    out << lhs << " to be at most " << format_number(upper) << ".";
  } else if (has_lower) {
    out << lhs << " to be at least " << format_number(lower) << ".";
  } else {
    out << lhs << " without a finite bound.";
  }

  return out.str();
}

} // namespace

BoundedIISResult computeBoundedIIS(
    const LPModel& model,
    const FeasibilityOracle& feasibility_oracle,
    const BoundedIISOptions& options) {
  constexpr double kHardTimeLimitSec = 2.5;
  constexpr int kHardIterationLimit = 20;

  if (!feasibility_oracle) {
    throw std::invalid_argument(
        "computeBoundedIIS requires a valid feasibility oracle");
  }
  if (options.time_limit_sec <= 0.0) {
    throw std::invalid_argument(
        "Bounded IIS time limit must be positive");
  }
  if (options.max_iterations <= 0) {
    throw std::invalid_argument(
        "Bounded IIS iteration limit must be positive");
  }

  const double time_budget =
      std::min(options.time_limit_sec, kHardTimeLimitSec);
  const int iteration_budget =
      std::min(options.max_iterations, kHardIterationLimit);

  BoundedIISResult result;
  const auto start = Clock::now();
  const auto deadline =
      start + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(time_budget));

  std::vector<int> active_rows;
  active_rows.reserve(model.A.rows);
  for (std::size_t i = 0; i < model.A.rows; ++i) {
    active_rows.push_back(static_cast<int>(i));
  }

  double remaining = remaining_seconds(deadline);
  if (remaining <= 0.0) {
    result.status = BoundedIISStatus::TIME_LIMIT;
    result.time_limit_hit = true;
    result.elapsed_sec =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
  }

  const FeasibilityStatus original_status =
      feasibility_oracle(model, active_rows, remaining);

  if (original_status == FeasibilityStatus::Feasible) {
    result.status = BoundedIISStatus::ORIGINAL_MODEL_FEASIBLE;
    result.elapsed_sec =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
  }

  if (original_status != FeasibilityStatus::Infeasible) {
    result.status = BoundedIISStatus::ORIGINAL_MODEL_UNKNOWN;
    result.elapsed_sec =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
  }

  result.original_infeasible = true;

  bool completed_irreducibility_pass = false;

  for (int pass = 0; pass < iteration_budget; ++pass) {
    remaining = remaining_seconds(deadline);
    if (remaining <= 0.0) {
      result.status = BoundedIISStatus::TIME_LIMIT;
      result.time_limit_hit = true;
      break;
    }

    ++result.iterations;
    bool removed_any = false;
    const std::vector<int> candidates = active_rows;

    for (int candidate_row : candidates) {
      remaining = remaining_seconds(deadline);
      if (remaining <= 0.0) {
        result.status = BoundedIISStatus::TIME_LIMIT;
        result.time_limit_hit = true;
        break;
      }

      const auto it =
          std::find(active_rows.begin(), active_rows.end(), candidate_row);
      if (it == active_rows.end()) continue;

      std::vector<int> trial_rows;
      trial_rows.reserve(active_rows.size() - 1);
      for (int row : active_rows) {
        if (row != candidate_row) trial_rows.push_back(row);
      }

      const FeasibilityStatus trial_status =
          feasibility_oracle(model, trial_rows, remaining);

      if (trial_status == FeasibilityStatus::Infeasible) {
        active_rows.swap(trial_rows);
        result.removed_rows.push_back(candidate_row);
        removed_any = true;
      }

      if (remaining_seconds(deadline) <= 0.0) {
        result.status = BoundedIISStatus::TIME_LIMIT;
        result.time_limit_hit = true;
        break;
      }
    }

    if (result.time_limit_hit) break;

    if (!removed_any) {
      completed_irreducibility_pass = true;
      break;
    }
  }

  result.conflicting_rows = active_rows;
  result.irreducible_candidate = completed_irreducibility_pass;

  if (!result.time_limit_hit &&
      !result.irreducible_candidate &&
      result.iterations >= options.max_iterations) {
    result.status = BoundedIISStatus::ITERATION_LIMIT;
  } else if (!result.time_limit_hit && result.irreducible_candidate) {
    result.status = BoundedIISStatus::BOUNDED_IIS_FOUND;
  }

  result.elapsed_sec =
      std::chrono::duration<double>(Clock::now() - start).count();

  std::ostringstream explanation;
  explanation
      << "BharatOpt found a bounded conflict explanation in "
      << format_number(result.elapsed_sec) << " seconds.\n\n"
      << "The original constraint set was proven infeasible. "
      << "The explainer removed " << result.removed_rows.size()
      << " constraints that were not required to preserve the "
         "detected conflict.\n\n";

  if (result.conflicting_rows.empty()) {
    explanation << "No conflicting business constraints remain in the "
                   "bounded result.";
  } else {
    explanation << "The remaining conflicting business requirements are:\n";
    for (std::size_t i = 0; i < result.conflicting_rows.size(); ++i) {
      explanation << "  " << (i + 1) << ". "
                  << describe_constraint(
                         model, result.conflicting_rows[i])
                  << "\n";
    }

    explanation << "\n";
    if (result.irreducible_candidate) {
      explanation
          << "Each remaining requirement was tested for deletion during "
             "the completed greedy passes; removing one did not produce "
             "a proven-feasible subsystem. This is an inclusion-minimal "
             "conflict set produced by the bounded filter, not a "
             "minimum-cardinality IIS.";
    } else if (result.time_limit_hit) {
      explanation
          << "The 2.5-second investigation budget was reached before "
             "full irreducibility could be established. Treat this as "
             "a bounded conflict candidate rather than a complete IIS "
             "certificate.";
    } else {
      explanation
          << "The 20-pass investigation limit was reached before full "
             "irreducibility could be established. Treat this as a "
             "bounded conflict candidate.";
    }
  }

  result.business_explanation = explanation.str();
  return result;
}

} // namespace bharatopt
