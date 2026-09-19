#pragma once
#include "bharatopt/model.hpp"
namespace bharatopt {
class BharatOptSolverCore {
public:
  SolverResult solve_lp(const LPModel& model, const SolverOptions& options = {});
  SolverResult solve_milp(const LPModel& model, const SolverOptions& options = {});
  SolverResult solve(const LPModel& model, const SolverOptions& options = {}) {
    return model.has_integer_variables() ? solve_milp(model, options) : solve_lp(model, options);
  }
};
}