#pragma once
#include "bharatopt/model.hpp"

namespace bharatopt {
class BharatOptSolverCore {
public:
  SolverResult solve_lp(const LPModel& model, const SolverOptions& options = {});
  SolverResult solve_milp(const LPModel& model, const SolverOptions& options = {});
  SolverResult solve_qp(const QPModel& model, const SolverOptions& options = {});
  SolverResult solve(const LPModel& model, const SolverOptions& options = {});
};
}

