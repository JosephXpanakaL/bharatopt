#pragma once
#include "bharatopt/model.hpp"

namespace bharatopt {
class BharatOptSolverCore {
public:
  SolverResult solve_lp(const LPModel& model, const SolverOptions& options = {});

  // ---------------------------------------------------------
  // MVP SCOPE FREEZE: MILP and QP deferred to 3-month roadmap
  // Stubbed to prevent linker errors and fail gracefully
  // ---------------------------------------------------------
  
  SolverResult solve_milp(const LPModel& model, const SolverOptions& options = {}) {
    SolverResult result;
    result.status = "UNSUPPORTED_MVP_FEATURE: MILP deferred to post-hackathon roadmap.";
    result.converged = false;
    return result;
  }

  // Assuming QPModel is defined in model.hpp. If not, comment this out entirely.
  SolverResult solve_qp(const QPModel& model, const SolverOptions& options = {}) {
    SolverResult result;
    result.status = "UNSUPPORTED_MVP_FEATURE: QP deferred to post-hackathon roadmap.";
    result.converged = false;
    return result;
  }

  SolverResult solve(const LPModel& model, const SolverOptions& options = {}) {
    // If the model has integers, it routes to the safe stub. Otherwise, real Simplex.
    return model.has_integer_variables() ? solve_milp(model, options) : solve_lp(model, options);
  }
};
}
