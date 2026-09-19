#include "bharatopt/simplex.hpp"
#include <stdexcept>

namespace bharatopt {

// ---------------------------------------------------------
// Core Solver Integration
// ---------------------------------------------------------

SolverResult BharatOptSolverCore::solve_lp(const LPModel& model, const SolverOptions& options) {
    // Instantiate our custom Phase 1 / Phase 2 Simplex engine
    DirectedRoundingSimplex simplex_engine(options);
    
    // Execute the solve and return the result, which now includes 
    // the certified_lower_bound and farkas_multipliers.
    return simplex_engine.solve(model);
}

// ---------------------------------------------------------
// MVP SCOPE FREEZE: MILP and QP Stubs (if not in header)
// ---------------------------------------------------------

/* 
 * If you did NOT inline these in the header, uncomment this block to 
 * satisfy the linker and prevent CMake errors.
 *
SolverResult BharatOptSolverCore::solve_milp(const LPModel& model, const SolverOptions& options) {
    SolverResult result;
    result.status = "UNSUPPORTED_MVP_FEATURE: MILP deferred to post-hackathon roadmap.";
    result.converged = false;
    return result;
}

SolverResult BharatOptSolverCore::solve_qp(const QPModel& model, const SolverOptions& options) {
    SolverResult result;
    result.status = "UNSUPPORTED_MVP_FEATURE: QP deferred to post-hackathon roadmap.";
    result.converged = false;
    return result;
}
*/

} // namespace bharatopt
