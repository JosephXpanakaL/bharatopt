#include "bharatopt/simplex.hpp"
#include "bharatopt/solver.hpp"
#include <stdexcept>

namespace bharatopt {

DirectedRoundingSimplex::DirectedRoundingSimplex(const SolverOptions& options)
    : options_(options) {}

SolverResult DirectedRoundingSimplex::solve(const LPModel& model) {
    BharatOptSolverCore core;
    SolverResult result = core.solve_lp(model, options_);
    result.backend = "Simplex-DirectedRounding";
    return result;
}

SolverResult solve_simplex(const LPModel& model, const SolverOptions& options) {
    DirectedRoundingSimplex simplex_engine(options);
    return simplex_engine.solve(model);
}

} // namespace bharatopt
