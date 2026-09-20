#pragma once

#include "bharatopt/model.hpp"
#include "bharatopt/solver.hpp"

#include <string>
#include <vector>

namespace bharatopt {

enum class SimplexPivotRule {
    Dantzig,
    Bland,
    SteepestEdge
};

struct SimplexOptions {
    int max_pivots{50000};
    double tolerance{1e-8};
    SimplexPivotRule pivot_rule{SimplexPivotRule::SteepestEdge};
    bool compute_farkas_certificate{true};
    double time_limit_sec{0.0};
};

struct SimplexResult {
    bool optimal{false};
    bool infeasible{false};
    bool unbounded{false};
    int pivots{0};
    double objective{0.0};
    double certified_lower_bound{0.0};
    std::vector<double> x;
    std::vector<double> farkas_multipliers;
    std::vector<double> dual_prices;
    std::string status;
};

class DirectedRoundingSimplex {
public:
    explicit DirectedRoundingSimplex(const SolverOptions& options);

    SolverResult solve(const LPModel& model);

private:
    SolverOptions options_;
};

SolverResult solve_simplex(const LPModel& model, const SolverOptions& options);

} // namespace bharatopt
