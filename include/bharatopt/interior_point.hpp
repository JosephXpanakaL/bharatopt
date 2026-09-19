#pragma once
#include "bharatopt/model.hpp"
namespace bharatopt {
struct InteriorPointOptions {
  int max_iterations{100};
  double tolerance{1e-8};
  int max_variables{400};
  int max_constraints{400};
};
SolverResult solve_interior_point(const LPModel& model, const InteriorPointOptions& options = {});
}