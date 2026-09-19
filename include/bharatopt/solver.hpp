#pragma once
#include "bharatopt/model.hpp"
namespace bharatopt {
class BharatOptSolverCore {
public:
  SolverResult solve(const LPModel& model, const SolverOptions& options = {});
};
}