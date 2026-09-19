#pragma once

#include "bharatopt/model.hpp"
#include "bharatopt/solver.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace bharatopt {

struct BilinearTerm {
  int left{-1};
  int right{-1};
  double coefficient{0.0};
};

struct PoolingModel {
  std::string name;
  LPModel linear;

  std::vector<BilinearTerm> objective_bilinear;
  std::vector<std::vector<BilinearTerm>> constraint_bilinear;

  bool valid() const;
};

struct PoolingSLPOptions {
  int max_iterations{20};

  double convergence_tolerance{1e-5};

  double initial_trust_radius{1.0};
  double minimum_trust_radius{1e-8};
  double maximum_trust_radius{100.0};

  double shrink_factor{0.5};
  double expand_factor{1.5};

  double poor_ratio{0.10};
  double excellent_ratio{0.75};

  double improvement_tolerance{1e-10};

  SolverOptions lp_options{};
};

struct PoolingSLPResult {
  bool converged{false};
  bool feasible{false};

  int iterations{0};

  double objective{std::numeric_limits<double>::quiet_NaN()};
  double initial_objective{std::numeric_limits<double>::quiet_NaN()};

  double predicted_improvement{0.0};
  double true_improvement{0.0};
  double improvement_ratio{0.0};

  double trust_radius{0.0};
  double final_step_norm{0.0};

  std::size_t accepted_steps{0};
  std::size_t rejected_steps{0};

  SolverResult last_lp_result;
  std::vector<double> x;
  std::string status;
};

struct McCormickTerm {
  int left{-1};
  int right{-1};
  int auxiliary_variable{-1};

  double left_lower{0.0};
  double left_upper{0.0};
  double right_lower{0.0};
  double right_upper{0.0};
};

struct McCormickRelaxation {
  LPModel relaxation;
  std::vector<McCormickTerm> terms;

  double global_lower_bound{INF};
  bool has_global_lower_bound{false};

  SolverResult solve_result;
};

McCormickRelaxation build_mccormick_relaxation(
    const PoolingModel& model);

McCormickRelaxation solve_mccormick_relaxation(
    const PoolingModel& model,
    BharatOptSolverCore& solver,
    const SolverOptions& options = {});

double mccormick_term_gap(
    double x,
    double y,
    double x_lower,
    double x_upper,
    double y_lower,
    double y_upper);

PoolingSLPResult solve_pooling_slp(
    const PoolingModel& model,
    BharatOptSolverCore& solver,
    const PoolingSLPOptions& options = {});

} // namespace bharatopt
