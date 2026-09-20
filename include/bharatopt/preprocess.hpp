#pragma once
#include "bharatopt/model.hpp"
namespace bharatopt {
struct PreprocessResult {
  LPModel model;
  std::vector<double> column_scale;
  bool feasible{true};
  double condition_number_estimate{1.0};
  std::string message;
};
PreprocessResult preprocess_lp(const LPModel& input, int scaling_passes = 5);
void recover_primal(const std::vector<double>& scaled_x, const std::vector<double>& column_scale, std::vector<double>& x);
double estimate_matrix_condition_number(const CsrMatrix& A, int max_iter = 30);
}