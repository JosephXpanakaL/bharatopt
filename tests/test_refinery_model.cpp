#include "bharatopt/refinery_model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  auto model = bharatopt::parse_refinery_json("examples/refinery_pooling.json");
  bharatopt::validate_refinery_model(model);
  assert(model.pooling.linear.A.rows == 2);
  assert(model.pooling.linear.A.cols == 3);
  assert(model.pooling.objective_bilinear.size() == 2);
  std::vector<double> x{2.0, 2.0, 1.0};
  assert(std::isfinite(bharatopt::pooling_objective_value(model.pooling, x)));
  assert(bharatopt::pooling_max_constraint_violation(model.pooling, x) >= 0.0);
  std::cout << "refinery JSON model test passed\n";
  return 0;
}
