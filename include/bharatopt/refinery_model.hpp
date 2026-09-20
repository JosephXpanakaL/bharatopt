#pragma once

#include "bharatopt/pooling.hpp"

#include <string>
#include <vector>

namespace bharatopt {

struct RefineryModelMetadata {
  struct Feedstock {
    std::string name;
    double availability{0.0};
    double cost{0.0};
    double sulfur{0.0};
  };
  struct Product {
    std::string name;
    double minimum_demand{0.0};
    double maximum_sulfur{0.0};
  };
  struct Unit {
    std::string name;
    double capacity{0.0};
  };

  std::vector<Feedstock> feedstocks;
  std::vector<Product> products;
  std::vector<Unit> units;
};

struct RefineryModel {
  PoolingModel pooling;
  RefineryModelMetadata metadata;
};

RefineryModel parse_refinery_json(const std::string& path);
void validate_refinery_model(const RefineryModel& model);

double pooling_objective_value(
    const PoolingModel& model,
    const std::vector<double>& x);

double pooling_max_constraint_violation(
    const PoolingModel& model,
    const std::vector<double>& x);

} // namespace bharatopt
