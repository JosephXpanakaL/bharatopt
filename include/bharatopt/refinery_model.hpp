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

struct ConstraintAuditDetail {
  std::string name;
  double lower{0.0};
  double upper{0.0};
  double actual_value{0.0};
  double violation{0.0};
  bool satisfied{true};
};

struct RefineryAuditReport {
  double objective{0.0};
  double max_bound_violation{0.0};
  double max_constraint_violation{0.0};
  std::vector<ConstraintAuditDetail> row_audits;
  bool all_satisfied{true};
  std::string summary() const;
};

RefineryAuditReport audit_refinery_solution(
    const PoolingModel& model,
    const std::vector<double>& x,
    double tolerance = 1e-5);

} // namespace bharatopt
