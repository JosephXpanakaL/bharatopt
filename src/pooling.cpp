#include "bharatopt/pooling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace bharatopt {
namespace {

constexpr double EPS = 1e-12;
using Vec = std::vector<double>;

double norm2(const Vec& x) {
  long double sum = 0.0L;
  for (double v : x) {
    if (!std::isfinite(v)) return INF;
    sum += static_cast<long double>(v) * static_cast<long double>(v);
  }
  return std::sqrt(static_cast<double>(sum));
}

double clamp_value(double v, double lo, double hi) {
  return std::min(std::max(v, lo), hi);
}

bool finite_bounds(double lo, double hi) {
  return std::isfinite(lo) && std::isfinite(hi) && lo <= hi;
}

void validate_term(const BilinearTerm& term, std::size_t n) {
  if (term.left < 0 ||
      term.right < 0 ||
      static_cast<std::size_t>(term.left) >= n ||
      static_cast<std::size_t>(term.right) >= n) {
    throw std::invalid_argument(
        "Pooling bilinear term references an invalid variable");
  }

  if (term.left == term.right) {
    throw std::invalid_argument(
        "Pooling layer requires distinct variables for bilinear terms");
  }

  if (!std::isfinite(term.coefficient)) {
    throw std::invalid_argument(
        "Pooling bilinear coefficient must be finite");
  }
}

bool model_dimensions_valid(const PoolingModel& model) {
  const LPModel& lp = model.linear;
  const std::size_t n = lp.A.cols;

  return lp.objective.size() == n &&
         lp.lower.size() == n &&
         lp.upper.size() == n &&
         lp.integer.size() == n &&
         lp.row_lower.size() == lp.A.rows &&
         lp.row_upper.size() == lp.A.rows &&
         lp.var_names.size() == n &&
         lp.rows.size() == lp.A.rows &&
         model.constraint_bilinear.size() <= lp.A.rows;
}

void validate_model(const PoolingModel& model) {
  if (!model_dimensions_valid(model)) {
    throw std::invalid_argument("Invalid PoolingModel dimensions");
  }

  const std::size_t n = model.linear.A.cols;

  for (std::size_t j = 0; j < n; ++j) {
    if (!finite_bounds(
            model.linear.lower[j],
            model.linear.upper[j])) {
      throw std::invalid_argument(
          "Pooling layer requires finite variable bounds");
    }

    if (model.linear.integer[j]) {
      throw std::invalid_argument(
          "Pooling SLP currently supports continuous variables only");
    }
  }

  for (const auto& term : model.objective_bilinear) {
    validate_term(term, n);
  }

  for (const auto& row : model.constraint_bilinear) {
    for (const auto& term : row) {
      validate_term(term, n);
    }
  }
}

double bilinear_value(
    const BilinearTerm& term,
    const Vec& x) {
  return term.coefficient *
         x[static_cast<std::size_t>(term.left)] *
         x[static_cast<std::size_t>(term.right)];
}

double true_objective(
    const PoolingModel& model,
    const Vec& x) {
  double value = model.linear.objective_offset;

  for (std::size_t j = 0;
       j < model.linear.objective.size();
       ++j) {
    value += model.linear.objective[j] * x[j];
  }

  for (const auto& term : model.objective_bilinear) {
    value += bilinear_value(term, x);
  }

  return value;
}

double objective_improvement(
    bool maximize,
    double current,
    double candidate) {
  return maximize ? candidate - current : current - candidate;
}

double linearized_term_value(
    const BilinearTerm& term,
    const Vec& x,
    const Vec& center) {
  const double xc =
      center[static_cast<std::size_t>(term.left)];
  const double yc =
      center[static_cast<std::size_t>(term.right)];
  const double xv =
      x[static_cast<std::size_t>(term.left)];
  const double yv =
      x[static_cast<std::size_t>(term.right)];

  return term.coefficient *
         (yc * xv + xc * yv - xc * yc);
}

double linearized_objective(
    const PoolingModel& model,
    const Vec& x,
    const Vec& center) {
  double value = model.linear.objective_offset;

  for (std::size_t j = 0;
       j < model.linear.objective.size();
       ++j) {
    value += model.linear.objective[j] * x[j];
  }

  for (const auto& term : model.objective_bilinear) {
    value += linearized_term_value(term, x, center);
  }

  return value;
}

double true_row_activity(
    const PoolingModel& model,
    int row,
    const Vec& x) {
  double activity = 0.0;

  for (int k = model.linear.A.row_ptr[row];
       k < model.linear.A.row_ptr[row + 1];
       ++k) {
    const int col = model.linear.A.col_index[k];
    activity += model.linear.A.values[k] *
                x[static_cast<std::size_t>(col)];
  }

  if (static_cast<std::size_t>(row) <
      model.constraint_bilinear.size()) {
    for (const auto& term :
         model.constraint_bilinear[static_cast<std::size_t>(row)]) {
      activity += bilinear_value(term, x);
    }
  }

  return activity;
}

double nonlinear_constraint_violation(
    const PoolingModel& model,
    const Vec& x) {
  double squared = 0.0;

  for (std::size_t i = 0;
       i < model.linear.A.rows;
       ++i) {
    const double activity =
        true_row_activity(model, static_cast<int>(i), x);

    double violation = 0.0;
    if (activity < model.linear.row_lower[i]) {
      violation = model.linear.row_lower[i] - activity;
    } else if (activity > model.linear.row_upper[i]) {
      violation = activity - model.linear.row_upper[i];
    }

    squared += violation * violation;
  }

  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] < model.linear.lower[j]) {
      const double v = model.linear.lower[j] - x[j];
      squared += v * v;
    }
    if (x[j] > model.linear.upper[j]) {
      const double v = x[j] - model.linear.upper[j];
      squared += v * v;
    }
  }

  return std::sqrt(squared);
}

bool nonlinear_feasible(
    const PoolingModel& model,
    const Vec& x,
    double tolerance) {
  return x.size() == model.linear.A.cols &&
         nonlinear_constraint_violation(model, x) <= tolerance;
}

void add_sparse_entry(
    std::vector<std::vector<std::pair<int, double>>>& rows,
    int row,
    int column,
    double value) {
  if (std::abs(value) > EPS) {
    rows[static_cast<std::size_t>(row)].push_back({column, value});
  }
}

SparseMatrixCSR build_csr(
    std::vector<std::vector<std::pair<int, double>>> rows,
    int columns) {
  SparseMatrixCSR matrix;
  matrix.rows = rows.size();
  matrix.cols = static_cast<std::size_t>(columns);
  matrix.row_ptr.assign(matrix.rows + 1, 0);

  for (auto& row : rows) {
    std::sort(
        row.begin(), row.end(),
        [](const auto& a, const auto& b) {
          return a.first < b.first;
        });

    std::vector<std::pair<int, double>> merged;
    merged.reserve(row.size());

    for (const auto& entry : row) {
      if (!merged.empty() &&
          merged.back().first == entry.first) {
        merged.back().second += entry.second;
      } else {
        merged.push_back(entry);
      }
    }

    row.swap(merged);
  }

  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (const auto& [column, value] : rows[i]) {
      if (std::abs(value) > EPS) {
        matrix.col_index.push_back(column);
        matrix.values.push_back(value);
      }
    }
    matrix.row_ptr[i + 1] =
        static_cast<int>(matrix.values.size());
  }

  return matrix;
}

void add_mccormick_constraints(
    std::vector<std::vector<std::pair<int, double>>>& rows,
    std::vector<double>& row_lower,
    std::vector<double>& row_upper,
    std::vector<ConstraintRow>& constraint_rows,
    int x,
    int y,
    int w,
    double lx,
    double ux,
    double ly,
    double uy,
    const std::string& prefix) {

  // w >= lx*y + ly*x - lx*ly
  {
    const int row = static_cast<int>(rows.size());
    rows.emplace_back();

    add_sparse_entry(rows, row, y, lx);
    add_sparse_entry(rows, row, x, ly);
    add_sparse_entry(rows, row, w, -1.0);

    row_lower.push_back(-INF);
    row_upper.push_back(lx * ly);

    constraint_rows.push_back({
        prefix + "_mc_lower_1",
        RowSense::LessEqual,
        row_upper.back()});
  }

  // w >= ux*y + uy*x - ux*uy
  {
    const int row = static_cast<int>(rows.size());
    rows.emplace_back();

    add_sparse_entry(rows, row, y, ux);
    add_sparse_entry(rows, row, x, uy);
    add_sparse_entry(rows, row, w, -1.0);

    row_lower.push_back(-INF);
    row_upper.push_back(ux * uy);

    constraint_rows.push_back({
        prefix + "_mc_lower_2",
        RowSense::LessEqual,
        row_upper.back()});
  }

  // w <= ux*y + ly*x - ux*ly
  {
    const int row = static_cast<int>(rows.size());
    rows.emplace_back();

    add_sparse_entry(rows, row, w, 1.0);
    add_sparse_entry(rows, row, y, -ux);
    add_sparse_entry(rows, row, x, -ly);

    row_lower.push_back(-INF);
    row_upper.push_back(-ux * ly);

    constraint_rows.push_back({
        prefix + "_mc_upper_1",
        RowSense::LessEqual,
        row_upper.back()});
  }

  // w <= lx*y + uy*x - lx*uy
  {
    const int row = static_cast<int>(rows.size());
    rows.emplace_back();

    add_sparse_entry(rows, row, w, 1.0);
    add_sparse_entry(rows, row, y, -lx);
    add_sparse_entry(rows, row, x, -uy);

    row_lower.push_back(-INF);
    row_upper.push_back(-lx * uy);

    constraint_rows.push_back({
        prefix + "_mc_upper_2",
        RowSense::LessEqual,
        row_upper.back()});
  }
}

LPModel build_linearized_lp(
    const PoolingModel& model,
    const Vec& center,
    double radius) {
  LPModel lp = model.linear;
  const std::size_t n = model.linear.A.cols;

  lp.name = model.name.empty()
              ? "pooling_slp_subproblem"
              : model.name + "_slp";
  lp.integer.assign(n, 0);

  // Strict trust-region box.
  for (std::size_t j = 0; j < n; ++j) {
    lp.lower[j] = std::max(
        model.linear.lower[j],
        center[j] - radius);
    lp.upper[j] = std::min(
        model.linear.upper[j],
        center[j] + radius);

    if (lp.lower[j] > lp.upper[j] + EPS) {
      throw std::runtime_error("Trust-region box is empty");
    }
  }

  std::vector<std::vector<std::pair<int, double>>> rows(
      model.linear.A.rows);

  for (std::size_t i = 0;
       i < model.linear.A.rows;
       ++i) {
    for (int k = model.linear.A.row_ptr[i];
         k < model.linear.A.row_ptr[i + 1];
         ++k) {
      rows[i].push_back({
          model.linear.A.col_index[k],
          model.linear.A.values[k]});
    }
  }

  lp.objective_offset = model.linear.objective_offset;

  // f(x,y) ~= y_k*x + x_k*y - x_k*y_k.
  for (const auto& term : model.objective_bilinear) {
    const double xc = center[
        static_cast<std::size_t>(term.left)];
    const double yc = center[
        static_cast<std::size_t>(term.right)];

    lp.objective[
        static_cast<std::size_t>(term.left)] +=
        term.coefficient * yc;

    lp.objective[
        static_cast<std::size_t>(term.right)] +=
        term.coefficient * xc;

    lp.objective_offset -=
        term.coefficient * xc * yc;
  }

  // Linearize nonlinear constraint rows.
  for (std::size_t i = 0;
       i < model.constraint_bilinear.size() &&
       i < rows.size();
       ++i) {
    for (const auto& term : model.constraint_bilinear[i]) {
      const double xc = center[
          static_cast<std::size_t>(term.left)];
      const double yc = center[
          static_cast<std::size_t>(term.right)];

      add_sparse_entry(
          rows, static_cast<int>(i), term.left,
          term.coefficient * yc);

      add_sparse_entry(
          rows, static_cast<int>(i), term.right,
          term.coefficient * xc);

      const double shift =
          term.coefficient * xc * yc;

      lp.row_lower[i] += shift;
      lp.row_upper[i] += shift;
    }
  }

  lp.A = build_csr(rows, static_cast<int>(n));
  return lp;
}

} // namespace

bool PoolingModel::valid() const {
  return model_dimensions_valid(*this);
}

McCormickRelaxation build_mccormick_relaxation(
    const PoolingModel& model) {
  validate_model(model);

  McCormickRelaxation result;
  result.relaxation = model.linear;

  const std::size_t original_n = model.linear.A.cols;

  std::vector<std::vector<std::pair<int, double>>> rows(
      model.linear.A.rows);

  for (std::size_t i = 0;
       i < model.linear.A.rows;
       ++i) {
    for (int k = model.linear.A.row_ptr[i];
         k < model.linear.A.row_ptr[i + 1];
         ++k) {
      rows[i].push_back({
          model.linear.A.col_index[k],
          model.linear.A.values[k]});
    }
  }

  std::vector<double> row_lower = model.linear.row_lower;
  std::vector<double> row_upper = model.linear.row_upper;
  std::vector<ConstraintRow> constraint_rows = model.linear.rows;

  struct ProductRef {
    BilinearTerm term;
    int auxiliary;
  };

  std::vector<ProductRef> products;

  for (const auto& term : model.objective_bilinear) {
    products.push_back({
        term,
        static_cast<int>(original_n + products.size())});
  }

  for (const auto& row : model.constraint_bilinear) {
    for (const auto& term : row) {
      products.push_back({
          term,
          static_cast<int>(original_n + products.size())});
    }
  }

  for (std::size_t product_index = 0;
       product_index < products.size();
       ++product_index) {
    const auto& product = products[product_index];
    const int w = product.auxiliary;
    const auto& term = product.term;

    const double lx =
        model.linear.lower[static_cast<std::size_t>(term.left)];
    const double ux =
        model.linear.upper[static_cast<std::size_t>(term.left)];
    const double ly =
        model.linear.lower[static_cast<std::size_t>(term.right)];
    const double uy =
        model.linear.upper[static_cast<std::size_t>(term.right)];

    const double p1 = lx * ly;
    const double p2 = lx * uy;
    const double p3 = ux * ly;
    const double p4 = ux * uy;

    const double product_lower =
        std::min(std::min(p1, p2), std::min(p3, p4));
    const double product_upper =
        std::max(std::max(p1, p2), std::max(p3, p4));

    result.relaxation.var_names.push_back(
        "mc_product_" + std::to_string(product_index));
    result.relaxation.objective.push_back(0.0);
    result.relaxation.lower.push_back(product_lower);
    result.relaxation.upper.push_back(product_upper);
    result.relaxation.integer.push_back(0);

    result.terms.push_back({
        term.left,
        term.right,
        w,
        lx,
        ux,
        ly,
        uy});
  }

  std::size_t cursor = 0;

  for (const auto& term : model.objective_bilinear) {
    const int w = products[cursor].auxiliary;
    result.relaxation.objective[
        static_cast<std::size_t>(w)] +=
        term.coefficient;
    ++cursor;
  }

  for (std::size_t row = 0;
       row < model.constraint_bilinear.size();
       ++row) {
    for (const auto& term : model.constraint_bilinear[row]) {
      const int w = products[cursor].auxiliary;

      add_sparse_entry(
          rows,
          static_cast<int>(row),
          w,
          term.coefficient);

      ++cursor;
    }
  }

  for (std::size_t k = 0; k < products.size(); ++k) {
    const auto& term = products[k].term;

    const double lx =
        model.linear.lower[static_cast<std::size_t>(term.left)];
    const double ux =
        model.linear.upper[static_cast<std::size_t>(term.left)];
    const double ly =
        model.linear.lower[static_cast<std::size_t>(term.right)];
    const double uy =
        model.linear.upper[static_cast<std::size_t>(term.right)];

    add_mccormick_constraints(
        rows,
        row_lower,
        row_upper,
        constraint_rows,
        term.left,
        term.right,
        products[k].auxiliary,
        lx, ux, ly, uy,
        "mc_" + std::to_string(k));
  }

  result.relaxation.A =
      build_csr(rows, static_cast<int>(original_n + products.size()));

  result.relaxation.row_lower =
      std::move(row_lower);
  result.relaxation.row_upper =
      std::move(row_upper);
  result.relaxation.rows =
      std::move(constraint_rows);

  result.relaxation.name =
      model.name.empty()
        ? "pooling_mccormick_relaxation"
        : model.name + "_mccormick";

  return result;
}

McCormickRelaxation solve_mccormick_relaxation(
    const PoolingModel& model,
    BharatOptSolverCore& solver,
    const SolverOptions& options) {
  McCormickRelaxation result =
      build_mccormick_relaxation(model);

  result.solve_result =
      solver.solve(result.relaxation, options);

  if (result.solve_result.x.size() == result.relaxation.A.cols &&
      std::isfinite(result.solve_result.objective)) {
    if (model.linear.maximize) {
      result.global_upper_bound = result.solve_result.objective;
      result.has_global_upper_bound = true;
    } else {
      result.global_lower_bound = result.solve_result.objective;
      result.has_global_lower_bound = true;
    }
  }

  return result;
}

double mccormick_term_gap(
    double x,
    double y,
    double x_lower,
    double x_upper,
    double y_lower,
    double y_upper) {
  if (!finite_bounds(x_lower, x_upper) ||
      !finite_bounds(y_lower, y_upper)) {
    return INF;
  }

  const double lower_1 =
      x_lower * y +
      y_lower * x -
      x_lower * y_lower;

  const double lower_2 =
      x_upper * y +
      y_upper * x -
      x_upper * y_upper;

  const double upper_1 =
      x_upper * y +
      y_lower * x -
      x_upper * y_lower;

  const double upper_2 =
      x_lower * y +
      y_upper * x -
      x_lower * y_upper;

  return std::max(
      0.0,
      std::min(upper_1, upper_2) -
      std::max(lower_1, lower_2));
}

PoolingSLPResult solve_pooling_slp(
    const PoolingModel& model,
    BharatOptSolverCore& solver,
    const PoolingSLPOptions& options) {
  validate_model(model);

  if (options.max_iterations <= 0) {
    throw std::invalid_argument(
        "Pooling SLP max_iterations must be positive");
  }

  if (options.convergence_tolerance <= 0.0) {
    throw std::invalid_argument(
        "Pooling SLP convergence_tolerance must be positive");
  }

  if (options.initial_trust_radius <= 0.0 ||
      options.minimum_trust_radius <= 0.0 ||
      options.maximum_trust_radius <= 0.0 ||
      options.minimum_trust_radius >
          options.maximum_trust_radius) {
    throw std::invalid_argument(
        "Invalid pooling trust-region radii");
  }

  if (options.shrink_factor <= 0.0 ||
      options.shrink_factor >= 1.0 ||
      options.expand_factor <= 1.0) {
    throw std::invalid_argument(
        "Invalid pooling trust-region adjustment factors");
  }

  if (options.poor_ratio < 0.0 ||
      options.excellent_ratio < options.poor_ratio) {
    throw std::invalid_argument(
        "Invalid pooling acceptance ratios");
  }

  const std::size_t n = model.linear.A.cols;

  Vec x(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = 0.5 * (model.linear.lower[j] + model.linear.upper[j]);
  }

  PoolingSLPResult result;
  result.x = x;
  result.objective = true_objective(model, x);
  result.initial_objective = result.objective;
  result.trust_radius = clamp_value(
      options.initial_trust_radius,
      options.minimum_trust_radius,
      options.maximum_trust_radius);

  // Deterministic feasibility restoration. This is not a global certificate;
  // it only searches for a valid nonlinear starting point for SLP.
  auto violation = [&](const Vec& z) {
    double worst = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      worst = std::max(worst, model.linear.lower[j] - z[j]);
      worst = std::max(worst, z[j] - model.linear.upper[j]);
    }
    for (std::size_t r = 0; r < model.linear.A.rows; ++r) {
      double az = 0.0;
      for (std::size_t k = model.linear.A.row_ptr[r];
           k < model.linear.A.row_ptr[r + 1]; ++k) {
        az += model.linear.A.values[k] *
              z[static_cast<std::size_t>(model.linear.A.col_index[k])];
      }
      for (const auto& t : model.constraint_bilinear[r]) {
        az += t.coefficient * z[static_cast<std::size_t>(t.left)] *
              z[static_cast<std::size_t>(t.right)];
      }
      if (std::isfinite(model.linear.row_lower[r]))
        worst = std::max(worst, model.linear.row_lower[r] - az);
      if (std::isfinite(model.linear.row_upper[r]))
        worst = std::max(worst, az - model.linear.row_upper[r]);
    }
    return std::max(0.0, worst);
  };

  double best_violation = violation(x);
  if (best_violation > options.lp_options.tolerance) {
    auto consider = [&](const Vec& candidate) {
      const double v = violation(candidate);
      if (v < best_violation) {
        best_violation = v;
        x = candidate;
      }
    };

    consider(model.linear.lower);
    consider(model.linear.upper);

    // Deterministic low-discrepancy samples plus coordinate restoration.
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    const int attempts = std::max(0, options.feasibility_search_attempts);
    for (int attempt = 0; attempt < attempts &&
         best_violation > options.lp_options.tolerance; ++attempt) {
      Vec candidate(n, 0.0);
      for (std::size_t j = 0; j < n; ++j) {
        state ^= state << 7;
        state ^= state >> 9;
        const double u = static_cast<double>(state & 0xFFFFFFULL) /
                         static_cast<double>(0x1000000ULL);
        candidate[j] = model.linear.lower[j] +
                       u * (model.linear.upper[j] - model.linear.lower[j]);
      }
      consider(candidate);
    }

    const double step_fraction =
        std::clamp(options.feasibility_search_step_fraction, 0.01, 0.5);
    for (int pass = 0; pass < 8 &&
         best_violation > options.lp_options.tolerance; ++pass) {
      bool improved = false;
      for (std::size_t j = 0; j < n; ++j) {
        const double lo = model.linear.lower[j];
        const double hi = model.linear.upper[j];
        const double span = hi - lo;
        for (double f : {0.0, step_fraction, 0.5, 1.0 - step_fraction, 1.0}) {
          Vec candidate = x;
          candidate[j] = lo + f * span;
          const double before = best_violation;
          consider(candidate);
          improved = improved || best_violation < before;
        }
      }
      if (!improved) break;
    }
  }

  if (!nonlinear_feasible(
          model, x, options.lp_options.tolerance)) {
    result.x = x;
    result.objective = true_objective(model, x);
    result.status = "NO_FEASIBLE_START_FOUND";
    result.feasible = false;
    return result;
  }

  result.x = x;
  result.objective = true_objective(model, x);
  result.initial_objective = result.objective;

  result.feasible = true;
  double radius = result.trust_radius;

  for (int iteration = 1;
       iteration <= options.max_iterations;
       ++iteration) {
    result.iterations = iteration;
    result.trust_radius = radius;

    const LPModel lp =
        build_linearized_lp(model, x, radius);

    SolverResult lp_result =
        solver.solve(lp, options.lp_options);

    result.last_lp_result = lp_result;

    if (lp_result.x.size() != n ||
        !std::isfinite(lp_result.objective)) {
      ++result.rejected_steps;
      radius = std::max(
          options.minimum_trust_radius,
          radius * options.shrink_factor);
      result.trust_radius = radius;

      if (radius <=
          options.minimum_trust_radius + EPS) {
        result.status =
            "SLP_SUBPROBLEM_FAILED_AT_MIN_RADIUS";
        break;
      }

      continue;
    }

    const Vec& candidate = lp_result.x;
    Vec step(n, 0.0);

    for (std::size_t j = 0; j < n; ++j) {
      step[j] = candidate[j] - x[j];
    }

    const double step_norm = norm2(step);
    result.final_step_norm = step_norm;

    /*
     * Because the linearization is exact at x_k, the candidate's
     * linearized objective difference is the model's predicted change.
     */
    const double predicted =
        objective_improvement(
            model.linear.maximize,
            linearized_objective(model, x, x),
            linearized_objective(model, candidate, x));

    const double current_true =
        true_objective(model, x);

    const double candidate_true =
        true_objective(model, candidate);

    const bool candidate_feasible =
        nonlinear_feasible(
            model,
            candidate,
            options.lp_options.tolerance);

    const double actual =
        candidate_feasible
          ? objective_improvement(
                model.linear.maximize,
                current_true,
                candidate_true)
          : -INF;

    const double ratio =
        candidate_feasible
          ? ((predicted > options.improvement_tolerance)
              ? actual / predicted
              : (actual > options.improvement_tolerance
                  ? INF
                  : 0.0))
          : -INF;

    result.predicted_improvement = predicted;
    result.true_improvement = actual;
    result.improvement_ratio = ratio;

    const bool poor_prediction =
        predicted > options.improvement_tolerance &&
        ratio < options.poor_ratio;

    const bool true_improvement_is_negative =
        predicted > options.improvement_tolerance &&
        actual < -options.improvement_tolerance;

    const bool reject =
        !candidate_feasible ||
        poor_prediction ||
        true_improvement_is_negative;

    if (reject) {
      ++result.rejected_steps;

      radius = std::max(
          options.minimum_trust_radius,
          radius * options.shrink_factor);

      result.trust_radius = radius;

      if (radius <=
          options.minimum_trust_radius + EPS) {
        result.status = candidate_feasible
          ? "TRUST_REGION_COLLAPSED"
          : "NONLINEAR_FEASIBILITY_BLOCKED";
        break;
      }

      continue;
    }

    /*
     * Accept step and move the operating point.
     */
    x = candidate;
    result.x = x;
    result.objective = candidate_true;
    ++result.accepted_steps;

    if (ratio >= options.excellent_ratio) {
      radius = std::min(
          options.maximum_trust_radius,
          radius * options.expand_factor);
    }

    result.trust_radius = radius;

    if (step_norm <= options.convergence_tolerance) {
      result.converged = true;
      result.status = "SLP_CONVERGED";
      break;
    }

    if (radius <=
        options.minimum_trust_radius + EPS) {
      result.status = "MIN_TRUST_RADIUS_REACHED";
      break;
    }
  }

  if (result.status.empty()) {
    result.status =
        result.converged
          ? "SLP_CONVERGED"
          : "SLP_ITERATION_LIMIT";
  }

  result.x = x;
  result.objective = true_objective(model, x);
  result.trust_radius = radius;
  result.feasible = nonlinear_feasible(
      model,
      x,
      options.lp_options.tolerance);

  return result;
}


GlobalPoolingResult solve_global_pooling(
    const PoolingModel& model,
    BharatOptSolverCore& solver,
    const GlobalPoolingOptions& options) {
  validate_model(model);
  if (options.max_nodes == 0) {
    throw std::invalid_argument("Global pooling max_nodes must be positive");
  }
  if (options.time_limit_sec <= 0.0) {
    throw std::invalid_argument("Global pooling time_limit_sec must be positive");
  }

  GlobalPoolingResult result;
  const auto started = std::chrono::steady_clock::now();

  struct Node {
    PoolingModel model;
    double bound{-INF};
    std::size_t depth{0};
  };

  auto gap_value = [&](double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return INF;
    return std::max(0.0, upper - lower);
  };

  // The current engine is a maximization-oriented refinery planner.
  // For minimization, retain the root relaxation result but do not claim
  // global certification from this branch-and-bound implementation.
  if (!model.linear.maximize) {
    auto slp = solve_pooling_slp(model, solver, options.slp_options);
    auto mc = solve_mccormick_relaxation(model, solver, options.relaxation_options);
    result.incumbent = slp;
    result.feasible = slp.feasible;
    result.objective = slp.objective;
    result.global_bound = mc.global_lower_bound;
    result.optimality_gap =
        std::isfinite(result.global_bound) && std::isfinite(result.objective)
          ? std::max(0.0, result.objective - result.global_bound)
          : INF;
    result.status = "MINIMIZATION_ROOT_RELAXATION_ONLY";
    return result;
  }

  auto root_slp = solve_pooling_slp(model, solver, options.slp_options);
  if (root_slp.feasible && std::isfinite(root_slp.objective)) {
    result.feasible = true;
    result.objective = root_slp.objective;
    result.x = root_slp.x;
    result.incumbent = root_slp;
  }

  auto root_mc = solve_mccormick_relaxation(
      model, solver, options.relaxation_options);
  if (!root_mc.has_global_upper_bound) {
    result.status = "ROOT_RELAXATION_FAILED";
    return result;
  }

  std::vector<Node> pending;
  pending.push_back({model, root_mc.global_upper_bound, 0});

  while (!pending.empty() &&
         result.nodes_explored < options.max_nodes) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - started).count();
    if (elapsed >= options.time_limit_sec) break;

    // Best-bound selection: process the node with the largest upper bound.
    auto best_it = std::max_element(
        pending.begin(), pending.end(),
        [](const Node& a, const Node& b) { return a.bound < b.bound; });
    Node node = std::move(*best_it);
    pending.erase(best_it);
    ++result.nodes_explored;

    const double required_gap =
        options.absolute_gap +
        options.relative_gap * (1.0 + std::abs(result.objective));

    if (result.feasible &&
        node.bound <= result.objective + required_gap) {
      ++result.nodes_pruned;
      continue;
    }

    auto slp = solve_pooling_slp(node.model, solver, options.slp_options);
    if (slp.feasible && std::isfinite(slp.objective) &&
        (!result.feasible || slp.objective > result.objective)) {
      result.feasible = true;
      result.objective = slp.objective;
      result.x = slp.x;
      result.incumbent = slp;
    }

    auto mc = solve_mccormick_relaxation(
        node.model, solver, options.relaxation_options);
    if (!mc.has_global_upper_bound) {
      ++result.nodes_pruned;
      continue;
    }

    const double upper = mc.global_upper_bound;
    const double incumbent_gap =
        result.feasible ? gap_value(upper, result.objective) : INF;

    const double current_required_gap =
        options.absolute_gap +
        options.relative_gap *
          (1.0 + (result.feasible ? std::abs(result.objective) : 0.0));

    if (result.feasible && incumbent_gap <= current_required_gap) {
      ++result.nodes_pruned;
      continue;
    }

    // Find the widest/loosest bilinear term and split its wider variable.
    int split_var = -1;
    double split_score = -1.0;
    auto inspect_term = [&](const BilinearTerm& term) {
      const double lx = node.model.linear.lower[static_cast<std::size_t>(term.left)];
      const double ux = node.model.linear.upper[static_cast<std::size_t>(term.left)];
      const double ly = node.model.linear.lower[static_cast<std::size_t>(term.right)];
      const double uy = node.model.linear.upper[static_cast<std::size_t>(term.right)];
      const double width_x = ux - lx;
      const double width_y = uy - ly;
      const double score = width_x * width_y;
      if (score > split_score) {
        split_score = score;
        split_var = width_x >= width_y ? term.left : term.right;
      }
    };

    for (const auto& term : node.model.objective_bilinear) inspect_term(term);
    for (const auto& row : node.model.constraint_bilinear)
      for (const auto& term : row) inspect_term(term);

    if (split_var < 0 || split_score <= 1e-14) {
      ++result.nodes_pruned;
      continue;
    }

    const std::size_t j = static_cast<std::size_t>(split_var);
    const double lo = node.model.linear.lower[j];
    const double hi = node.model.linear.upper[j];
    const double mid = 0.5 * (lo + hi);
    if (!(mid > lo && mid < hi)) {
      ++result.nodes_pruned;
      continue;
    }

    PoolingModel left = node.model;
    PoolingModel right = node.model;
    left.linear.upper[j] = mid;
    right.linear.lower[j] = mid;

    auto push_child = [&](PoolingModel child) {
      auto child_mc = solve_mccormick_relaxation(
          child, solver, options.relaxation_options);
      if (!child_mc.has_global_upper_bound) return;
      if (result.feasible) {
        const double required =
            options.absolute_gap +
            options.relative_gap * (1.0 + std::abs(result.objective));
        if (child_mc.global_upper_bound <= result.objective + required) {
          ++result.nodes_pruned;
          return;
        }
      }
      pending.push_back({
          std::move(child),
          child_mc.global_upper_bound,
          node.depth + 1});
    };

    push_child(std::move(left));
    push_child(std::move(right));
  }

  double remaining_upper = -INF;
  for (const auto& node : pending)
    remaining_upper = std::max(remaining_upper, node.bound);

  result.global_bound = pending.empty()
      ? (result.feasible ? result.objective : root_mc.global_upper_bound)
      : remaining_upper;

  if (!result.feasible) {
    result.status = "NO_FEASIBLE_SOLUTION_FOUND";
    result.optimality_gap = INF;
    return result;
  }

  result.optimality_gap =
      std::max(0.0, result.global_bound - result.objective);

  const double tolerance =
      options.absolute_gap +
      options.relative_gap * (1.0 + std::abs(result.objective));

  result.certified =
      result.optimality_gap <= tolerance &&
      pending.empty();

  if (result.certified) {
    result.status = "GLOBAL_OPTIMAL_WITHIN_TOLERANCE";
  } else if (pending.empty()) {
    result.status = "GLOBAL_SEARCH_EXHAUSTED";
  } else {
    result.status = "GLOBAL_SEARCH_LIMIT_REACHED";
  }

  return result;
}

} // namespace bharatopt
