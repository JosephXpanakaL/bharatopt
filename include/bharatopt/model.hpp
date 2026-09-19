#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
namespace bharatopt {
constexpr double INF = std::numeric_limits<double>::infinity();
enum class RowSense { LessEqual, GreaterEqual, Equal };
struct SparseMatrixCSR { std::size_t rows{0},cols{0}; std::vector<double> values; std::vector<int> col_index,row_ptr; };
struct ConstraintRow { std::string name; RowSense sense{RowSense::Equal}; double rhs{0.0}; };
struct LPModel {
  std::string name;
  std::vector<std::string> var_names;
  std::vector<double> objective,lower,upper;
  bool maximize{false};
  double objective_offset{0.0};
  std::vector<std::uint8_t> integer;
  std::vector<ConstraintRow> rows;
  SparseMatrixCSR A;
  std::vector<double> row_lower,row_upper;
  bool has_integer_variables() const { for(auto v:integer) if(v)return true; return false; }
};
struct QPModel {
  std::string name;
  LPModel linear;
  SparseMatrixCSR Q;
  bool convex_psd{true};
};
struct SolverOptions {
  int max_iterations{50000};
  double tolerance{1e-6};
  double tau{0.5},sigma{0.5},theta{1.0};
  bool use_cuda{false},verbose{false};
  int max_nodes{256};
  double mip_gap{1e-4};
  double integrality_tolerance{1e-6};
  int scaling_passes{5};
  bool presolve{true};
  double time_limit_sec{0.0};
};
struct SolverResult {
  bool converged{false};
  int iterations{0};
  std::size_t nodes{0};
  double objective{0.0},best_bound{INF},mip_gap{INF};
  double primal_residual{INF},dual_residual{INF},solve_time_sec{0.0};
  std::vector<double> x;
  std::string status,backend;
};
}