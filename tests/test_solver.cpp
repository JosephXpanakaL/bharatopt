#include "bharatopt/mps.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
int main(){
  bharatopt::BharatOptSolverCore s;
  bharatopt::SolverOptions o;o.max_iterations=30000;o.tolerance=1e-4;

  auto lp=bharatopt::make_refinery_demo();auto r=s.solve_lp(lp,o);
  assert(r.converged);assert(std::isfinite(r.objective));assert(r.primal_residual<=1e-4);
  assert(std::abs(r.objective-72.0)<0.25);

  auto mip=bharatopt::make_milp_demo();o.max_nodes=128;o.mip_gap=1e-3;
  auto mr=s.solve_milp(mip,o);
  assert(mr.converged);assert(!mr.x.empty());assert(std::isfinite(mr.objective));assert(std::abs(mr.objective+13.0)<1e-3);

  auto qp=bharatopt::make_qp_demo();auto qr=s.solve_qp(qp,o);
  assert(qr.converged);assert(std::isfinite(qr.objective));assert(qr.primal_residual<=1e-4);
  assert(std::abs(qr.objective+9.5)<2e-2);

  const char* file="test_ranges_max.mps";
  {
    std::ofstream f(file);
    f<<"NAME TEST\nOBJSENSE\n MAX\nROWS\n N OBJ\n L C1\nCOLUMNS\n X OBJ 5 C1 1\nRHS\n RHS1 C1 5 OBJ 0\nRANGES\n RNG1 C1 1\nBOUNDS\n UP B1 X 10\nENDATA\n";
  }
  auto parsed=bharatopt::parse_mps(file);
  std::remove(file);
  assert(parsed.maximize);
  assert(parsed.rows.size()==1);
  assert(std::abs(parsed.row_lower[0]-4.0)<1e-12);
  assert(std::abs(parsed.row_upper[0]-5.0)<1e-12);

  std::cout<<"LP, MILP, QP and MPS parser tests passed\n";
  return 0;
}
