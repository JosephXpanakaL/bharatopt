#include "bharatopt/mps.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
  bharatopt::BharatOptSolverCore s;
  bharatopt::SolverOptions o;o.max_iterations=30000;o.tolerance=1e-4;

  auto lp=bharatopt::make_refinery_demo();auto r=s.solve_lp(lp,o);
  assert(r.converged);assert(std::isfinite(r.objective));assert(r.primal_residual<=1e-4);
  assert(std::abs(r.objective-72.0)<0.2);

  auto mip=bharatopt::make_milp_demo();o.max_nodes=128;o.mip_gap=1e-3;
  auto mr=s.solve_milp(mip,o);
  assert(mr.converged);assert(!mr.x.empty());assert(std::isfinite(mr.objective));assert(std::abs(mr.objective+13.0)<1e-3);

  auto qp=bharatopt::make_qp_demo();auto qr=s.solve_qp(qp,o);
  assert(qr.converged);assert(std::isfinite(qr.objective));assert(qr.primal_residual<=1e-4);
  assert(std::abs(qr.objective+9.5)<2e-2);

  std::cout<<"LP, MILP and QP tests passed\n";
  return 0;
}
