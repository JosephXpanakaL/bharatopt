#include "bharatopt/mps.hpp"
#include "bharatopt/solver.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 auto lp=bharatopt::make_refinery_demo(); bharatopt::BharatOptSolverCore s; bharatopt::SolverOptions o;o.max_iterations=30000;o.tolerance=1e-4;
 auto r=s.solve_lp(lp,o);assert(r.converged);assert(std::isfinite(r.objective));assert(r.primal_residual<=1e-4);
 auto mip=bharatopt::make_milp_demo();o.max_nodes=64;o.mip_gap=1e-3;o.tolerance=1e-4;
 auto mr=s.solve_milp(mip,o);assert(!mr.x.empty());assert(std::isfinite(mr.objective));assert(mr.nodes>0);
 std::cout<<"LP and MILP tests passed\n";return 0;
}