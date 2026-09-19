#include "bharatopt/mps.hpp"
#include "bharatopt/solver.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){auto m=bharatopt::make_refinery_demo();bharatopt::BharatOptSolverCore s;bharatopt::SolverOptions o;o.max_iterations=20000;o.tolerance=1e-4;auto r=s.solve(m,o);assert(r.converged);assert(std::isfinite(r.objective));assert(r.primal_residual<=1e-4);std::cout<<"solver test passed\n";}