#include "bharatopt/cuda_backend.hpp"

namespace bharatopt::cuda_backend {

bool cuda_available(){ return false; }
std::string cuda_device_name(){ return "disabled"; }

struct CudaPdhgSolver::Impl {};
CudaPdhgSolver::CudaPdhgSolver(): impl_(new Impl{}) {}
CudaPdhgSolver::~CudaPdhgSolver(){ delete impl_; }
bool CudaPdhgSolver::solve(const LPModel&, const SolverOptions&, SolverResult&){ return false; }

}