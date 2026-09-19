#pragma once
#include <string>
#include <vector>
#include "bharatopt/model.hpp"

namespace bharatopt::cuda_backend {

bool cuda_available();
std::string cuda_device_name();

class CudaPdhgSolver {
public:
  CudaPdhgSolver();
  ~CudaPdhgSolver();
  CudaPdhgSolver(const CudaPdhgSolver&) = delete;
  CudaPdhgSolver& operator=(const CudaPdhgSolver&) = delete;

  bool solve(const LPModel& model, const SolverOptions& options, SolverResult& result);

private:
  struct Impl;
  Impl* impl_;
};

}