#pragma once
#include <string>
#include <vector>
#include "bharatopt/model.hpp"
namespace bharatopt::cuda_backend {
bool cuda_available();
std::string cuda_device_name();
bool spmv(const SparseMatrixCSR&, const std::vector<double>&, std::vector<double>&);
bool project_box(std::vector<double>&, const std::vector<double>&, const std::vector<double>&);
}