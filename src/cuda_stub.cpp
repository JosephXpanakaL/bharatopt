#include "bharatopt/cuda_backend.hpp"
namespace bharatopt::cuda_backend {
bool cuda_available(){return false;}
std::string cuda_device_name(){return "disabled";}
bool spmv(const SparseMatrixCSR&,const std::vector<double>&,std::vector<double>&){return false;}
bool project_box(std::vector<double>&,const std::vector<double>&,const std::vector<double>&){return false;}
}
