#include "bharatopt/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace bharatopt::cuda_backend {
namespace {

inline void ck(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
inline void cb(cublasStatus_t e,const char*where){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(where)+": cuBLAS error");}
inline void cs(cusparseStatus_t e,const char*where){if(e!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(std::string(where)+": cuSPARSE error");}

__global__ void dual_update(int m,double sigma,const double* ax,const double* y,const double* lo,const double* hi,double* yout){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
  double q=y[i]+sigma*ax[i];
  double z=q/sigma;
  double p=fmin(fmax(z,lo[i]),hi[i]);
  yout[i]=q-sigma*p;
}
__global__ void primal_update(int n,double tau,const double* x,const double* aty,const double* c,const double* lo,const double* hi,double* xout){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  double z=x[j]-tau*(c[j]+aty[j]);
  xout[j]=fmin(fmax(z,lo[j]),hi[j]);
}
__global__ void extrapolate(int n,double theta,const double* x,const double* xprev,double* xbar){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  xbar[j]=x[j]+theta*(x[j]-xprev[j]);
}
__global__ void projected_stationarity(int n,const double* x,const double* grad,const double* lo,const double* hi,double* out){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  double z=x[j]-grad[j];
  double p=fmin(fmax(z,lo[j]),hi[j]);
  out[j]=x[j]-p;
}
__global__ void feasibility_violation(int m,const double* ax,const double* lo,const double* hi,double* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
  double v=ax[i],w=0.0;
  if(v<lo[i])w=lo[i]-v; else if(v>hi[i])w=v-hi[i];
  out[i]=w;
}
__global__ void gradient_add_q(int n,const double* qx,const double* aty,const double* c,double* grad){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  grad[j]=qx[j]+aty[j]+c[j];
}

struct CsrDevice {
  int m{0},n{0},nnz{0};
  double *val{nullptr};
  int *col{nullptr},*row{nullptr};
  ~CsrDevice(){if(val)cudaFree(val);if(col)cudaFree(col);if(row)cudaFree(row);}
};

struct Vec {
  double* p{nullptr};std::size_t n{0};
  ~Vec(){if(p)cudaFree(p);}
};

}

struct CudaPdhgSolver::Impl {
  cublasHandle_t blas{nullptr};
  cusparseHandle_t sparse{nullptr};
  cusparseSpMatDescr_t A{nullptr};
  cusparseDnVecDescr_t dx{nullptr},dy{nullptr};
  void* buffer{nullptr};
  std::size_t buffer_size{0};
  CsrDevice csr;
  Vec d_c,d_lo,d_hi,d_rlo,d_rhi,d_x,d_xprev,d_xbar,d_y,d_ax,d_aty,d_grad,d_qx,d_res;
  ~Impl(){
    if(dx)cusparseDestroyDnVec(dx);if(dy)cusparseDestroyDnVec(dy);if(A)cusparseDestroySpMat(A);
    if(buffer)cudaFree(buffer);if(sparse)cusparseDestroy(sparse);if(blas)cublasDestroy(blas);
  }
  void alloc(Vec&v,std::size_t n){v.n=n;ck(cudaMalloc(&v.p,n*sizeof(double)),"cudaMalloc");}
  void upload(const Vec&v,const std::vector<double>&h){if(v.n!=h.size())throw std::runtime_error("GPU vector size mismatch");ck(cudaMemcpy(v.p,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice),"H2D");}
  void download(const Vec&v,std::vector<double>&h){h.resize(v.n);ck(cudaMemcpy(h.data(),v.p,v.n*sizeof(double),cudaMemcpyDeviceToHost),"D2H");}
  void spmv(bool trans,const Vec&x,Vec&y){
    cusparseDnVecDescr_t* yin=&dy; if(trans){} // descriptor objects are rebound below
    (void)yin;
  }
};

bool cuda_available(){
  int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;
}
std::string cuda_device_name(){
  if(!cuda_available())return "none";
  cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return "unknown";return p.name;
}

CudaPdhgSolver::CudaPdhgSolver():impl_(new Impl{}){}
CudaPdhgSolver::~CudaPdhgSolver(){delete impl_;}

bool CudaPdhgSolver::solve(const LPModel&m,const SolverOptions&o,SolverResult&r){
  if(!cuda_available())return false;
  if(m.A.rows>static_cast<std::size_t>(std::numeric_limits<int>::max())||m.A.cols>static_cast<std::size_t>(std::numeric_limits<int>::max())||m.A.values.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))return false;
  try{
    auto&g=*impl_;r=SolverResult{};r.backend="CUDA-resident-PDHG";
    int rows=(int)m.A.rows,cols=(int)m.A.cols,nnz=(int)m.A.values.size();
    cudaMemGetInfo_t dummy; (void)dummy;
    std::size_t free_b=0,total_b=0;ck(cudaMemGetInfo(&free_b,&total_b),"cudaMemGetInfo");
    std::size_t estimate=nnz*(sizeof(double)+sizeof(int))+std::size_t(rows+1)*sizeof(int)+
      std::size_t(4*rows+5*cols)*sizeof(double);
    if(estimate>free_b*7/10)return false;

    cb(cublasCreate(&g.blas),"cublasCreate");cs(cusparseCreate(&g.sparse),"cusparseCreate");

    g.csr.m=rows;g.csr.n=cols;g.csr.nnz=nnz;
    ck(cudaMalloc(&g.csr.val,nnz*sizeof(double)),"cudaMalloc values");
    ck(cudaMalloc(&g.csr.col,nnz*sizeof(int)),"cudaMalloc col");
    ck(cudaMalloc(&g.csr.row,(rows+1)*sizeof(int)),"cudaMalloc row");
    ck(cudaMemcpy(g.csr.val,m.A.values.data(),nnz*sizeof(double),cudaMemcpyHostToDevice),"copy values");
    ck(cudaMemcpy(g.csr.col,m.A.col_index.data(),nnz*sizeof(int),cudaMemcpyHostToDevice),"copy col");
    ck(cudaMemcpy(g.csr.row,m.A.row_ptr.data(),(rows+1)*sizeof(int),cudaMemcpyHostToDevice),"copy row");

    cs(cusparseCreateCsr(&g.A,rows,cols,nnz,g.csr.row,g.csr.col,g.csr.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F),"cusparseCreateCsr");

    g.alloc(g.d_c,cols);g.alloc(g.d_lo,cols);g.alloc(g.d_hi,cols);g.alloc(g.d_rlo,rows);g.alloc(g.d_rhi,rows);
    g.alloc(g.d_x,cols);g.alloc(g.d_xprev,cols);g.alloc(g.d_xbar,cols);g.alloc(g.d_y,rows);g.alloc(g.d_ax,rows);g.alloc(g.d_aty,cols);
    g.alloc(g.d_grad,cols);g.alloc(g.d_qx,cols);g.alloc(g.d_res,std::max(rows,cols));
    g.upload(g.d_c,m.objective);g.upload(g.d_lo,m.lower);g.upload(g.d_hi,m.upper);g.upload(g.d_rlo,m.row_lower);g.upload(g.d_rhi,m.row_upper);

    std::vector<double>x0(cols,0.0),xbar0(cols,0.0);for(int j=0;j<cols;j++){x0[j]=std::min(std::max(0.0,m.lower[j]),m.upper[j]);xbar0[j]=x0[j];}
    std::vector<double>y0(rows,0.0);
    g.upload(g.d_x,x0);g.upload(g.d_xprev,x0);g.upload(g.d_xbar,xbar0);g.upload(g.d_y,y0);

    double L=1.0,tau=o.tau,sigma=o.sigma;
    // A CPU power estimate is deliberately avoided here: use a conservative bound from row/column coefficient sums.
    std::vector<double>row_sum(rows,0.0);for(int i=0;i<rows;i++)for(int k=m.A.row_ptr[i];k<m.A.row_ptr[i+1];k++)row_sum[i]+=std::abs(m.A.values[k]);
    double Lbound=0;for(double s:row_sum)Lbound=std::max(Lbound,s);
    L=std::max(1.0,Lbound);
    if(tau*sigma*L*L>=0.8){double scale=std::sqrt(0.7/(tau*sigma*L*L));tau*=scale;sigma*=scale;}

    cs(cusparseCreateDnVec(&g.dx,cols,g.d_xbar.p,CUDA_R_64F),"vec x");
    cs(cusparseCreateDnVec(&g.dy,rows,g.d_ax.p,CUDA_R_64F),"vec y");
    cs(cusparseSpMV_bufferSize(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&tau,g.A,g.dx,&tau,g.dy,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,&g.buffer_size),"SpMV buffer size");
    ck(cudaMalloc(&g.buffer,g.buffer_size),"SpMV buffer");

    std::vector<double>ones(cols,1.0),zero_cols(cols,0.0),zero_rows(rows,0.0);g.upload(g.d_aty,zero_cols);g.upload(g.d_ax,zero_rows);
    auto dot_norm=[&](const Vec&v,int n){double out=0;cb(cublasDnrm2(g.blas,n,v.p,1,&out),"Dnrm2");return out;};

    auto t0=std::chrono::steady_clock::now();
    std::vector<double>host_x;
    for(int it=1;it<=o.max_iterations;it++){
      cs(cusparseDnVecSetValues(g.dx,g.d_xbar.p),"set dx");cs(cusparseDnVecSetValues(g.dy,g.d_ax.p),"set dax");
      double one=1.0,minus=0.0;
      cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,g.A,g.dx,&minus,g.dy,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"SpMV Ax");
      dual_update<<<(rows+255)/256,256>>>(rows,sigma,g.d_ax.p,g.d_y.p,g.d_rlo.p,g.d_rhi.p,g.d_y.p);ck(cudaGetLastError(),"dual kernel");

      cs(cusparseDnVecSetValues(g.dx,g.d_y.p),"set y as input");
      cs(cusparseDnVecSetValues(g.dy,g.d_aty.p),"set aty"); // dy is now cols-sized: descriptor shape mismatch, recreate below
      if(g.dy){cusparseDestroyDnVec(g.dy);g.dy=nullptr;}
      cs(cusparseCreateDnVec(&g.dy,cols,g.d_aty.p,CUDA_R_64F),"vec aty out");
      cs(cusparseDnVecSetValues(g.dx,g.d_y.p),"y input");
      cs(cusparseDestroyDnVec(g.dx));g.dx=nullptr;
      cs(cusparseCreateDnVec(&g.dx,rows,g.d_y.p,CUDA_R_64F),"vec y in");
      // Transpose SpMV uses A^T * y -> aty.
      cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_TRANSPOSE,&one,g.A,g.dx,&minus,g.dy,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"SpMV ATy");
      cusparseDestroyDnVec(g.dx);g.dx=nullptr;
      cs(cusparseCreateDnVec(&g.dx,cols,g.d_x.p,CUDA_R_64F),"restore dx");

      // QP helper is not in the LP CUDA path; grad = c + A^T y.
      int block=(cols+255)/256;
      gradient_add_q<<<block,256>>>(cols,g.d_qx.p,g.d_aty.p,g.d_c.p,g.d_grad.p);
      ck(cudaGetLastError(),"gradient kernel");
      primal_update<<<block,256>>>(cols,tau,g.d_x.p,g.d_aty.p,g.d_c.p,g.d_lo.p,g.d_hi.p,g.d_xprev.p);
      ck(cudaGetLastError(),"primal kernel");
      // swap x and xprev by device pointer exchange; xprev becomes old x.
      std::swap(g.d_x.p,g.d_xprev.p);
      extrapolate<<<block,256>>>(cols,o.theta,g.d_x.p,g.d_xprev.p,g.d_xbar.p);
      ck(cudaGetLastError(),"extrapolate kernel");

      cs(cusparseDnVecSetValues(g.dx,g.d_x.p),"set dx residual");
      cs(cusparseDnVecSetValues(g.dy,g.d_ax.p),"set ax residual");
      cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,g.A,g.dx,&minus,g.dy,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"SpMV residual Ax");

      double normax=dot_norm(g.d_ax,rows);
      feasibility_violation<<<(rows+255)/256,256>>>(rows,g.d_ax.p,g.d_rlo.p,g.d_rhi.p,g.d_res.p);ck(cudaGetLastError(),"feasibility kernel");
      double normviol=dot_norm(g.d_res,rows);
      r.primal_residual=normviol/(1.0+normax);

      projected_stationarity<<<block,256>>>(cols,g.d_x.p,g.d_grad.p,g.d_lo.p,g.d_hi.p,g.d_res.p);ck(cudaGetLastError(),"stationarity kernel");
      double normstat=dot_norm(g.d_res,cols);double normc=dot_norm(g.d_c,cols);r.dual_residual=normstat/(1.0+normc);
      double obj=0;cb(cublasDdot(g.blas,cols,g.d_c.p,1,g.d_x.p,1,&obj),"objective");r.objective=obj;
      r.iterations=it;
      if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}
    }
    g.download(g.d_x,host_x);r.x=std::move(host_x);
    r.best_bound=r.objective;r.mip_gap=0;
    r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    r.status=r.converged?"OPTIMALITY_TOL_REACHED":"ITERATION_LIMIT";
    return true;
  }catch(...){
    return false;
  }
}

}
