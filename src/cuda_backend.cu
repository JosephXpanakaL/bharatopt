#include "bharatopt/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace bharatopt::cuda_backend {
namespace {
inline void ck(cudaError_t e,const char* w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
inline void cb(cublasStatus_t e,const char* w){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(w)+": cuBLAS error");}
inline void cs(cusparseStatus_t e,const char* w){if(e!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(std::string(w)+": cuSPARSE error");}

__global__ void dual_update(int m,double sigma,const double* ax,const double* y,const double* lo,const double* hi,double* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
  double q=y[i]+sigma*ax[i];
  double p=fmin(fmax(q/sigma,lo[i]),hi[i]);
  out[i]=q-sigma*p;
}
__global__ void primal_update(int n,double tau,const double* x,const double* grad,const double* lo,const double* hi,double* out){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  double z=x[j]-tau*grad[j];
  out[j]=fmin(fmax(z,lo[j]),hi[j]);
}
__global__ void extrapolate(int n,double theta,const double* x,const double* xprev,double* xbar){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)xbar[j]=x[j]+theta*(x[j]-xprev[j]);
}
__global__ void add_gradient(int n,const double* aty,const double* c,double* grad){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)grad[j]=aty[j]+c[j];
}
__global__ void projected_stationarity(int n,const double* x,const double* grad,const double* lo,const double* hi,double* out){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
  double p=fmin(fmax(x[j]-grad[j],lo[j]),hi[j]);out[j]=x[j]-p;
}
__global__ void violation(int m,const double* ax,const double* lo,const double* hi,double* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
  double v=ax[i],w=0.0;if(v<lo[i])w=lo[i]-v;else if(v>hi[i])w=v-hi[i];out[i]=w;
}

struct DeviceVec{
  double* p{nullptr};std::size_t n{0};
  ~DeviceVec(){if(p)cudaFree(p);}
  void alloc(std::size_t s){n=s;ck(cudaMalloc(&p,n*sizeof(double)),"cudaMalloc vector");}
};
double dual_lower_bound_host(const LPModel&m,const std::vector<double>&y){
  double support=0.0;
  for(std::size_t i=0;i<m.A.rows;i++){
    double yi=y[i];
    if(yi>0){if(!std::isfinite(m.row_upper[i]))return -INF;support+=m.row_upper[i]*yi;}
    else if(yi<0){if(!std::isfinite(m.row_lower[i]))return -INF;support+=m.row_lower[i]*yi;}
  }
  std::vector<double>aty(m.A.cols,0.0);
  for(std::size_t i=0;i<m.A.rows;i++)for(int k=m.A.row_ptr[i];k<m.A.row_ptr[i+1];k++)aty[m.A.col_index[k]]+=m.A.values[k]*y[i];
  double value=-support;
  for(std::size_t j=0;j<m.A.cols;j++){
    double g=m.objective[j]+aty[j];
    if(g>=0){if(!std::isfinite(m.lower[j]))return -INF;value+=g*m.lower[j];}
    else {if(!std::isfinite(m.upper[j]))return -INF;value+=g*m.upper[j];}
  }
  return value;
}

struct DeviceCsr{
  double* val{nullptr};int* col{nullptr};int* row{nullptr};
  ~DeviceCsr(){if(val)cudaFree(val);if(col)cudaFree(col);if(row)cudaFree(row);}
};

struct CudaPdhgSolver::Impl{
  cublasHandle_t blas{nullptr};
  cusparseHandle_t sparse{nullptr};
  cusparseSpMatDescr_t A{nullptr};
  cusparseDnVecDescr_t xbar_vec{nullptr},ax_vec{nullptr},y_vec{nullptr},aty_vec{nullptr};
  void* buffer{nullptr};std::size_t buffer_size{0};
  DeviceCsr csr;
  DeviceVec c,lo,hi,rlo,rhi,x,xprev,xbar,y,ax,aty,grad,res;
  ~Impl(){
    if(xbar_vec)cusparseDestroyDnVec(xbar_vec);
    if(ax_vec)cusparseDestroyDnVec(ax_vec);
    if(y_vec)cusparseDestroyDnVec(y_vec);
    if(aty_vec)cusparseDestroyDnVec(aty_vec);
    if(A)cusparseDestroySpMat(A);
    if(buffer)cudaFree(buffer);
    if(sparse)cusparseDestroy(sparse);
    if(blas)cublasDestroy(blas);
  }
  void alloc(DeviceVec&v,std::size_t n){v.alloc(n);}
  void upload(const DeviceVec&v,const std::vector<double>&h){if(v.n!=h.size())throw std::runtime_error("GPU vector length mismatch");ck(cudaMemcpy(v.p,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice),"cudaMemcpy H2D");}
  void download(const DeviceVec&v,std::vector<double>&h){h.resize(v.n);ck(cudaMemcpy(h.data(),v.p,v.n*sizeof(double),cudaMemcpyDeviceToHost),"cudaMemcpy D2H");}
};

}
bool cuda_available(){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}
std::string cuda_device_name(){if(!cuda_available())return "none";cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return "unknown";return p.name;}

CudaPdhgSolver::CudaPdhgSolver():impl_(new Impl{}){}
CudaPdhgSolver::~CudaPdhgSolver(){delete impl_;}

bool CudaPdhgSolver::solve(const LPModel&m,const SolverOptions&o,SolverResult&r){
 if(!cuda_available())return false;
 if(m.A.rows>static_cast<std::size_t>(std::numeric_limits<int>::max())||m.A.cols>static_cast<std::size_t>(std::numeric_limits<int>::max())||m.A.values.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))return false;
 try{
  auto&g=*impl_;r=SolverResult{};r.backend="CUDA-resident-PDHG";
  const int M=(int)m.A.rows,N=(int)m.A.cols,NNZ=(int)m.A.values.size();
  std::size_t free_b=0,total_b=0;ck(cudaMemGetInfo(&free_b,&total_b),"cudaMemGetInfo");
  std::size_t estimate=NNZ*(sizeof(double)+sizeof(int))+std::size_t(M+1)*sizeof(int)+std::size_t(5*N+5*M)*sizeof(double);
  if(estimate>free_b*7/10)return false;

  cb(cublasCreate(&g.blas),"cublasCreate");cs(cusparseCreate(&g.sparse),"cusparseCreate");
  g.csr.val=nullptr;ck(cudaMalloc(&g.csr.val,NNZ*sizeof(double)),"values");ck(cudaMalloc(&g.csr.col,NNZ*sizeof(int)),"columns");ck(cudaMalloc(&g.csr.row,(M+1)*sizeof(int)),"rowptr");
  ck(cudaMemcpy(g.csr.val,m.A.values.data(),NNZ*sizeof(double),cudaMemcpyHostToDevice),"values H2D");
  ck(cudaMemcpy(g.csr.col,m.A.col_index.data(),NNZ*sizeof(int),cudaMemcpyHostToDevice),"columns H2D");
  ck(cudaMemcpy(g.csr.row,m.A.row_ptr.data(),(M+1)*sizeof(int),cudaMemcpyHostToDevice),"rowptr H2D");
  cs(cusparseCreateCsr(&g.A,M,N,NNZ,g.csr.row,g.csr.col,g.csr.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F),"cusparseCreateCsr");

  g.alloc(g.c,N);g.alloc(g.lo,N);g.alloc(g.hi,N);g.alloc(g.rlo,M);g.alloc(g.rhi,M);g.alloc(g.x,N);g.alloc(g.xprev,N);g.alloc(g.xbar,N);g.alloc(g.y,M);g.alloc(g.ax,M);g.alloc(g.aty,N);g.alloc(g.grad,N);g.alloc(g.res,std::max(M,N));
  g.upload(g.c,m.objective);g.upload(g.lo,m.lower);g.upload(g.hi,m.upper);g.upload(g.rlo,m.row_lower);g.upload(g.rhi,m.row_upper);

  std::vector<double>x0(N,0.0);for(int j=0;j<N;j++)x0[j]=std::min(std::max(0.0,m.lower[j]),m.upper[j]);
  g.upload(g.x,x0);g.upload(g.xprev,x0);g.upload(g.xbar,x0);std::vector<double>y0(M,0.0);g.upload(g.y,y0);

  double tau=o.tau,sigma=o.sigma;
  double Lbound=0.0;for(int i=0;i<M;i++){double rs=0;for(int k=m.A.row_ptr[i];k<m.A.row_ptr[i+1];k++)rs+=std::abs(m.A.values[k]);Lbound=std::max(Lbound,rs);}
  double L=std::max(1.0,Lbound);if(tau*sigma*L*L>=0.8){double s=std::sqrt(0.7/(tau*sigma*L*L));tau*=s;sigma*=s;}

  cs(cusparseCreateDnVec(&g.xbar_vec,N,g.xbar.p,CUDA_R_64F),"xbar desc");
  cs(cusparseCreateDnVec(&g.ax_vec,M,g.ax.p,CUDA_R_64F),"ax desc");
  cs(cusparseCreateDnVec(&g.y_vec,M,g.y.p,CUDA_R_64F),"y desc");
  cs(cusparseCreateDnVec(&g.aty_vec,N,g.aty.p,CUDA_R_64F),"aty desc");
  double alpha=1.0,beta=0.0;
  std::size_t b0=0,b1=0;
  cs(cusparseSpMV_bufferSize(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,g.A,g.xbar_vec,&beta,g.ax_vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,&b0),"SpMV NN buffer");
  cs(cusparseSpMV_bufferSize(g.sparse,CUSPARSE_OPERATION_TRANSPOSE,&alpha,g.A,g.y_vec,&beta,g.aty_vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,&b1),"SpMV T buffer");
  g.buffer_size=std::max(b0,b1);
  if(g.buffer_size>0)ck(cudaMalloc(&g.buffer,g.buffer_size),"SpMV buffer");

  auto nrm2=[&](const DeviceVec&v,int n){double z=0;cb(cublasDnrm2(g.blas,n,v.p,1,&z),"Dnrm2");return z;};
  auto t0=std::chrono::steady_clock::now();

  for(int it=1;it<=o.max_iterations;it++){
    // Axbar
    cs(cusparseDnVecSetValues(g.xbar_vec,g.xbar.p),"set xbar");
    cs(cusparseDnVecSetValues(g.ax_vec,g.ax.p),"set ax");
    cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,g.A,g.xbar_vec,&beta,g.ax_vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"Axbar");

    dual_update<<<(M+255)/256,256>>>(M,sigma,g.ax.p,g.y.p,g.rlo.p,g.rhi.p,g.y.p);ck(cudaGetLastError(),"dual update");

    // ATy
    cs(cusparseDnVecSetValues(g.y_vec,g.y.p),"set y");
    cs(cusparseDnVecSetValues(g.aty_vec,g.aty.p),"set aty");
    cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_TRANSPOSE,&alpha,g.A,g.y_vec,&beta,g.aty_vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"ATy");

    add_gradient<<<(N+255)/256,256>>>(N,g.aty.p,g.c.p,g.grad.p);ck(cudaGetLastError(),"gradient");
    primal_update<<<(N+255)/256,256>>>(N,tau,g.x.p,g.grad.p,g.lo.p,g.hi.p,g.xprev.p);ck(cudaGetLastError(),"primal");
    std::swap(g.x.p,g.xprev.p);
    extrapolate<<<(N+255)/256,256>>>(N,o.theta,g.x.p,g.xprev.p,g.xbar.p);ck(cudaGetLastError(),"extrapolate");

    // Ax at current x
    cs(cusparseDnVecSetValues(g.xbar_vec,g.x.p),"set x");
    cs(cusparseDnVecSetValues(g.ax_vec,g.ax.p),"set ax");
    cs(cusparseSpMV(g.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,g.A,g.xbar_vec,&beta,g.ax_vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,g.buffer),"Ax");

    double normax=nrm2(g.ax,M);
    violation<<<(M+255)/256,256>>>(M,g.ax.p,g.rlo.p,g.rhi.p,g.res.p);ck(cudaGetLastError(),"violation");
    double normv=nrm2(g.res,M);r.primal_residual=normv/(1.0+normax);

    projected_stationarity<<<(N+255)/256,256>>>(N,g.x.p,g.grad.p,g.lo.p,g.hi.p,g.res.p);ck(cudaGetLastError(),"stationarity");
    double ns=nrm2(g.res,N),nc=nrm2(g.c,N);r.dual_residual=ns/(1.0+nc);
    double obj=0;cb(cublasDdot(g.blas,N,g.c.p,1,g.x.p,1,&obj),"objective");r.objective=obj;r.iterations=it;
    if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}
  }
  g.download(g.x,r.x);std::vector<double>host_y;g.download(g.y,host_y);double db=dual_lower_bound_host(m,host_y);r.dual_bound=db;r.best_bound=m.maximize?(-db+m.objective_offset):(db+m.objective_offset);r.mip_gap=0;r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();r.status=r.converged?"OPTIMALITY_TOL_REACHED":"ITERATION_LIMIT";return true;
 }catch(...){return false;}
}

}
