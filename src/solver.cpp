#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
namespace bharatopt { namespace {
double n2(const std::vector<double>&v){long double s=0;for(double a:v)s+=(long double)a*a;return std::sqrt((double)s);}
void proj(std::vector<double>&v,const std::vector<double>&lo,const std::vector<double>&hi){for(size_t i=0;i<v.size();++i)v[i]=std::min(std::max(v[i],lo[i]),hi[i]);}
void Ax(const SparseMatrixCSR&A,const std::vector<double>&x,std::vector<double>&y){y.assign(A.rows,0);for(size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)y[i]+=A.values[k]*x[A.col_index[k]];}
void ATy(const SparseMatrixCSR&A,const std::vector<double>&y,std::vector<double>&x){x.assign(A.cols,0);for(size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)x[A.col_index[k]]+=A.values[k]*y[i];}
double opnorm(const SparseMatrixCSR&A){if(A.cols==0)return 0;std::vector<double>x(A.cols,1),y,z;for(int it=0;it<12;it++){Ax(A,x,y);ATy(A,y,z);double n=n2(z);if(n==0)return 0;for(double&v:z)v/=n;x=z;}Ax(A,x,y);return n2(y)/std::max(n2(x),1e-30);}
bool finite_solution(const SolverResult&r){return !r.x.empty()&&std::isfinite(r.objective)&&std::isfinite(r.primal_residual);}
bool integer_feasible(const LPModel&m,const std::vector<double>&x,double tol,int&branch){branch=-1;for(size_t i=0;i<m.integer.size();++i)if(m.integer[i]){double nearest=std::round(x[i]);if(std::abs(x[i]-nearest)>tol){branch=(int)i;return false;}}return true;}
}
SolverResult BharatOptSolverCore::solve_lp(const LPModel&m,const SolverOptions&o){
  if(m.A.cols!=m.objective.size()||m.A.rows!=m.row_lower.size()||m.A.rows!=m.row_upper.size()||m.lower.size()!=m.A.cols||m.upper.size()!=m.A.cols)throw std::runtime_error("Inconsistent model dimensions");
#ifdef BHARATOPT_CUDA_ENABLED
  if(o.use_cuda){SolverResult gpu;if(cuda_backend::CudaPdhgSolver().solve(m,o,gpu))return gpu;}
#endif
  int n=(int)m.A.cols,rc=(int)m.A.rows;SolverResult r;r.x.assign(n,0);proj(r.x,m.lower,m.upper);
  std::vector<double>xbar=r.x,xprev=r.x,y(rc),a,aty;double L=opnorm(m.A);if(L<1e-12)L=1;double tau=o.tau,sigma=o.sigma;
  if(tau*sigma*L*L>=.95){double s=std::sqrt(.9/(tau*sigma*L*L));tau*=s;sigma*=s;}
  auto t0=std::chrono::steady_clock::now();r.backend="CPU-PDHG";
  for(int it=1;it<=o.max_iterations;it++){
    Ax(m.A,xbar,a);
    for(int i=0;i<rc;i++){double q=y[i]+sigma*a[i];double p=std::min(std::max(q/sigma,m.row_lower[i]),m.row_upper[i]);y[i]=q-sigma*p;}
    ATy(m.A,y,aty);
    xprev=r.x;
    for(int j=0;j<n;j++)r.x[j]-=tau*(m.objective[j]+aty[j]);
    proj(r.x,m.lower,m.upper);
    for(int j=0;j<n;j++)xbar[j]=r.x[j]+o.theta*(r.x[j]-xprev[j]);
    Ax(m.A,r.x,a);
    double ps=0;for(int i=0;i<rc;i++){double v=a[i],w=0;if(v<m.row_lower[i])w=m.row_lower[i]-v;else if(v>m.row_upper[i])w=v-model.row_upper[i];ps+=w*w;}r.primal_residual=std::sqrt(ps)/(1+n2(a));
    double ds=0;for(int j=0;j<n;j++){double z=r.x[j]-(m.objective[j]+aty[j]);double p=std::min(std::max(z,m.lower[j]),m.upper[j]);double d=r.x[j]-p;ds+=d*d;}r.dual_residual=std::sqrt(ds)/(1+n2(m.objective));r.iterations=it;
    if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}
  }
  for(int j=0;j<n;j++)r.objective+=m.objective[j]*r.x[j];r.best_bound=r.objective;r.mip_gap=0;
  r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();r.status=r.converged?"OPTIMALITY_TOL_REACHED":"ITERATION_LIMIT";return r;
}
SolverResult BharatOptSolverCore::solve_milp(const LPModel&m,const SolverOptions&o){
 auto t0=std::chrono::steady_clock::now();SolverResult out;out.backend="CPU-branch-and-bound";out.x.assign(m.A.cols,0);double incumbent=INF,best_bound=INF;size_t nodes=0;bool hit_limit=false;
 std::function<void(const LPModel&)>dfs=[&](const LPModel&node){
  if((int)nodes>=o.max_nodes){hit_limit=true;return;}++nodes;
  SolverOptions lp=o;lp.use_cuda=false;lp.max_iterations=std::min(o.max_iterations,10000);lp.tolerance=std::max(o.tolerance,1e-5);
  SolverResult rr=solve_lp(node,lp);if(!finite_solution(rr)||!rr.converged)return;
  best_bound=std::min(best_bound,rr.objective);if(rr.objective>=incumbent-1e-9)return;
  int branch=-1;if(integer_feasible(node,rr.x,o.integrality_tolerance,branch)){if(rr.objective<incumbent){incumbent=rr.objective;out.x=rr.x;}return;}
  if(branch<0)return;double v=rr.x[branch],fl=std::floor(v),ce=std::ceil(v);
  if(fl>=node.lower[branch]){LPModel left=node;left.upper[branch]=std::min(left.upper[branch],fl);if(left.lower[branch]<=left.upper[branch])dfs(left);}
  if(ce<=node.upper[branch]){LPModel right=node;right.lower[branch]=std::max(right.lower[branch],ce);if(right.lower[branch]<=right.upper[branch])dfs(right);}
 };
 dfs(m);out.nodes=nodes;out.iterations=(int)nodes;out.best_bound=best_bound;out.objective=incumbent;
 if(std::isfinite(incumbent)){double denom=1+std::abs(incumbent);out.mip_gap=std::isfinite(best_bound)?std::max(0.0,(incumbent-best_bound)/denom):INF;out.converged=!hit_limit&&out.mip_gap<=o.mip_gap;out.status=out.converged?"MIP_OPTIMALITY_GAP_REACHED":(hit_limit?"MIP_NODE_LIMIT":"MIP_INCUMBENT_FOUND");}
 else out.status=hit_limit?"MIP_NODE_LIMIT_NO_INCUMBENT":"MIP_NO_FEASIBLE_INCUMBENT";
 out.primal_residual=0;out.dual_residual=0;out.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return out;
}
}