#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include "bharatopt/preprocess.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace bharatopt {
namespace {
double n2(const std::vector<double>&v){long double s=0;for(double a:v){if(!std::isfinite(a))return INF;s+=(long double)a*a;}return std::sqrt((double)s);}
void proj(std::vector<double>&v,const std::vector<double>&lo,const std::vector<double>&hi){for(std::size_t i=0;i<v.size();++i)v[i]=std::min(std::max(v[i],lo[i]),hi[i]);}
void Ax(const SparseMatrixCSR&A,const std::vector<double>&x,std::vector<double>&y){y.assign(A.rows,0.0);for(std::size_t i=0;i<A.rows;i++){double s=0;for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)s+=A.values[k]*x[A.col_index[k]];y[i]=s;}}
void ATy(const SparseMatrixCSR&A,const std::vector<double>&y,std::vector<double>&x){x.assign(A.cols,0.0);for(std::size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)x[A.col_index[k]]+=A.values[k]*y[i];}
double opnorm(const SparseMatrixCSR&A){if(A.cols==0)return 0;std::vector<double>x(A.cols,1.0),y,z;for(int it=0;it<12;it++){Ax(A,x,y);ATy(A,y,z);double q=n2(z);if(!std::isfinite(q)||q==0)return 0;for(double&v:z)v/=q;x=z;}Ax(A,x,y);return n2(y)/std::max(n2(x),1e-30);}
void certify_original(const LPModel&orig,SolverResult&r){std::vector<double>ax;Ax(orig,r.x,ax);double viol=0;for(std::size_t i=0;i<orig.A.rows;i++){double w=0; if(ax[i]<orig.row_lower[i])w=orig.row_lower[i]-ax[i];else if(ax[i]>orig.row_upper[i])w=ax[i]-orig.row_upper[i];viol+=w*w;}r.primal_residual=std::sqrt(viol)/(1.0+n2(ax));double internal=0;for(std::size_t j=0;j<r.x.size();j++)internal+=orig.objective[j]*r.x[j];r.objective=(orig.maximize?-internal:internal)+orig.objective_offset;r.best_bound=r.objective;}
bool finite_solution(const SolverResult&r){return !r.x.empty()&&std::isfinite(r.objective)&&std::isfinite(r.primal_residual)&&std::isfinite(r.dual_residual);}
bool integer_feasible(const LPModel&m,const std::vector<double>&x,double tol,int&branch){branch=-1;for(std::size_t i=0;i<m.integer.size();i++)if(m.integer[i]){double n=std::round(x[i]);if(std::abs(x[i]-n)>tol){branch=(int)i;return false;}}return true;}
}
SolverResult BharatOptSolverCore::solve_lp(const LPModel&input,const SolverOptions&o){
  if(input.A.cols!=input.objective.size()||input.A.rows!=input.row_lower.size()||input.A.rows!=input.row_upper.size()||input.lower.size()!=input.A.cols||input.upper.size()!=input.A.cols)throw std::runtime_error("Inconsistent model dimensions");
  auto pp=preprocess_lp(input,o.presolve?o.scaling_passes:0);
  if(!pp.feasible){SolverResult r;r.status="INFEASIBLE_PRESOLVE";r.backend="presolve";return r;}
  LPModel m=std::move(pp.model);
#ifdef BHARATOPT_CUDA_ENABLED
  if(o.use_cuda){SolverResult g;if(cuda_backend::CudaPdhgSolver().solve(m,o,g)){recover_primal(g.x,pp.column_scale,g.x);certify_original(input,g);return g;}}
#endif
  const int n=(int)m.A.cols,rc=(int)m.A.rows;
  SolverResult r;r.x.assign(n,0.0);proj(r.x,m.lower,m.upper);
  std::vector<double>xbar=r.x,xprev=r.x,y(rc,0.0),a,aty;double L=opnorm(m.A);if(!std::isfinite(L))L=1.0;if(L<1e-12)L=1.0;
  double tau=o.tau,sigma=o.sigma;if(tau<=0||sigma<=0)throw std::runtime_error("tau and sigma must be positive");
  if(tau*sigma*L*L>=0.95){double s=std::sqrt(0.9/(tau*sigma*L*L));tau*=s;sigma*=s;}
  const auto t0=std::chrono::steady_clock::now();r.backend="CPU-PDHG";
  for(int it=1;it<=o.max_iterations;it++){
    Ax(m.A,xbar,a);
    for(int i=0;i<rc;i++){double q=y[i]+sigma*a[i];double p=std::min(std::max(q/sigma,m.row_lower[i]),m.row_upper[i]);y[i]=q-sigma*p;}
    ATy(m.A,y,aty);xprev=r.x;
    for(int j=0;j<n;j++)r.x[j]-=tau*(m.objective[j]+aty[j]);
    proj(r.x,m.lower,m.upper);
    for(int j=0;j<n;j++)xbar[j]=r.x[j]+o.theta*(r.x[j]-xprev[j]);
    Ax(m.A,r.x,a);
    double ps=0;for(int i=0;i<rc;i++){double v=a[i],w=0;if(v<m.row_lower[i])w=m.row_lower[i]-v;else if(v>m.row_upper[i])w=v-m.row_upper[i];ps+=w*w;}
    r.primal_residual=std::sqrt(ps)/(1+n2(a));
    double ds=0;for(int j=0;j<n;j++){double z=r.x[j]-(m.objective[j]+aty[j]);double p=std::min(std::max(z,m.lower[j]),m.upper[j]);double d=r.x[j]-p;ds+=d*d;}
    r.dual_residual=std::sqrt(ds)/(1+n2(m.objective));r.iterations=it;
    if(!std::isfinite(r.primal_residual)||!std::isfinite(r.dual_residual)){r.status="NUMERICAL_FAILURE";break;}
    if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}
    if(o.time_limit_sec>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>=o.time_limit_sec){r.status="TIME_LIMIT";break;}
  }
  certify_original(input,r);
  if(r.status.empty())r.status=r.converged?"OPTIMALITY_TOL_REACHED":"ITERATION_LIMIT";
  r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  return r;
}
SolverResult BharatOptSolverCore::solve_milp(const LPModel&m,const SolverOptions&o){
  auto t0=std::chrono::steady_clock::now();SolverResult out;out.backend="CPU-branch-and-bound";out.x.assign(m.A.cols,0);double incumbent=m.maximize?-INF:INF,best_bound=m.maximize?-INF:INF;std::size_t nodes=0;bool limit=false;
  std::function<void(const LPModel&)>dfs=[&](const LPModel&node){
    if(nodes>=(std::size_t)std::max(1,o.max_nodes)){limit=true;return;}
    if(o.time_limit_sec>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>=o.time_limit_sec){limit=true;return;}
    ++nodes;
    SolverOptions lp=o;lp.use_cuda=false;lp.max_iterations=std::min(o.max_iterations,20000);lp.time_limit_sec=0.0;
    auto rr=solve_lp(node,lp);
    if(rr.status=="INFEASIBLE_PRESOLVE"||!finite_solution(rr)||!rr.converged)return;
    best_bound=m.maximize?std::max(best_bound,rr.objective):std::min(best_bound,rr.objective);
    if((!m.maximize&&rr.objective>=incumbent-o.integrality_tolerance)||(m.maximize&&rr.objective<=incumbent+o.integrality_tolerance))return;
    int branch=-1;if(integer_feasible(node,rr.x,o.integrality_tolerance,branch)){if((!m.maximize&&rr.objective<incumbent)||(m.maximize&&rr.objective>incumbent)){incumbent=rr.objective;out.x=rr.x;}return;}
    if(branch<0)return;
    double v=rr.x[branch],fl=std::floor(v),ce=std::ceil(v);
    if(fl>=node.lower[branch]){LPModel left=node;left.upper[branch]=std::min(left.upper[branch],fl);if(left.lower[branch]<=left.upper[branch])dfs(left);}
    if(ce<=node.upper[branch]){LPModel right=node;right.lower[branch]=std::max(right.lower[branch],ce);if(right.lower[branch]<=right.upper[branch])dfs(right);}
  };
  dfs(m);out.nodes=nodes;out.iterations=(int)nodes;out.best_bound=best_bound;out.objective=incumbent;
  if(std::isfinite(incumbent)){out.mip_gap=std::isfinite(best_bound)?std::abs(incumbent-best_bound)/(1+std::abs(incumbent)):INF;out.converged=!limit&&out.mip_gap<=o.mip_gap;out.status=out.converged?"MIP_OPTIMALITY_GAP_REACHED":(limit?"MIP_LIMIT":"MIP_INCUMBENT_FOUND");}
  else out.status=limit?"MIP_LIMIT_NO_INCUMBENT":"MIP_NO_FEASIBLE_INCUMBENT";
  out.primal_residual=0;out.dual_residual=0;out.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return out;
}
}
