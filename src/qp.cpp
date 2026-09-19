#include "bharatopt/solver.hpp"
#include "bharatopt/qp.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace bharatopt { namespace {
double n2(const std::vector<double>&v){long double s=0;for(double a:v)s+=(long double)a*a;return std::sqrt((double)s);}
void Ax(const SparseMatrixCSR&A,const std::vector<double>&x,std::vector<double>&y){y.assign(A.rows,0);for(size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)y[i]+=A.values[k]*x[A.col_index[k]];}
void AT(const SparseMatrixCSR&A,const std::vector<double>&y,std::vector<double>&x){x.assign(A.cols,0);for(size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)x[A.col_index[k]]+=A.values[k]*y[i];}
void proj(std::vector<double>&x,const std::vector<double>&lo,const std::vector<double>&hi){for(size_t i=0;i<x.size();i++)x[i]=std::min(std::max(x[i],lo[i]),hi[i]);}
double opnorm(const SparseMatrixCSR&A){if(A.cols==0)return 0;std::vector<double>x(A.cols,1),y,z;for(int it=0;it<10;it++){Ax(A,x,y);AT(A,y,z);double q=n2(z);if(q==0)return 0;for(double&v:z)v/=q;x=z;}Ax(A,x,y);return n2(y)/std::max(n2(x),1e-30);}
double objective(const QPModel&m,const std::vector<double>&x){std::vector<double>qx;Ax(m.Q,x,qx);double s=0;for(size_t i=0;i<x.size();i++)s+=0.5*x[i]*qx[i]+m.linear.objective[i]*x[i];return s;}
}
QPModel make_qp_demo(){
 QPModel q;q.name="BHARATOPT-CONVEX-QP-DEMO";q.linear.name=q.name;q.linear.var_names={"flow_A","flow_B"};q.linear.objective={-4,-2};q.linear.lower={0,0};q.linear.upper={3,3};q.linear.integer={0,0};
 q.linear.rows={{"minimum_flow",RowSense::GreaterEqual,3}};q.linear.A.rows=1;q.linear.A.cols=2;q.linear.A.row_ptr={0,2};q.linear.A.col_index={0,1};q.linear.A.values={1,1};q.linear.row_lower={3};q.linear.row_upper={INF};
 q.Q.rows=2;q.Q.cols=2;q.Q.row_ptr={0,1,2};q.Q.col_index={0,1};q.Q.values={1,1};return q;
}
SolverResult BharatOptSolverCore::solve_qp(const QPModel&m,const SolverOptions&o){
 if(!m.convex_psd)throw std::runtime_error("QP model is not declared convex PSD");
 const LPModel&l=m.linear; if(m.Q.rows!=l.A.cols||m.Q.cols!=l.A.cols)throw std::runtime_error("Q dimension mismatch");
 if(l.objective.size()!=m.Q.cols||l.lower.size()!=m.Q.cols||l.upper.size()!=m.Q.cols)throw std::runtime_error("QP vector dimension mismatch");
 int n=(int)m.Q.cols,rc=(int)l.A.rows;SolverResult r;r.x.assign(n,0);proj(r.x,l.lower,l.upper);std::vector<double>xbar=r.x,xprev=r.x,y(rc),a,aty,qx,grad;
 double La=opnorm(l.A);if(La<1e-12)La=1;double Lq=opnorm(m.Q);double tau=o.tau,sigma=o.sigma;if(Lq>1e-12)tau=std::min(tau,0.8/Lq);if(tau*sigma*La*La>=.95){double s=std::sqrt(.9/(tau*sigma*La*La));tau*=s;sigma*=s;}
 auto t0=std::chrono::steady_clock::now();r.backend="CPU-QP-PDHG";
 for(int it=1;it<=o.max_iterations;it++){
  Ax(l.A,xbar,a);
  for(int i=0;i<rc;i++){double z=y[i]+sigma*a[i];double p=std::min(std::max(z/sigma,l.row_lower[i]),l.row_upper[i]);y[i]=z-sigma*p;}
  AT(l.A,y,aty);Ax(m.Q,r.x,qx);
  xprev=r.x;for(int j=0;j<n;j++)grad[j]=l.objective[j]+qx[j]+aty[j];
  for(int j=0;j<n;j++)r.x[j]-=tau*grad[j];proj(r.x,l.lower,l.upper);
  for(int j=0;j<n;j++)xbar[j]=r.x[j]+o.theta*(r.x[j]-xprev[j]);
  Ax(l.A,r.x,a);double ps=0;for(int i=0;i<rc;i++){double v=a[i],w=0;if(v<l.row_lower[i])w=l.row_lower[i]-v;else if(v>l.row_upper[i])w=v-l.row_upper[i];ps+=w*w;}r.primal_residual=std::sqrt(ps)/(1+n2(a));
  double ds=0;for(int j=0;j<n;j++){double z=r.x[j]-grad[j],p=std::min(std::max(z,l.lower[j]),l.upper[j]),d=r.x[j]-p;ds+=d*d;}r.dual_residual=std::sqrt(ds)/(1+n2(grad));r.iterations=it;
  if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}
 }
 r.objective=objective(m,r.x);r.best_bound=r.objective;r.mip_gap=0;r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();r.status=r.converged?"QP_OPTIMALITY_TOL_REACHED":"QP_ITERATION_LIMIT";return r;
}
}