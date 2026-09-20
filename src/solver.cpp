#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include "bharatopt/preprocess.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace bharatopt {
namespace {
double n2(const std::vector<double>&v){long double s=0;for(double a:v){if(!std::isfinite(a))return INF;s+=(long double)a*a;}return std::sqrt((double)s);}
void proj(std::vector<double>&v,const std::vector<double>&lo,const std::vector<double>&hi){for(std::size_t i=0;i<v.size();++i)v[i]=std::min(std::max(v[i],lo[i]),hi[i]);}
void Ax(const SparseMatrixCSR&A,const std::vector<double>&x,std::vector<double>&y){y.assign(A.rows,0.0);for(std::size_t i=0;i<A.rows;i++){double s=0;for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)s+=A.values[k]*x[A.col_index[k]];y[i]=s;}}
void ATy(const SparseMatrixCSR&A,const std::vector<double>&y,std::vector<double>&x){x.assign(A.cols,0.0);for(std::size_t i=0;i<A.rows;i++)for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)x[A.col_index[k]]+=A.values[k]*y[i];}
double opnorm(const SparseMatrixCSR&A){if(A.cols==0)return 0;std::vector<double>x(A.cols,1.0),y,z;for(int it=0;it<12;it++){Ax(A,x,y);ATy(A,y,z);double q=n2(z);if(!std::isfinite(q)||q==0)return 0;for(double&v:z)v/=q;x=z;}Ax(A,x,y);return n2(y)/std::max(n2(x),1e-30);}
double model_objective(const LPModel&m,const std::vector<double>&x){
  double v=m.objective_offset;
  for(std::size_t j=0;j<x.size()&&j<m.objective.size();j++)v+=m.objective[j]*x[j];
  return v;
}
void certify_original(const LPModel&orig,SolverResult&r){
  if(r.x.size()!=orig.A.cols)return;
  std::vector<double>ax;Ax(orig.A,r.x,ax);
  double viol=0, max_viol=0;
  for(std::size_t j=0;j<orig.A.cols;j++){
    if(orig.lower[j]-r.x[j]>max_viol) max_viol=orig.lower[j]-r.x[j];
    if(r.x[j]-orig.upper[j]>max_viol) max_viol=r.x[j]-orig.upper[j];
  }
  for(std::size_t i=0;i<orig.A.rows;i++){
    double w=0;
    if(ax[i]<orig.row_lower[i])w=orig.row_lower[i]-ax[i];
    else if(ax[i]>orig.row_upper[i])w=ax[i]-orig.row_upper[i];
    viol+=w*w;
    if(w>max_viol) max_viol=w;
  }
  r.primal_residual=std::sqrt(viol)/(1.0+n2(ax));
  r.max_constraint_violation=max_viol;
  r.objective=model_objective(orig,r.x);
}
double dual_lower_bound(const LPModel&m,const std::vector<double>&y){
  double support=0.0;
  for(std::size_t i=0;i<m.A.rows;i++){
    double yi=y[i];
    if(!std::isfinite(m.row_lower[i])&&std::isfinite(m.row_upper[i])) yi=std::max(0.0,yi);
    else if(std::isfinite(m.row_lower[i])&&!std::isfinite(m.row_upper[i])) yi=std::min(0.0,yi);
    else if(!std::isfinite(m.row_lower[i])&&!std::isfinite(m.row_upper[i])) yi=0.0;
    if(yi>0) support+=m.row_upper[i]*yi;
    else if(yi<0) support+=m.row_lower[i]*yi;
  }
  std::vector<double>aty;ATy(m.A,y,aty);double value=-support;
  for(std::size_t j=0;j<m.A.cols;j++){
    double g=m.objective[j]+aty[j];
    if(g>=0){if(!std::isfinite(m.lower[j]))return -INF;value+=g*m.lower[j];}
    else {if(!std::isfinite(m.upper[j]))return -INF;value+=g*m.upper[j];}
  }
  return value;
}
bool finite_solution(const SolverResult&r){return !r.x.empty()&&std::isfinite(r.objective)&&std::isfinite(r.primal_residual)&&std::isfinite(r.dual_residual);}
bool integer_feasible(const LPModel&m,const std::vector<double>&x,double tol,int&branch){branch=-1;for(std::size_t i=0;i<m.integer.size();i++)if(m.integer[i]){double n=std::round(x[i]);if(std::abs(x[i]-n)>tol){branch=(int)i;return false;}}return true;}
bool feasible_solution(const LPModel&m,std::vector<double>&x,double tol){
  if(x.size()!=m.A.cols)return false;
  for(std::size_t j=0;j<x.size();j++){
    if(m.integer[j])x[j]=std::round(x[j]);
    if(x[j]<m.lower[j]-tol||x[j]>m.upper[j]+tol)return false;
  }
  std::vector<double>ax;Ax(m.A,x,ax);
  for(std::size_t i=0;i<ax.size();i++)if(ax[i]<m.row_lower[i]-tol||ax[i]>m.row_upper[i]+tol)return false;
  return true;
}
LPModel internal_min_model(const LPModel&input){
  LPModel m=input;
  if(input.maximize){
    for(double&v:m.objective)v=-v;
    m.objective_offset=-input.objective_offset;
    m.maximize=false;
  }
  return m;
}
double to_original_bound(const LPModel&input,const LPModel&internal,double internal_bound){
  if(!std::isfinite(internal_bound))return input.maximize?(internal_bound==INF?-INF:INF):internal_bound;
  return input.maximize?-internal_bound:internal_bound;
}
}
SolverResult BharatOptSolverCore::solve_lp(const LPModel&input,const SolverOptions&o){
  if(input.A.cols!=input.objective.size()||input.A.rows!=input.row_lower.size()||input.A.rows!=input.row_upper.size()||input.lower.size()!=input.A.cols||input.upper.size()!=input.A.cols)throw std::runtime_error("Inconsistent model dimensions");
  const LPModel normalized=internal_min_model(input);
  auto pp=preprocess_lp(normalized,o.presolve?o.scaling_passes:0);
  if(!pp.feasible){SolverResult r;r.status="INFEASIBLE_PRESOLVE";r.backend="presolve";return r;}
  LPModel m=std::move(pp.model);
#ifdef BHARATOPT_CUDA_ENABLED
  if(o.use_cuda){
    SolverResult g;
    if(cuda_backend::CudaPdhgSolver().solve(m,o,g)){
      recover_primal(g.x,pp.column_scale,g.x);
      double internal_bound=g.best_bound;
      certify_original(input,g);
      if(std::isfinite(internal_bound)){
        double actual_bound=to_original_bound(input,m,internal_bound);
        g.dual_bound=actual_bound;
        g.best_bound=actual_bound;
      }else{
        g.dual_bound=input.maximize?-INF:INF;
        g.best_bound=g.objective;
      }
      return g;
    }
  }
#endif
  const int n=(int)m.A.cols,rc=(int)m.A.rows;
  SolverResult r;r.x.assign(n,0.0);proj(r.x,m.lower,m.upper);
  std::vector<double>xbar=r.x,xprev=r.x,y(rc,0.0),a,aty;
  double L=opnorm(m.A);if(!std::isfinite(L))L=1.0;if(L<1e-12)L=1.0;
  double tau=o.tau,sigma=o.sigma;if(tau<=0||sigma<=0)throw std::runtime_error("tau and sigma must be positive");
  if(tau*sigma*L*L>=0.95){double s=std::sqrt(0.9/(tau*sigma*L*L));tau*=s;sigma*=s;}
  const auto t0=std::chrono::steady_clock::now();r.backend="CPU-PDHG";
  for(int it=1;it<=o.max_iterations;it++){
    Ax(m.A,xbar,a);
    for(int i=0;i<rc;i++){
      double q=y[i]+sigma*a[i];
      double p=std::min(std::max(q/sigma,m.row_lower[i]),m.row_upper[i]);
      y[i]=q-sigma*p;
    }
    ATy(m.A,y,aty);xprev=r.x;
    for(int j=0;j<n;j++)r.x[j]-=tau*(m.objective[j]+aty[j]);
    proj(r.x,m.lower,m.upper);
    for(int j=0;j<n;j++)xbar[j]=r.x[j]+o.theta*(r.x[j]-xprev[j]);
    Ax(m.A,r.x,a);
    double ps=0;
    for(int i=0;i<rc;i++){
      double v=a[i],w=0;
      if(v<m.row_lower[i])w=m.row_lower[i]-v;
      else if(v>m.row_upper[i])w=v-m.row_upper[i];
      ps+=w*w;
    }
    r.primal_residual=std::sqrt(ps)/(1+n2(a));
    double ds=0;
    for(int j=0;j<n;j++){
      double z=r.x[j]-(m.objective[j]+aty[j]);
      double p=std::min(std::max(z,m.lower[j]),m.upper[j]);
      double d=r.x[j]-p;ds+=d*d;
    }
    r.dual_residual=std::sqrt(ds)/(1+n2(m.objective));r.iterations=it;
    if(!std::isfinite(r.primal_residual)||!std::isfinite(r.dual_residual)){r.status="NUMERICAL_FAILURE";break;}
    if(std::max(r.primal_residual,r.dual_residual)<=o.tolerance){r.converged=true;break;}

    // Adaptive step-size residual balancing (PDHG / Barzilai-Borwein style)
    if(it % 50 == 0){
      if(r.primal_residual > 5.0 * r.dual_residual && tau > 1e-6){
        tau /= 1.25; sigma *= 1.25;
      } else if(r.dual_residual > 5.0 * r.primal_residual && sigma > 1e-6){
        tau *= 1.25; sigma /= 1.25;
      }
      if(tau * sigma * L * L >= 0.95){
        double s = std::sqrt(0.9 / (tau * sigma * L * L));
        tau *= s; sigma *= s;
      }
    }

    if(o.time_limit_sec>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>=o.time_limit_sec){r.status="TIME_LIMIT";break;}
  }
  double db=dual_lower_bound(m,y);
  double internal_bound=(o.scaling_passes==0&&std::isfinite(db))?db+m.objective_offset:INF;
  certify_original(input,r);
  if(std::isfinite(internal_bound)){
    r.dual_bound=to_original_bound(input,m,internal_bound);
    r.best_bound=r.dual_bound;
  } else {
    r.dual_bound=input.maximize?-INF:INF;
    r.best_bound=r.objective;
  }
  if(r.status.empty())r.status=r.converged?"OPTIMALITY_TOL_REACHED":"ITERATION_LIMIT";
  r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  return r;
}
SolverResult BharatOptSolverCore::solve(const LPModel& model,const SolverOptions& options){
  // Central dispatch used by the CLI and presentation layer.
  // Integer variables require the MILP branch-and-bound path; otherwise solve as an LP.
  if(model.has_integer_variables()) return solve_milp(model,options);
  return solve_lp(model,options);
}

SolverResult BharatOptSolverCore::solve_milp(const LPModel&m,const SolverOptions&o){
  struct Node{LPModel model;SolverResult relaxation;double bound{0};};
  struct Cmp{bool maximize;bool operator()(const Node&a,const Node&b)const{return maximize?a.bound<b.bound:a.bound>b.bound;}};
  std::priority_queue<Node,std::vector<Node>,Cmp> open((Cmp{m.maximize}));
  auto t0=std::chrono::steady_clock::now();
  SolverResult out;out.backend="CPU-best-bound-branch-and-bound";out.x.assign(m.A.cols,0);
  double incumbent=m.maximize?-INF:INF;std::size_t nodes=0;bool limit=false,gap_reached=false;

  auto exhausted=[&](){
    if(nodes>=(std::size_t)std::max(1,o.max_nodes))return true;
    return o.time_limit_sec>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>=o.time_limit_sec;
  };
  auto dominates_incumbent=[&](double bound){
    if(!std::isfinite(incumbent))return false;
    return m.maximize?(bound<=incumbent+o.integrality_tolerance):(bound>=incumbent-o.integrality_tolerance);
  };
  auto make_node=[&](const LPModel&model)->Node{
    Node n;n.model=model;
    if(exhausted()){limit=true;return n;}
    ++nodes;
    SolverOptions lp=o;lp.use_cuda=false;lp.scaling_passes=0;lp.presolve=true;lp.max_iterations=std::min(o.max_iterations,20000);lp.time_limit_sec=0.0;
    auto rr=solve_lp(model,lp);
    if(rr.status=="INFEASIBLE_PRESOLVE"||!rr.converged||!std::isfinite(rr.best_bound))return n;
    n.relaxation=std::move(rr);n.bound=n.relaxation.best_bound;return n;
  };
  auto valid_node=[&](const Node&n){return !n.relaxation.x.empty()&&std::isfinite(n.bound);};

  Node root=make_node(m);
  if(valid_node(root)&&!dominates_incumbent(root.bound))open.push(std::move(root));

  while(!open.empty()&&!limit){
    if(exhausted()){limit=true;break;}
    if(std::isfinite(incumbent)){
      double bound=open.top().bound;
      double gap=std::isfinite(bound)?std::abs(incumbent-bound)/(1.0+std::abs(incumbent)):INF;
      if(gap<=o.mip_gap){gap_reached=true;break;}
    }
    Node node=open.top();open.pop();
    if(dominates_incumbent(node.bound))continue;

    int branch=-1;double frac=0.0;
    for(std::size_t j=0;j<node.model.integer.size();j++)if(node.model.integer[j]){
      double f=std::abs(node.relaxation.x[j]-std::round(node.relaxation.x[j]));
      double score=std::min(f,1.0-f);
      if(f>o.integrality_tolerance&&score>frac){frac=score;branch=(int)j;}
    }
    if(branch<0){
      std::vector<double>candidate=node.relaxation.x;
      if(feasible_solution(node.model,candidate,std::max(10.0*o.tolerance,o.integrality_tolerance))){
        double obj=model_objective(m,candidate);
        if((!m.maximize&&obj<incumbent)||(m.maximize&&obj>incumbent)){incumbent=obj;out.x=candidate;}
      }
      while(!open.empty()&&dominates_incumbent(open.top().bound))open.pop();
      continue;
    }

    double v=node.relaxation.x[branch],fl=std::floor(v),ce=std::ceil(v);
    if(fl>=node.model.lower[branch]){
      LPModel left=node.model;left.upper[branch]=std::min(left.upper[branch],fl);
      if(left.lower[branch]<=left.upper[branch]){
        Node child=make_node(left);
        if(valid_node(child)&&!dominates_incumbent(child.bound))open.push(std::move(child));
      }
    }
    if(ce<=node.model.upper[branch]&&!limit){
      LPModel right=node.model;right.lower[branch]=std::max(right.lower[branch],ce);
      if(right.lower[branch]<=right.upper[branch]){
        Node child=make_node(right);
        if(valid_node(child)&&!dominates_incumbent(child.bound))open.push(std::move(child));
      }
    }
  }

  out.nodes=nodes;out.iterations=(int)nodes;
  if(std::isfinite(incumbent)){
    out.objective=incumbent;
    if(!open.empty()){
      out.best_bound=open.top().bound;
      out.dual_bound=out.best_bound;
      out.mip_gap=std::abs(incumbent-out.best_bound)/(1+std::abs(incumbent));
    } else {
      out.best_bound=incumbent;out.dual_bound=incumbent;out.mip_gap=0.0;
    }
    out.converged=!limit&&(open.empty()||gap_reached);
    out.status=open.empty()&&!limit?"MIP_OPTIMALITY_PROVED":(gap_reached?"MIP_GAP_REACHED":"MIP_LIMIT");
  }else{
    out.objective=m.maximize?-INF:INF;out.best_bound=m.maximize?-INF:INF;out.dual_bound=out.best_bound;
    out.mip_gap=INF;out.status=limit?"MIP_LIMIT_NO_INCUMBENT":"MIP_NO_FEASIBLE_INCUMBENT";
  }
  out.primal_residual=0;out.dual_residual=0;
  out.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  return out;
}
}
