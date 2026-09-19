#include "bharatopt/interior_point.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace bharatopt {
namespace {

struct VarRep {
  double constant{0.0};
  std::vector<std::pair<int,double>> terms;
  bool has_upper{false};
  double upper_span{0.0};
};

struct StdLP {
  int m{0}, n{0};
  std::vector<double> B,b,c;
  std::vector<VarRep> vars;
  double offset{0.0};
};

void append_column(std::vector<std::vector<double>>& rows){for(auto& r:rows)r.push_back(0.0);}

void add_ineq(std::vector<std::vector<double>>& rows,std::vector<double>& rhs,int& n,
              const std::vector<double>& coef,double bound){
  std::vector<double> row=coef;
  row.resize(n,0.0);
  row.push_back(1.0);
  for(auto& r:rows)r.push_back(0.0);
  rows.push_back(std::move(row));
  rhs.push_back(bound);
  ++n;
}

StdLP standardize(const LPModel& m){
  StdLP s;s.vars.resize(m.A.cols);s.offset=m.objective_offset;
  std::vector<std::vector<double>> rows;std::vector<double> rhs;int n=0;

  for(std::size_t j=0;j<m.A.cols;j++){
    double l=m.lower[j],u=m.upper[j],c=m.objective[j];
    if(std::isfinite(l)){
      s.vars[j].constant=l;s.offset+=c*l;
      if(std::isfinite(u)&&std::abs(u-l)<=1e-12){continue;}
      int z=n++;append_column(rows);s.c.push_back(c);s.vars[j].terms.push_back({z,1.0});
      if(std::isfinite(u)){s.vars[j].has_upper=true;s.vars[j].upper_span=u-l;}
    }else if(std::isfinite(u)){
      s.vars[j].constant=u;s.offset+=c*u;
      int z=n++;append_column(rows);s.c.push_back(-c);s.vars[j].terms.push_back({z,-1.0});
    }else{
      int zp=n++;append_column(rows);s.c.push_back(c);s.vars[j].terms.push_back({zp,1.0});
      int zm=n++;append_column(rows);s.c.push_back(-c);s.vars[j].terms.push_back({zm,-1.0});
    }
  }

  auto build_row=[&](std::size_t i){
    std::vector<double> coef(n,0.0);double constant=0.0;
    for(int k=m.A.row_ptr[i];k<m.A.row_ptr[i+1];k++){
      int j=m.A.col_index[k];double a=m.A.values[k];constant+=a*s.vars[j].constant;
      for(auto [z,v]:s.vars[j].terms)coef[z]+=a*v;
    }
    return std::pair<std::vector<double>,double>{std::move(coef),constant};
  };

  for(std::size_t i=0;i<m.A.rows;i++){
    auto [coef,constant]=build_row(i);
    double lo=m.row_lower[i],u=m.row_upper[i];
    if(std::isfinite(lo)&&std::isfinite(u)&&std::abs(lo-u)<=1e-12){
      rows.push_back(coef);rhs.push_back(lo-constant);
    }else{
      if(std::isfinite(u))add_ineq(rows,rhs,n,coef,u-constant);
      if(std::isfinite(lo)){for(double&v:coef)v=-v;add_ineq(rows,rhs,n,coef,-lo+constant);}
    }
  }
  for(std::size_t j=0;j<s.vars.size();j++)if(s.vars[j].has_upper){
    std::vector<double> coef(n,0.0);for(auto [z,v]:s.vars[j].terms)coef[z]+=v;
    add_ineq(rows,rhs,n,coef,s.vars[j].upper_span);
  }

  s.m=(int)rows.size();s.n=n;s.b=rhs;s.B.assign((std::size_t)s.m*s.n,0.0);s.c.resize(s.n,0.0);
  for(int i=0;i<s.m;i++)for(int j=0;j<s.n;j++)s.B[(std::size_t)i*s.n+j]=rows[i][j];
  return s;
}

double n2(const std::vector<double>&v){long double s=0;for(double x:v)s+=(long double)x*x;return std::sqrt((double)s);}
void Bz(const StdLP&s,const std::vector<double>&z,std::vector<double>&out){out.assign(s.m,0);for(int i=0;i<s.m;i++)for(int j=0;j<s.n;j++)out[i]+=s.B[(std::size_t)i*s.n+j]*z[j];}
void BtY(const StdLP&s,const std::vector<double>&y,std::vector<double>&out){out.assign(s.n,0);for(int i=0;i<s.m;i++)for(int j=0;j<s.n;j++)out[j]+=s.B[(std::size_t)i*s.n+j]*y[i];}

std::vector<double> solve_dense(std::vector<double> A,std::vector<double> b,int n){
  for(int i=0;i<n;i++)A[(std::size_t)i*n+i]+=1e-10;
  for(int k=0;k<n;k++){
    int p=k;double best=std::abs(A[(std::size_t)k*n+k]);
    for(int i=k+1;i<n;i++){double v=std::abs(A[(std::size_t)i*n+k]);if(v>best){best=v;p=i;}}
    if(best<1e-14)throw std::runtime_error("Interior-point factorization failed: singular normal equations");
    if(p!=k){for(int j=k;j<n;j++)std::swap(A[(std::size_t)k*n+j],A[(std::size_t)p*n+j]);std::swap(b[k],b[p]);}
    for(int i=k+1;i<n;i++){double f=A[(std::size_t)i*n+k]/A[(std::size_t)k*n+k];if(std::abs(f)<1e-18)continue;for(int j=k;j<n;j++)A[(std::size_t)i*n+j]-=f*A[(std::size_t)k*n+j];b[i]-=f*b[k];}
  }
  std::vector<double>x(n,0);for(int i=n-1;i>=0;i--){double v=b[i];for(int j=i+1;j<n;j++)v-=A[(std::size_t)i*n+j]*x[j];x[i]=v/A[(std::size_t)i*n+i];}return x;
}

void direction(const StdLP&s,const std::vector<double>&z,const std::vector<double>&slack,
               const std::vector<double>&rp,const std::vector<double>&rd,const std::vector<double>&rc,
               std::vector<double>&dz,std::vector<double>&dy,std::vector<double>&ds){
  std::vector<double>D(s.n),q(s.n);
  for(int j=0;j<s.n;j++){D[j]=z[j]/slack[j];q[j]=rc[j]/slack[j]+D[j]*rd[j];}
  std::vector<double>Bq;Bz(s,q,Bq);std::vector<double>rhs(s.m);for(int i=0;i<s.m;i++)rhs[i]=rp[i]-Bq[i];
  std::vector<double>H((std::size_t)s.m*s.m,0.0);
  for(int i=0;i<s.m;i++)for(int j=0;j<s.m;j++)for(int k=0;k<s.n;k++)H[(std::size_t)i*s.m+j]+=s.B[(std::size_t)i*s.n+k]*D[k]*s.B[(std::size_t)j*s.n+k];
  dy=s.m?solve_dense(H,rhs,s.m):std::vector<double>{};
  std::vector<double>Btdy;BtY(s,dy,Btdy);dz.resize(s.n);ds.resize(s.n);
  for(int j=0;j<s.n;j++){ds[j]=Btdy[j]+rd[j];dz[j]=-rc[j]/slack[j]-D[j]*ds[j];}
}

double step_to_boundary(const std::vector<double>&x,const std::vector<double>&dx,double eta){
  double a=1.0;for(std::size_t i=0;i<x.size();i++)if(dx[i]<0)a=std::min(a,eta*(-x[i]/dx[i]));return a;
}

}

SolverResult solve_interior_point(const LPModel&m,const InteriorPointOptions&opt){
  SolverResult r;r.backend="CPU-Mehrotra-IP";
  if(m.has_integer_variables()){r.status="INTEGER_MODEL_NOT_SUPPORTED_BY_IP";return r;}
  StdLP s=standardize(m);
  if(s.n>opt.max_variables || s.m>opt.max_constraints){r.status="IP_SIZE_LIMIT";return r;}

  if(s.n==0){
    r.x.assign(m.A.cols,0.0);for(std::size_t j=0;j<m.A.cols;j++)r.x[j]=s.vars[j].constant;
    r.objective=m.objective_offset;for(std::size_t j=0;j<r.x.size();j++)r.objective+=m.objective[j]*r.x[j];
    r.best_bound=r.objective;r.converged=true;r.status="OPTIMALITY_TOL_REACHED";return r;
  }

  std::vector<double>z(s.n,1.0),slack(s.n,1.0),y(s.m,0.0),rp,rd,rc,dz,dy,ds,tmp;
  const auto t0=std::chrono::steady_clock::now();
  for(int it=1;it<=opt.max_iterations;it++){
    Bz(s,z,tmp);rp.resize(s.m);for(int i=0;i<s.m;i++)rp[i]=tmp[i]-s.b[i];
    BtY(s,y,tmp);rd.resize(s.n);for(int j=0;j<s.n;j++)rd[j]=tmp[j]+s.c[j]-slack[j];
    double mu=0;for(int j=0;j<s.n;j++)mu+=z[j]*slack[j];mu/=s.n;
    double obj=s.offset;for(int j=0;j<s.n;j++)obj+=s.c[j]*z[j];
    r.iterations=it;r.objective=obj;r.primal_residual=n2(rp)/(1+n2(s.b));r.dual_residual=n2(rd)/(1+n2(s.c));
    if(std::max({r.primal_residual,r.dual_residual,mu/(1+std::abs(obj))})<=opt.tolerance){r.converged=true;break;}

    rc.resize(s.n);for(int j=0;j<s.n;j++)rc[j]=z[j]*slack[j];
    direction(s,z,slack,rp,rd,rc,dz,dy,ds);
    double aaff=std::min(step_to_boundary(z,dz,0.995),step_to_boundary(slack,ds,0.995)),mu_aff=0;
    for(int j=0;j<s.n;j++)mu_aff+=(z[j]+aaff*dz[j])*(slack[j]+aaff*ds[j]);mu_aff/=s.n;
    double cent=std::clamp(std::pow(std::max(0.0,mu_aff/mu),3.0),0.0,1.0);
    for(int j=0;j<s.n;j++)rc[j]=z[j]*slack[j]+dz[j]*ds[j]-cent*mu;
    direction(s,z,slack,rp,rd,rc,dz,dy,ds);
    double a=std::min(step_to_boundary(z,dz,0.995),step_to_boundary(slack,ds,0.995));
    if(a<1e-9){r.status="IP_STALLED";break;}
    for(int j=0;j<s.n;j++)z[j]+=a*dz[j];
    for(int i=0;i<s.m;i++)y[i]+=a*dy[i];
    for(int j=0;j<s.n;j++)slack[j]+=a*ds[j];
  }

  r.x.resize(m.A.cols);
  for(std::size_t j=0;j<m.A.cols;j++){r.x[j]=s.vars[j].constant;for(auto [k,v]:s.vars[j].terms)r.x[j]+=v*z[k];}
  r.objective=m.objective_offset;for(std::size_t j=0;j<r.x.size();j++)r.objective+=m.objective[j]*r.x[j];
  r.best_bound=r.objective;r.solve_time_sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  if(r.status.empty())r.status=r.converged?"OPTIMALITY_TOL_REACHED":"IP_ITERATION_LIMIT";
  return r;
}
}
