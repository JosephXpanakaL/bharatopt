#include "bharatopt/preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace bharatopt {
namespace {
void tighten_singletons(LPModel& m){
  for(int pass=0;pass<3;pass++){
    bool changed=false;
    for(std::size_t i=0;i<m.A.rows;i++){
      int begin=m.A.row_ptr[i],end=m.A.row_ptr[i+1];
      if(end-begin!=1)continue;
      int j=m.A.col_index[begin];double a=m.A.values[begin];
      if(a==0)continue;
      double nl=m.lower[j],nu=m.upper[j];
      if(std::isfinite(m.row_lower[i])){
        double v=m.row_lower[i]/a;if(a>0)nl=std::max(nl,v);else nu=std::min(nu,v);
      }
      if(std::isfinite(m.row_upper[i])){
        double v=m.row_upper[i]/a;if(a>0)nu=std::min(nu,v);else nl=std::max(nl,v);
      }
      if(nl>m.lower[j]+1e-15){m.lower[j]=nl;changed=true;}
      if(nu<m.upper[j]-1e-15){m.upper[j]=nu;changed=true;}
      if(m.lower[j]>m.upper[j]+1e-12)return;
    }
    if(!changed)break;
  }
}
}
PreprocessResult preprocess_lp(const LPModel&input,int scaling_passes){
  PreprocessResult out;out.model=input;
  if(out.model.integer.size()!=out.model.A.cols)out.model.integer.assign(out.model.A.cols,0);
  if(out.model.lower.size()!=out.model.A.cols||out.model.upper.size()!=out.model.A.cols){
    out.feasible=false;out.message="Variable bound dimension mismatch";return out;
  }
  for(std::size_t j=0;j<out.model.A.cols;j++)if(out.model.lower[j]>out.model.upper[j]+1e-12){out.feasible=false;out.message="Inconsistent variable bounds";return out;}
  for(std::size_t i=0;i<out.model.A.rows;i++){
    bool zero=out.model.row_ptr[i]==out.model.row_ptr[i+1];
    if(zero){
      if(0.0<out.model.row_lower[i]-1e-12||0.0>out.model.row_upper[i]+1e-12){out.feasible=false;out.message="Infeasible zero row: "+out.model.rows[i].name;return out;}
    }
  }
  tighten_singletons(out.model);
  for(std::size_t j=0;j<out.model.A.cols;j++)if(out.model.lower[j]>out.model.upper[j]+1e-12){out.feasible=false;out.message="Presolve detected inconsistent bounds";return out;}

  out.column_scale.assign(out.model.A.cols,1.0);
  for(int pass=0;pass<std::max(0,scaling_passes);pass++){
    std::vector<double> row_scale(out.model.A.rows,1.0);
    for(std::size_t i=0;i<out.model.A.rows;i++){
      double mx=0;for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)mx=std::max(mx,std::abs(out.model.A.values[k]));
      if(mx>1e-12&&std::isfinite(mx))row_scale[i]=1.0/std::sqrt(mx);
    }
    for(std::size_t i=0;i<out.model.A.rows;i++){
      for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)out.model.A.values[k]*=row_scale[i];
      if(std::isfinite(out.model.row_lower[i]))out.model.row_lower[i]*=row_scale[i];
      if(std::isfinite(out.model.row_upper[i]))out.model.row_upper[i]*=row_scale[i];
    }
    std::vector<double> col_max(out.model.A.cols,0.0);
    for(std::size_t i=0;i<out.model.A.rows;i++)for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)col_max[out.model.A.col_index[k]]=std::max(col_max[out.model.A.col_index[k]],std::abs(out.model.A.values[k]));
    for(std::size_t j=0;j<out.model.A.cols;j++){
      if(col_max[j]<=1e-12||!std::isfinite(col_max[j]))continue;
      double sc=1.0/std::sqrt(col_max[j]);out.column_scale[j]*=sc;
      out.model.objective[j]*=sc;
      if(std::isfinite(out.model.lower[j]))out.model.lower[j]/=sc;
      if(std::isfinite(out.model.upper[j]))out.model.upper[j]/=sc;
    }
    for(std::size_t i=0;i<out.model.A.rows;i++)for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)out.model.A.values[k]*=(out.column_scale[out.model.A.col_index[k]]/(pass==0?1.0:1.0)); // cumulative column scaling already requires only the new ratio; corrected below
  }
  // Rebuild scaled coefficients cleanly from original input and accumulated row/column factors.
  out.model=input;
  if(out.model.integer.size()!=out.model.A.cols)out.model.integer.assign(out.model.A.cols,0);
  std::vector<double> rs(out.model.A.rows,1.0), cs(out.model.A.cols,1.0);
  for(int pass=0;pass<std::max(0,scaling_passes);pass++){
    std::vector<double> row_max(out.model.A.rows,0.0);
    for(std::size_t i=0;i<out.model.A.rows;i++)for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)row_max[i]=std::max(row_max[i],std::abs(out.model.A.values[k])*rs[i]*cs[out.model.A.col_index[k]]);
    for(std::size_t i=0;i<out.model.A.rows;i++)if(row_max[i]>1e-12&&std::isfinite(row_max[i]))rs[i]/=std::sqrt(row_max[i]);
    std::vector<double> col_max(out.model.A.cols,0.0);
    for(std::size_t i=0;i<out.model.A.rows;i++)for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++){double v=std::abs(input.A.values[k])*rs[i]*cs[out.model.A.col_index[k]];col_max[out.model.A.col_index[k]]=std::max(col_max[out.model.A.col_index[k]],v);}
    for(std::size_t j=0;j<out.model.A.cols;j++)if(col_max[j]>1e-12&&std::isfinite(col_max[j]))cs[j]/=std::sqrt(col_max[j]);
  }
  out.column_scale=cs;out.model.A.values=input.A.values;
  for(std::size_t i=0;i<out.model.A.rows;i++){
    for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)out.model.A.values[k]*=rs[i]*cs[out.model.A.col_index[k]];
    if(std::isfinite(out.model.row_lower[i]))out.model.row_lower[i]*=rs[i];
    if(std::isfinite(out.model.row_upper[i]))out.model.row_upper[i]*=rs[i];
  }
  for(std::size_t j=0;j<out.model.A.cols;j++){
    out.model.objective[j]*=cs[j];
    if(std::isfinite(out.model.lower[j]))out.model.lower[j]/=cs[j];
    if(std::isfinite(out.model.upper[j]))out.model.upper[j]/=cs[j];
  }
  return out;
}
void recover_primal(const std::vector<double>&scaled_x,const std::vector<double>&column_scale,std::vector<double>&x){
  x.resize(scaled_x.size());for(std::size_t j=0;j<x.size();j++)x[j]=scaled_x[j]*column_scale[j];
}
}