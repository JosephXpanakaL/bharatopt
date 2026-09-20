#include "bharatopt/preprocess.hpp"
#include <algorithm>
#include <cmath>
namespace bharatopt {
namespace {
void tighten_singletons(LPModel& m){
  for(int pass=0;pass<4;pass++){
    bool changed=false;
    for(std::size_t i=0;i<m.A.rows;i++){
      int b=m.A.row_ptr[i],e=m.A.row_ptr[i+1];
      if(e-b!=1)continue;
      int j=m.A.col_index[b];double a=m.A.values[b];if(a==0)continue;
      double lo=m.lower[j],hi=m.upper[j];
      if(std::isfinite(m.row_lower[i])){double z=m.row_lower[i]/a;if(a>0)lo=std::max(lo,z);else hi=std::min(hi,z);}
      if(std::isfinite(m.row_upper[i])){double z=m.row_upper[i]/a;if(a>0)hi=std::min(hi,z);else lo=std::max(lo,z);}
      if(lo>m.lower[j]+1e-12)m.lower[j]=lo,changed=true;
      if(hi<m.upper[j]-1e-12)m.upper[j]=hi,changed=true;
      if(m.lower[j]>m.upper[j]+1e-12)return;
    }
    if(!changed)break;
  }
}

void tighten_implied_bounds(LPModel& m) {
  for (int pass = 0; pass < 3; pass++) {
    bool changed = false;
    for (std::size_t i = 0; i < m.A.rows; i++) {
      int b = m.A.row_ptr[i], e = m.A.row_ptr[i + 1];
      if (e - b <= 1) continue;

      bool all_finite = true;
      double min_act = 0.0, max_act = 0.0;
      for (int k = b; k < e; k++) {
        int j = m.A.col_index[k];
        double a = m.A.values[k];
        if (a > 0) {
          if (!std::isfinite(m.lower[j])) { all_finite = false; break; }
          min_act += a * m.lower[j];
          if (std::isfinite(m.upper[j])) max_act += a * m.upper[j];
          else all_finite = false;
        } else {
          if (!std::isfinite(m.upper[j])) { all_finite = false; break; }
          min_act += a * m.upper[j];
          if (std::isfinite(m.lower[j])) max_act += a * m.lower[j];
          else all_finite = false;
        }
      }
      if (!all_finite) continue;

      if (std::isfinite(m.row_upper[i])) {
        double slack = m.row_upper[i] - min_act;
        if (slack >= 0.0) {
          for (int k = b; k < e; k++) {
            int j = m.A.col_index[k];
            double a = m.A.values[k];
            if (a > 0) {
              double implied_hi = m.lower[j] + slack / a;
              if (implied_hi < m.upper[j] - 1e-12) {
                m.upper[j] = implied_hi;
                changed = true;
              }
            } else if (a < 0) {
              double implied_lo = m.upper[j] + slack / a;
              if (implied_lo > m.lower[j] + 1e-12) {
                m.lower[j] = implied_lo;
                changed = true;
              }
            }
          }
        }
      }

      if (std::isfinite(m.row_lower[i])) {
        double slack = max_act - m.row_lower[i];
        if (slack >= 0.0) {
          for (int k = b; k < e; k++) {
            int j = m.A.col_index[k];
            double a = m.A.values[k];
            if (a > 0) {
              double implied_lo = m.upper[j] - slack / a;
              if (implied_lo > m.lower[j] + 1e-12) {
                m.lower[j] = implied_lo;
                changed = true;
              }
            } else if (a < 0) {
              double implied_hi = m.lower[j] - slack / a;
              if (implied_hi < m.upper[j] - 1e-12) {
                m.upper[j] = implied_hi;
                changed = true;
              }
            }
          }
        }
      }
    }
    if (!changed) break;
  }
}
}
PreprocessResult preprocess_lp(const LPModel&input,int passes){
  PreprocessResult out;out.model=input;
  if(out.model.integer.size()!=out.model.A.cols)out.model.integer.assign(out.model.A.cols,0);
  if(out.model.lower.size()!=out.model.A.cols||out.model.upper.size()!=out.model.A.cols){out.feasible=false;out.message="Variable bound dimension mismatch";return out;}
  if(out.model.row_lower.size()!=out.model.A.rows||out.model.row_upper.size()!=out.model.A.rows){out.feasible=false;out.message="Constraint bound dimension mismatch";return out;}
  for(std::size_t j=0;j<out.model.A.cols;j++)if(out.model.lower[j]>out.model.upper[j]+1e-12){out.feasible=false;out.message="Inconsistent variable bounds";return out;}
  for(std::size_t i=0;i<out.model.A.rows;i++)if(out.model.A.row_ptr[i]==out.model.A.row_ptr[i+1] && (0.0<out.model.row_lower[i]-1e-12||0.0>out.model.row_upper[i]+1e-12)){out.feasible=false;out.message="Infeasible zero row: "+out.model.rows[i].name;return out;}
  tighten_singletons(out.model);
  tighten_implied_bounds(out.model);
  for(std::size_t j=0;j<out.model.A.cols;j++)if(out.model.lower[j]>out.model.upper[j]+1e-12){out.feasible=false;out.message="Presolve detected inconsistent bounds";return out;}

  out.column_scale.assign(out.model.A.cols,1.0);
  if(passes<=0)return out;

  std::vector<double> rs(out.model.A.rows,1.0),cs(out.model.A.cols,1.0);
  passes=std::min(passes,8);
  for(int pass=0;pass<passes;pass++){
    std::vector<double> rmax(out.model.A.rows,0.0);
    for(std::size_t i=0;i<out.model.A.rows;i++)
      for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)
        rmax[i]=std::max(rmax[i],std::abs(out.model.A.values[k])*rs[i]*cs[out.model.A.col_index[k]]);
    for(std::size_t i=0;i<rmax.size();i++)if(rmax[i]>1e-12&&std::isfinite(rmax[i]))rs[i]/=std::sqrt(rmax[i]);

    std::vector<double> cmax(out.model.A.cols,0.0);
    for(std::size_t i=0;i<out.model.A.rows;i++)
      for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++){
        int j=out.model.A.col_index[k];
        cmax[j]=std::max(cmax[j],std::abs(out.model.A.values[k])*rs[i]*cs[j]);
      }
    for(std::size_t j=0;j<cmax.size();j++)if(cmax[j]>1e-12&&std::isfinite(cmax[j]))cs[j]/=std::sqrt(cmax[j]);
  }

  out.column_scale=cs;
  for(std::size_t i=0;i<out.model.A.rows;i++){
    if(std::isfinite(out.model.row_lower[i]))out.model.row_lower[i]*=rs[i];
    if(std::isfinite(out.model.row_upper[i]))out.model.row_upper[i]*=rs[i];
    for(int k=out.model.A.row_ptr[i];k<out.model.A.row_ptr[i+1];k++)out.model.A.values[k]*=rs[i]*cs[out.model.A.col_index[k]];
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
