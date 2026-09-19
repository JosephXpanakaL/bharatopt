#include "bharatopt/mps.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
namespace bharatopt {
namespace { std::vector<std::string> split(const std::string& s){ std::istringstream is(s); std::vector<std::string> v; std::string t; while(is>>t)v.push_back(t); return v; } }
LPModel parse_mps(const std::string& path){
  std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open MPS file: "+path);
  LPModel m; enum class Sec{NONE,ROWS,COLUMNS,RHS,BOUNDS}; Sec sec=Sec::NONE;
  std::vector<std::tuple<std::string,std::string,double>> entries; std::unordered_map<std::string,int> vi,ri; std::string obj;
  std::string line;
  while(std::getline(in,line)){ if(line.empty()||line[0]=='*')continue; auto t=split(line); if(t.empty())continue;
    if(t[0]=="NAME"){if(t.size()>1)m.name=t[1];continue;} if(t[0]=="ROWS"){sec=Sec::ROWS;continue;} if(t[0]=="COLUMNS"){sec=Sec::COLUMNS;continue;}
    if(t[0]=="RHS"){sec=Sec::RHS;continue;} if(t[0]=="BOUNDS"){sec=Sec::BOUNDS;continue;} if(t[0]=="ENDATA")break;
    if(sec==Sec::ROWS){ if(t.size()<2)continue; char s=t[0][0]; ConstraintRow r; r.name=t[1]; if(s=='N')obj=r.name; else {r.sense=s=='L'?RowSense::LessEqual:(s=='G'?RowSense::GreaterEqual:RowSense::Equal);ri[r.name]=(int)m.rows.size();m.rows.push_back(r);} }
    else if(sec==Sec::COLUMNS){ if(t.size()<3)continue; std::string var=t[0]; if(!vi.count(var)){int id=(int)m.var_names.size();vi[var]=id;m.var_names.push_back(var);m.objective.push_back(0);m.lower.push_back(0);m.upper.push_back(INF);} int j=vi[var]; for(size_t k=1;k+1<t.size();k+=2){double val=std::stod(t[k+1]); if(t[k]==obj)m.objective[j]+=val; else entries.emplace_back(t[k],var,val);} }
    else if(sec==Sec::RHS){for(size_t k=1;k+1<t.size();k+=2){auto it=ri.find(t[k]);if(it!=ri.end())m.rows[it->second].rhs=std::stod(t[k+1]);}}
    else if(sec==Sec::BOUNDS){if(t.size()<3)continue; std::string type=t[0],var=t[2]; if(!vi.count(var)){int id=(int)m.var_names.size();vi[var]=id;m.var_names.push_back(var);m.objective.push_back(0);m.lower.push_back(0);m.upper.push_back(INF);} int j=vi[var]; double v=t.size()>3?std::stod(t[3]):0; if(type=="LO")m.lower[j]=v;else if(type=="UP")m.upper[j]=v;else if(type=="FX")m.lower[j]=m.upper[j]=v;else if(type=="FR"||type=="MI")m.lower[j]=-INF;else if(type=="BV"){m.lower[j]=0;m.upper[j]=1;}}
  }
  int n=(int)m.var_names.size(), p=(int)m.rows.size(); m.A.rows=p;m.A.cols=n;m.A.row_ptr.assign(p+1,0); std::vector<std::vector<std::pair<int,double>>> rows(p);
  for(auto& e:entries){auto a=ri.find(std::get<0>(e)),b=vi.find(std::get<1>(e));if(a!=ri.end()&&b!=vi.end())rows[a->second].push_back({b->second,std::get<2>(e)});}
  for(int i=0;i<p;i++){std::sort(rows[i].begin(),rows[i].end());m.A.row_ptr[i+1]=m.A.row_ptr[i]+(int)rows[i].size();for(auto [j,v]:rows[i]){m.A.col_index.push_back(j);m.A.values.push_back(v);}}
  m.row_lower.resize(p);m.row_upper.resize(p);for(int i=0;i<p;i++){auto s=m.rows[i].sense;if(s==RowSense::LessEqual){m.row_lower[i]=-INF;m.row_upper[i]=m.rows[i].rhs;}else if(s==RowSense::GreaterEqual){m.row_lower[i]=m.rows[i].rhs;m.row_upper[i]=INF;}else m.row_lower[i]=m.row_upper[i]=m.rows[i].rhs;}
  return m;
}
LPModel make_refinery_demo(){
  LPModel m; m.name="MRPL-refinery-blending-demo"; m.var_names={"crude_A","crude_B"};m.objective={12,9};m.lower={0,0};m.upper={10,10};
  m.rows={{"sulfur",RowSense::LessEqual,3.5},{"capacity",RowSense::LessEqual,10},{"demand",RowSense::GreaterEqual,8}};
  m.A.rows=3;m.A.cols=2;m.A.row_ptr={0,2,4,6};m.A.col_index={0,1,0,1,0,1};m.A.values={1,2.5,1,1,1,1};m.row_lower={-INF,-INF,8};m.row_upper={3.5,10,INF};return m;
}
}