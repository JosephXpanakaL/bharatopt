#include "bharatopt/mps.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace bharatopt {
namespace {
std::vector<std::string> split(const std::string&s){std::istringstream is(s);std::vector<std::string>v;std::string t;while(is>>t)v.push_back(t);return v;}
bool has_marker(const std::vector<std::string>&t,const char*marker){for(const auto&x:t)if(x.find(marker)!=std::string::npos)return true;return false;}
}
LPModel parse_mps(const std::string&path){
 std::ifstream in(path);if(!in)throw std::runtime_error("Cannot open MPS file: "+path);
 LPModel m;enum class Sec{NONE,OBJSENSE,OBJNAME,ROWS,COLUMNS,RHS,RANGES,BOUNDS,OTHER};Sec sec=Sec::NONE;
 std::vector<std::tuple<std::string,std::string,double>> entries;std::vector<double> ranges;std::unordered_map<std::string,int>vi,ri;std::string obj;std::string rhs_vector,range_vector,bound_vector;bool integer_mode=false;
 std::string line;
 while(std::getline(in,line)){
  if(line.empty()||line[0]=='*')continue;auto t=split(line);if(t.empty())continue;
  if(t[0]=="NAME"){if(t.size()>1)m.name=t[1];continue;}
  if(t[0]=="OBJSENSE"){sec=Sec::OBJSENSE;continue;}
  if(t[0]=="OBJNAME"){sec=Sec::OBJNAME;continue;}
  if(t[0]=="ROWS"){sec=Sec::ROWS;continue;}
  if(t[0]=="COLUMNS"){sec=Sec::COLUMNS;continue;}
  if(t[0]=="RHS"){sec=Sec::RHS;continue;}
  if(t[0]=="RANGES"){sec=Sec::RANGES;ranges.assign(m.rows.size(),0.0);continue;}
  if(t[0]=="BOUNDS"){sec=Sec::BOUNDS;continue;}
  if(t[0]=="ENDATA")break;
  if(sec==Sec::OBJSENSE){std::string s=t.back();if(s=="MAX")m.maximize=true;else if(s=="MIN")m.maximize=false;else throw std::runtime_error("Unsupported OBJSENSE: "+s);sec=Sec::NONE;continue;}
  if(sec==Sec::OBJNAME){if(t.size()>0)obj=t.back();sec=Sec::NONE;continue;}
  if(sec==Sec::ROWS){
   if(t.size()<2)continue;char s=t[0][0];ConstraintRow row;row.name=t[1];
   if(s=='N'){if(obj.empty())obj=row.name;}else{row.sense=s=='L'?RowSense::LessEqual:(s=='G'?RowSense::GreaterEqual:RowSense::Equal);ri[row.name]=(int)m.rows.size();m.rows.push_back(row);ranges.push_back(0.0);}
  }else if(sec==Sec::COLUMNS){
   if(has_marker(t,"MARKER")){if(has_marker(t,"INTORG"))integer_mode=true;if(has_marker(t,"INTEND"))integer_mode=false;continue;}
   if(t.size()<3)continue;std::string var=t[0];
   if(!vi.count(var)){int id=(int)m.var_names.size();vi[var]=id;m.var_names.push_back(var);m.objective.push_back(0.0);m.lower.push_back(0.0);m.upper.push_back(INF);m.integer.push_back(integer_mode?1:0);}
   else if(integer_mode)m.integer[vi[var]]=1;
   int j=vi[var];
   for(std::size_t k=1;k+1<t.size();k+=2){double value=std::stod(t[k+1]);if(t[k]==obj)m.objective[j]+=value;else entries.emplace_back(t[k],var,value);}
  }else if(sec==Sec::RHS){
   if(rhs_vector.empty())rhs_vector=t[0];
   if(t[0]!=rhs_vector)continue;
   for(std::size_t k=1;k+1<t.size();k+=2){auto it=ri.find(t[k]);if(it!=ri.end())m.rows[it->second].rhs=std::stod(t[k+1]);else if(t[k]==obj)m.objective_offset-=std::stod(t[k+1]);}
  }else if(sec==Sec::RANGES){
   if(range_vector.empty())range_vector=t[0];
   if(t[0]!=range_vector)continue;
   for(std::size_t k=1;k+1<t.size();k+=2){auto it=ri.find(t[k]);if(it!=ri.end())ranges[it->second]=std::stod(t[k+1]);}
  }else if(sec==Sec::BOUNDS){
   if(bound_vector.empty())bound_vector=t[1];
   if(t[1]!=bound_vector)continue;
   if(t.size()<3)continue;std::string type=t[0],var=t[2];
   if(!vi.count(var)){int id=(int)m.var_names.size();vi[var]=id;m.var_names.push_back(var);m.objective.push_back(0.0);m.lower.push_back(0.0);m.upper.push_back(INF);m.integer.push_back(0);}
   int j=vi[var];double v=t.size()>3?std::stod(t[3]):0.0;
   if(type=="LO"||type=="LI")m.lower[j]=v;
   else if(type=="UP"||type=="UI"){m.upper[j]=v;if(type=="UI")m.integer[j]=1;}
   else if(type=="FX")m.lower[j]=m.upper[j]=v;
   else if(type=="FR")m.lower[j]=-INF,m.upper[j]=INF;
   else if(type=="MI")m.lower[j]=-INF;
   else if(type=="PL")m.upper[j]=INF;
   else if(type=="BV")m.lower[j]=0,m.upper[j]=1,m.integer[j]=1;
  }
 }
 int n=(int)m.var_names.size(),p=(int)m.rows.size();m.A.rows=p;m.A.cols=n;m.A.row_ptr.assign(p+1,0);std::vector<std::vector<std::pair<int,double>>>rows(p);
 for(auto&e:entries){auto a=ri.find(std::get<0>(e)),b=vi.find(std::get<1>(e));if(a!=ri.end()&&b!=vi.end())rows[a->second].push_back({b->second,std::get<2>(e)});}
 for(int i=0;i<p;i++){std::sort(rows[i].begin(),rows[i].end());m.A.row_ptr[i+1]=m.A.row_ptr[i]+(int)rows[i].size();for(auto [j,v]:rows[i]){m.A.col_index.push_back(j);m.A.values.push_back(v);}}
 m.row_lower.resize(p);m.row_upper.resize(p);
 for(int i=0;i<p;i++){auto s=m.rows[i].sense;double b=m.rows[i].rhs;double rr=ranges[i];if(rr==0){if(s==RowSense::LessEqual)m.row_lower[i]=-INF,m.row_upper[i]=b;else if(s==RowSense::GreaterEqual)m.row_lower[i]=b,m.row_upper[i]=INF;else m.row_lower[i]=m.row_upper[i]=b;}else if(s==RowSense::GreaterEqual){m.row_lower[i]=b;m.row_upper[i]=b+std::abs(rr);}else if(s==RowSense::LessEqual){m.row_lower[i]=b-std::abs(rr);m.row_upper[i]=b;}else if(rr>0){m.row_lower[i]=b;m.row_upper[i]=b+rr;}else{m.row_lower[i]=b+rr;m.row_upper[i]=b;}}
 if(m.maximize)for(double&v:m.objective)v=-v;
 return m;
}
LPModel make_refinery_demo(){
 LPModel m;m.name="MRPL-refinery-blending-demo";m.var_names={"crude_A","crude_B"};m.objective={12,9};m.lower={0,0};m.upper={10,10};m.integer={0,0};
 m.rows={{"sulfur",RowSense::LessEqual,3.5},{"capacity",RowSense::LessEqual,10},{"demand",RowSense::GreaterEqual,8}};
 m.A.rows=3;m.A.cols=2;m.A.row_ptr={0,2,4,6};m.A.col_index={0,1,0,1,0,1};m.A.values={0.2,0.35,1,1,1,1};m.row_lower={-INF,-INF,8};m.row_upper={3.5,10,INF};return m;
}
LPModel make_milp_demo(){
 LPModel m;m.name="MRPL-batch-selection-milp-demo";m.var_names={"unit_A","unit_B","unit_C"};m.objective={-7,-6,-5};m.lower={0,0,0};m.upper={1,1,1};m.integer={1,1,1};
 m.rows={{"budget",RowSense::LessEqual,2},{"coverage",RowSense::GreaterEqual,1}};m.A.rows=2;m.A.cols=3;m.A.row_ptr={0,3,6};m.A.col_index={0,1,2,0,1,2};m.A.values={1,1,1,1,1,0};m.row_lower={-INF,1};m.row_upper={2,INF};return m;
}
}
