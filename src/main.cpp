#include "bharatopt/mps.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
static std::string js(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';o+=c;}o+='\"';return o;}
static void jnum(std::ostream&o,double v){if(std::isfinite(v))o<<v;else o<<"null";}
int main(int argc,char**argv){
 bool demo=false,mip_demo=false,qp_demo=false,use_cuda=false,json=false,lp_only=false;std::string mps;bharatopt::SolverOptions opt;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--demo")demo=true;else if(a=="--mip-demo")mip_demo=true;else if(a=="--qp-demo")qp_demo=true;else if(a=="--cuda")use_cuda=true;else if(a=="--json")json=true;else if(a=="--lp-only")lp_only=true;else if(a=="--mps"&&i+1<argc)mps=argv[++i];else if(a=="--max-iters"&&i+1<argc)opt.max_iterations=std::stoi(argv[++i]);else if(a=="--tol"&&i+1<argc)opt.tolerance=std::stod(argv[++i]);else if(a=="--max-nodes"&&i+1<argc)opt.max_nodes=std::stoi(argv[++i]);else if(a=="--mip-gap"&&i+1<argc)opt.mip_gap=std::stod(argv[++i]);else{std::cerr<<"Unknown argument: "<<a<<"\n";return 2;}}
 try{
  bharatopt::BharatOptSolverCore solver;opt.use_cuda=use_cuda;
  if(qp_demo){auto q=bharatopt::make_qp_demo();auto r=solver.solve_qp(q,opt);
   if(json){std::cout<<"{\"model\":"<<js(q.name)<<",\"rows\":"<<q.linear.A.rows<<",\"cols\":"<<q.linear.A.cols<<",\"nnz\":"<<q.linear.A.values.size()<<",\"status\":"<<js(r.status)<<",\"converged\":"<<(r.converged?"true":"false")<<",\"iterations\":"<<r.iterations<<",\"objective\":";jnum(std::cout,r.objective);std::cout<<",\"best_bound\":";jnum(std::cout,r.best_bound);std::cout<<",\"mip_gap\":";jnum(std::cout,r.mip_gap);std::cout<<",\"primal_residual\":";jnum(std::cout,r.primal_residual);std::cout<<",\"dual_residual\":";jnum(std::cout,r.dual_residual);std::cout<<",\"solve_time_sec\":";jnum(std::cout,r.solve_time_sec);std::cout<<",\"backend\":"<<js(r.backend)<<"}\n";}
   else{std::cout<<"BharatOpt | model="<<q.name<<" rows="<<q.linear.A.rows<<" cols="<<q.linear.A.cols<<" nnz="<<q.linear.A.values.size()<<"\n";std::cout<<"status="<<r.status<<" converged="<<r.converged<<" iterations="<<r.iterations<<" objective="<<r.objective<<" primal_residual="<<r.primal_residual<<" dual_residual="<<r.dual_residual<<" time_sec="<<r.solve_time_sec<<" backend="<<r.backend<<"\n";std::cout<<"x:";for(double v:r.x)std::cout<<" "<<v;std::cout<<"\n";}return r.converged?0:3;}
  bharatopt::LPModel m;if(demo)m=bharatopt::make_refinery_demo();else if(mip_demo)m=bharatopt::make_milp_demo();else if(!mps.empty())m=bharatopt::parse_mps(mps);else{std::cerr<<"Use --demo, --mip-demo, --qp-demo or --mps file.mps\n";return 2;}
  auto r=lp_only?solver.solve_lp(m,opt):solver.solve(m,opt);
  if(json){std::cout<<"{\"model\":"<<js(m.name)<<",\"rows\":"<<m.A.rows<<",\"cols\":"<<m.A.cols<<",\"nnz\":"<<m.A.values.size()<<",\"status\":"<<js(r.status)<<",\"converged\":"<<(r.converged?"true":"false")<<",\"iterations\":"<<r.iterations<<",\"nodes\":"<<r.nodes<<",\"objective\":";jnum(std::cout,r.objective);std::cout<<",\"best_bound\":";jnum(std::cout,r.best_bound);std::cout<<",\"mip_gap\":";jnum(std::cout,r.mip_gap);std::cout<<",\"primal_residual\":";jnum(std::cout,r.primal_residual);std::cout<<",\"dual_residual\":";jnum(std::cout,r.dual_residual);std::cout<<",\"solve_time_sec\":";jnum(std::cout,r.solve_time_sec);std::cout<<",\"backend\":"<<js(r.backend)<<"}\n";}
  else{std::cout<<"BharatOpt | model="<<m.name<<" rows="<<m.A.rows<<" cols="<<m.A.cols<<" nnz="<<m.A.values.size()<<"\n";if(use_cuda)std::cout<<"CUDA device: "<<bharatopt::cuda_backend::cuda_device_name()<<"\n";std::cout<<"status="<<r.status<<" converged="<<r.converged<<" iterations="<<r.iterations<<" nodes="<<r.nodes<<"\n";std::cout<<"objective="<<r.objective<<" best_bound="<<r.best_bound<<" mip_gap="<<r.mip_gap<<" primal_residual="<<r.primal_residual<<" dual_residual="<<r.dual_residual<<" time_sec="<<r.solve_time_sec<<" backend="<<r.backend<<"\n";std::cout<<"x:";for(double v:r.x)std::cout<<" "<<v;std::cout<<"\n";}
  return r.converged?0:3;
 }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}