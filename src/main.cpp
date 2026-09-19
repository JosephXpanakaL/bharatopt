#include "bharatopt/mps.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <iostream>
#include <string>
int main(int argc,char**argv){
 bool demo=false,mip_demo=false,qp_demo=false,use_cuda=false,json=false,lp_only=false; std::string mps;
 bharatopt::SolverOptions o;
 for(int i=1;i<argc;i++){std::string a=argv[i];
  if(a=="--demo")demo=true;else if(a=="--mip-demo")mip_demo=true;else if(a=="--qp-demo")qp_demo=true;else if(a=="--cuda")use_cuda=true;else if(a=="--json")json=true;else if(a=="--lp-only")lp_only=true;
  else if(a=="--mps"&&i+1<argc)mps=argv[++i];else if(a=="--max-iters"&&i+1<argc)o.max_iterations=std::stoi(argv[++i]);
  else if(a=="--tol"&&i+1<argc)o.tolerance=std::stod(argv[++i]);else if(a=="--max-nodes"&&i+1<argc)o.max_nodes=std::stoi(argv[++i]);
  else if(a=="--mip-gap"&&i+1<argc)o.mip_gap=std::stod(argv[++i]);else if(a=="--verbose")o.verbose=true;
  else{std::cerr<<"Unknown argument: "<<a<<"\n";return 2;}
 }
 try{
  bharatopt::LPModel m;if(demo)m=bharatopt::make_refinery_demo();else if(mip_demo)m=bharatopt::make_milp_demo();else if(!mps.empty())m=bharatopt::parse_mps(mps);else if(qp_demo){auto q=bharatopt::make_qp_demo();o.use_cuda=use_cuda;auto r=s.solve_qp(q,o);std::cout<<"BharatOpt | model="<<q.name<<" rows="<<q.linear.A.rows<<" cols="<<q.linear.A.cols<<" nnz="<<q.linear.A.values.size()<<"\n";std::cout<<"status="<<r.status<<" converged="<<r.converged<<" iterations="<<r.iterations<<"\n";std::cout<<"objective="<<r.objective<<" primal_residual="<<r.primal_residual<<" dual_residual="<<r.dual_residual<<" time_sec="<<r.solve_time_sec<<" backend="<<r.backend<<"\n";std::cout<<"x:";for(double v:r.x)std::cout<<" "<<v;std::cout<<"\n";return r.converged?0:3;}else{std::cerr<<"Use --demo, --mip-demo, --qp-demo or --mps file.mps\n";return 2;}
  o.use_cuda=use_cuda; bharatopt::BharatOptSolverCore s; auto r=lp_only?s.solve_lp(m,o):s.solve(m,o);
  if(json){
   std::cout<<"{";
   std::cout<<"\"model\":\""<<m.name<<"\",\"rows\":"<<m.A.rows<<",\"cols\":"<<m.A.cols<<",\"nnz\":"<<m.A.values.size();
   std::cout<<",\"status\":\""<<r.status<<"\",\"converged\":"<<(r.converged?"true":"false");
   std::cout<<",\"iterations\":"<<r.iterations<<",\"nodes\":"<<r.nodes<<",\"objective\":"<<r.objective<<",\"best_bound\":"<<r.best_bound;
   std::cout<<",\"mip_gap\":"<<r.mip_gap<<",\"primal_residual\":"<<r.primal_residual<<",\"dual_residual\":"<<r.dual_residual;
   std::cout<<",\"solve_time_sec\":"<<r.solve_time_sec<<",\"backend\":\""<<r.backend<<"\"}\n";
  }else{
   std::cout<<"BharatOpt | model="<<m.name<<" rows="<<m.A.rows<<" cols="<<m.A.cols<<" nnz="<<m.A.values.size()<<"\n";
   if(use_cuda)std::cout<<"CUDA device: "<<bharatopt::cuda_backend::cuda_device_name()<<"\n";
   std::cout<<"status="<<r.status<<" converged="<<r.converged<<" iterations="<<r.iterations<<" nodes="<<r.nodes<<"\n";
   std::cout<<"objective="<<r.objective<<" best_bound="<<r.best_bound<<" mip_gap="<<r.mip_gap<<" primal_residual="<<r.primal_residual<<" dual_residual="<<r.dual_residual<<" time_sec="<<r.solve_time_sec<<" backend="<<r.backend<<"\n";
   if(!r.x.empty()){std::cout<<"x:";for(double v:r.x)std::cout<<" "<<v;std::cout<<"\n";}
  }
  return r.converged?0:3;
 }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}