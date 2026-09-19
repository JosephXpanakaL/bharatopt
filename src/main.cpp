#include "bharatopt/mps.hpp"
#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <iostream>
#include <string>
int main(int argc,char**argv){
 bool demo=false,cuda=false;std::string mps; bharatopt::SolverOptions o;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--demo")demo=true;else if(a=="--cuda")cuda=true;else if(a=="--mps"&&i+1<argc)mps=argv[++i];else if(a=="--max-iters"&&i+1<argc)o.max_iterations=std::stoi(argv[++i]);else if(a=="--tol"&&i+1<argc)o.tolerance=std::stod(argv[++i]);else if(a=="--verbose")o.verbose=true;else{std::cerr<<"Unknown argument: "<<a<<"\n";return 2;}}
 try{bharatopt::LPModel m;if(demo)m=bharatopt::make_refinery_demo();else if(!mps.empty())m=bharatopt::parse_mps(mps);else{std::cerr<<"Use --demo or --mps file.mps\n";return 2;}o.use_cuda=cuda;std::cout<<"Model: "<<m.name<<" rows="<<m.A.rows<<" cols="<<m.A.cols<<" nnz="<<m.A.values.size()<<"\n";if(cuda)std::cout<<"CUDA device: "<<bharatopt::cuda_backend::cuda_device_name()<<"\n";bharatopt::BharatOptSolverCore s;auto r=s.solve(m,o);std::cout<<"status="<<r.status<<" converged="<<r.converged<<" iterations="<<r.iterations<<"\n";std::cout<<"objective="<<r.objective<<" primal_residual="<<r.primal_residual<<" dual_residual="<<r.dual_residual<<" time_sec="<<r.solve_time_sec<<"\n";std::cout<<"x: ";for(double v:r.x)std::cout<<v<<" ";std::cout<<"\n";}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
