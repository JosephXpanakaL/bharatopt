# BharatOpt — Indigenous GPU-Accelerated Optimization Solver Prototype

SIH26119 • Team NovaKin • MRPL

BharatOpt is a from-scratch optimization-engine prototype aimed at the technologies required for a sovereign industrial LP/MILP/QP solver. It focuses on sparse mathematical optimization, numerical robustness, adaptive CPU/GPU execution, and reproducible testing.

> This repository is a prototype. It is not a drop-in replacement for CPLEX, Gurobi, or Xpress.

## What you can run now

- Sparse CSR LP model representation
- Practical MPS parser with LP bounds and integer-marker/BV recognition
- CPU primal-dual LP solver (PDHG-style)
- Native CUDA kernels for sparse Ax and bound projection
- CPU-safe CUDA stub and runtime fallback
- Small MILP branch-and-bound layer
- CLI with human-readable and JSON output
- FastAPI web console with MPS upload
- CPU and CUDA Docker images
- GitHub Actions CPU build/test
- Repeatable benchmark CSV script
- MRPL-style blending LP and batch-selection MILP demos

## Quick start

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHARATOPT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bharatopt_cli --demo
./build/bharatopt_cli --mip-demo --max-nodes 64
~~~

See DEPLOY.md for the web console and Docker path.

## CUDA

On an NVIDIA machine with CUDA installed:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHARATOPT_ENABLE_CUDA=ON
cmake --build build -j
./build/bharatopt_cli --demo --cuda
~~~

The current GPU path is deliberately labelled hybrid prototype: Ax and bound projection can execute on GPU, but A^T y remains CPU and device buffers are not yet persistent.

## Mathematical core

The LP form is

min c^T x

subject to

l_c <= A x <= u_c,  l_x <= x <= u_x.

The continuous prototype uses projected primal-dual iterations and stops using normalized feasibility plus projected stationarity residuals.

The MILP prototype adds branch-and-bound around LP relaxations for small integer models. Exact industrial certificates, advanced cutting planes, strong presolve and robust incumbent/bound management remain future work.

## Deployment

CPU Docker:

~~~bash
docker build -t bharatopt:cpu .
docker run --rm -p 8000:8000 bharatopt:cpu
~~~

RTX Docker:

~~~bash
docker build -f Dockerfile.cuda -t bharatopt:cuda .
docker run --rm --gpus all -p 8000:8000 bharatopt:cuda
~~~

## Benchmark protocol

Record model dimensions/NNZ, load time separately from solve time, runtime, iterations, objective, feasibility/KKT residuals, MILP gap/node count, peak RAM/VRAM, cold/warm run and exact build settings. Never invent or mix results across machines.

## Roadmap

1. Persistent GPU-resident vectors/matrix
2. cuSPARSE SpMV + cuBLAS reductions
3. Reversible Ruiz scaling and numerical-risk diagnostics
4. FP32 throughput + periodic FP64 validation/refinement
5. Revised simplex and interior-point engines
6. Stronger MILP branch-and-bound, cuts, heuristics and presolve
7. General convex QP / MIQP support
8. Netlib, MIPLIB, Mittelmann and QPLIB benchmark harness
9. MRPL refinery scheduling/blending case study
10. Thin Python bindings and richer monitoring
