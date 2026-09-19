# BharatOpt — Indigenous GPU-Accelerated Optimization Solver Prototype

**SIH26119 · Team NovaKin · MRPL**

BharatOpt is a from-scratch optimization-engine prototype aimed at the technologies required for a sovereign industrial LP/MILP/QP solver. It focuses on sparse mathematical optimization, numerical robustness, adaptive CPU/GPU execution, and a transparent benchmarking path.

> This repository is a prototype. It is not a drop-in replacement for CPLEX, Gurobi, or Xpress.

## Current prototype
- Canonical LP model representation with sparse CSR matrices
- Practical MPS parser for LP-style models
- CPU projected primal-dual method (PDHG-style)
- Native CUDA kernels for CSR SpMV and box projection
- Hybrid CUDA mode: Ax and projection can use GPU; transpose multiply remains CPU
- CPU fallback
- Solver certificate with objective, primal feasibility, and projected stationarity residual
- Synthetic MRPL-style crude-blending demo
- Unit test and CMake build

## Mathematical core
For min c^T x subject to l_c <= A x <= u_c and l_x <= x <= u_x, the prototype uses a primal-dual projected method:

q = y_k + sigma A xbar_k
y_(k+1) = q - sigma P_[l_c,u_c](q/sigma)
x_(k+1) = P_[l_x,u_x](x_k - tau(c+A^T y_(k+1)))
xbar_(k+1) = x_(k+1) + theta(x_(k+1)-xbar_k)

Stopping uses normalized primal feasibility and projected stationarity residuals.

## Build
CPU:
cmake -S . -B build -DBHARATOPT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bharatopt_cli --demo

CUDA:
cmake -S . -B build -DBHARATOPT_ENABLE_CUDA=ON
cmake --build build -j
./build/bharatopt_cli --demo --cuda

Target development hardware includes an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) and Apple Silicon M4 CPU baseline.

## Roadmap
1. Reversible Ruiz scaling and numerical risk diagnostics
2. Persistent GPU-resident vectors and sparse matrix storage
3. cuSPARSE SpMV + cuBLAS vector operations
4. FP32 throughput + periodic FP64 validation/refinement
5. Interior-point and revised-simplex continuous engines
6. MILP branch-and-bound, cuts, presolve and heuristics
7. QP / MIQP support
8. MIPLIB, Netlib, Mittelmann and QPLIB benchmark harness
9. MRPL refinery scheduling/blending benchmark model
10. Python bindings and dashboard as a thin orchestration layer

## Benchmark discipline
Report model dimensions/NNZ, loading time separately from solve time, runtime, iterations, objective, primal/KKT residuals, MILP gap/node count when available, peak RAM/VRAM, and cold vs warm run. External solvers are baselines, not dependencies of the core engine.
