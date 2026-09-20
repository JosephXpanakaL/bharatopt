# BharatOpt — Indigenous GPU-Accelerated Optimization Solver

SIH26119 • Team NovaKin • MRPL

BharatOpt is a from-scratch sparse mathematical optimization engine. The core is C++20; the NVIDIA path uses CUDA, cuSPARSE and cuBLAS. The project is designed to become a sovereign industrial LP/MILP/QP stack rather than a wrapper around an existing solver.

> Current state: an executable research prototype with working LP, small MILP, convex QP, MPS input, configurable nonlinear pooling/refinery JSON models, SLP, McCormick relaxations, constraint auditing, dashboard integration, tests and CI. It is not yet a production replacement for CPLEX/Gurobi/Xpress.

## Implemented engine

### LP
- General sparse CSR matrix representation.
- LP form: min c^T x subject to l_c <= A x <= u_c and l_x <= x <= u_x.
- Projected primal-dual hybrid-gradient style iterations.
- Normalized primal feasibility and projected stationarity residuals.
- Reversible Ruiz-style row/column equilibration.
- Safe presolve bound tightening for singleton rows, zero-row infeasibility detection and bound consistency checks.
- Numerical-failure detection plus iteration and wall-clock limits.

### MPS
- NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA.
- First RHS/range/bound vector selection.
- LO, LI, UP, UI, FX, FR, MI, PL, BV.
- Integer INTORG / INTEND markers.
- OBJSENSE MIN/MAX with internal minimization normalization.
- Objective offset from the RHS objective row.

### MILP
- LP-relaxation branch-and-bound for small/medium prototype instances.
- Integrality tolerance, node limit, time limit and relative gap controls.
- Experimental: not yet certificate-grade compared with mature industrial MIP solvers.

### QP
- Convex quadratic objective prototype with sparse Q.
- Projected primal-dual iterations with the quadratic gradient term.
- QP demo and regression test.

### Nonlinear pooling / refinery models
- Configurable JSON model input with named continuous variables, bounds, linear objective terms, bilinear objective terms, linear constraints and bilinear constraint terms.
- Sequential Linear Programming (SLP) with trust-region acceptance/rejection based on predicted versus true nonlinear improvement.
- McCormick convex-hull relaxation for bilinear terms; for maximization, its solved objective is reported as a global upper bound on the nonlinear optimum.
- Post-solve nonlinear constraint audit reports the maximum true-model violation and the actual variable values used by the SLP solution.
- The web console accepts `.json` refinery models directly; the built-in pooling benchmark remains available only as an engine regression check.

Example:
    ./build/bharatopt --refinery-json examples/refinery_pooling.json --output result.json --mode certified

### Interior point
- Small-model Mehrotra predictor-corrector primal-dual method.
- Standard-form transformation for bounds, interval constraints, slacks and free-variable splitting.
- Sparse model input is accepted, with dense normal-equation factorization intentionally limited to small instances.

### GPU
The NVIDIA path is GPU-resident PDHG:
- CSR matrix and solver vectors uploaded once.
- cuSPARSE for A x and A^T y.
- cuBLAS for dot products and norms.
- Custom CUDA kernels for projection, dual/primal updates, extrapolation and residual diagnostics.
- FP64 arithmetic in the current path.
- Runtime VRAM estimate with CPU fallback when the model does not safely fit.

The GPU implementation is deliberately conservative. Mixed precision, sparse factorization and full production-grade MIP acceleration remain future work.

## Quick start

Linux/macOS CPU:
    ./setup.sh
    ./build/bharatopt_cli --demo
    ./build/bharatopt_cli --mip-demo
    ./build/bharatopt_cli --qp-demo
    ./build/bharatopt_cli --demo --ip

Windows PowerShell:
    .\setup.ps1
    .\build\bharatopt_cli.exe --demo

NVIDIA GPU:
    ./setup.sh --cuda
    ./build/bharatopt_cli --demo --cuda

MPS:
    ./build/bharatopt_cli --mps examples/refinery_blending.mps
    ./build/bharatopt_cli --mps examples/max_demo.mps

JSON:
    ./build/bharatopt_cli --demo --json

## Web console

See DEPLOY.md. The FastAPI service exposes /health, /api/solve/demo, /api/solve/qp-demo and /api/solve/mps.

## Stress testing

Generate a deterministic feasible sparse refinery-like LP:
    python scripts/generate_refinery_benchmark.py --variables 10000 --constraints 5000 --nnz-per-row 20 --output data/refinery_10k.mps

Run repeated measurements:
    python scripts/benchmark.py data/refinery_10k.mps --repeats 5
    python scripts/benchmark.py data/refinery_10k.mps --repeats 5 --cuda

Never fabricate benchmark results. Record the model generator settings, machine, compiler, CUDA version, tolerance and build type with every measurement.

## Build validation

GitHub Actions validates C++ build, tests, LP/MILP/QP smoke tests, maximization MPS parsing, JSON output, Python syntax and CUDA compilation in an NVIDIA CUDA development container.

## Roadmap
1. Stronger presolve: redundancy removal, implied bounds, coefficient tightening and probing.
2. Sparse factorization path for an interior-point method.
3. Revised simplex with sparse basis updates.
4. Parallel cut generation and MILP heuristics.
5. Better node queueing, strong branching and warm starts.
6. Mixed FP32/FP64 execution with iterative refinement.
7. GPU/CPU structure-aware selection and out-of-core execution.
8. Netlib/Mittelmann/QPLIB/MIPLIB benchmark harness and external-baseline runner.
9. Feasible-start/Phase-1 construction for nonlinear refinery models so models do not require a feasible midpoint.
10. Multi-pool refinery flows, product-quality equations and unit-operation constraints driven by plant data.
11. Global spatial branch-and-bound / tighter relaxations for stronger nonlinear optimality certificates.
12. Stable C API and Python bindings over the native engine.

## Project layout

    include/bharatopt/
      model.hpp
      mps.hpp
      qp.hpp
      preprocess.hpp
      solver.hpp
      cuda_backend.hpp
    src/
      main.cpp
      mps.cpp
      preprocess.cpp
      qp.cpp
      solver.cpp
      cuda_backend.cu
      cuda_stub.cpp
    scripts/
      benchmark.py
      generate_refinery_benchmark.py
      preflight.py
    benchmarks/
    DEPLOY.md
    Dockerfile
    Dockerfile.cuda

## License

MIT
