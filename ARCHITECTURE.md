# BharatOpt architecture

## Execution layers

1. Model layer: LPModel stores variables, bounds, objective, constraint intervals and CSR data.
2. Input layer: MPS parser normalizes rows into lower/upper intervals and records integrality.
3. Presolve/scaling: singleton bound tightening, zero-row infeasibility checks and reversible row/column equilibration.
4. Continuous engine: projected primal-dual hybrid-gradient style iterations.
5. GPU engine: persistent device CSR plus cuSPARSE SpMV, cuBLAS reductions and custom CUDA update/projection kernels.
6. Discrete engine: branch-and-bound around continuous LP relaxations with integrality branching and dual-bound pruning.
7. QP engine: convex quadratic objective prototype using sparse Q.
8. Interface layer: native CLI, JSON output and FastAPI dashboard.

## Scaling map

For positive diagonal column scaling C and row scaling R:

    A_s = R A C
    x = C x_s
    c_s = C c
    l_x,s = C^-1 l_x
    u_x,s = C^-1 u_x
    l_c,s = R l_c
    u_c,s = R u_c

The original primal solution is recovered with x = C x_s.

## Dual lower bound

For interval constraints and box bounds, any multiplier vector y gives the Lagrangian lower bound

    g(y) = -support_[l_c,u_c](y) + min_[l_x,u_x] (c + A^T y)^T x

when the support and box minimization are finite. BharatOpt uses this bound as a certificate signal for branch-and-bound. The current LP method remains first-order, so primal and dual tolerances must be reported with benchmark results.

## GPU memory policy

CSR uses FP64 values and 32-bit row/column indices in the current engine. The GPU backend estimates the resident footprint and declines GPU execution when the estimate exceeds a safety fraction of free VRAM. The caller then falls back to CPU.

## NVIDIA path

The current GPU implementation keeps matrix data and the main iteration vectors on device memory. cuSPARSE handles both A*x and A^T*y; cuBLAS handles norms/dots; custom CUDA kernels handle primal projection and update logic.

Future performance work: diagonal preconditioning, mixed-precision FP32/FP64 refinement, SpMV preprocessing/plans, fused diagnostics, multi-stream transfers for out-of-core cases and sparse factorization for an interior-point method.

## Industrial roadmap

Production parity requires a stronger continuous stack (interior-point + revised simplex), full presolve and basis management, robust MIP heuristics/cuts, warm starts, complete numerical certificates and broad benchmark validation.