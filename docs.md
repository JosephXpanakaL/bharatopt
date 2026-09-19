# BharatOpt Technical Notes

## Hardware strategy
- NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM: CUDA development and GPU benchmark target
- Apple Silicon M4: CPU/orchestration baseline

## Sparse representation
CSR is the primary representation because row-wise Ax is natural. A future backend may retain CSC or an auxiliary transpose structure for A^T y.

## GPU memory
For FP64 values with int32 indices, CSR storage is approximately:
8*NNZ + 4*NNZ + 4*(M+1) bytes,
before vectors, bounds, objective, residual buffers and workspace.
The implementation should query free VRAM at runtime and keep a safety margin. Oversized models should switch to hybrid or CPU mode.

## Numerical robustness
The prototype uses a projected stationarity residual for box-constrained LPs. Full KKT validation for inequality dual signs, complementary slackness, presolve mappings and MILP certificates is future work.

Ruiz scaling is intentionally not active yet because production-quality scaling requires reversible transformations for coefficients, RHS, bounds and objective.

## GPU architecture roadmap
1. Upload model once
2. Keep x, xbar, y, residual and work vectors resident
3. Use cuSPARSE for CSR SpMV
4. Use cuBLAS for reductions and vector updates
5. Keep custom CUDA kernels for projection, fused updates and diagnostics
6. Periodically run FP64 validation and iterative refinement
