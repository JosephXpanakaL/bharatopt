# Synthetic refinery-like stress benchmarks

Use `scripts/generate_refinery_benchmark.py` to generate reproducible sparse LPs without committing large datasets.

Example:

    python scripts/generate_refinery_benchmark.py --variables 10000 --constraints 5000 --nnz-per-row 20 --output data/refinery_10k.mps

The generator is synthetic and is not MRPL operational data. Record generator settings, compiler version, CPU/GPU, CUDA version, solver tolerance and timing when reporting results.
