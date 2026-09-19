# BharatOpt deployment and test guide

## Local CPU test

Requirements: C++20 compiler, CMake 3.20+ and Python 3.10+.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHARATOPT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bharatopt_cli --demo
./build/bharatopt_cli --mip-demo --max-nodes 64
./build/bharatopt_cli --mps examples/milp_demo.mps --max-nodes 64
~~~

## Web console

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BHARATOPT_BIN="$PWD/build/bharatopt_cli"
uvicorn api:app --host 127.0.0.1 --port 8000
~~~

Windows PowerShell:

~~~powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:BHARATOPT_BIN="$PWD\build\bharatopt_cli.exe"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000

## CPU Docker

~~~bash
docker build -t bharatopt:cpu .
docker run --rm -p 8000:8000 bharatopt:cpu
~~~

## RTX 3050 Docker

With a working NVIDIA driver and NVIDIA Container Toolkit:

~~~bash
docker build -f Dockerfile.cuda -t bharatopt:cuda .
docker run --rm --gpus all -p 8000:8000 bharatopt:cuda
~~~

Then tick Use CUDA backend in the dashboard.

## Benchmark

~~~bash
./build/bharatopt_cli --demo --json
python scripts/benchmark.py examples/refinery_blending.mps 5
~~~

The benchmark script writes benchmark_results.csv locally. Do not commit measured numbers until the machine, CUDA driver, build type, tolerance and model set are recorded.

## Current boundaries

The CUDA path is a functional hybrid prototype: sparse Ax and box projection use native CUDA kernels, while A^T y remains on CPU. The current CUDA wrapper is intentionally simple and allocates device buffers per call, so it is not suitable for claiming production GPU speedups yet.

MILP uses a small branch-and-bound layer over approximate LP relaxations. It is intended for prototype instances and is not a certificate-grade industrial MIP engine.

The next production step is persistent GPU-resident state with cuSPARSE/cuBLAS, followed by numerical scaling, stronger presolve/KKT validation, and a more complete MILP stack.
