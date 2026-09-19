# BharatOpt deployment and test guide

## 1. Windows RTX 3050

Open PowerShell in the cloned repository:

    .\setup.ps1 -Cuda

The script configures a Release build, compiles the C++/CUDA engine, runs CTest, and executes the refinery demo.

Confirm the GPU driver/CUDA installation first:

    python scripts/preflight.py

Then:

    .\build\bharatopt_cli.exe --demo --cuda
    .\build\bharatopt_cli.exe --mps examples/max_demo.mps --cuda

The RTX 3050 path targets CUDA compute capability 8.6.

## 2. CPU build

Linux/macOS:

    ./setup.sh

Windows:

    .\setup.ps1

Useful direct commands:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHARATOPT_ENABLE_CUDA=OFF
    cmake --build build
    ctest --test-dir build --output-on-failure

## 3. CLI

    ./build/bharatopt_cli --demo
    ./build/bharatopt_cli --mip-demo --max-nodes 128
    ./build/bharatopt_cli --qp-demo
    ./build/bharatopt_cli --demo --ip
    ./build/bharatopt_cli --mps examples/refinery_blending.mps
    ./build/bharatopt_cli --demo --json

Control numerical work:

    --max-iters N
    --tol T
    --time-limit SECONDS
    --max-nodes N
    --mip-gap T
    --no-presolve
    --scaling-passes N
    --cuda

## 4. Browser console

Create the Python environment:

    python -m venv .venv

Linux/macOS:

    source .venv/bin/activate

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Set the solver binary:

    export BHARATOPT_BIN="$PWD/build/bharatopt_cli"

Windows PowerShell:

    $env:BHARATOPT_BIN="$PWD\build\bharatopt_cli.exe"

Start:

    python -m uvicorn api:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000

## 5. Docker

CPU:

    docker build -t bharatopt:cpu .
    docker run --rm -p 8000:8000 bharatopt:cpu

NVIDIA:

    docker build -f Dockerfile.cuda -t bharatopt:cuda .
    docker run --rm --gpus all -p 8000:8000 bharatopt:cuda

## 6. Generate a feasible stress model

    python scripts/generate_refinery_benchmark.py --variables 10000 --constraints 5000 --nnz-per-row 20 --output data/refinery_10k.mps

Run:

    python scripts/benchmark.py data/refinery_10k.mps --repeats 5
    python scripts/benchmark.py data/refinery_10k.mps --repeats 5 --cuda

## 7. Optional external baseline

HiGHS is a benchmark-only dependency; it is not part of the BharatOpt engine.

    python -m pip install highspy
    python scripts/compare_highs.py data/refinery_10k.mps --binary build/bharatopt_cli

HiGHS can read MPS directly through its executable or Python interface, so the comparison is based on the same model file. citeturn982299search0turn982299search1

## Current technical boundary

BharatOpt is now a real native optimization engine prototype, but the continuous method is still a first-order PDHG-family method and the MILP layer is experimental. It is not appropriate to claim production parity with mature industrial solvers until broader benchmark coverage, stronger presolve, simplex/interior-point methods, complete MIP certificates and extensive numerical testing are completed.
