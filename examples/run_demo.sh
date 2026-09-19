#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build -DBHARATOPT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bharatopt_cli --demo