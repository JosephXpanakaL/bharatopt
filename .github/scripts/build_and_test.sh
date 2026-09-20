#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cmake -S . -B build -G Ninja
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure

echo "Build and tests completed. Build tree: build/"
