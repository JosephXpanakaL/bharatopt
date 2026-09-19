#!/usr/bin/env bash
set -euo pipefail

CUDA=OFF
BUILD_TYPE=Release
for arg in "$@"; do
  case "$arg" in
    --cuda) CUDA=ON ;;
    --debug) BUILD_TYPE=Debug ;;
    *) echo "usage: ./setup.sh [--cuda] [--debug]"; exit 2 ;;
  esac
done

echo "== BharatOpt build =="
cmake -S . -B build -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBHARATOPT_ENABLE_CUDA="$CUDA"
cmake --build build --parallel
ctest --test-dir build --output-on-failure

echo
echo "Build complete."
if [[ "$CUDA" == "ON" ]]; then
  echo "Run: ./build/bharatopt_cli --demo --cuda"
else
  echo "Run: ./build/bharatopt_cli --demo"
fi
echo "Web console: install requirements.txt and run uvicorn api:app"
