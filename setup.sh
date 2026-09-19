#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-bharatopt}"
mkdir -p "$PROJECT_DIR"
echo "BharatOpt project bootstrap for $PROJECT_DIR"
echo "Repository: https://github.com/JosephXpanakaL/bharatopt"
echo "Use the repository contents directly, then build with CMake."
