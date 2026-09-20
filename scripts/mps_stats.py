#!/usr/bin/env python3
"""
MPS Model Sparsity, Conditioning and Matrix Statistics Analyzer
"""

import sys
from pathlib import Path

def analyze_mps(mps_path):
    p = Path(mps_path)
    if not p.exists():
        print(f"Error: file {mps_path} not found.")
        return 1

    section = None
    rows = {}
    cols = set()
    int_vars = set()
    nnz = 0
    coeffs = []

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("*"): continue
            first_word = line.split()[0].upper()
            if first_word in ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA", "OBJSENSE"):
                section = first_word
                continue

            parts = line.split()
            if section == "ROWS":
                sense = parts[0].upper()
                name = parts[1]
                if sense != "N": rows[name] = sense
            elif section == "COLUMNS":
                if "'MARKER'" in line.upper():
                    continue
                var_name = parts[0]
                cols.add(var_name)
                # Parse pairs
                i = 1
                while i + 1 < len(parts):
                    try:
                        val = float(parts[i+1])
                        coeffs.append(abs(val))
                        nnz += 1
                    except ValueError:
                        pass
                    i += 2
            elif section == "BOUNDS":
                btype = parts[0].upper()
                var_name = parts[2] if len(parts) > 2 else ""
                if btype in ("BV", "UI", "LI") and var_name:
                    int_vars.add(var_name)

    m = len(rows)
    n = len(cols)
    density = (nnz / (m * n) * 100.0) if (m > 0 and n > 0) else 0.0

    print("==================================================")
    print(f"  BharatOpt MPS Model Analysis: {p.name}")
    print("==================================================")
    print(f"  Total Constraints (Rows)  : {m:,}")
    print(f"    - Less-than or equal (<=) : {sum(1 for s in rows.values() if s == 'L'):,}")
    print(f"    - Greater-than (>=)      : {sum(1 for s in rows.values() if s == 'G'):,}")
    print(f"    - Equality (==)          : {sum(1 for s in rows.values() if s == 'E'):,}")
    print(f"  Total Variables (Cols)    : {n:,}")
    print(f"    - Continuous Variables   : {n - len(int_vars):,}")
    print(f"    - Discrete / Binary Vars : {len(int_vars):,}")
    print(f"  Nonzero Elements (NNZ)    : {nnz:,}")
    print(f"  Matrix Density            : {density:.4f}%")

    if coeffs:
        nonzeros = [c for c in coeffs if c > 0]
        if nonzeros:
            min_c = min(nonzeros)
            max_c = max(nonzeros)
            ratio = max_c / min_c
            print(f"  Matrix Coefficient Range  : [{min_c:.2e}, {max_c:.2e}]")
            print(f"  Dynamic Conditioning Ratio: {ratio:.2e}")
            if ratio > 1e6:
                print("  [WARNING] High dynamic range: Ruiz equilibration scaling strongly recommended.")

    print("==================================================")
    return 0

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "examples/refinery_blending.mps"
    sys.exit(analyze_mps(target))
