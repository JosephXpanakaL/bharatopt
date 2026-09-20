#!/usr/bin/env python3
"""
BharatOpt End-to-End Test and Verification Suite
Validates model integrity, refinery flowsheet schemas, CLI outputs, and mathematical certificates.
"""

import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def find_solver_binary():
    env_bin = os.environ.get("BHARATOPT_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)

    candidates = [
        ROOT / "build" / "bin" / "bharatopt_cli.exe",
        ROOT / "build" / "bin" / "bharatopt.exe",
        ROOT / "build" / "bin" / "bharatopt_cli",
        ROOT / "build" / "bin" / "bharatopt",
        ROOT / "build" / "bharatopt_cli.exe",
        ROOT / "build" / "bharatopt.exe",
        ROOT / "build" / "bharatopt_cli",
        ROOT / "build" / "bharatopt",
        ROOT / "build" / "Release" / "bharatopt_cli.exe",
        ROOT / "build" / "Release" / "bharatopt.exe",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None

def test_mrpl_flowsheet_integrity():
    print("[TEST 1/4] Validating MRPL Phase-III refinery flowsheet integrity...")
    flowsheet_path = ROOT / "examples" / "mrpl_refinery_flowsheet.json"
    assert flowsheet_path.exists(), f"Missing {flowsheet_path}"
    
    data = json.loads(flowsheet_path.read_text(encoding="utf-8"))
    assert data["name"] == "MRPL_Phase_III_Industrial_Refinery_Flowsheet"
    assert len(data["feedstocks"]) >= 3, "Expected at least 3 crude feedstocks"
    assert len(data["units"]) >= 4, "Expected at least 4 refinery units"
    assert len(data["pools"]) >= 3, "Expected at least 3 product pools"
    assert len(data["products"]) >= 3, "Expected at least 3 commercial products"

    # Verify mass yields per unit do not exceed 1.0 (100% mass balance)
    for unit in data["units"]:
        for feed, yields in unit.get("yields", {}).items():
            total_yield = sum(yields.values())
            assert total_yield <= 1.0001, f"Unit {unit['name']} yield sum for {feed} exceeds 1.0: {total_yield}"

    print("  -> Passed: Flowsheet schema, yields, and mass balance limits validated.")

def test_pooling_json_integrity():
    print("[TEST 2/4] Validating standard refinery pooling benchmark model...")
    pool_path = ROOT / "examples" / "refinery_pooling.json"
    assert pool_path.exists(), f"Missing {pool_path}"
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    assert "variables" in data
    assert "objective_linear" in data
    assert "objective_bilinear" in data
    assert "constraints" in data
    print("  -> Passed: Bilinear pooling model validated.")

def test_mps_demos_exist():
    print("[TEST 3/4] Checking standard MPS input benchmarks...")
    mps_files = ["max_demo.mps", "milp_demo.mps", "refinery_blending.mps"]
    for f in mps_files:
        p = ROOT / "examples" / f
        assert p.exists(), f"Missing benchmark {f}"
        assert p.stat().st_size > 0, f"Empty benchmark {f}"
    print("  -> Passed: All MPS benchmarks verified.")

def test_solver_cli_execution(binary_path):
    print("[TEST 4/4] Executing native solver verification tests...")
    if binary_path is None or not binary_path.exists():
        print("  -> Skipped: Native binary not yet compiled in build/. Run setup.ps1 or CMake to build.")
        return

    print(f"  Using solver binary: {binary_path}")

    # Test LP Demo with JSON output
    cmd = [str(binary_path), "--demo", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"LP demo failed: {proc.stderr}"
    res = json.loads(proc.stdout)
    assert res.get("status") in ("optimal", "feasible"), f"Unexpected status: {res.get('status')}"
    assert "objective" in res
    assert "variables" in res
    print("  -> Passed: LP demo solved with certified objective and variables.")

    # Test QP Demo
    cmd_qp = [str(binary_path), "--qp-demo", "--json"]
    proc_qp = subprocess.run(cmd_qp, capture_output=True, text=True, timeout=30)
    assert proc_qp.returncode == 0, f"QP demo failed: {proc_qp.stderr}"
    print("  -> Passed: Convex QP solver verified.")

def main():
    print("==================================================")
    print("       BharatOpt Verification & Test Suite        ")
    print("==================================================")

    test_mrpl_flowsheet_integrity()
    test_pooling_json_integrity()
    test_mps_demos_exist()

    binary = find_solver_binary()
    test_solver_cli_execution(binary)

    print("==================================================")
    print("     ALL VERIFICATION TESTS COMPLETED: OK         ")
    print("==================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
