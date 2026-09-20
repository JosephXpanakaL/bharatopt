#!/usr/bin/env python3
"""
BharatOpt Benchmark Evaluation Runner
Benchmarking harness for Netlib / Mittelmann LP/MILP and Refinery Flowsheet problem suites.
Executes the sovereign BharatOpt engine across benchmark instances and generates performance profiles.
"""

import sys
import os
import glob
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List

def find_bharatopt_binary() -> str:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "bin" / "Release" / "bharatopt.exe",
        root / "build" / "bin" / "bharatopt.exe",
        root / "build" / "bin" / "bharatopt",
        root / "build" / "bharatopt.exe",
        root / "build" / "bharatopt",
        root / "bin" / "bharatopt.exe",
        root / "bin" / "bharatopt",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)
    return "bharatopt"

def run_instance(binary: str, instance_path: str, max_iter: int = 50000, tol: float = 1e-6) -> Dict[str, Any]:
    cmd = [binary, instance_path, "--json", f"--max-iterations={max_iter}", f"--tol={tol}"]
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        wall_time = time.perf_counter() - start_time
        
        # Parse JSON output from stdout
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    data["wall_time_sec"] = wall_time
                    data["returncode"] = proc.returncode
                    data["instance"] = os.path.basename(instance_path)
                    return data
                except json.JSONDecodeError:
                    pass

        return {
            "instance": os.path.basename(instance_path),
            "status": "PARSE_ERROR" if proc.returncode == 0 else "EXEC_FAILED",
            "returncode": proc.returncode,
            "wall_time_sec": wall_time,
            "raw_stdout": proc.stdout[:300],
            "raw_stderr": proc.stderr[:300],
        }
    except subprocess.TimeoutExpired:
        return {
            "instance": os.path.basename(instance_path),
            "status": "TIMEOUT",
            "returncode": -1,
            "wall_time_sec": 120.0
        }
    except Exception as e:
        return {
            "instance": os.path.basename(instance_path),
            "status": f"ERROR: {str(e)}",
            "returncode": -2,
            "wall_time_sec": 0.0
        }

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 95)
    print(f"{'Instance':<30} | {'Status':<18} | {'Objective':<14} | {'Iter':<8} | {'Time (s)':<10}")
    print("=" * 95)
    for r in results:
        inst = r.get("instance", "unknown")[:30]
        status = r.get("status", "N/A")[:18]
        obj = r.get("objective")
        obj_str = f"{obj:14.4f}" if isinstance(obj, (int, float)) else "N/A"
        iters = r.get("iterations", "N/A")
        t = r.get("time_sec", r.get("wall_time_sec", 0.0))
        t_str = f"{t:10.4f}"
        print(f"{inst:<30} | {status:<18} | {obj_str:<14} | {str(iters):<8} | {t_str:<10}")
    print("=" * 95 + "\n")

def main():
    parser = argparse.ArgumentParser(description="BharatOpt Benchmark Runner")
    parser.add_argument("--dir", default=None, help="Directory containing .mps or flowsheet .json instances")
    parser.add_argument("--binary", default=None, help="Path to bharatopt binary")
    parser.add_argument("--json-out", default=None, help="Save benchmark results to JSON")
    parser.add_argument("--csv-out", default=None, help="Save benchmark results to CSV")
    parser.add_argument("--max-iterations", type=int, default=50000, help="Max iterations per solve")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    args = parser.parse_args()

    binary = args.binary or find_bharatopt_binary()
    print(f"[BharatOpt Benchmark] Using solver binary: {binary}")

    root = Path(__file__).resolve().parent.parent
    instances_dir = Path(args.dir) if args.dir else root / "examples"

    files = sorted(list(instances_dir.glob("*.mps")) + list(instances_dir.glob("*.json")))
    if not files:
        print(f"[!] No instances found in {instances_dir}")
        sys.exit(1)

    print(f"[BharatOpt Benchmark] Found {len(files)} instances in {instances_dir}")
    results = []

    for f in files:
        print(f" -> Benchmarking {f.name}...", end="", flush=True)
        res = run_instance(binary, str(f), max_iter=args.max_iterations, tol=args.tol)
        results.append(res)
        print(f" {res.get('status', 'DONE')} in {res.get('wall_time_sec', 0):.3f}s")

    print_summary_table(results)

    if args.json_out:
        with open(args.json_out, "w") as out_f:
            json.dump(results, out_f, indent=2)
        print(f"[+] Saved JSON report to {args.json_out}")

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["Instance", "Status", "Objective", "Iterations", "SolveTimeSec", "WallTimeSec"])
            for r in results:
                writer.writerow([
                    r.get("instance"),
                    r.get("status"),
                    r.get("objective", ""),
                    r.get("iterations", ""),
                    r.get("time_sec", ""),
                    r.get("wall_time_sec", "")
                ])
        print(f"[+] Saved CSV report to {args.csv_out}")

if __name__ == "__main__":
    main()
