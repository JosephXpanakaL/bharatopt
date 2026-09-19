#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("usage: benchmark.py <mps-file> [repeats]", file=sys.stderr)
        return 2
    mps=Path(sys.argv[1])
    repeats=int(sys.argv[2]) if len(sys.argv)>2 else 3
    binary=Path(__file__).resolve().parents[1]/"build"/"bharatopt_cli"
    if sys.platform=="win32": binary=Path(str(binary)+".exe")
    rows=[]
    for i in range(repeats):
        t=time.perf_counter()
        p=subprocess.run([str(binary),"--mps",str(mps),"--json"],capture_output=True,text=True)
        wall=time.perf_counter()-t
        if not p.stdout.strip():
            raise RuntimeError(p.stderr or "solver returned no JSON")
        d=json.loads(p.stdout)
        d["wall_sec"]=wall
        d["run"]=i+1
        rows.append(d)
    out=Path("benchmark_results.csv")
    fields=["run","model","rows","cols","nnz","status","objective","best_bound","mip_gap",
            "primal_residual","dual_residual","solve_time_sec","wall_sec","iterations","nodes","backend"]
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
