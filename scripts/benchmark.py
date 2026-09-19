#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description="Run repeatable BharatOpt solver benchmarks.")
    ap.add_argument("mps",type=Path)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--cuda",action="store_true")
    ap.add_argument("--max-iters",type=int,default=50000)
    ap.add_argument("--tol",type=float,default=1e-6)
    ap.add_argument("--time-limit",type=float,default=0.0)
    ap.add_argument("--output",type=Path,default=Path("benchmark_results.csv"))
    args=ap.parse_args()
    binary=Path(__file__).resolve().parents[1]/"build"/"bharatopt_cli"
    if binary.with_suffix(".exe").exists(): binary=binary.with_suffix(".exe")
    rows=[]
    for run in range(1,args.repeats+1):
        cmd=[str(binary),"--mps",str(args.mps),"--json","--max-iters",str(args.max_iters),"--tol",str(args.tol)]
        if args.time_limit>0: cmd += ["--time-limit",str(args.time_limit)]
        if args.cuda: cmd.append("--cuda")
        t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True);wall=time.perf_counter()-t
        if not p.stdout.strip(): raise RuntimeError(p.stderr or "solver returned no JSON")
        d=json.loads(p.stdout);d["run"]=run;d["wall_sec"]=wall;d["cuda_requested"]=args.cuda
        rows.append(d)
        print("run",run,"status",d.get("status"),"time",d.get("solve_time_sec"),"backend",d.get("backend"))
    fields=["run","model","rows","cols","nnz","status","objective","best_bound","mip_gap","primal_residual","dual_residual","solve_time_sec","wall_sec","iterations","nodes","backend","cuda_requested"]
    with args.output.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print("wrote",args.output)
if __name__=="__main__":raise SystemExit(main())
