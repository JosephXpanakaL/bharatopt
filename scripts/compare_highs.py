#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def bharatopt(binary,mps,args):
    cmd=[str(binary),"--mps",str(mps),"--json","--max-iters",str(args.max_iters),"--tol",str(args.tol)]
    if args.cuda: cmd.append("--cuda")
    t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True);wall=time.perf_counter()-t
    return {"solver":"BharatOpt","wall_sec":wall,"returncode":p.returncode,"result":json.loads(p.stdout) if p.stdout.strip() else None,"stderr":p.stderr.strip()}

def highs(mps):
    try:
        import highspy
    except ImportError:
        raise SystemExit("Install optional baseline with: python -m pip install highspy")
    h=highspy.Highs()
    t=time.perf_counter();status=h.readModel(str(mps));read=time.perf_counter()-t
    if status!=highspy.HighsStatus.kOk: raise RuntimeError(f"HiGHS readModel failed: {status}")
    t=time.perf_counter();status=h.run();wall=time.perf_counter()-t
    info=h.getInfo()
    return {"solver":"HiGHS","read_sec":read,"wall_sec":wall,"status":h.modelStatusToString(h.getModelStatus()),"objective":info.objective_function_value,"run_status":str(status)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mps",type=Path)
    ap.add_argument("--binary",type=Path,default=Path("build/bharatopt_cli"))
    ap.add_argument("--cuda",action="store_true")
    ap.add_argument("--max-iters",type=int,default=50000)
    ap.add_argument("--tol",type=float,default=1e-6)
    args=ap.parse_args()
    binary=args.binary
    if binary.with_suffix(".exe").exists(): binary=binary.with_suffix(".exe")
    out={"bharatopt":bharatopt(binary,args.mps,args),"highs":highs(args.mps)}
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
