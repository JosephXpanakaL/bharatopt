#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description="Generate deterministic MRPL-like sparse LP benchmarks.")
    ap.add_argument("--variables",type=int,default=1000)
    ap.add_argument("--constraints",type=int,default=500)
    ap.add_argument("--nnz-per-row",type=int,default=12)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--output",type=Path,default=Path("data/refinery_stress.mps"))
    args=ap.parse_args()
    rng=random.Random(args.seed)
    n=max(2,args.variables);m=max(1,args.constraints);k=max(1,min(args.nnz_per_row,n))
    vars=["X%06d"%j for j in range(n)]
    rows=[]
    for i in range(m):
        sense="L" if i%3==0 else ("G" if i%3==1 else "E")
        rows.append((sense,"R%06d"%i,float(rng.randint(500,5000))))
    obj=[rng.uniform(1,20) for _ in range(n)]
    entries=[[] for _ in range(m)]
    for i in range(m):
        for j in rng.sample(range(n),k): entries[i].append((j,rng.uniform(0.05,2.0)))
    inv=[[] for _ in range(n)]
    for i in range(m):
        for j,v in entries[i]: inv[j].append((i,v))
    lines=["NAME          BHARATOPT-REFINERY-STRESS","ROWS"," N  COST"]
    lines += [" %s  %s"%(s,name) for s,name,_ in rows]
    lines += ["COLUMNS"]
    for j,var in enumerate(vars):
        lines.append("    %-10s  %-12s %g"%(var,"COST",obj[j]))
        for i,v in inv[j]: lines.append("    %-10s  %-12s %g"%(var,rows[i][1],v))
    lines.append("RHS")
    for start in range(0,m,2):
        pieces=[]
        for i in range(start,min(start+2,m)): pieces.append("    %-10s %-10s %g"%("RHS1",rows[i][1],rows[i][2]))
        lines.extend(pieces)
    lines.append("BOUNDS")
    for var in vars: lines.append(" UP BND1      %-10s 1000"%var)
    lines += ["ENDATA",""]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print(args.output,"variables=",n,"constraints=",m,"nnz=",sum(len(x) for x in entries))

if __name__=="__main__": main()
