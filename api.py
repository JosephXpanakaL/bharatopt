import json
import os
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import sys

ROOT = Path(__file__).resolve().parent

def find_solver_bin():
    env_bin = os.environ.get("BHARATOPT_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)
    candidates = [
        ROOT / "build" / "bin" / "bharatopt_cli",
        ROOT / "build" / "bin" / "bharatopt_cli.exe",
        ROOT / "build" / "bin" / "bharatopt",
        ROOT / "build" / "bin" / "bharatopt.exe",
        ROOT / "build" / "bharatopt_cli",
        ROOT / "build" / "bharatopt_cli.exe",
        ROOT / "build" / "bharatopt",
        ROOT / "build" / "bharatopt.exe",
        ROOT / "build" / "Release" / "bharatopt_cli.exe",
        ROOT / "build" / "Release" / "bharatopt.exe",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return ROOT / "build" / ("bharatopt_cli.exe" if sys.platform == "win32" else "bharatopt_cli")

BIN = str(find_solver_bin())
WEB = ROOT / "web" / "index.html"
app = FastAPI(title="BharatOpt API", version="0.2.0")

def run_solver(args):
    if not Path(BIN).exists():
        raise HTTPException(status_code=503, detail=f"Solver binary not found: {BIN}")
    p = subprocess.run([BIN, *args, "--json"], text=True, capture_output=True, timeout=300)
    if not p.stdout.strip():
        raise HTTPException(status_code=500, detail=p.stderr.strip() or "Solver produced no output")
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=p.stderr.strip() or f"Invalid solver JSON: {exc}")
    data["process_returncode"] = p.returncode
    if p.stderr.strip():
        data["stderr"] = p.stderr.strip()
    return JSONResponse(data)

@app.get("/health")
def health():
    return {"status": "ok", "solver_binary": BIN, "solver_present": Path(BIN).exists()}

@app.get("/")
def home():
    if not WEB.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(WEB)

@app.post("/api/solve/demo")
def solve_demo(cuda: bool = Form(False), max_iters: int = Form(50000), tolerance: float = Form(1e-6), time_limit: float = Form(0.0)):
    args = ["--demo", "--max-iters", str(max_iters), "--tol", str(tolerance), "--time-limit", str(time_limit)]
    if cuda: args.append("--cuda")
    return run_solver(args)

@app.post("/api/solve/mps")
async def solve_mps(file: UploadFile = File(...), cuda: bool = Form(False),
                    max_iters: int = Form(50000), tolerance: float = Form(1e-6),
                    max_nodes: int = Form(256), mip_gap: float = Form(1e-4), time_limit: float = Form(0.0)):
    name = Path(file.filename or "model.mps").name
    if not name.lower().endswith(".mps"):
        raise HTTPException(status_code=400, detail="Upload an .mps file")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="MPS file exceeds 50 MiB limit")
    with tempfile.TemporaryDirectory(prefix="bharatopt-") as td:
        path = Path(td) / name
        path.write_bytes(content)
        args = ["--mps", str(path), "--max-iters", str(max_iters), "--tol", str(tolerance),
                "--max-nodes", str(max_nodes), "--mip-gap", str(mip_gap), "--time-limit", str(time_limit)]
        if cuda: args.append("--cuda")
        return run_solver(args)


@app.post("/api/solve/qp-demo")
def solve_qp_demo(max_iters: int = Form(50000), tolerance: float = Form(1e-6)):
    return run_solver(["--qp-demo", "--max-iters", str(max_iters), "--tol", str(tolerance)])

@app.post("/api/solve/flowsheet")
async def solve_flowsheet(
    file: UploadFile = File(...),
    global_nodes: int = Form(200),
    global_gap: float = Form(1e-5),
    time_limit: float = Form(30.0)
):
    name = Path(file.filename or "flowsheet.json").name
    if not name.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Upload a refinery flowsheet .json file")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 20 MiB limit")
    with tempfile.TemporaryDirectory(prefix="bharatopt-flowsheet-") as td:
        path = Path(td) / name
        path.write_bytes(content)
        args = [
            "--refinery-json", str(path),
            "--mode", "certified",
            "--global-nodes", str(global_nodes),
            "--global-gap", str(global_gap),
            "--global-time-limit", str(time_limit)
        ]
        return run_solver(args)

