import json
import os
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
BIN = os.environ.get("BHARATOPT_BIN", str(ROOT / "build" / "bharatopt_cli"))
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
