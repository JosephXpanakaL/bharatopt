#!/usr/bin/env python3
import platform
import shutil
import subprocess

def run(cmd):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        return p.returncode,p.stdout.strip(),p.stderr.strip()
    except Exception as e:
        return 1,"",str(e)

print("BharatOpt preflight")
print("OS:",platform.platform())
print("Python:",platform.python_version())
for name in ("cmake","g++","clang++","nvcc","nvidia-smi"):
    path=shutil.which(name)
    print(name+": "+(path or "not found"))
if shutil.which("nvidia-smi"):
    code,out,err=run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap","--format=csv,noheader"])
    if out: print("GPU:",out)
    elif err: print("nvidia-smi error:",err)
print()
print("Build CPU with: ./setup.sh")
print("Build CUDA with: ./setup.sh --cuda")
