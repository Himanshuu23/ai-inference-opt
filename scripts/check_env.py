"""Environment check: python scripts/check_env.py  -> tells you exactly what is missing."""
import importlib
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {type(e).__name__}>"


missing = []
print(f"OS: {platform.platform()}\npython: {sys.version.split()[0]}")
print("\n--- nvidia-smi ---")
smi = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
print(smi)
if smi.startswith("<failed"):
    missing.append("NVIDIA driver / nvidia-smi (install the latest Game Ready/Studio driver)")
print("\n--- nvcc ---")
nvcc = run(["nvcc", "--version"])
print(nvcc.splitlines()[-1] if not nvcc.startswith("<failed") else nvcc)
if nvcc.startswith("<failed"):
    print("(nvcc is NOT required: PyTorch/TensorRT wheels bundle their own CUDA runtime. Only needed if you build custom kernels.)")

print("\n--- python packages ---")
for mod, pipname, required in [("torch", "torch torchvision (CUDA wheel, see README)", True), ("torchvision", "torchvision", True),
                               ("transformers", "transformers", True), ("onnx", "onnx", True),
                               ("onnxruntime", "onnxruntime-gpu", False), ("tensorrt", "tensorrt", True),
                               ("numpy", "numpy", True), ("pandas", "pandas", True), ("matplotlib", "matplotlib", True),
                               ("PIL", "pillow", True)]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK       {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  MISSING  {mod}  ({type(e).__name__})")
        if required:
            missing.append(f"pip install {pipname}")
try:
    import torch
    print(f"\ntorch.cuda.is_available() = {torch.cuda.is_available()}  (torch built for CUDA {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}  cc={p.major}.{p.minor}  VRAM={p.total_memory / 2**30:.2f} GiB")
        if (p.major, p.minor) != (8, 6):
            print("  NOTE: expected compute capability 8.6 for an RTX 2050 (Ampere); results/engines are device-specific.")
    else:
        missing.append("CUDA-enabled PyTorch: pip uninstall torch; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
except ImportError:
    pass
try:
    import onnxruntime as ort
    print("onnxruntime providers:", ort.get_available_providers())
except ImportError:
    pass

print("\n--- command line tools ---")
for tool, hint in [("trtexec", "optional (we build engines via the Python API)"),
                   ("nsys", "needed for scripts/profile_nsys.sh (Nsight Systems)"),
                   ("ncu", "optional, scripts/profile_ncu.sh (Nsight Compute)")]:
    path = shutil.which(tool)
    print(f"  {'OK     ' if path else 'MISSING'} {tool:8s} {path or hint}")

print("\n=== TO INSTALL MANUALLY ===" if missing else "\nAll required components found.")
for m in dict.fromkeys(missing):
    print(" -", m)
