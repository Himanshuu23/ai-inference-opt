#!/usr/bin/env bash
# Nsight Systems on ResNet-50, batch 16, configs A (eager FP32), B (FP16), D (TensorRT FP16).
# Run from the project root:  bash scripts/profile_nsys.sh
# Needs: nsys on PATH (Nsight Systems, ships with the CUDA toolkit / separate installer).
# Permissions: see README "Profiling permissions".
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/nsys; mkdir -p "$OUT"
command -v nsys >/dev/null || { echo "nsys not found on PATH -> install Nsight Systems (see README)"; exit 1; }

# Build the TensorRT engine first so the build is not in the way
python -m src.build_engines --model resnet50 --precisions fp16

for CFG in A B D; do
  TAG="resnet50_${CFG}_bs16"
  echo "=== nsys: $TAG ==="
  nsys profile -o "$OUT/$TAG" --force-overwrite true \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --export=sqlite \
    python -m src.profile_target --model resnet50 --config "$CFG" --batch 16 --iters 20 --tag "$TAG" --outdir "$OUT"
done
python scripts/analyze_profile.py
