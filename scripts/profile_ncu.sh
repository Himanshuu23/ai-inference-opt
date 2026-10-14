#!/usr/bin/env bash
# OPTIONAL: Nsight Compute on one conv kernel (ResNet-50 FP16, N=1) and one GEMM kernel (BERT FP16, N=1).
# Run from the project root: bash scripts/profile_ncu.sh      Needs ncu on PATH and profiling permissions.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/ncu; mkdir -p "$OUT" results/nsys
command -v ncu >/dev/null || { echo "ncu not found on PATH -> install Nsight Compute (see README)"; exit 1; }

METRICS="sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
gpu__time_duration.sum"

# 1 sample (N=1), FP16 eager (config B). -k filters by kernel name; -c 1 profiles the first match only.
ncu --profile-from-start off --kernel-name-base demangled -k "regex:(?i)(xmma_fprop|s16816fprop)" -c 1 \
    --metrics "$METRICS" --csv --log-file "$OUT/resnet50_conv_bs1.csv" \
    python -m src.profile_target --model resnet50 --config B --batch 1 --iters 1 --tag ncu_resnet50 --outdir results/nsys

ncu --profile-from-start off --kernel-name-base demangled -k "regex:(?i)(gemm|cutlass|cublas|ampere_|sm80_|sm86_)" -c 1 \
    --metrics "$METRICS" --csv --log-file "$OUT/bert_gemm_bs1.csv" \
    python -m src.profile_target --model bert --config B --batch 1 --iters 1 --tag ncu_bert --outdir results/nsys
python scripts/analyze_profile.py
