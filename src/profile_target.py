"""Workload executed under Nsight Systems / Nsight Compute.

    python -m src.profile_target --model resnet50 --config D --batch 16 --iters 20 --tag resnet50_D_bs16

Warmup runs happen outside the capture range; only the `iters` measured inferences sit between
cudaProfilerStart/Stop (use nsys --capture-range=cudaProfilerApi / ncu --profile-from-start off).
A sidecar JSON with the iteration count is written so the analyser can compute per-inference numbers.
"""
from __future__ import annotations

import argparse
import json

import torch

from .backends import make_runner
from .common import RESULTS
from .models import get_inputs, load_bundle, take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "bert"])
    ap.add_argument("--config", default="A", choices=list("ABCDE"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bert-model", default="bert-base-uncased")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(RESULTS / "nsys"))
    a = ap.parse_args()

    torch.cuda.set_stream(torch.cuda.Stream())  # non-default stream (TensorRT warns otherwise)
    b = load_bundle(a.model, a.bert_model)
    inputs = take(get_inputs(b, max(a.batch, 64)), a.batch)
    runner = make_runner(a.config, b, inputs)
    runner.bind(inputs)
    for _ in range(a.warmup):
        runner.run()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"{a.model}_{a.config}_bs{a.batch}")
    for _ in range(a.iters):
        runner.run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()

    tag = a.tag or f"{a.model}_{a.config}_bs{a.batch}"
    with open(f"{a.outdir}/{tag}.meta.json", "w") as f:
        json.dump({"model": a.model, "config": a.config, "batch": a.batch, "iters": a.iters, "note": runner.note}, f)
    print(f"PROFILE_DONE tag={tag} iters={a.iters}")
    runner.close()


if __name__ == "__main__":
    main()
