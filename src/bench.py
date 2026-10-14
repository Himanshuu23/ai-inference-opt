"""Benchmark harness.

    python -m src.bench                          # everything
    python -m src.bench --models resnet50 --configs A B D --batch-sizes 1 16

Per (model, config, batch):
  * latency: `iters` single inferences, each bracketed by CUDA events after a full device sync
    (so it is the latency of an isolated request) -> p50 / p99 / mean
  * throughput: a separate back-to-back loop of `iters` inferences (queue kept full) -> samples/s
  * memory: device-wide used-memory delta (torch.cuda.mem_get_info) from before the runner was built to after
    the timed loops. Includes weights, activations, cuDNN/cuBLAS/TensorRT workspaces, allocator cache.
    It excludes the one-time CUDA context. Also torch's own max_memory_allocated (empty for TensorRT).
  * accuracy vs the FP32 baseline (config A) on a fixed 64-sample set: max |logit diff|, top-1 agreement.
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

from .backends import make_runner
from .common import BATCH_SIZES, CONFIGS, RESULTS
from .models import get_inputs, load_bundle, take
from .results_io import load_rows, merge_save, table_md

MB = 2 ** 20
EVAL_N, EVAL_CHUNK = 64, 16


def gpu_state() -> dict:
    """SM clock / temperature / power right now (to spot thermal or power throttling)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().splitlines()[0].split(",")
        return {"gpu": out[0].strip(), "sm_clock_mhz": out[1].strip(), "temp_c": out[2].strip(), "power_w": out[3].strip()}
    except Exception:  # noqa: BLE001
        return {"gpu": torch.cuda.get_device_name(0), "sm_clock_mhz": "", "temp_c": "", "power_w": ""}


def is_oom(e: Exception) -> bool:
    s = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in s or "out_of_memory" in s or "failed to allocate" in s


def collect_logits(runner, inputs: dict, chunk: int = EVAL_CHUNK) -> torch.Tensor:
    n = len(next(iter(inputs.values())))
    outs = []
    for i in range(0, n, chunk):
        runner.bind(take(inputs, chunk, i))
        out = runner.run()
        torch.cuda.synchronize()
        outs.append(out.float().cpu().clone())
    return torch.cat(outs)


def accuracy(base: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    return float((base - got).abs().max()), float((base.argmax(-1) == got.argmax(-1)).float().mean() * 100)


def time_runner(runner, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        runner.run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        runner.run()
        e.record()
        e.synchronize()
        lat.append(s.elapsed_time(e))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    lat = np.asarray(lat)
    return {"p50_ms": np.percentile(lat, 50), "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(), "wall_s": wall}


def bench_model(model: str, configs, batches, iters, warmup, bert_id, rebuild, out_csv):
    b = load_bundle(model, bert_id)
    eval_in = get_inputs(b, EVAL_N)
    base_logits = None
    # FP32 baseline logits, always computed (even if A is not benchmarked)
    r = make_runner("A", b, take(eval_in, EVAL_CHUNK))
    base_logits = collect_logits(r, eval_in)
    r.close()
    rows_all = []

    for cfg in configs:
        if cfg == "E" and model != "resnet50":
            continue
        common = {"model": model, "config": cfg, "config_desc": CONFIGS[cfg], "acc_data": b.data_source,
                  "timestamp": datetime.now().isoformat(timespec="seconds")}
        acc = None
        for bs in batches:
            row = dict(common, batch=bs)
            runner = None
            try:
                gc.collect(); torch.cuda.empty_cache()
                free0, _ = torch.cuda.mem_get_info()
                torch.cuda.reset_peak_memory_stats()
                inputs = take(get_inputs(b, max(bs, EVAL_N)), bs)
                runner = make_runner(cfg, b, inputs, rebuild)
                row["notes"] = runner.note
                if acc is None:
                    acc = accuracy(base_logits, collect_logits(runner, eval_in))
                runner.bind(inputs)
                t = time_runner(runner, iters, warmup)
                free1, _ = torch.cuda.mem_get_info()
                row.update(status="ok", p50_ms=f"{t['p50_ms']:.3f}", p99_ms=f"{t['p99_ms']:.3f}", mean_ms=f"{t['mean_ms']:.3f}",
                           throughput_per_s=f"{bs * iters / t['wall_s']:.1f}",
                           device_mem_delta_mb=f"{(free0 - free1) / MB:.0f}",
                           torch_peak_alloc_mb=f"{torch.cuda.max_memory_allocated() / MB:.0f}",
                           max_abs_diff=f"{acc[0]:.4g}", top1_agree_pct=f"{acc[1]:.1f}", **gpu_state())
                print(f"  {model:8s} {cfg} bs={bs:<3d} p50={t['p50_ms']:.2f} ms  p99={t['p99_ms']:.2f} ms  "
                      f"{bs * iters / t['wall_s']:.0f}/s  {row['device_mem_delta_mb']} MB  [{row['notes']}]")
            except Exception as ex:  # noqa: BLE001
                oom = is_oom(ex)
                row.update(status="OOM" if oom else f"ERROR: {str(ex)[:150]}")
                print(f"  {model:8s} {cfg} bs={bs:<3d} {row['status']}")
                if not oom and bs == batches[0]:
                    row["notes"] = "TODO: fix, see status"
            finally:
                if runner is not None:
                    runner.close()
                gc.collect(); torch.cuda.empty_cache()
            rows_all.append(row)
            if row["status"] == "OOM":  # larger batches will not fit either
                break
        merge_save(out_csv, [r for r in rows_all if r["config"] == cfg and r["model"] == model])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "bert"], choices=["resnet50", "bert"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    ap.add_argument("--iters", type=int, default=200, help="timed iterations (>=100 recommended; p99 needs many samples)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bert-model", default="bert-base-uncased", help="or distilbert-base-uncased / gpt2 if memory is a problem")
    ap.add_argument("--rebuild", action="store_true", help="rebuild TensorRT engines")
    ap.add_argument("--out", default=str(RESULTS / "results.csv"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU not available: install the CUDA build of PyTorch (README -> Setup)"
    torch.cuda.set_stream(torch.cuda.Stream())  # non-default stream (TensorRT warns otherwise)
    print(f"GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    for m in a.models:
        print(f"== {m} ==")
        bench_model(m, a.configs, a.batch_sizes, a.iters, a.warmup, a.bert_model, a.rebuild, a.out)
    rows = load_rows(a.out)
    print("\n" + table_md(rows))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
