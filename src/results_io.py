"""CSV read/merge/print helpers (no torch import, so they work anywhere)."""
from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ["model", "config", "config_desc", "batch", "status", "p50_ms", "p99_ms", "mean_ms",
          "throughput_per_s", "device_mem_delta_mb", "torch_peak_alloc_mb", "max_abs_diff",
          "top1_agree_pct", "acc_data", "sm_clock_mhz", "temp_c", "power_w", "gpu", "notes", "timestamp"]


def load_rows(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _bkey(r):
    b = str(r.get("batch", ""))
    return int(b) if b.lstrip("-").isdigit() else -1


def merge_save(path, new_rows: list[dict]) -> None:
    """Replace rows with the same (model, config, batch); keep everything else."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (str(r["model"]), str(r["config"]), str(r.get("batch", "")))
    new_keys = {key(r) for r in new_rows}
    rows = [r for r in load_rows(path) if key(r) not in new_keys] + list(new_rows)
    rows.sort(key=lambda r: (r["model"], r["config"], _bkey(r)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


TABLE_COLS = ["model", "config", "batch", "status", "p50_ms", "p99_ms", "throughput_per_s",
              "device_mem_delta_mb", "max_abs_diff", "top1_agree_pct"]


def table_md(rows: list[dict], cols=TABLE_COLS) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, ""))[:30] for c in cols) + " |")
    return "\n".join(out)
