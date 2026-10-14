"""results/results.csv -> results/latency.png, results/throughput.png, results/speedups.md (+ README injection).

    python scripts/make_plots.py [--update-readme]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import CONFIGS  # noqa: E402
from src.results_io import load_rows, table_md  # noqa: E402


def ok_rows(rows):
    return [r for r in rows if r.get("status") == "ok"]


def plot(rows, metric, ylabel, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(6.5 * max(len(models), 1), 4.8), squeeze=False)
    for ax, m in zip(axes[0], models):
        for cfg in CONFIGS:
            pts = sorted((int(r["batch"]), float(r[metric])) for r in rows if r["model"] == m and r["config"] == cfg)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=f"{cfg}: {CONFIGS[cfg]}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 32])
        ax.set_xticklabels([1, 4, 16, 32])
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{m}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / fname, dpi=140)
    plt.close(fig)


def speedups_md(rows):
    out = ["Speedup vs config A (eager FP32), computed from this machine's `results.csv` (throughput ratio; p50 latency ratio in brackets):", ""]
    for m in sorted({r["model"] for r in rows}):
        batches = sorted({int(r["batch"]) for r in rows if r["model"] == m})
        out += [f"**{m}**", "", "| config | " + " | ".join(f"bs={b}" for b in batches) + " |", "|---|" + "---|" * len(batches)]
        base = {int(r["batch"]): r for r in rows if r["model"] == m and r["config"] == "A"}
        for cfg in CONFIGS:
            cells = []
            for b in batches:
                r = next((x for x in rows if x["model"] == m and x["config"] == cfg and int(x["batch"]) == b), None)
                if r is None or b not in base:
                    cells.append("n/a")
                else:
                    t = float(r["throughput_per_s"]) / float(base[b]["throughput_per_s"])
                    l = float(base[b]["p50_ms"]) / float(r["p50_ms"])
                    cells.append(f"{t:.2f}x ({l:.2f}x)")
            if any(c != "n/a" for c in cells):
                out.append(f"| {cfg} | " + " | ".join(cells) + " |")
        out.append("")
    acc = []
    for m in sorted({r["model"] for r in rows}):
        for cfg in CONFIGS:
            r = next((x for x in rows if x["model"] == m and x["config"] == cfg), None)
            if r:
                acc.append(f"| {m} | {cfg} | {r['max_abs_diff']} | {r['top1_agree_pct']} | {r['acc_data']} |")
    out += ["Accuracy vs FP32 baseline (64-sample set):", "", "| model | config | max abs logit diff | top-1 agreement % | data |", "|---|---|---|---|---|"] + acc
    return "\n".join(out)


def inject(readme: Path, marker: str, content: str):
    s = readme.read_text()
    a, b = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if a in s and b in s:
        head, rest = s.split(a, 1)
        _, tail = rest.split(b, 1)
        readme.write_text(head + a + "\n" + content + "\n" + b + tail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true")
    a = ap.parse_args()
    rows_all = load_rows(ROOT / "results" / "results.csv")
    rows = ok_rows(rows_all)
    if not rows:
        print("TODO: results/results.csv has no successful rows; run `python -m src.bench` first.")
        sys.exit(0)
    plot(rows, "p50_ms", "p50 latency (ms, log)", "latency.png", "Inference latency vs batch size (p50)")
    plot(rows, "throughput_per_s", "samples / s (log)", "throughput.png", "Throughput vs batch size")
    sp = speedups_md(rows)
    (ROOT / "results" / "speedups.md").write_text(sp)
    print(table_md(rows_all))
    print("\n" + sp)
    if a.update_readme:
        inject(ROOT / "README.md", "RESULTS", table_md(rows_all))
        inject(ROOT / "README.md", "SPEEDUPS", sp)
        print("README.md updated")
