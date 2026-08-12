"""Charts for reports/compute_profile.md, built from benchmark.json + grid_summary.json.

Every figure traces to a measured file -- nothing here is illustrative. Palette is the
validated categorical default (blue/orange/aqua), which clears CVD separation and the
normal-vision floor on adjacent pairs; the aqua slot sits below 3:1 against the surface, so
each chart carries direct labels and the report repeats the values in tables.

    python src/report_charts.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"

FIX_FULL, PER_FULL = 15.8, 0.1618     # full-phase fit over the batch sweep, R^2 = 0.999
FIX_FROZ, PER_FROZ = 7.2, 0.0655


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left", fontweight="600")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURFACE)


def fig(w=7.6, h=4.3):
    f, ax = plt.subplots(figsize=(w, h), dpi=170)
    f.patch.set_facecolor(SURFACE)
    return f, ax


def chart_batch_scaling(bench, out):
    f, ax = fig()
    for phase, color, label in (("full", BLUE, "trunk + head (full)"),
                                ("frozen", ORANGE, "head only (frozen trunk)")):
        rows = [r for r in bench["batch"][phase] if not r.get("oom")]
        x = [r["batch_size"] for r in rows]
        y = [r["mol_per_s"] for r in rows]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{y[-1]:,.0f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 3), color=color, fontsize=9, fontweight="600")
    ax.axvline(256, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("batch 256\n(what the grid ran)", (256, ax.get_ylim()[1] * 0.55),
                textcoords="offset points", xytext=(8, 0), color=MUTED, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(55, 7600)          # headroom so the end-of-line labels are not clipped
    style(ax, "The grid ran well below saturation",
          "batch size (molecules per step)", "molecules / s  (GPU-only)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def chart_fixed_overhead(out):
    f, ax = fig(7.6, 4.0)
    batches = [256, 1024, 4096]
    fixed = [FIX_FULL] * 3
    variable = [PER_FULL * b for b in batches]
    ypos = np.arange(len(batches))
    ax.barh(ypos, fixed, color=ORANGE, height=0.46, label="fixed per-step overhead")
    ax.barh(ypos, variable, left=[v + 6 for v in fixed], color=BLUE, height=0.46,
            label="GPU work (scales with batch)")
    for i, (fx, var) in enumerate(zip(fixed, variable)):
        total = fx + var
        ax.annotate(f"{total:,.0f} ms · {fx/total*100:.0f}% fixed · "
                    f"max {total/fx:.1f}× from faster silicon",
                    (total + 25, i), va="center", color=INK2, fontsize=9)
    ax.set_yticks(ypos); ax.set_yticklabels([f"batch {b}" for b in batches], color=INK2)
    ax.set_xlim(0, 1350)
    style(ax, "Fixed overhead caps the speedup any faster GPU can deliver",
          "milliseconds per training step (full phase)", "")
    # Legend below the plot: every in-axes corner is occupied by a bar or its label.
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, ncol=2,
              loc="upper left", bbox_to_anchor=(0, -0.16))
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def chart_clocks(csv, out):
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["utilization.gpu [%]"] > 50].copy()
    df["t"] = pd.to_datetime(df["timestamp"])
    f, ax = fig()
    for gpu, color in zip(sorted(df["index"].unique()), (BLUE, ORANGE, AQUA)):
        sub = df[df["index"] == gpu].sort_values("t")
        mins = (sub["t"] - sub["t"].iloc[0]).dt.total_seconds() / 60
        clk = sub["clocks.current.sm [MHz]"].astype(float)
        ax.plot(mins, clk, color=color, linewidth=1.6, label=f"GPU {gpu}")
        ax.annotate(f"GPU {gpu}", (mins.iloc[-1], clk.iloc[-1]), textcoords="offset points",
                    xytext=(6, -2), color=color, fontsize=9, fontweight="600")
    ax.axhline(2100, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("2100 MHz max boost", (0.3, 2100), textcoords="offset points",
                xytext=(0, 6), color=MUTED, fontsize=8.5)
    ax.set_ylim(1400, 2200)
    ax.set_xlim(-0.3, mins.max() * 1.12)     # room for the end-of-line labels
    # NOT "still falling": the sawtooth is the four job waves (recoveries at ~3.5 and ~7
    # min are the idle gaps between waves), so a linear slope over this window is
    # confounded. What the data does support is the level, not a trend.
    style(ax, "SM clocks hold 14–29% below max boost under three-GPU load",
          "minutes under load", "SM clock (MHz)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower left")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def chart_projection(out):
    f, ax = fig(7.6, 4.2)
    batches = [256, 1024, 4096]
    # The conservative bar is shown, not just tabulated: this figure will be screenshotted
    # out of the report, and a bandwidth-only projection overstates by up to 3x.
    ratios = {"RTX A6000 (measured)": 1.0,
              "conservative (compute-bound, 1.73×)": 1.73,
              "H100 SXM (bandwidth-bound, 4.36×)": 4.36,
              "H200 SXM (bandwidth-bound, 6.25×)": 6.25}
    colors = [MUTED, ORANGE, BLUE, AQUA]
    width, x = 0.2, np.arange(len(batches))
    for i, ((label, r), color) in enumerate(zip(ratios.items(), colors)):
        vals = [b / ((FIX_FULL + PER_FULL * b / r) / 1000) for b in batches]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=color, label=label)
        for b_, v in zip(bars, vals):
            ax.annotate(f"{v/1000:.0f}k", (b_.get_x() + b_.get_width() / 2, v),
                        ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"batch {b}" for b in batches], color=INK2)
    ax.set_ylim(0, 42000)
    style(ax, "Projected per-GPU throughput — PROJECTED, not measured on TamIA",
          "", "molecules / s  (GPU-only)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8.5, ncol=2,
              loc="upper left", bbox_to_anchor=(0, 1.0))
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def chart_concurrency(reports, out):
    """Aggregate throughput vs concurrent processes on ONE GPU. It never rises."""
    f, ax = fig(7.6, 4.2)
    for name, color, label in (
            ("concurrency.json", BLUE, "batch 1024"),
            ("concurrency_b256.json", ORANGE, "batch 256")):
        path = reports / name
        if not path.exists():
            continue
        pts = [p for p in json.loads(path.read_text())["points"] if not p.get("failed")]
        x = [p["procs"] for p in pts]
        y = [p["aggregate_mol_per_s"] for p in pts]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        # The change relative to a single process -- negative, i.e. work LOST by splitting.
        ax.annotate(f"{(y[1]/y[0]-1)*100:+.0f}% vs 1 process", (x[1], y[1]),
                    textcoords="offset points", xytext=(10, -14), color=color, fontsize=9)
    ax.set_ylim(0, 7000)
    ax.set_xticks([1, 2, 3, 4, 6, 8])
    style(ax, "Splitting one GPU never increases total work done",
          "concurrent training processes on one GPU", "aggregate molecules / s")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    bench = json.loads((args.reports / "benchmark.json").read_text())["results"]

    chart_batch_scaling(bench, args.reports / "fig_batch_scaling.png")
    chart_fixed_overhead(args.reports / "fig_fixed_overhead.png")
    chart_projection(args.reports / "fig_projection.png")
    chart_concurrency(args.reports, args.reports / "fig_concurrency.png")
    tele = args.reports / "gpu_telemetry.csv"
    if tele.exists():
        chart_clocks(tele, args.reports / "fig_clocks.png")
    print(f"wrote figures to {args.reports}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
