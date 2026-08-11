"""Generate pProp distribution histograms for each fold.

One PNG per fold, showing:
- Histogram of pProp values (log-scale y-axis)
- Vertical lines at pProp 3.5 and 5.0
- Counts for bins: [0, 3.5), [3.5, 5.0), [5.0, ∞)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "figure.dpi": 100})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("assignments", type=Path, help="path to assignments.csv")
    p.add_argument("-o", "--outdir", type=Path,
                   help="output directory for PNG files (default: siblings of assignments)")
    p.add_argument("--bins", type=int, default=40, help="histogram bins")
    p.add_argument("--width", type=float, default=6, help="figure width (inches)")
    p.add_argument("--height", type=float, default=4, help="figure height (inches)")
    return p.parse_args(argv)


def make_fold_histogram(fold_id, fold_pprop, outfile, bins, width, height):
    """Create one fold histogram PNG."""
    fig, ax = plt.subplots(figsize=(width, height))

    counts, edges, patches = ax.hist(
        fold_pprop, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.7, color="steelblue"
    )
    ax.set_yscale("log")
    ax.set_xlabel("pProp", fontsize=12)
    ax.set_ylabel("count (log scale)", fontsize=12)
    ax.set_title(f"Fold {fold_id} pProp Distribution (n={len(fold_pprop)})", fontsize=14, fontweight="bold")

    ax.axvline(3.5, color="red", linestyle="--", linewidth=2, alpha=0.8, label="3.5")
    ax.axvline(5.0, color="darkred", linestyle="--", linewidth=2, alpha=0.8, label="5.0")

    n_lo = (fold_pprop < 3.5).sum()
    n_mid = ((fold_pprop >= 3.5) & (fold_pprop < 5.0)).sum()
    n_hi = (fold_pprop >= 5.0).sum()

    text = f"[0, 3.5):  {n_lo:,}\n[3.5, 5):  {n_mid:,}\n[5, ∞):    {n_hi:,}"
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.8),
        family="monospace",
    )

    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.assignments)
    outdir = args.outdir or args.assignments.parent

    print(f"generating fold histograms to {outdir}/")

    for fold_id in range(5):
        fold_data = df[df["fold"] == fold_id]
        pprop = fold_data["pprop"].values
        outfile = outdir / f"fold_{fold_id}_pprop_distribution.png"
        make_fold_histogram(fold_id, pprop, outfile, args.bins, args.width, args.height)
        print(f"  fold {fold_id}: {outfile}")

    print("done: all 5 folds")


if __name__ == "__main__":
    main()
