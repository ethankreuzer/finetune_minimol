"""Generate pProp distribution histograms for each cluster.

One PNG per cluster, showing:
- Histogram of pProp values (log-scale y-axis)
- Vertical lines at pProp 3.5 and 5.0
- Counts for bins: [0, 3.5), [3.5, 5.0), [5.0, ∞)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "figure.dpi": 100})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("assignments", type=Path, help="path to assignments.csv")
    p.add_argument("-o", "--outdir", type=Path,
                   help="output directory for PNG files (default: siblings of assignments)")
    p.add_argument("--bins", type=int, default=30, help="histogram bins")
    p.add_argument("--width", type=float, default=4, help="figure width (inches)")
    p.add_argument("--height", type=float, default=3, help="figure height (inches)")
    p.add_argument("-j", type=int, default=8, help="parallel workers")
    return p.parse_args(argv)


def make_histogram(cluster_id, cluster_pprop, outfile, bins, width, height):
    """Create one histogram PNG."""
    fig, ax = plt.subplots(figsize=(width, height))

    counts, edges, patches = ax.hist(
        cluster_pprop, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.7
    )
    ax.set_yscale("log")
    ax.set_xlabel("pProp")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Cluster {cluster_id} (n={len(cluster_pprop)})", fontsize=10, fontweight="bold")

    ax.axvline(3.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="3.5")
    ax.axvline(5.0, color="darkred", linestyle="--", linewidth=1, alpha=0.7, label="5.0")

    n_lo = (cluster_pprop < 3.5).sum()
    n_mid = ((cluster_pprop >= 3.5) & (cluster_pprop < 5.0)).sum()
    n_hi = (cluster_pprop >= 5.0).sum()

    text = f"[0, 3.5): {n_lo}\n[3.5, 5): {n_mid}\n[5, ∞): {n_hi}"
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return outfile


def _worker(args_tuple):
    cluster_id, cluster_pprop, outfile, bins, width, height = args_tuple
    try:
        make_histogram(cluster_id, cluster_pprop, outfile, bins, width, height)
        return outfile
    except Exception as e:
        print(f"ERROR cluster {cluster_id}: {e}")
        return None


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.assignments)
    outdir = args.outdir or args.assignments.parent / "cluster_histograms"
    outdir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("cluster_id")
    n_clusters = len(grouped)
    print(f"generating {n_clusters:,} histograms to {outdir}/", flush=True)

    from multiprocessing import Pool

    tasks = []
    for cluster_id, group in grouped:
        outfile = outdir / f"cluster_{cluster_id:06d}.png"
        pprop = group["pprop"].values
        tasks.append((cluster_id, pprop, outfile, args.bins, args.width, args.height))

    with Pool(args.j) as p:
        results = p.map(_worker, tasks)

    ok = sum(1 for r in results if r is not None)
    print(f"done: {ok:,} / {n_clusters:,} succeeded")


if __name__ == "__main__":
    main()
