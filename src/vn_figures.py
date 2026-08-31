"""Violin figures: how well each MiniMol embedding predicts pProp, across bootstrap models.

One PNG per metric at `figures/<sweep_id>/<metric>.png`. Each figure puts the six embeddings on
the x axis and, for every embedding, one violin per configuration showing the distribution over
that configuration's 10 bootstrap models.

Reads the tidy CSV `src/vn_probe.py` writes, so re-styling a figure never refits a forest.

Two deliberate choices a reader should know about:

**The y axis is scaled to the data, not to [0, 1].** These are in-sample scores and they
saturate near 1.0; on a 0-1 axis all eighteen violins would collapse onto one flat line. The
tick labels carry the actual range, so nothing is hidden -- but the visual spread is not a
spread on the natural scale of the metric.

**The ten models are drawn as points on top of each violin.** A violin is a kernel density, and
a kernel density over ten samples is mostly bandwidth. Showing the samples keeps the figure
honest about how little is behind each shape, and doubles as the visible-data relief the aqua
series needs (it sits at 2.74:1 against this surface, under the 3:1 bar).

The two classification metrics are handled apart from the three regression ones -- they refuse
to plot in sample, and they carry a chance line. See `CLASSIFICATION` below for why both.

Usage:
    python src/vn_figures.py --metrics-csv figures/lyc0lh2d/probe_metrics.csv --suffix _oob
    python src/vn_figures.py --metrics-csv figures/lyc0lh2d/probe_metrics.csv \
        --metrics average_precision auc --suffix _oob
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vn_taps import DISPLAY_NAMES, EMBEDDING_ORDER      # noqa: E402

# Categorical slots 1-3 of the validated reference palette. Checked with
# `validate_palette.js --mode light --pairs all`: all-pairs CVD dE 9.2, normal-vision 24.0, both
# clear. One WARN -- aqua is 2.74:1 against the surface -- answered by the overlaid points and
# the CSV table view, per the relief rule.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e5e1"

METRICS = {
    # vn_probe.py -- random forest reading pProp off each embedding
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "r2": "R²",
    "average_precision": "Average precision",
    "auc": "AUC",
    # vn_geometry.py -- geometry, against ECFP4 rather than against pProp
    "tanimoto_spearman": "Embedding distance vs ECFP4 dissimilarity (ρ)",
    "tanimoto_partial": "Structural correlation, prediction partialled out",
    "scalarness": "Scalarness — distance vs |Δ predicted pProp|",
    "knn20_jaccard": "kNN-20 neighbourhood overlap with ECFP4",
    "emb_effective_rank": "Effective rank",
    # vn_cluster_probe.py -- held-out-CLUSTER utility, two probes side by side
    "knn_spearman": "Held-out-cluster Spearman (kNN)",
    "knn_ap": "Held-out-cluster average precision (kNN)",
    "ridge_spearman": "Held-out-cluster Spearman (ridge)",
    "ridge_ap": "Held-out-cluster average precision (ridge)",
}
DEFAULT_METRICS = ("pearson", "spearman", "r2")

# The two metrics scored against the pProp >= 3.5 label. They are handled apart from the three
# regression metrics for two reasons, both of which would otherwise produce a plausible figure
# rather than an error.
#
# THEY SATURATE IN SAMPLE. A random forest at `min_samples_leaf=1` memorises the fold, and
# measured over all 30 models x 6 embeddings the in-sample AP and AUC are 1.000 to within
# 2e-16. Plotted on this module's data-scaled y axis, that float noise becomes eighteen
# violins of apparent structure. Only the `_oob` columns discriminate, so `--suffix _oob` is
# required rather than merely advisable.
#
# THEY NEED A CHANCE LINE. A correlation is read against 0 and everyone knows it; average
# precision is read against the positive rate, which here is 0.0095. Without it 0.059 looks
# like a failure when it is six times chance.
CLASSIFICATION = {"average_precision", "auc"}

# Metrics read against the positive rate rather than against zero. `average_precision` is
# `vn_probe.py`'s; `knn_ap` / `ridge_ap` are `vn_cluster_probe.py`'s, on cluster-disjoint halves.
# Saturation is a property of the in-sample forest, not of AP itself, so the cluster-probe
# columns are held-out by construction and are not subject to the `--suffix _oob` guard above.
AP_METRICS = {"average_precision", "knn_ap", "ridge_ap"}

# Below this many models a series is drawn as bare points rather than as a violin. A kernel
# density over ten samples is already mostly bandwidth; over one it is undefined.
MIN_VIOLIN = 3

# Series that are REFERENCES rather than configurations: the un-fine-tuned trunk, and the
# run-independent baselines scored once. They are drawn in neutral ink, deliberately outside the
# three-slot categorical palette -- with four series a `SERIES[j % 3]` would hand `pretrained`
# cfg0's blue, and two unrelated things in one colour is the one thing a categorical palette
# exists to prevent. Ink also reads as an annotation, which is what they are.
REFERENCE_CONFIGS = ("pretrained", "shared")
REFERENCE_COLOUR = "#52514e"


def chance_level(metric, df):
    """The no-skill value for a classification metric, or None if the metric has no such level.

    AUC's is 0.5 by construction. Average precision's is the positive rate, which is a property
    of the data and NOT a constant: it moves with the fold and with `PPROP_EDGE`. It is read
    from the CSV -- `vn_probe.py` records `positive_rate` per row for exactly this -- so a CSV
    from another sweep, another fold or another threshold draws its own line. An older CSV
    without the column gets no line rather than a wrong one.
    """
    if metric == "auc":
        return 0.5
    if metric in AP_METRICS and "positive_rate" in df.columns:
        return float(df["positive_rate"].iloc[0])
    return None


def embedding_order(df):
    """X-axis order: the known stack order first, then anything else in first-appearance order.

    `EMBEDDING_ORDER` names the six tensors `vn_taps.py` defines. Later drivers score
    featurizations it does not know about -- concatenations, an ECFP baseline, a pretrained
    reference -- and reindexing on the constant would drop precisely the rows that carry the
    result. Unknown keys are kept, at the right-hand end, rather than silently discarded.
    """
    seen = list(dict.fromkeys(df["embedding"].tolist()))
    known = [k for k in EMBEDDING_ORDER if k in seen]
    return known + [k for k in seen if k not in known]


def display_label(df, key):
    """The label the CSV gives this key, falling back to `vn_taps.DISPLAY_NAMES`."""
    if "embedding_label" in df.columns:
        hit = df.loc[df["embedding"] == key, "embedding_label"]
        if len(hit):
            return str(hit.iloc[0])
    return DISPLAY_NAMES.get(key, key)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-csv", type=Path, required=True,
                   help="probe_metrics.csv from src/vn_probe.py")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: the directory holding --metrics-csv")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                   help=f"any of {list(METRICS)}; default {list(DEFAULT_METRICS)}")
    p.add_argument("--suffix", default="",
                   help="score column suffix, e.g. '_oob' to plot the out-of-bag scores")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--only", nargs="+", default=None,
                   help="restrict the x axis to these embedding keys. vn_cluster_probe.py "
                        "writes 23 featurizations and all of them on one axis is unreadable; "
                        "naming the ~10 that carry the comparison is the fix")
    p.add_argument("--allow-in-sample", action="store_true",
                   help="plot average_precision / auc without --suffix _oob. They saturate at "
                        "1.000 in sample; this exists so the saturation can be shown, not so "
                        "it can be ignored")
    return p.parse_args(argv)


def one_figure(df, metric, column, sweep_id, configs, out_path, dpi, chance=None,
               source="probe_metrics.csv"):
    # Width follows the category count. At six embeddings 11 inches is right; at twenty the
    # labels collide into an unreadable smear, which is a figure that lies by omission rather
    # than one that is merely cramped.
    n_cat = df["embedding"].nunique()
    fig, ax = plt.subplots(figsize=(max(11.0, 1.05 * n_cat + 4.0), 6.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Horizontal hairlines only, solid, one shade off the surface, behind the marks.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)

    order = embedding_order(df)
    labels = [display_label(df, k) for k in order]
    n_cfg = len(configs)
    # A gap between adjacent violins rather than an outline to separate them.
    slot = 0.78 / n_cfg
    width = slot * 0.82
    rng = np.random.default_rng(0)
    extent = []          # y-extents of the drawn violin bodies, in DATA coordinates

    # Categorical slots are assigned over the CONFIGURATIONS only, so adding a reference
    # series never shifts the colour of cfg0/1/2 and never wraps onto one of them.
    palette, k = {}, 0
    for cfg in configs:
        if cfg not in REFERENCE_CONFIGS:
            palette[cfg] = SERIES[k % len(SERIES)]
            k += 1

    for j, cfg in enumerate(configs):
        offset = (j - (n_cfg - 1) / 2) * slot
        colour = palette.get(cfg, REFERENCE_COLOUR)
        data, pos = [], []
        for i, key in enumerate(order):
            v = df[(df["config"] == cfg) & (df["embedding"] == key)][column].to_numpy()
            # Non-finite values are DROPPED, not plotted. The pretrained baseline has no head,
            # so its three prediction-dependent geometry columns are legitimately nan; feeding
            # those to violinplot gives a kernel density over nan, and leaving the series in
            # gives the legend an entry with nothing behind it.
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            data.append(v)
            pos.append(i + offset)
        if not data:
            continue

        # A series with too few models gets NO violin. The pretrained baseline is one
        # deterministic model, and `violinplot` on a single sample builds a kernel density from
        # a singular covariance -- it either raises or draws a shape whose width is pure
        # bandwidth. Drawing the points alone states what is actually there: a reference
        # reading with no spread, which is what it is.
        if min(len(v) for v in data) >= MIN_VIOLIN:
            parts = ax.violinplot(data, positions=pos, widths=width,
                                  showmeans=False, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                v = body.get_paths()[0].vertices
                extent += [float(v[:, 1].min()), float(v[:, 1].max())]
                body.set_facecolor(colour)
                body.set_alpha(0.30)
                body.set_edgecolor(colour)
                body.set_linewidth(1.2)
        else:
            extent += [float(np.min(v)) for v in data] + [float(np.max(v)) for v in data]

        for p, v in zip(pos, data):
            # The ten models themselves. Jittered so ties do not hide each other.
            ax.scatter(p + rng.uniform(-width * 0.16, width * 0.16, len(v)), v,
                       s=9, color=colour, edgecolor=SURFACE, linewidth=0.5,
                       zorder=3, alpha=0.95)
            # Median as a short rule -- selective, one per violin, no per-point numbers.
            med = float(np.median(v))
            ax.plot([p - width * 0.36, p + width * 0.36], [med, med],
                    color=colour, linewidth=2.0, solid_capstyle="round", zorder=4)
        ax.plot([], [], color=colour, linewidth=6, alpha=0.45, label=cfg)

    # Pad the y range so a violin's tail is never clipped. The extents come from the violin
    # bodies as they were drawn -- a violin reaches past its data by up to half a kernel
    # bandwidth, so padding the data range alone would still clip. Scatter collections are
    # deliberately NOT consulted: a marker's path is a unit circle in marker space, and mixing
    # those numbers into a data-coordinate range silently forces the axis to +/-0.5.
    # The no-skill reference, drawn BEFORE the y range is set so it is inside `extent` and the
    # axis opens up to include it. That widening is the point: on average precision it costs
    # ~15% of vertical range and buys an axis with a meaningful floor, so a violin at 0.06 can
    # be read as "six times chance" instead of "about zero".
    if chance is not None:
        extent.append(float(chance))
        ax.axhline(chance, color=INK_2, linewidth=0.9, linestyle=(0, (5, 4)),
                   zorder=1, alpha=0.75)

    if extent:
        lo, hi = min(extent), max(extent)
        pad = (hi - lo) * 0.10 or 0.01
        ax.set_ylim(lo - pad, hi + pad)

    if chance is not None:
        # Labelled at the left edge, above its own line, so it reads as an annotation on the
        # axis rather than as a nineteenth series.
        ax.text(-0.55, chance + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012,
                f"chance = {chance:.4g}", fontsize=8.5, color=INK_2, va="bottom", ha="left")

    ax.set_xticks(range(len(labels)))
    rotate = len(labels) > 8
    ax.set_xticklabels(labels, fontsize=11 if not rotate else 9, color=INK,
                       rotation=30 if rotate else 0, ha="right" if rotate else "center")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(METRICS[metric], fontsize=11, color=INK_2)

    # The subtitle is assembled from what the CSV actually carries. Three drivers write these
    # figures and they describe different measurements, so a fixed sentence naming a random
    # forest would be wrong on two of them -- and `n_components` does not even exist there.
    n_boot = int(df.groupby(["config", "embedding"]).size().max())
    n_boot_note = f"one violin per configuration, {n_boot} bootstrap models each"
    if "n_components" in df.columns:                       # vn_probe.py
        scoring = "out-of-bag" if column.endswith("_oob") else "in-sample"
        n_pop = int(df["n_rows"].iloc[0])
        subtitle = (f"Random forest on PCA-{int(df['n_components'].iloc[0])} of each embedding, "
                    f"{scoring} on {n_pop:,} held-out-fold molecules · {n_boot_note}")
    elif "n_per_half" in df.columns:                       # vn_cluster_probe.py
        n_pop = int(df["n_per_half"].iloc[0])
        subtitle = (f"Fit on one cluster-disjoint half of fold {int(df['fold'].iloc[0])}, scored "
                    f"on the other, both directions averaged · {n_pop:,} molecules per half · "
                    f"{n_boot_note}")
    elif "n_readout" in df.columns:                        # vn_geometry.py
        n_pop = int(df["n_readout"].iloc[0])
        subtitle = (f"Against ECFP4 Tanimoto over {n_pop:,} molecules of fold "
                    f"{int(df['fold'].iloc[0])} · {n_boot_note}")
    else:
        n_pop = int(df["n_rows"].iloc[0]) if "n_rows" in df.columns else 0
        subtitle = n_boot_note

    if metric in AP_METRICS and {"n_positive", "pprop_edge"} <= set(df.columns):
        # The class balance belongs beside a classification score, not in a caption elsewhere.
        edge = float(df["pprop_edge"].iloc[0])
        n_pos = int(df["n_positive"].iloc[0])
        subtitle += (f"\nLabel is pProp ≥ {edge:g} · {n_pos:,} of {n_pop:,} positive "
                     f"({100 * n_pos / n_pop:.2f}%)")
    ax.set_title(f"{sweep_id} — {METRICS[metric]}", fontsize=15, color=INK,
                 loc="left", pad=26 if "\n" not in subtitle else 38, fontweight="bold")
    ax.text(0, 1.03, subtitle,
            transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=n_cfg,
                    frameon=False, fontsize=10, handlelength=1.6,
                    labelcolor=INK_2, title=None)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # Names the table view, which is what discharges the aqua contrast WARN.
    fig.text(0.995, 0.012, f"table view: {source}", ha="right",
             fontsize=8, color=INK_2, alpha=0.7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.metrics_csv)
    if args.only:
        missing = [k for k in args.only if k not in set(df["embedding"])]
        if missing:
            raise SystemExit(f"--only names {missing}, not in {args.metrics_csv}")
        df = df[df["embedding"].isin(args.only)]
    out_dir = args.out_dir or args.metrics_csv.parent
    sweep_id = str(df["sweep_id"].iloc[0])
    configs = sorted(df["config"].unique())

    print(f"{len(df)} rows | sweep {sweep_id} | configs {configs} | "
          f"{df['embedding'].nunique()} embeddings")

    written = []
    for metric in args.metrics:
        if metric not in METRICS:
            raise SystemExit(f"unknown metric {metric!r}; choose from {list(METRICS)}")
        if metric in CLASSIFICATION and args.suffix != "_oob" and not args.allow_in_sample:
            raise SystemExit(
                f"{metric!r} is saturated in sample -- a forest at min_samples_leaf=1 scores "
                "1.000 on every embedding, so the figure would plot float noise on a "
                "data-scaled axis. Pass --suffix _oob (or --allow-in-sample to override).")
        column = metric + args.suffix
        if column not in df.columns:
            raise SystemExit(f"column {column!r} is not in {args.metrics_csv}. "
                             "Re-run src/vn_probe.py with --oob if you want OOB scores.")
        name = metric + (".png" if not args.suffix else f"{args.suffix}.png")
        chance = chance_level(metric, df)
        if metric in AP_METRICS and chance is None:
            print(f"  note: {args.metrics_csv} has no positive_rate column, so the average "
                  "precision figure gets no chance line. Re-run src/vn_probe.py to add it.")
        p = one_figure(df, metric, column, sweep_id, configs, out_dir / name, args.dpi,
                       chance=chance, source=args.metrics_csv.name)
        spread = df.groupby("embedding")[column].mean()
        written.append(p)
        print(f"  {p}  (across-embedding mean spread "
              f"{spread.max() - spread.min():.4f})")
    print(f"\n{len(written)} figure(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
