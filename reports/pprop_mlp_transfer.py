"""What pProp_MLP's sweeps say about this repo's hyperparameter ranges.

`pProp_MLP` trained a dual-head MLP on **frozen** MiniMol embeddings for the same target
(NOTES §12). Its wandb project `ethan_personal/pprop-mlp-minimol-multitask` holds 11,784
runs. This script reduces them to the subset that is comparable under one objective
revision, and prints every table in `reports/pprop_mlp_transfer.md`.

Two paths:

    python reports/pprop_mlp_transfer.py --fetch     # wandb -> reports/pprop_mlp_runs.csv
    python reports/pprop_mlp_transfer.py             # csv -> the tables

`--fetch` needs network and the wandb package; the default path needs only numpy, scipy and
the stdlib, so it runs under `my_conda_env` with no model stack. The CSV is the evidence
file the report cites -- regenerating it is a network operation, reading it is not.

WHY THE SEGMENTATION MATTERS. Three sweeps live in that project and only two are mutually
comparable. `nihehqst` (9,466 runs) used a different split and a 6-class scheme, and its
`goal_metric` spans -4.38..0.40 against 1.30..1.45 for the others -- a different objective
revision under the same metric name. That is precisely the hazard `src/objective.py`'s
`OBJECTIVE_VERSION` hash exists to prevent, showing up in the source data. Pooling the three
would average incomparables, so `comparable()` drops `nihehqst` and every table below is
built from the remaining 2,247 finished runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "ethan_personal/pprop-mlp-minimol-multitask"
DEFAULT_CSV = Path(__file__).with_name("pprop_mlp_runs.csv")

#: Sweeps sharing one objective revision, one split and one class scheme.
COMPARABLE_SWEEPS = ("j6z4dh1u", "9okf022y")
#: Dropped: `split_6`, 6-class weights, no `objective_classes` -- a different revision.
EXCLUDED_SWEEPS = ("nihehqst",)

METRIC = "val/goal_metric"

HP_NUMERIC = ("init_lr", "epochs", "batch_size", "dropout", "weight_decay", "huber_delta",
              "w_cls", "w_pair", "w_std", "hidden_dim", "n_layers", "cls_hidden_dim",
              "cls_n_layers", "reg_hidden_dim", "reg_n_layers")
HP_TEXT = ("pprop_norm", "split_dir", "seed")
METRICS = (METRIC, "val/ap_star", "val/pearson_star", "val/mae_skill_star",
           "val/pearson", "val/weighted_mae")

#: Swept log-uniformly there, so bin and correlate them in log space.
LOG_SCALED = {"init_lr", "weight_decay", "huber_delta", "w_cls", "w_pair", "w_std",
              "batch_size", "hidden_dim", "cls_hidden_dim", "reg_hidden_dim", "epochs"}

#: "Near-best" tolerance. 0.005 is ~2.5x the measured cost of final-epoch selection
#: (median 0.0019, see `report_final_epoch_gap`), so it separates a genuinely good
#: configuration from one that merely got a lucky last epoch.
NEAR_BEST_TOL = 0.005

COLUMNS = ("run_id", "sweep", "state", "created") + HP_TEXT + HP_NUMERIC + METRICS


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------

def fetch(csv_path: Path) -> None:
    """Pull every run from wandb and write the flat CSV the analysis reads."""
    import wandb

    api = wandb.Api(timeout=120)
    runs = api.runs(PROJECT, per_page=500)

    rows, n_seen = [], 0
    for run in runs:
        cfg, summary = run.config, run.summary
        row = {"run_id": run.id, "sweep": run.sweep.id if run.sweep else "",
               "state": run.state, "created": str(run.created_at)}
        for key in HP_TEXT:
            row[key] = cfg.get(key, "")
        for key in HP_NUMERIC:
            row[key] = _number(cfg.get(key))
        for key in METRICS:
            row[key] = _number(summary.get(key))
        rows.append(row)
        n_seen += 1
        if n_seen % 1000 == 0:
            print(f"  {n_seen} runs", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (r["sweep"], r["run_id"]))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs -> {csv_path}", file=sys.stderr)
    _write_meta(csv_path, rows)


def _number(value):
    """Coerce to float, mapping wandb's nulls and nans to ''. Keeps the CSV free of 'nan'."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def _write_meta(csv_path: Path, rows: list[dict]) -> None:
    """Sibling meta.json, per the repo convention (CLAUDE.md 'Conventions')."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["sweep"]] = counts.get(row["sweep"], 0) + 1
    meta = {
        "source_project": PROJECT,
        "n_runs": len(rows),
        "runs_per_sweep": counts,
        "comparable_sweeps": list(COMPARABLE_SWEEPS),
        "excluded_sweeps": list(EXCLUDED_SWEEPS),
        "columns": list(COLUMNS),
        "argv": sys.argv,
        "git_commit": _git_commit(),
    }
    csv_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------

def load(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in HP_NUMERIC + METRICS:
            row[key] = float(row[key]) if row[key] not in ("", None) else None
    return rows


def comparable(rows: list[dict]) -> list[dict]:
    """Finished runs from one objective revision, with the objective actually recorded."""
    return [r for r in rows
            if r["sweep"] in COMPARABLE_SWEEPS
            and r["state"] == "finished"
            and r[METRIC] is not None]


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------

def report_segmentation(rows: list[dict]) -> None:
    print("=" * 88)
    print("SEGMENTATION -- which runs are comparable at all")
    print("=" * 88)
    print(f"{'sweep':>10} {'n':>7} {'finished':>9} {'split':>14} {'goal min':>10} "
          f"{'goal max':>10}  comparable")
    for sweep in sorted({r["sweep"] for r in rows}):
        group = [r for r in rows if r["sweep"] == sweep]
        done = [r for r in group if r["state"] == "finished" and r[METRIC] is not None]
        goals = [r[METRIC] for r in done]
        split = group[0]["split_dir"] or "?"
        flag = "YES" if sweep in COMPARABLE_SWEEPS else "no -- different revision"
        print(f"{sweep:>10} {len(group):7d} {len(done):9d} {split:>14} "
              f"{min(goals):10.4f} {max(goals):10.4f}  {flag}")
    print(f"\ncomparable total: {len(comparable(rows))} finished runs")


def report_near_best(rows: list[dict], key: str, edges: list[float]) -> None:
    """P(within NEAR_BEST_TOL of the global best) per band -- the hit-rate view.

    Reported alongside p90 rather than alone: a band can hold the single best run by luck
    while being a bad place to sample from, and the hit rate is what a sweep actually buys.
    """
    values, goals = _column(rows, key), _column(rows, METRIC)
    best = goals.max()
    print(f"\n{key}: P(within {NEAR_BEST_TOL} of global best {best:.4f})")
    print(f"{'band':>26} {'n':>6} {'P(near-best)':>13} {'p90':>9} {'max':>9}")
    for lo, hi in zip(edges, edges[1:]):
        mask = (values >= lo) & (values < hi)
        if mask.sum() < 5:
            continue
        subset = goals[mask]
        print(f"{lo:11.3g} .. {hi:11.3g} {mask.sum():6d} "
              f"{(subset >= best - NEAR_BEST_TOL).mean():12.1%} "
              f"{np.percentile(subset, 90):9.4f} {subset.max():9.4f}")


def report_regret(rows: list[dict], key: str, n_bins: int = 8) -> float:
    """Best-achievable per quantile bin. Returns the largest regret across bins.

    Regret = global best - best in bin, i.e. what you give up by restricting to that band.
    This is the statistic that decides whether a range can be cut: a flat regret profile
    means the axis carries no signal, whatever its rank correlation says.
    """
    values, goals = _column(rows, key), _column(rows, METRIC)
    use_log = key in LOG_SCALED and (values > 0).all()
    scaled = np.log10(values) if use_log else values
    edges = np.unique(np.quantile(scaled, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0

    best = goals.max()
    print(f"\n{key}  ({'log' if use_log else 'linear'} quantile bins)")
    print(f"{'band':>26} {'n':>6} {'max':>9} {'p95':>9} {'median':>9} {'regret':>9}")
    worst = 0.0
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        last = i == len(edges) - 2
        mask = (scaled >= lo) & (scaled <= hi if last else scaled < hi)
        if mask.sum() < 5:
            continue
        subset = goals[mask]
        regret = best - subset.max()
        worst = max(worst, regret)
        rlo, rhi = (10 ** lo, 10 ** hi) if use_log else (lo, hi)
        print(f"{rlo:11.4g} .. {rhi:11.4g} {mask.sum():6d} {subset.max():9.4f} "
              f"{np.percentile(subset, 95):9.4f} {np.median(subset):9.4f} {regret:9.4f}")
    print(f"{'':>26} {'':>6} max regret across bins: {worst:.4f}")
    return worst


def report_spearman(rows: list[dict], keys: tuple[str, ...]) -> None:
    goals = _column(rows, METRIC)
    print(f"\n{'hyperparameter':>16} {'spearman':>10} {'p':>10}   (log-scaled where swept so)")
    for key in keys:
        values = _column(rows, key)
        mask = ~np.isnan(values)
        if mask.sum() < 20:
            continue
        scaled = values[mask]
        if key in LOG_SCALED and (scaled > 0).all():
            scaled = np.log10(scaled)
        rho, p = stats.spearmanr(scaled, goals[mask])
        print(f"{key:>16} {rho:10.3f} {p:10.2g}")


def report_interaction(rows: list[dict], key: str, key_edges: list[float],
                       other: str, other_edges: list[float]) -> None:
    """Best achievable on a 2-D grid -- used to show an optimum does NOT move.

    The question this answers is narrow but load-bearing: `batch_size` is pinned at 1200
    here while pProp swept it 1000-10000, so any prior taken from pProp is only valid if
    its optimum is batch-independent. A pattern that holds down every column says it is.
    """
    values, others, goals = _column(rows, key), _column(rows, other), _column(rows, METRIC)
    print(f"\nmax goal_metric by ({key} x {other})")
    header = f"{key:>24} |" + "".join(
        f"{f'{other_edges[j]:g}-{other_edges[j+1]:g}':>16}" for j in range(len(other_edges) - 1))
    print(header)
    print("-" * len(header))
    for lo, hi in zip(key_edges, key_edges[1:]):
        line = f"{lo:10.3g} .. {hi:10.3g} |"
        for olo, ohi in zip(other_edges, other_edges[1:]):
            mask = (values >= lo) & (values < hi) & (others >= olo) & (others < ohi)
            cell = f"{goals[mask].max():.4f}({mask.sum()})" if mask.sum() >= 5 else "-"
            line += f"{cell:>16}"
        print(line)


def report_categorical(rows: list[dict], key: str) -> None:
    print(f"\n{key}:")
    for value in sorted({r[key] for r in rows}):
        goals = np.array([r[METRIC] for r in rows if r[key] == value])
        print(f"  {value:>10}: n={len(goals):5d}  median={np.median(goals):.4f}  "
              f"p95={np.percentile(goals, 95):.4f}  max={goals.max():.4f}")


def report_inherited_defaults(rows: list[dict]) -> None:
    """The run this repo's `train.py` loss defaults were copied from.

    Checked rather than asserted in prose: if these four stop matching, the whole premise
    that pProp's population is a prior for this repo's ranges needs re-examining.
    """
    best = max(rows, key=lambda r: r[METRIC])
    expected = {"w_cls": 0.4418, "w_pair": 7.486, "w_std": 0.7911, "huber_delta": 1.0513}
    print("\n" + "=" * 88)
    print(f"INHERITED DEFAULTS -- best comparable run is {best['run_id']} "
          f"(sweep {best['sweep']}, goal {best[METRIC]:.4f})")
    print("=" * 88)
    print(f"{'param':>14} {'pProp winner':>16} {'train.py default':>18}  match")
    for key, want in expected.items():
        got = best[key]
        ok = abs(got - want) < 5e-4 * max(1.0, abs(want))
        print(f"{key:>14} {got:16.6g} {want:18}  {'OK' if ok else 'MISMATCH'}")


def report_final_epoch_gap(rows: list[dict], top_n: int = 20) -> None:
    """How much final-epoch selection costs, and where the peak actually lands.

    Needs per-epoch history, so this one path does hit wandb. It is what calibrates
    NEAR_BEST_TOL and what tells you whether the cosine tail is doing anything.
    """
    import wandb

    api = wandb.Api(timeout=120)
    top = sorted(rows, key=lambda r: -r[METRIC])[:top_n]
    print("\n" + "=" * 88)
    print(f"FINAL-EPOCH SELECTION COST -- top {top_n} comparable runs")
    print("=" * 88)
    print(f"{'run':>10} {'epochs':>7} {'final':>9} {'best':>9} {'gap':>8} {'peak@':>7}")

    gaps, positions = [], []
    for row in top:
        history = api.run(f"{PROJECT}/{row['run_id']}").history(keys=[METRIC], pandas=False)
        curve = np.array([h[METRIC] for h in history if h.get(METRIC) is not None])
        if len(curve) < 3:
            continue
        peak = int(curve.argmax())
        gaps.append(curve.max() - curve[-1])
        positions.append((peak + 1) / len(curve))
        print(f"{row['run_id']:>10} {len(curve):7d} {curve[-1]:9.4f} {curve.max():9.4f} "
              f"{curve.max() - curve[-1]:8.4f} {positions[-1]:7.2f}")

    print(f"\nbest-final gap:  median {np.median(gaps):.4f}  mean {np.mean(gaps):.4f}  "
          f"max {np.max(gaps):.4f}")
    print(f"peak position:   median {np.median(positions):.2f} of the schedule; "
          f"{np.mean([p > 0.8 for p in positions]):.0%} peaked in the final fifth; "
          f"{np.mean([p < 1.0 for p in positions]):.0%} peaked before the last epoch")


# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fetch", action="store_true",
                        help="re-pull from wandb and rewrite the CSV (needs network)")
    parser.add_argument("--history", action="store_true",
                        help="also measure the final-epoch gap (needs network)")
    args = parser.parse_args(argv)

    if args.fetch:
        fetch(args.csv)

    rows = load(args.csv)
    report_segmentation(rows)

    good = comparable(rows)
    report_inherited_defaults(good)

    print("\n" + "=" * 88)
    print("THE LEARNING RATE -- pProp's `init_lr` is this repo's phase-1 `head_lr`")
    print("=" * 88)
    report_near_best(good, "init_lr", [5e-5, 1.5e-4, 3e-4, 6e-4, 1.2e-3, 3.1e-3])

    print("\n" + "=" * 88)
    print("PER-AXIS REGRET -- how much a range can be cut without losing the optimum")
    print("=" * 88)
    latest = [r for r in good if r["sweep"] == "j6z4dh1u"]
    for key in ("init_lr", "weight_decay", "huber_delta", "w_std", "w_cls", "w_pair",
                "dropout", "batch_size", "epochs", "hidden_dim", "n_layers"):
        report_regret(latest, key)

    print("\n" + "=" * 88)
    print("RANK CORRELATION (context for the regret tables, not a substitute)")
    print("=" * 88)
    report_spearman(latest, HP_NUMERIC)

    print("\n" + "=" * 88)
    print("EPOCH BUDGET -- does a longer schedule raise the ceiling, or the hit rate?")
    print("=" * 88)
    report_near_best(good, "epochs", [20, 35, 50, 70, 101])

    print("\n" + "=" * 88)
    print("BATCH INDEPENDENCE -- does pinning batch_size at 1200 invalidate the priors?")
    print("=" * 88)
    report_interaction(latest, "init_lr", [5e-5, 2e-4, 5e-4, 1e-3, 3.1e-3],
                       "batch_size", [1000, 1600, 2600, 10000])
    report_interaction(latest, "w_pair", [0.005, 0.5, 2, 5, 10.1],
                       "batch_size", [1000, 1600, 2600, 10000])

    print("\n" + "=" * 88)
    print("CONFIRMATIONS of decisions already made in this repo")
    print("=" * 88)
    report_categorical(latest, "pprop_norm")

    if args.history:
        report_final_epoch_gap(good)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
