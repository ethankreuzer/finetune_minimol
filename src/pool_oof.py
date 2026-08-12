"""Pool out-of-fold predictions across the 5 folds and score the tail once, properly.

    .venv/bin/python src/pool_oof.py                        # every run under outputs/
    .venv/bin/python src/pool_oof.py --runs outputs/grid_a  # one sweep bucket

WHY THIS EXISTS
---------------
Per-fold tail metrics are noise. Each validation fold holds exactly **20** molecules at
pProp >= 5.0 -- and that is not a sampling accident, it is what the stratified split was
built to guarantee (NOTES §11.3). A correlation or an enrichment factor over 20 points says
almost nothing.

But the 5 validation folds cover all 331,480 rows **exactly once**, so concatenating one
seed's five runs reconstructs a full-length prediction vector in which every molecule was
predicted by a model that never saw it. That gives one honest estimate over all **100**
potent molecules; two seeds give two such estimates, and the spread between them is the
optimisation variance (NOTES §11.4).

`train.py` has always written `val_predictions.npy` + `val_indices.npy` per run for exactly
this purpose. Until now nothing consumed them.

WHAT IT CHECKS BEFORE IT REPORTS
--------------------------------
The pooling is only valid if the folds really do tile the dataset, so that is asserted
rather than assumed: the union of a seed's row indices must be every row exactly once, with
no duplicates and nothing missing. A seed with a failed or missing run is reported and
skipped, never silently pooled from four folds -- which would look like a slightly smaller
dataset rather than like an error.

Runs are also grouped by their provenance triple (`objective_version`, `split_sha256`,
`input_sha256`). pProp_MLP's `runs/` became unreadable because an objective revision and an
in-place split regeneration both went unrecorded, and old checkpoints then scored new data
and printed plausible wrong numbers. Runs whose triples disagree are never pooled together.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import PPROP_EDGE, build_sample_weights                         # noqa: E402
from metrics import enrichment_factor, flavoured_metrics, group_split_metrics  # noqa: E402
from objective import REQUIRED_FLAVOURS, compute_objective                  # noqa: E402

# The tail edge the per-fold metrics cannot speak to. 100 molecules pooled, 20 per fold.
TAIL_EDGE = 5.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("outputs"),
                   help="directory searched recursively for runs with a meta.json")
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--out", type=Path, default=Path("reports/pooled_oof.json"))
    p.add_argument("--markdown", type=Path, default=Path("reports/pooled_oof.md"))
    p.add_argument("--expect-folds", type=int, default=5)
    return p.parse_args(argv)


def load_runs(root):
    """Every run under `root` that has both a meta.json and saved validation predictions."""
    runs = []
    for meta_path in sorted(Path(root).rglob("meta.json")):
        d = meta_path.parent
        preds, idx = d / "val_predictions.npy", d / "val_indices.npy"
        if not (preds.exists() and idx.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        cfg = meta.get("config", {})
        logits = d / "val_logits.npy"
        runs.append({
            "dir": d,
            "fold": cfg.get("fold"),
            "seed": cfg.get("seed"),
            "frozen_baseline": meta.get("frozen_baseline"),
            "provenance": (meta.get("objective_version"), meta.get("split_sha256"),
                           meta.get("input_sha256")),
            "pred": np.load(preds),
            # Falls back to the regression output when logits were not saved (runs from
            # before they were). Both rank the same direction, so AP is still meaningful --
            # but it is then the regression head's ranking, not the classifier's, so the
            # substitution is recorded rather than silent.
            "logits": np.load(logits) if logits.exists() else np.load(preds),
            "has_logits": logits.exists(),
            "rows": np.load(idx),
        })
    return runs


def group_key(run):
    """Runs pool together only if they share a seed, an arm, and a provenance triple."""
    return (run["seed"], bool(run["frozen_baseline"]), run["provenance"])


def pool_group(runs, n_rows, expect_folds):
    """Concatenate one group's folds into full-length vectors. Returns (arrays, problems)."""
    problems = []
    folds = sorted(r["fold"] for r in runs)
    if len(runs) != expect_folds:
        problems.append(f"has {len(runs)} runs (folds {folds}), expected {expect_folds}")
    if len(set(folds)) != len(folds):
        problems.append(f"duplicate folds: {folds}")

    rows = np.concatenate([r["rows"] for r in runs])
    if len(np.unique(rows)) != len(rows):
        problems.append(f"{len(rows) - len(np.unique(rows))} rows predicted more than once")
    if len(rows) != n_rows:
        problems.append(f"covers {len(rows):,} rows, dataset has {n_rows:,}")
    missing = n_rows - len(np.unique(rows))
    if missing:
        problems.append(f"{missing:,} rows never predicted")
    if problems:
        return None, problems

    order = np.argsort(rows)
    return ({"rows": rows[order],
             "pred": np.concatenate([r["pred"] for r in runs])[order],
             "logits": np.concatenate([r["logits"] for r in runs])[order]},
            [])


def score_pooled(y, pooled):
    """Every metric on the pooled vector, plus the tail-edge split the folds cannot support."""
    pred, logits = pooled["pred"], pooled["logits"]
    flavours = {f: build_sample_weights(f, y) for f in REQUIRED_FLAVOURS}

    m = flavoured_metrics(y, pred, logits, flavours, edge=PPROP_EDGE)
    m.update(compute_objective(m))
    m.update(enrichment_factor(y, logits))
    # Two group splits, and the second is the point of pooling: at PPROP_EDGE the per-fold
    # number was already meaningful (~630 molecules), at TAIL_EDGE it was not (20).
    for edge, tag in ((PPROP_EDGE, "e35"), (TAIL_EDGE, "e50")):
        for k, v in group_split_metrics(y, pred, group_edge=edge).items():
            m[f"{k}_{tag}"] = v
    return m


def main(argv=None):
    args = parse_args(argv)
    y_all = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)
    n_rows = len(y_all)

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs with saved predictions under {args.runs}")

    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)

    print(f"{len(runs)} runs under {args.runs} in {len(groups)} group(s)\n")

    results, skipped = {}, {}
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        seed, frozen, provenance = key
        arm = "frozen_baseline" if frozen else "finetuned"
        label = f"seed{seed}_{arm}"

        pooled, problems = pool_group(members, n_rows, args.expect_folds)
        if problems:
            skipped[label] = problems
            print(f"[SKIP] {label}: " + "; ".join(problems))
            continue

        y = y_all[pooled["rows"]]
        m = score_pooled(y, pooled)
        m["n_tail"] = int((y >= TAIL_EDGE).sum())
        m["objective_version"], m["split_sha256"], m["input_sha256"] = provenance
        m["all_logits_saved"] = all(r["has_logits"] for r in members)
        results[label] = m

        print(f"[OK]   {label}: {len(pooled['rows']):,} rows, "
              f"{m['n_positive']:,} at pProp>={PPROP_EDGE}, {m['n_tail']} at >={TAIL_EDGE}")
        print(f"         goal {m['goal_metric']:.4f} | ap {m['ap_uniform']:.4f} | "
              f"tail spearman {m['spearman_group_ge_e50']:.4f} | "
              f"EF@5.0/top1% {m['ef_p5.0_top0.01']:.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "skipped": skipped,
                                    "n_rows": n_rows, "tail_edge": TAIL_EDGE,
                                    "pprop_edge": PPROP_EDGE},
                                   indent=2, default=float) + "\n")

    lines = ["# Pooled out-of-fold metrics", "",
             f"Each row pools one seed's {args.expect_folds} validation folds into a "
             f"full-length prediction vector covering all {n_rows:,} molecules exactly "
             "once, so the tail is scored over all "
             f"{int((y_all >= TAIL_EDGE).sum())} molecules at pProp >= {TAIL_EDGE} "
             "rather than 20 per fold.", ""]
    if results:
        keys = ["goal_metric", "ap_uniform", "pearson_uniform", "mae_uniform",
                "spearman_group_ge_e50", "mae_group_ge_e50", "ef_p5.0_top0.01"]
        lines += ["| group | " + " | ".join(f"`{k}`" for k in keys) + " |",
                  "|---" * (len(keys) + 1) + "|"]
        for label, m in results.items():
            lines.append(f"| {label} | " +
                         " | ".join(f"{m.get(k, float('nan')):.4f}" for k in keys) + " |")
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- **{k}**: {'; '.join(v)}" for k, v in skipped.items()]
    args.markdown.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {args.out} and {args.markdown}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
