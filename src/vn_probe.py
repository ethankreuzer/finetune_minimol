"""Score each of MiniMol's internal embeddings by how well a random forest reads pProp from it.

The question this branch exists to answer is whether features from different depths of MiniMol
carry pProp predictive ability. `extract_embeddings.py` produced the features; this scores them.

Per (configuration, bootstrap, embedding): reduce the embedding to 32 principal components, fit
a random forest on the validation fold, and score its predictions on that same fold. Five
metrics, from two forests -- a regressor for Pearson / Spearman / R^2, and a classifier on the
pProp >= 3.5 label for average precision and AUC.

**Scoring is in-sample, by design (decided with Ethan 2026-08-27).** The forest is fit and
evaluated on the same molecules, so these numbers are not estimates of generalisation and must
not be read as such. Measured on this data, at default forest settings, in-sample Spearman runs
~0.13-0.25 above a cluster-held-out estimate, and -- because the optimism is larger for the
narrower embeddings -- it *compresses* the differences between embeddings that the figures are
meant to show. The comparison across the six is the thing at risk, not the absolute level. Both
`--oob` and `--cluster-cv` exist here to quantify that whenever it is worth revisiting.

30 models x 6 embeddings, ~4 s each, so a full pass is ~15 min on 128 cores.

Usage:
    python src/vn_probe.py --root outputs/vn_analysis/lyc0lh2d
    /     # adds OOB columns
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import pearsonr, spearmanr                              # noqa: E402
from sklearn.decomposition import PCA                                    # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score  # noqa: E402

from losses import PPROP_EDGE                                          # noqa: E402
from vn_taps import DISPLAY_NAMES, EMBEDDING_ORDER                       # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="outputs/vn_analysis/<sweep_id>")
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oob", action="store_true",
                   help="also record out-of-bag scores, which cost nothing and are a held-out "
                        "reading on the same fit")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="default: figures/<sweep_id>/probe_metrics.csv")
    return p.parse_args(argv)


def discover(root):
    """`(config, bootstrap, path)` for every extracted run, in a stable order.

    Layout is `<root>/cfg<rank>/boot<nn>/vn_embeddings.npz`, written by retrain_bootstrap.py.
    """
    out = []
    for p in sorted(root.rglob("vn_embeddings.npz")):
        out.append({"config": p.parent.parent.name, "bootstrap": p.parent.name, "path": p})
    if not out:
        raise SystemExit(f"no vn_embeddings.npz under {root} -- run extract_embeddings.py first")
    return out


def score_one(Z, y, y_bin, args):
    """Two forests on one 32-d feature matrix -> the five metrics."""
    rf = RandomForestRegressor(n_estimators=args.n_estimators,
                               min_samples_leaf=args.min_samples_leaf,
                               n_jobs=args.n_jobs, random_state=args.seed,
                               oob_score=args.oob)
    rf.fit(Z, y)
    pred = rf.predict(Z)

    clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                 min_samples_leaf=args.min_samples_leaf,
                                 n_jobs=args.n_jobs, random_state=args.seed,
                                 oob_score=args.oob)
    clf.fit(Z, y_bin)
    proba = clf.predict_proba(Z)[:, 1]

    m = {
        "pearson": float(pearsonr(pred, y).statistic),
        "spearman": float(spearmanr(pred, y).statistic),
        "r2": float(r2_score(y, pred)),
        "average_precision": float(average_precision_score(y_bin, proba)),
        "auc": float(roc_auc_score(y_bin, proba)),
    }
    if args.oob:
        # The out-of-bag prediction for a row comes only from the ~37% of trees that never saw
        # it, so these are held out at the row level (though not at the cluster level -- a
        # structural analog can still be in-bag).
        op, oq = rf.oob_prediction_, clf.oob_decision_function_[:, 1]
        m.update({
            "pearson_oob": float(pearsonr(op, y).statistic),
            "spearman_oob": float(spearmanr(op, y).statistic),
            "r2_oob": float(r2_score(y, op)),
            "average_precision_oob": float(average_precision_score(y_bin, oq)),
            "auc_oob": float(roc_auc_score(y_bin, oq)),
        })
    return m


def main(argv=None):
    args = parse_args(argv)
    sweep_id = args.root.name
    runs = discover(args.root)

    y_all = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy()
    ref_rows = None
    records = []
    t0 = time.time()

    print(f"sweep {sweep_id} | {len(runs)} models x {len(EMBEDDING_ORDER)} embeddings = "
          f"{len(runs) * len(EMBEDDING_ORDER)} fits | PCA -> {args.n_components}d | "
          f"in-sample scoring")

    for i, r in enumerate(runs, 1):
        d = np.load(r["path"])
        rows = d["row_indices"]
        if ref_rows is None:
            ref_rows = rows
            y = y_all[rows]
            y_bin = (y >= PPROP_EDGE).astype(int)
            print(f"{len(rows):,} validation molecules | {int(y_bin.sum())} positive at "
                  f"pProp >= {PPROP_EDGE}\n")
        elif not np.array_equal(rows, ref_rows):
            # The same guard extract_embeddings.py applies, restated because this script can be
            # pointed at a directory assembled by hand.
            raise SystemExit(f"{r['path']} has a different row order from {runs[0]['path']}; "
                             "these models are not scored on the same molecules")

        line = f"[{i:>2}/{len(runs)}] {r['config']}/{r['bootstrap']}"
        for key in EMBEDDING_ORDER:
            X = d[key].astype(np.float32)
            pca = PCA(n_components=args.n_components, random_state=args.seed)
            Z = pca.fit_transform(X)
            m = score_one(Z, y, y_bin, args)
            records.append({
                "sweep_id": sweep_id, "config": r["config"], "bootstrap": r["bootstrap"],
                "embedding": key, "embedding_label": DISPLAY_NAMES[key],
                "n_rows": int(len(rows)), "input_dim": int(X.shape[1]),
                # The positive class, carried in the CSV rather than only printed. Average
                # precision is uninterpretable without it -- 0.06 is six times chance here and
                # reads as nothing without the 0.0095 beside it -- and a figure that hardcoded
                # the rate would be silently wrong on a different fold or a different edge.
                "pprop_edge": PPROP_EDGE,
                "n_positive": int(y_bin.sum()),
                "positive_rate": float(y_bin.mean()),
                "n_components": args.n_components,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                **m,
            })
            line += f" {key}:{m['spearman']:.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    out = args.out or Path("figures") / sweep_id / "probe_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "script": "src/vn_probe.py", "argv": sys.argv, "sweep_id": sweep_id,
        "n_models": len(runs), "embeddings": list(EMBEDDING_ORDER),
        "scoring": "in-sample: the forest is fit and evaluated on the same molecules",
        "pprop_edge": PPROP_EDGE,
        "rf": {"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
               "n_components": args.n_components, "seed": args.seed},
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))

    print(f"\n{len(df)} rows -> {out}  ({(time.time() - t0)/60:.1f} min)")
    print("\nmean over the 30 models, by embedding:")
    cols = ["pearson", "spearman", "r2", "average_precision", "auc", "pca_explained_variance"]
    summary = df.groupby("embedding_label", sort=False)[cols].mean()
    print(summary.reindex([DISPLAY_NAMES[k] for k in EMBEDDING_ORDER]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
