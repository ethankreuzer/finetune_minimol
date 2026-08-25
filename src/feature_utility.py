"""Does the exported 32-d embedding actually work as a featurizer?

    .venv/bin/python src/feature_utility.py --runs outputs/rank_v1 outputs/rank_v2 \
        -o reports/feature_utility.csv

S1 and S2 establish that the 32 slots are OCCUPIED (effective rank 28.6/32) and that they are
occupied by CHEMISTRY rather than noise or restated pProp (`tanimoto_partial` 0.139 against a
0.022 control). Neither establishes that the dimensions are USEFUL. `vic` is unsupervised:
chemical variation irrelevant to AmpC binding satisfies it exactly as well as variation that
matters, so nothing in the loss steers the recovered dimensions toward relevance. A random 32-d
projection of ECFP4 bits would score well on the structural readout while being no better as a
featurizer than the fingerprints it came from.

This module probes the featurization directly: fit a cheap model on labelled molecules in one
set of clusters, predict pProp for molecules in clusters it never saw, and compare.

HELD-OUT CLUSTERS, NEVER HELD-OUT MOLECULES. The deliverable is consumed by a GP that an active
learning loop will query on novel chemistry. Fold 0's validation clusters are split 50/50, the
probe fits on one half and is scored on the other, then the halves swap and the two directions
are averaged. A random molecule split would leave near-duplicates on both sides and flatter
every featurization here, the 32-d one most of all.

TWO PROBES, BECAUSE THEY ASK DIFFERENT QUESTIONS.
  * kNN (k=20, distance-weighted) is the PRIMARY. A GP kernel consumes distances, so "is
    distance in this space informative about pProp" is the question the deliverable turns on.
  * Ridge tests linear extractability -- the property `head.py`'s linear task heads exist to
    guarantee, and the geometry an RBF/Matern kernel prefers.

THE BASELINES ARE THE POINT. The embedding's own number means nothing alone.
  * `pred1` -- the model's own predicted pProp, as a 1-d featurization. THE FLOOR. If 32
    dimensions cannot beat their own scalar output, nothing else in this experiment matters.
  * `pca{k}` -- the top k principal components. If k=32 does not beat k=2, the rank recovered
    by the `vic` term is cosmetic for prediction.
  * `minimol512` -- the raw frozen trunk embedding, i.e. the workflow this repo exists to beat.
  * `ecfp4` -- chemistry with no model in it at all.

The pre-registered rules U1/U2/U3 are in `reports/embedding_collapse_experiment.md` S2.5 and
were written before this file was run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

KNN_K = 20
POSITIVE_EDGE = 3.5
PCA_KS = (1, 2, 4, 8, 16, 32)


def load_reference(split_dir, csv_path, fold, n_per_half, seed):
    """Fold `fold`'s validation rows, split 50/50 BY CLUSTER into two disjoint halves.

    Clusters are dealt whole, so no molecule in one half has its own cluster-mates in the
    other. Dealing is by shuffled cluster id rather than by size, which leaves the halves
    unequal in count; that is fine because each half is then subsampled to `n_per_half`.
    """
    import splits as splits_mod                       # noqa: E402 -- path set above
    _, val_idx = splits_mod.load_fold(str(split_dir), fold=fold)
    assign = pd.read_csv(Path(split_dir) / "assignments.csv", usecols=["row_idx", "cluster_id"])
    cluster = assign.set_index("row_idx")["cluster_id"]
    y = pd.read_csv(csv_path, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)

    val_idx = np.asarray(sorted(val_idx))
    cl = cluster.loc[val_idx].to_numpy()
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(uniq)
    left = set(uniq[: len(uniq) // 2].tolist())
    mask = np.array([c in left for c in cl])

    halves = []
    for take in (mask, ~mask):
        rows = val_idx[take]
        if len(rows) > n_per_half:
            rows = np.sort(rng.choice(rows, n_per_half, replace=False))
        halves.append(rows)
    return val_idx, halves, y


def knn_predict(fit_x, fit_y, test_x, k=KNN_K, block=2048):
    """Distance-weighted kNN regression, blocked so the distance matrix never materialises."""
    fit_x = np.ascontiguousarray(fit_x, dtype=np.float32)
    test_x = np.ascontiguousarray(test_x, dtype=np.float32)
    fit_sq = (fit_x ** 2).sum(1)
    out = np.empty(len(test_x), dtype=np.float64)
    for s in range(0, len(test_x), block):
        q = test_x[s:s + block]
        d2 = fit_sq[None, :] - 2.0 * (q @ fit_x.T) + (q ** 2).sum(1)[:, None]
        np.maximum(d2, 0, out=d2)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        dist = np.sqrt(np.take_along_axis(d2, nn, axis=1))
        # Inverse-distance weights, floored so an exact duplicate does not become infinite.
        w = 1.0 / np.maximum(dist, 1e-6)
        out[s:s + block] = (w * fit_y[nn]).sum(1) / w.sum(1)
    return out


def ridge_predict(fit_x, fit_y, test_x, alpha=1.0):
    m = Ridge(alpha=alpha)
    m.fit(fit_x, fit_y)
    return m.predict(test_x)


def score(true, pred):
    """Held-out skill. Spearman is the pre-registered response; the rest are context."""
    if np.std(pred) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "ap": np.nan, "mae": np.nan}
    return {"spearman": float(spearmanr(true, pred).statistic),
            "pearson": float(np.corrcoef(true, pred)[0, 1]),
            "ap": float(average_precision_score(true >= POSITIVE_EDGE, pred)),
            "mae": float(np.abs(true - pred).mean())}


def pca_truncations(z_fit, z_test, ks):
    """Top-k principal components, basis fitted on the PROBE-FIT half only.

    Fitting the basis on the test half too would leak the test set's covariance into the
    featurization -- small here, but it is exactly the kind of leak that makes a k-curve
    climb for the wrong reason.
    """
    mu = z_fit.mean(0)
    _, _, vt = np.linalg.svd(z_fit - mu, full_matrices=False)
    out = {}
    for k in ks:
        if k > z_fit.shape[1]:
            continue
        p = vt[:k]
        out[f"pca{k}"] = ((z_fit - mu) @ p.T, (z_test - mu) @ p.T)
    return out


def minimol_512(rows, features_dir, batch_size=512):
    """The raw frozen trunk embedding for `rows` -- the baseline this repo exists to beat."""
    import torch
    from torch.utils.data import DataLoader, Subset
    import trunk as trunk_mod                          # noqa: E402
    from features import load_features                 # noqa: E402

    tk = trunk_mod.MiniMolTrunk()
    tk.eval()
    ds = load_features(str(features_dir))
    loader = DataLoader(Subset(ds, list(rows)), batch_size=batch_size, shuffle=False,
                        collate_fn=tk.collate)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(tk(tk.to_device(batch)).detach().cpu().numpy())
    return np.concatenate(chunks, 0).astype(np.float64)


def featurizations_for_run(run_dir, halves):
    """The featurizations that DEPEND ON THE RUN, as {name: (fit_matrix, test_matrix)}.

    `ecfp4` and `minimol512` are deliberately not here: neither depends on the run, so scoring
    them once per run would report a single measurement eighteen times and give them an
    artificial zero variance in any aggregate. They are scored once, as shared baselines.
    """
    val_indices = np.load(run_dir / "val_indices.npy")
    z = np.load(run_dir / "val_embeddings.npy")
    pred = np.load(run_dir / "val_predictions.npy")
    order = np.argsort(val_indices)
    sorted_idx = val_indices[order]

    def positions(rows):
        pos = np.searchsorted(sorted_idx, rows)
        if not np.array_equal(sorted_idx[pos], rows):
            raise ValueError(f"{run_dir}: val set does not contain the reference rows")
        return order[pos]

    pf, pt = positions(halves[0]), positions(halves[1])
    feats = {"pred1": (pred[pf].reshape(-1, 1).astype(np.float64),
                       pred[pt].reshape(-1, 1).astype(np.float64))}
    feats.update(pca_truncations(z[pf].astype(np.float64), z[pt].astype(np.float64), PCA_KS))
    feats["emb32"] = (z[pf].astype(np.float64), z[pt].astype(np.float64))
    return feats


def score_featurizations(feats, y_fit, y_test):
    """Both probes on every featurization; one row per (feat, probe)."""
    out = []
    for name, (xf, xt) in feats.items():
        for probe, fn in (("knn", knn_predict), ("ridge", ridge_predict)):
            out.append({"feat": name, "probe": probe, **score(y_test, fn(xf, y_fit, xt))})
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path,
                   default=[Path("outputs/rank_v1"), Path("outputs/rank_v2")])
    p.add_argument("--cells", nargs="+",
                   default=["A_base", "C_w1", "D_w3", "E_w10"],
                   help="cells to probe; the k-curve is read against the w_vic dose")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv-path", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-per-half", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--no-minimol", action="store_true",
                   help="skip the raw 512-d baseline (needs a GPU forward pass)")
    p.add_argument("-o", "--out", type=Path, default=Path("reports/feature_utility.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    val_idx, halves, y = load_reference(args.splits, args.csv_path, args.fold,
                                        args.n_per_half, args.seed)
    print(f"fold {args.fold}: {len(val_idx)} val rows -> halves of "
          f"{len(halves[0])} / {len(halves[1])}, split by cluster")

    fp = np.load(args.splits / "fingerprints.npy")
    fp_halves = [np.unpackbits(fp[h], axis=1).astype(np.float32) for h in halves]

    extra = {}
    if not args.no_minimol:
        print("computing the raw 512-d MiniMol baseline (one forward pass, no backward)...")
        mm = [minimol_512(h, args.features) for h in halves]
        extra["minimol512"] = (mm[0], mm[1])

    runs = []
    for root in args.runs:
        for emb in sorted(Path(root).rglob("val_embeddings.npy")):
            d = emb.parent
            cell = d.parent.name
            if cell in args.cells and (d / "meta.json").exists():
                runs.append(d)

    rows = []
    # ---- shared baselines, scored once ------------------------------------------------
    shared = []
    for direction, hs in enumerate((halves, halves[::-1])):
        fp_d = fp_halves if direction == 0 else fp_halves[::-1]
        feats = {"ecfp4": (fp_d[0], fp_d[1])}
        feats.update({k: (v if direction == 0 else v[::-1]) for k, v in extra.items()})
        shared += score_featurizations(feats, y[hs[0]], y[hs[1]])
    for _, r in pd.DataFrame(shared).groupby(["feat", "probe"]).mean().reset_index().iterrows():
        rows.append({"cell": "(shared)", "seed": -1, "run": "", "w_vic": None, **r.to_dict()})
        if r["probe"] == "knn":
            print(f"{'(shared)':>12}        knn spearman: {r['feat']}={r['spearman']:.4f}")

    for d in runs:
        meta = json.loads((d / "meta.json").read_text())
        cfg = meta["config"]
        if int(cfg["fold"]) != args.fold:
            continue
        # Both directions: fit on half 0 / test on half 1, then the reverse. Averaging the two
        # uses every molecule once as training and once as test, and halves the split variance.
        per_dir = []
        for hs in (halves, halves[::-1]):
            per_dir += score_featurizations(featurizations_for_run(d, hs), y[hs[0]], y[hs[1]])
        agg = (pd.DataFrame(per_dir).groupby(["feat", "probe"]).mean().reset_index())
        for _, r in agg.iterrows():
            rows.append({"cell": d.parent.name, "seed": int(cfg["seed"]),
                         "run": str(d), "w_vic": cfg.get("w_vic"), **r.to_dict()})
        best = agg[(agg.feat == "emb32") & (agg.probe == "knn")]["spearman"].iloc[0]
        floor = agg[(agg.feat == "pred1") & (agg.probe == "knn")]["spearman"].iloc[0]
        print(f"{d.parent.name:>12} seed{cfg['seed']}  knn spearman: emb32={best:.4f} "
              f"pred1={floor:.4f}  delta={best - floor:+.4f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"script": "src/feature_utility.py", "argv": sys.argv[1:], "fold": args.fold,
         "n_per_half": int(len(halves[0])), "seed": args.seed, "knn_k": KNN_K,
         "n_runs": int(df["run"].nunique()) if len(df) else 0}, indent=2))
    print(f"\nwrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
