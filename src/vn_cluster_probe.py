"""Does a virtual node tap add anything to the pooled embedding, on chemistry it has not seen?

Two things separate this from `vn_probe.py`, and both change what the answer means.

**HELD-OUT CLUSTERS, NOT HELD-OUT ROWS.** `vn_probe.py`'s `_oob` columns hold a molecule out of
the trees that scored it, but its structural analogs stay in -- and this library is full of
analogs. The deliverable is consumed by a GP that an active-learning loop will query on novel
chemistry, so the question is generalisation to clusters the probe never saw. Fold 0's
validation clusters are dealt 50/50, the probe fits on one half and is scored on the other, and
then the halves swap. `feature_utility.load_reference` already does exactly this dealing.

**CONCATENATIONS, NOT JUST SINGLE TENSORS.** The encoder question is not "is the virtual node
better than the pooled embedding" -- `vn_probe.py` answered that, and it is not -- but "does
`pooled ⊕ vn` beat `pooled`". A gain there changes the deliverable; a gain for the virtual node
alone does not.

THE NULL IS WHAT MAKES THE CONCATENATION A RESULT
-------------------------------------------------
`pooled512 ⊕ randproj(pooled512 -> 336)` is the same width as `pooled512 ⊕ vn15` and carries
exactly the same information as `pooled512` alone -- a random linear projection adds no bits.
Without it, "the concatenation scored higher" is confounded with "kNN liked the extra 336
dimensions", and the two are indistinguishable from the number alone.

BLOCK SCALING, AND WHY THE ORDER MATTERS
----------------------------------------
The virtual node grows ~17x in norm across the stack (`Minimol_architecture_overview.md` §9), so
a raw `pooled ⊕ vn15` would hand the Euclidean distance to whichever block happens to be larger
and the comparison would be about scale rather than content. Each block is z-scored per
dimension and then divided by `sqrt(d_block)`, so every block contributes equal total variance
to the distance.

**z-score first, then divide.** The other order is not a different convention, it is a no-op:
dividing before z-scoring is undone by the z-score, and the result would silently be
"unweighted concatenation" while claiming to be balanced. Statistics come from the FIT half
only, for the reason `feature_utility.pca_truncations` already gives -- fitting on both halves
leaks the test half's covariance into the featurization.

WHAT IS SCORED
--------------
    vn03..vn15, pooled512   the six tensors, alone, raw -- comparable to each other
    z                       the head export -- the CURRENT deliverable, not pooled512
    pooled@scaled           the baseline the two concatenations are read against
    pooled+vn15             the complementarity test
    pooled+randproj         the dimension-matched null
    ecfp4                   chemistry with no model in it. Run-independent, so scored ONCE
    pred1                   the model's own predicted pProp, as a 1-d featurization

**Read the concatenation against `pooled@scaled`, never against `pooled512`.** The three
scaled rows differ only in what the second block contains; `pooled512` differs from all of
them in preprocessing too.

`pred1` is a BAR, not a floor. The sealed 32-d arc called it the floor because a 32-d
bottleneck that cannot beat its own scalar output is worthless -- but the model that produced
it never saw fold 0, so as a featurization it is strong, and measured here it is competitive
with every embedding. An embedding losing to `pred1` is a finding about the embedding.

Every one is scored twice: kNN (primary -- a GP kernel consumes distances) and ridge (linear
extractability). `--pca` adds a width-matched 32-component variant of each, since 336 vs 512 vs
848 is itself a confound.

The `pretrained/frozen` row has no head, so `z` and `pred1` do not exist for it and are skipped
rather than faked.

Usage:
    python src/vn_cluster_probe.py --root outputs/vn_analysis/lyc0lh2d
    python src/vn_cluster_probe.py --root outputs/vn_analysis/lyc0lh2d --limit 1  # smoke gate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_utility import (KNN_K, POSITIVE_EDGE, knn_predict,  # noqa: E402
                             load_reference, ridge_predict, score)
from vn_probe import discover                                    # noqa: E402
from vn_taps import DISPLAY_NAMES, EMBEDDING_ORDER, POOLED_KEY   # noqa: E402

# The tap carried into the concatenation. VN15 rather than all five: it is the deepest tap and
# the one `vn_probe.py` measured as strongest on every out-of-bag column, so if any virtual node
# adds to the pooled embedding this is the one that does. Scanning all five would multiply the
# run for a comparison the depth trend has already ordered.
CONCAT_TAP = "vn15"

PROBES = (("knn", knn_predict), ("ridge", ridge_predict))
PCA_K = 32
PROJ_SEED = 20260828


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="outputs/vn_analysis/<sweep_id>")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-per-half", type=int, default=5000,
                   help="molecules per cluster-disjoint half. kNN is O(n^2 d) and this runs "
                        "over ~20 featurizations x 2 directions x every model")
    p.add_argument("--pca", action="store_true",
                   help="also score a width-matched PCA-32 variant of every featurization")
    p.add_argument("--seed", type=int, default=20260819,
                   help="the cluster deal; matches feature_utility.py's default so the halves "
                        "are the same molecules that probe used")
    p.add_argument("--limit", type=int, default=None, help="score only the first N models")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="default: figures/<sweep_id>/cluster_probe.csv")
    return p.parse_args(argv)


def block_scale(fit, test):
    """Z-score per dimension on the FIT half, then divide by sqrt(d).

    Returns both halves under one transform. The division is what makes concatenated blocks
    comparable: after z-scoring, each of d dimensions has unit variance, so a block contributes
    d to the squared distance; dividing by sqrt(d) normalises that to 1 regardless of width.
    """
    mu = fit.mean(0)
    sd = fit.std(0)
    # A dead dimension would divide by zero and produce inf, which kNN turns into a silent
    # single-dimension distance. Flooring leaves it contributing nothing instead.
    sd = np.where(sd > 1e-12, sd, 1.0)
    scale = np.sqrt(fit.shape[1])
    return ((fit - mu) / sd / scale, (test - mu) / sd / scale)


def pca32(fit, test, k=PCA_K):
    """Top-k components, basis fitted on the FIT half only. None if the block is narrower."""
    if fit.shape[1] <= k:
        return None
    mu = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    p = vt[:k]
    return ((fit - mu) @ p.T, (test - mu) @ p.T)


def featurizations(d, z, pred, pf, pt, rng):
    """{name: (fit_matrix, test_matrix)} for one model at one direction of the split.

    `z` and `pred` are None for the pretrained baseline, which has no head; those two entries
    are then absent rather than filled with a stand-in.
    """
    def block(key):
        return (d[key][pf].astype(np.float64), d[key][pt].astype(np.float64))

    feats = {k: block(k) for k in EMBEDDING_ORDER}

    pooled_f, pooled_t = block(POOLED_KEY)
    vn_f, vn_t = block(CONCAT_TAP)
    sp_f, sp_t = block_scale(pooled_f, pooled_t)
    sv_f, sv_t = block_scale(vn_f, vn_t)
    feats[f"pooled+{CONCAT_TAP}"] = (np.hstack([sp_f, sv_f]), np.hstack([sp_t, sv_t]))

    # The BASELINE FOR THE CONCATENATIONS, and not the same thing as `pooled512` above.
    # `pooled512` is scored raw, like the other five tensors, so those six stay comparable to
    # each other. The concatenations are necessarily block-scaled -- and neither kNN nor ridge
    # at a fixed alpha is scale-invariant, so comparing a scaled concatenation against a raw
    # pooled512 would mix a change of content with a change of preprocessing. Measured at
    # n=1500: raw pooled512 scores ridge spearman 0.8245 while pooled+randproj, which carries
    # exactly the same information, scores 0.8523. That gap is entirely preprocessing, and
    # without this row it would have read as the null beating the baseline.
    feats["pooled@scaled"] = (sp_f, sp_t)

    # The null: the same width, built from pooled512 alone, so it carries no information the
    # pooled block does not already have. One projection matrix for both halves -- a different
    # one per half would not be a featurization at all.
    proj = rng.standard_normal((pooled_f.shape[1], vn_f.shape[1])) / np.sqrt(pooled_f.shape[1])
    rp_f, rp_t = block_scale(pooled_f @ proj, pooled_t @ proj)
    feats["pooled+randproj"] = (np.hstack([sp_f, rp_f]), np.hstack([sp_t, rp_t]))

    if z is not None:
        feats["z"] = (z[pf].astype(np.float64), z[pt].astype(np.float64))
    if pred is not None:
        feats["pred1"] = (pred[pf].reshape(-1, 1).astype(np.float64),
                          pred[pt].reshape(-1, 1).astype(np.float64))
    return feats


def score_all(feats, y_fit, y_test, use_pca):
    """Both probes on every featurization -> {name: {knn_spearman: .., ridge_ap: .., ..}}.

    One row per featurization with the two probes side by side, rather than one row per
    (featurization, probe): it keeps the CSV directly plottable by `vn_figures.py`, whose
    violins key on one row per model per embedding.
    """
    out = {}
    for name, (xf, xt) in feats.items():
        variants = {name: (xf, xt)}
        if use_pca:
            red = pca32(xf, xt)
            if red is not None:
                variants[f"{name}@pca{PCA_K}"] = red
        for vname, (a, b) in variants.items():
            row = {}
            for probe, fn in PROBES:
                row.update({f"{probe}_{k}": v for k, v in
                            score(y_test, fn(a, y_fit, b)).items()})
            out[vname] = row
    return out


def mean_of_directions(a, b):
    """Average the two fit/test directions. Missing in one direction means missing."""
    return {k: {m: float(np.mean([a[k][m], b[k][m]])) for m in a[k]}
            for k in a if k in b}


def main(argv=None):
    args = parse_args(argv)
    sweep_id = args.root.name
    runs = discover(args.root)
    if args.limit:
        runs = runs[:args.limit]

    val_idx, halves, y_all = load_reference(args.splits, args.csv, args.fold,
                                            args.n_per_half, args.seed)
    y = [y_all[h] for h in halves]
    n_pos = [int((h >= POSITIVE_EDGE).sum()) for h in y]
    t0 = time.time()
    print(f"sweep {sweep_id} | {len(runs)} models | fold {args.fold}: {len(val_idx):,} val rows "
          f"-> cluster-disjoint halves of {len(halves[0]):,} / {len(halves[1]):,}")
    print(f"positives at pProp >= {POSITIVE_EDGE}: {n_pos[0]} / {n_pos[1]}  "
          f"| kNN k={KNN_K}, both directions averaged\n")

    # ECFP4 does not depend on the model, so it is scored ONCE. Repeating it per model would
    # report a single measurement 31 times and hand it an artificial zero variance in any
    # aggregate -- the mistake feature_utility.py calls out by name.
    fp = np.load(args.splits / "fingerprints.npy")
    ecfp = [np.unpackbits(fp[h], axis=1).astype(np.float64) for h in halves]
    shared = mean_of_directions(
        score_all({"ecfp4": (ecfp[0], ecfp[1])}, y[0], y[1], args.pca),
        score_all({"ecfp4": (ecfp[1], ecfp[0])}, y[1], y[0], args.pca))
    print(f"ecfp4 (run-independent, scored once): "
          f"knn spearman {shared['ecfp4']['knn_spearman']:.4f} "
          f"({time.time() - t0:.0f}s)\n")

    records = [{"sweep_id": sweep_id, "config": "shared", "bootstrap": "shared",
                "embedding": k, "embedding_label": k, "fold": args.fold,
                "n_per_half": len(halves[0]), "pprop_edge": POSITIVE_EDGE,
                "n_positive": int(np.mean(n_pos)),
                "positive_rate": float(np.mean([n / len(halves[0]) for n in n_pos])),
                **v} for k, v in shared.items()]

    for i, r in enumerate(runs, 1):
        d = np.load(r["path"])
        rows = d["row_indices"]
        order = np.argsort(rows)
        srt = rows[order]

        def positions(want):
            pos = np.searchsorted(srt, want)
            if not np.array_equal(srt[pos], want):
                raise SystemExit(f"{r['path']}: validation rows do not contain the reference "
                                 "half -- this model was not scored on the frozen split")
            return order[pos]

        p0, p1 = positions(halves[0]), positions(halves[1])
        run_dir = r["path"].parent
        has_head = (run_dir / "val_embeddings.npy").exists()
        z = np.load(run_dir / "val_embeddings.npy") if has_head else None
        pred = np.load(run_dir / "val_predictions.npy") if has_head else None

        # One projection per model, drawn from a seeded generator, so the null is a fresh draw
        # rather than one lucky matrix reused 31 times -- its spread across models is then a
        # real null distribution to read the concatenation against.
        rng = np.random.default_rng(PROJ_SEED + i)
        both = mean_of_directions(
            score_all(featurizations(d, z, pred, p0, p1, rng), y[0], y[1], args.pca),
            score_all(featurizations(d, z, pred, p1, p0, rng), y[1], y[0], args.pca))

        for name, m in both.items():
            records.append({
                "sweep_id": sweep_id, "config": r["config"], "bootstrap": r["bootstrap"],
                "embedding": name, "embedding_label": DISPLAY_NAMES.get(name, name),
                "fold": args.fold, "n_per_half": len(halves[0]),
                # Average precision is read against the positive rate and is meaningless
                # without it. Carried per row so a figure never has to hardcode a number that
                # moves with the fold, the threshold and the half size.
                "pprop_edge": POSITIVE_EDGE, "n_positive": int(np.mean(n_pos)),
                "positive_rate": float(np.mean([n / len(halves[0]) for n in n_pos])), **m})
        # The three numbers printed are the ones that must be read together: the scaled
        # baseline, the test, and the null. Printing raw pooled512 here would invite exactly
        # the comparison `pooled@scaled` exists to prevent.
        print(f"[{i:>2}/{len(runs)}] {r['config']}/{r['bootstrap']}  knn: "
              f"pooled@scaled {both['pooled@scaled']['knn_spearman']:.4f} | "
              f"+{CONCAT_TAP} {both[f'pooled+{CONCAT_TAP}']['knn_spearman']:.4f} | "
              f"+randproj {both['pooled+randproj']['knn_spearman']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    out = args.out or Path("figures") / sweep_id / "cluster_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps({
        "script": "src/vn_cluster_probe.py", "argv": sys.argv, "sweep_id": sweep_id,
        "n_models": len(runs), "fold": args.fold, "n_per_half": len(halves[0]),
        "n_positive_per_half": n_pos, "positive_edge": POSITIVE_EDGE,
        "knn_k": KNN_K, "deal_seed": args.seed, "proj_seed": PROJ_SEED,
        "concat_tap": CONCAT_TAP, "pca_k": PCA_K if args.pca else None,
        "scoring": "cluster-disjoint halves, both directions averaged",
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))

    print(f"\n{len(df)} rows -> {out}  ({(time.time() - t0) / 60:.1f} min)")
    ft = df[~df["config"].isin(["shared", "pretrained"])]
    if len(ft):
        cols = ["knn_spearman", "knn_ap", "ridge_spearman", "ridge_ap"]
        print("\nmean over the fine-tuned models:")
        print(ft.groupby("embedding", sort=False)[cols].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
