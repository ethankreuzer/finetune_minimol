"""Score exported 32-d embeddings against chemistry, not against pProp.

    .venv/bin/python src/emb_readout.py --runs outputs/rank_v1 -o reports/rank_v1_runs.csv
    .venv/bin/python src/emb_readout.py --runs outputs/wvic_scan outputs/_no_sweep  # P6.4

One CSV row per run: every hyperparameter from `meta.json["config"]`, every `val/*` from the
final epoch, the bottleneck's geometry, and a structural readout built from ECFP4.

WHY THIS EXISTS
---------------
`emb_effective_rank` is circular as a success criterion. It is computed from the same
covariance the `vic` term minimises, so a loss term that inflates it has, by construction,
optimised the metric that scores it. Rank could rise because the embedding gained chemical
information, or because the term manufactured 30 directions of orthogonal noise, and the
eigenspectrum cannot tell those apart.

The deliverable is consumed by a deep-kernel-learning GP, which uses exactly one property of
this space: *distance between molecules*. So the honest question is whether embedding distance
tracks structural dissimilarity. This module measures that against ECFP4 Tanimoto -- a
fingerprint family the model never saw, computed from `fingerprints.npy`, which the frozen
splits already persist.

THE CONTROL IS THE WHOLE POINT
------------------------------
A model that predicts pProp well will show *some* correlation between embedding distance and
structural distance for free, because similar molecules dock similarly. `pred_tanimoto_spearman`
-- rho(|delta predicted pProp|, 1 - T) -- measures exactly that free correlation, and it is
flat at ~0.07 across every run measured so far. `tanimoto_partial` removes it: it is the
correlation between rank(1 - T) and the part of rank(embedding distance) that |delta pred|
does not already explain,

    tanimoto_partial = (rho_AC - rho_AB * rho_BC) / sqrt(1 - rho_AB ** 2)

with A = embedding distance, B = |delta pred|, C = 1 - T. That is the number the
pre-registered decision rule ranks configurations on: rank without structural information is a
noise embedding and does not count.

`scalarness` -- rho(embedding distance, |delta pred|) -- states the failure mode directly. At
0.9986 (measured, seed 1, `w_vic=0`) the GP's notion of "these two molecules are far apart"
IS "these two molecules have different predicted pProp", so posterior variance stops tracking
genuine ignorance and active learning has nothing to steer on.

MECHANICS THAT MATTER
---------------------
* **Sampling is by dataset row id, never by position.** The reference sample for a fold is
  drawn from the frozen split (`splits.load_fold`), not from any run's `val_indices.npy`, so
  every run scoring that fold is scored on the same molecules. Each run maps ids to its own
  positions with `np.searchsorted` and asserts the mapping is exact -- a run whose val set
  disagrees with the frozen split is an error, not a silently different sample.
* **Tanimoto by Gram matrix.** `inter = B @ B.T` over unpacked bits, then
  `T = inter / (|a| + |b| - inter)`; distances by `x^2 + y^2 - 2xy`. Never an `[n, n, 32]`
  broadcast.
* **`T` and `rank(1 - T)` are cached per fold**, not per run -- ranking millions of pairs is
  the expensive step and it does not depend on the model.
* **Spearman throughout.** Euclidean-vs-Tanimoto is monotone at best, and Tanimoto is heavily
  right-skewed (median ~0.13), so Pearson would report the skew.
* **No pair-count standard errors.** Millions of pairs are not independent observations;
  effective n is the molecule count. Run-to-run variability comes from the seed replicates,
  like every other number in this experiment.

`knn20_jaccard` is reported beside the global rho because it is what a kernel actually uses:
the mean overlap between each molecule's 20 nearest neighbours by ECFP and its 20 nearest by
embedding. Global rho can be carried by the far tail of the distance distribution while the
local neighbourhoods -- the only part a GP with a length scale ever sees -- disagree. Treat
disagreement between the two as a finding rather than picking whichever is higher.

Geometry columns come from the FULL `val_embeddings.npy`, not from the sample and not from
`val/vic`: `train.py` computes `val/vic` on a subsample, and the diagnosis table in
`reports/embedding_collapse_experiment.md` turns on whether per-dimension stds pile up below
gamma or at it, which needs the real per-dimension spread.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import embedding_metrics                        # noqa: E402
from splits import load_fingerprints, load_fold              # noqa: E402

SAMPLE_SEED = 20260818          # + fold; fixed so the sample is stable across invocations
KNN_K = 20


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, default=[Path("outputs/rank_v1")],
                   help="roots to search for run directories (val_embeddings.npy + meta.json)")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--n-sample", type=int, default=5000,
                   help="molecules per fold for the structural readout; pairs grow as n^2")
    p.add_argument("-o", "--out", type=Path, default=Path("reports/rank_v1_runs.csv"))
    p.add_argument("--no-meta", action="store_true", help="skip the sibling meta.json")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- geometry

def geometry_columns(z, gamma):
    """Bottleneck geometry from the full validation embedding.

    `embedding_metrics` supplies the three logged numbers so this file cannot drift from what
    `train.py` reports; the rest is the per-dimension spread the diagnosis table needs.
    """
    out = dict(embedding_metrics(z))
    std = z.std(axis=0, ddof=1)
    out["emb_trace"] = float((std ** 2).sum())
    for q in (5, 25, 50, 75, 95):
        out[f"emb_std_p{q}"] = float(np.percentile(std, q))
    out["emb_std_max"] = float(std.max())
    out["n_dims_below_0.1"] = int((std < 0.1).sum())
    # Both: `gamma` answers "is this run's own hinge saturated", 0.5 is a fixed rung so cells
    # swept at different gamma stay comparable in the diagnosis table.
    out["n_dims_below_0.5"] = int((std < 0.5).sum())
    out["n_dims_below_gamma"] = int((std < gamma).sum())
    return out


# --------------------------------------------------------------- structural readout

def tanimoto_matrix(bits):
    """Pairwise Tanimoto over unpacked ECFP bits, by Gram matrix."""
    b = bits.astype(np.float32)
    inter = b @ b.T
    popcount = np.diag(inter).copy()
    union = popcount[:, None] + popcount[None, :] - inter
    return inter / np.maximum(union, 1.0)


def euclidean_matrix(x):
    """Pairwise Euclidean distance by the `x^2 + y^2 - 2xy` identity."""
    x = np.asarray(x, dtype=np.float64)
    sq = (x ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def upper(m):
    """The strict upper triangle, flattened -- each pair once, no self-pairs."""
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def knn_sets(m, k, larger_is_closer):
    """Indices of each row's k nearest neighbours, self excluded."""
    m = m.copy()
    np.fill_diagonal(m, -np.inf if larger_is_closer else np.inf)
    order = np.argpartition(-m if larger_is_closer else m, k, axis=1)[:, :k]
    return order


def knn_jaccard(a_idx, b_idx, k):
    """Mean Jaccard overlap of two neighbour listings, molecule by molecule."""
    overlaps = np.empty(len(a_idx))
    for i in range(len(a_idx)):
        inter = len(set(a_idx[i].tolist()) & set(b_idx[i].tolist()))
        overlaps[i] = inter / (2 * k - inter)
    return float(overlaps.mean())


class FoldReference:
    """The per-fold sample, its Tanimoto matrix, and the ranks derived from it.

    Built once per fold and reused by every run scoring that fold: `rankdata` over millions of
    pairs is the expensive step here and it does not depend on the model.
    """

    def __init__(self, split_dir, fold, n_sample):
        _, val_idx = load_fold(split_dir, fold)
        val_idx = np.sort(np.asarray(val_idx))
        rng = np.random.default_rng(SAMPLE_SEED + fold)
        n = min(n_sample, len(val_idx))
        self.fold = fold
        self.rows = np.sort(rng.choice(val_idx, n, replace=False))

        fp = load_fingerprints(split_dir)
        bits = np.unpackbits(fp[self.rows], axis=1)
        self.tan = tanimoto_matrix(bits)
        self.dis_rank = rankdata(upper(1.0 - self.tan))
        self.tan_knn = knn_sets(self.tan, KNN_K, larger_is_closer=True)

    def positions_in(self, val_indices):
        """Map this fold's sampled row ids onto one run's own row order."""
        order = np.argsort(val_indices)
        pos = order[np.searchsorted(val_indices, self.rows, sorter=order)]
        if not np.array_equal(np.asarray(val_indices)[pos], self.rows):
            raise ValueError("run's val_indices do not contain the frozen split's rows -- "
                             "this run was not trained on the split it claims")
        return pos


def structural_columns(ref, z_sample, pred_sample):
    """The Tanimoto readout for one run, against one fold's cached reference."""
    emb_rank = rankdata(upper(euclidean_matrix(z_sample)))
    pred = np.asarray(pred_sample, dtype=np.float64)
    pred_rank = rankdata(upper(np.abs(pred[:, None] - pred[None, :])))

    def rho(a, b):
        return float(np.corrcoef(a, b)[0, 1])         # Pearson on ranks == Spearman

    r_ac = rho(emb_rank, ref.dis_rank)                # embedding vs structure
    r_ab = rho(emb_rank, pred_rank)                   # the failure mode
    r_bc = rho(pred_rank, ref.dis_rank)               # the control
    denom = np.sqrt(max(1.0 - r_ab ** 2, 1e-12))

    emb_knn = knn_sets(euclidean_matrix(z_sample), KNN_K, larger_is_closer=False)
    return {"tanimoto_spearman": r_ac,
            "pred_tanimoto_spearman": r_bc,
            "scalarness": r_ab,
            "tanimoto_partial": (r_ac - r_ab * r_bc) / denom,
            "knn20_jaccard": knn_jaccard(ref.tan_knn, emb_knn, KNN_K),
            "n_readout": len(ref.rows)}


# ------------------------------------------------------------------------------ driver

def find_runs(roots):
    seen, runs = set(), []
    for root in roots:
        for emb in sorted(Path(root).rglob("val_embeddings.npy")):
            d = emb.parent
            if d in seen or not (d / "meta.json").exists():
                continue
            seen.add(d)
            runs.append(d)
    return runs


def score_run(run_dir, refs, split_dir, n_sample):
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta["config"]
    fold = int(cfg["fold"])
    z = np.load(run_dir / "val_embeddings.npy")
    val_indices = np.load(run_dir / "val_indices.npy")
    pred = np.load(run_dir / "val_predictions.npy")

    row = {"run": str(run_dir),
           # The cell label is the directory, but nothing is READ from it: every
           # hyperparameter below comes from meta.json, so a mislabelled directory is a
           # visible inconsistency rather than a silent one.
           "cell": run_dir.parent.name if run_dir.name.startswith("fold") else run_dir.name,
           "objective_version": meta.get("objective_version"),
           "split_sha256": meta.get("split_sha256"),
           "input_sha256": meta.get("input_sha256"),
           "n_val": len(val_indices), "n_epochs": len(meta["history"])}
    row.update({f"cfg_{k}": v for k, v in cfg.items()
                if not isinstance(v, (list, dict)) or k in ("wandb_tags",)})
    row.update({k: v for k, v in meta["history"][-1].items() if k.startswith("val/")})
    row.update(geometry_columns(z, float(cfg.get("vic_gamma", 1.0))))

    if len(z) < len(val_indices):
        raise ValueError(f"{run_dir}: {len(z)} embeddings for {len(val_indices)} val rows")
    if cfg.get("subset"):
        row["readout_skipped"] = "subset run"
        return row
    if fold not in refs:
        refs[fold] = FoldReference(split_dir, fold, n_sample)
    ref = refs[fold]
    pos = ref.positions_in(val_indices)
    row.update(structural_columns(ref, z[pos], pred[pos]))
    return row


def main(argv=None):
    args = parse_args(argv)
    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs with val_embeddings.npy under {args.runs}")

    refs, rows, failed = {}, [], []
    for d in runs:
        try:
            rows.append(score_run(d, refs, args.splits, args.n_sample))
            r = rows[-1]
            print(f"{r['cell']:>12} seed{r.get('cfg_seed')} "
                  f"w_vic={r.get('cfg_w_vic'):<6} rank={r.get('emb_effective_rank', float('nan')):6.3f} "
                  f"tan={r.get('tanimoto_spearman', float('nan')):.3f} "
                  f"partial={r.get('tanimoto_partial', float('nan')):+.3f} "
                  f"scalar={r.get('scalarness', float('nan')):.3f} "
                  f"knn={r.get('knn20_jaccard', float('nan')):.3f}")
        except Exception as exc:                             # noqa: BLE001 -- reported, not swallowed
            failed.append((str(d), repr(exc)))
            print(f"FAILED {d}: {exc}")

    df = pd.DataFrame(rows).sort_values(["cell", "cfg_seed"], kind="stable")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} runs, {len(df.columns)} columns)")

    triples = df[["objective_version", "split_sha256", "input_sha256"]].drop_duplicates()
    if len(triples) > 1:
        print("WARNING: runs span more than one provenance triple; they are not comparable:")
        print(triples.to_string(index=False))

    if not args.no_meta:
        meta_path = args.out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(
            {"script": "src/emb_readout.py", "argv": sys.argv[1:],
             "n_sample": args.n_sample, "sample_seed": SAMPLE_SEED, "knn_k": KNN_K,
             "runs": [str(d) for d in runs], "failed": failed,
             "folds_sampled": {str(f): ref.rows.tolist()[:5] for f, ref in refs.items()},
             "provenance_triples": triples.to_dict("records")}, indent=2))
        print(f"wrote {meta_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
