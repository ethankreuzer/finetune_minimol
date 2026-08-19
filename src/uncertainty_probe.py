"""Does the exported geometry buy usable *uncertainty*, not just usable predictions?

    .venv/bin/python src/uncertainty_probe.py -o reports/uncertainty_probe.csv

THE CLAIM THIS TESTS. The whole anti-collapse effort rests on a four-link chain:

    collapsed embedding -> GP distance degenerates into "difference in predicted pProp"
    -> posterior variance stops tracking genuine ignorance -> active learning has nothing
    to steer on

S1/S2 established the first link's premise (`scalarness` 0.960 at the control, 0.283 at the
pin). S2.5 showed the fix COSTS 0.009 held-out Spearman. Nothing has ever measured what it
BUYS. This module does, under the V1/V2/V3 rules pre-registered in
`reports/embedding_collapse_experiment.md` S2.6 before it was run.

WHY A GP AND NOT A CHEAPER PROBE. Posterior variance is the object under test, and only a
probabilistic model has one. Exact GP regression, Matern 5/2 plus a white-noise term, a single
isotropic length-scale fitted per run. `n_fit` is small (~1,000) because GPs are O(N^3) and
because early active learning genuinely has few labels -- that is the regime the deliverable
exists for, not a compromise.

EMBEDDINGS ARE FED RAW. Standardising would rescale the control's ~30 near-dead dimensions to
unit variance and pour pure noise into the kernel, so `A_base` would fail for a preprocessing
artefact rather than for the pathology under test. "Deploy the frozen embedding into a GP"
means the raw tensor. `--scaling zscore` runs the robustness check.

THE THREE MEASUREMENTS.
  V1 `calib_rho`     rho(posterior std, |actual error|). Is the variance about anything at all?
  V2 `novelty_rho`   rho(posterior std, 1 - max Tanimoto to any training molecule). Does
                     "uncertain" mean "structurally unlike what I have seen"? THE ONE THAT
                     MATTERS -- reported beside `predext_rho`, rho(std, |pred - median pred|),
                     which is the failure mode in its own currency: variance that is merely
                     extremity along the prediction axis.
  V3 `ucb_hits`      simulated batched acquisition; cumulative pProp >= 3.5 found, against
                     random and against greedy-on-mean.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utility import load_reference                      # noqa: E402

POSITIVE_EDGE = 3.5


def fit_gp(x, y, seed):
    """Exact GP with the length-scale fitted, not assumed.

    A fixed length-scale would silently favour whichever embedding happened to match it, and
    the cells differ in operating scale by 9x (`emb_trace` 4.0 at the control, 34.9 at
    gamma=1.0) -- which is exactly the axis S2 found to matter.
    """
    k = (ConstantKernel(1.0, (1e-3, 1e3))
         * Matern(length_scale=np.sqrt(x.shape[1]), length_scale_bounds=(1e-2, 1e4), nu=2.5)
         + WhiteKernel(0.1, (1e-6, 1e1)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0,
                                  random_state=seed)
    gp.fit(x, y)
    return gp


def tanimoto_novelty(test_bits, fit_bits, block=1024):
    """1 - max Tanimoto to any training molecule. Gram trick; never an [n, m, 2048] array."""
    fit_pop = fit_bits.sum(1)
    out = np.empty(len(test_bits), dtype=np.float64)
    for s in range(0, len(test_bits), block):
        q = test_bits[s:s + block]
        inter = q @ fit_bits.T
        union = q.sum(1)[:, None] + fit_pop[None, :] - inter
        out[s:s + block] = 1.0 - (inter / np.maximum(union, 1.0)).max(1)
    return out


def nearest_distance(test_x, fit_x, block=2048):
    fit_sq = (fit_x ** 2).sum(1)
    out = np.empty(len(test_x), dtype=np.float64)
    for s in range(0, len(test_x), block):
        q = test_x[s:s + block]
        d2 = fit_sq[None, :] - 2.0 * (q @ fit_x.T) + (q ** 2).sum(1)[:, None]
        out[s:s + block] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def rho(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def acquisition_sim(x_fit, y_fit, x_pool, y_pool, kernel, seed, n0=200, rounds=8, batch=50):
    """Batched active learning over the held-out pool, one curve per strategy.

    The kernel is FIXED to the one already fitted on the full training half rather than
    re-optimised each round: re-fitting hyperparameters on 200 points would make the early
    rounds a test of hyperparameter estimation rather than of the embedding.
    """
    rng = np.random.default_rng(seed)
    start = rng.choice(len(x_fit), n0, replace=False)
    hits = {}
    for strategy in ("random", "greedy", "ucb", "maxvar"):
        tx, ty = x_fit[start].copy(), y_fit[start].copy()
        taken = np.zeros(len(x_pool), dtype=bool)
        found = []
        srng = np.random.default_rng(seed + 1)
        for _ in range(rounds):
            avail = np.flatnonzero(~taken)
            if strategy == "random":
                pick = srng.choice(avail, min(batch, len(avail)), replace=False)
            else:
                gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
                gp.fit(tx, ty)
                mu, sd = gp.predict(x_pool[avail], return_std=True)
                sc = {"greedy": mu, "ucb": mu + 2.0 * sd, "maxvar": sd}[strategy]
                pick = avail[np.argsort(-sc)[:batch]]
            taken[pick] = True
            tx = np.concatenate([tx, x_pool[pick]])
            ty = np.concatenate([ty, y_pool[pick]])
            found.append(int((y_pool[taken] >= POSITIVE_EDGE).sum()))
        hits[strategy] = found
    return hits


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path,
                   default=[Path("outputs/rank_v1"), Path("outputs/rank_v2")])
    p.add_argument("--cells", nargs="+", default=["A_base", "C_w1", "D_w3", "E_w10"])
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv-path", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-per-half", type=int, default=5000)
    p.add_argument("--n-fit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--scaling", choices=["raw", "zscore"], default="raw")
    p.add_argument("--no-acquisition", action="store_true", help="skip V3")
    p.add_argument("-o", "--out", type=Path, default=Path("reports/uncertainty_probe.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _, halves, y = load_reference(args.splits, args.csv_path, args.fold,
                                  args.n_per_half, args.seed)
    fp = np.load(args.splits / "fingerprints.npy")
    bits = [np.unpackbits(fp[h], axis=1).astype(np.float32) for h in halves]
    print(f"fold {args.fold}: halves of {len(halves[0])} / {len(halves[1])} by cluster; "
          f"n_fit={args.n_fit}, scaling={args.scaling}")

    runs = [d for root in args.runs for emb in sorted(Path(root).rglob("val_embeddings.npy"))
            for d in [emb.parent]
            if d.parent.name in args.cells and (d / "meta.json").exists()]

    rows = []
    for d in runs:
        meta = json.loads((d / "meta.json").read_text())
        cfg = meta["config"]
        if int(cfg["fold"]) != args.fold:
            continue
        val_indices = np.load(d / "val_indices.npy")
        z_all = np.load(d / "val_embeddings.npy").astype(np.float64)
        pred_all = np.load(d / "val_predictions.npy").astype(np.float64)
        order = np.argsort(val_indices)
        sorted_idx = val_indices[order]

        def pos(rows_):
            j = np.searchsorted(sorted_idx, rows_)
            if not np.array_equal(sorted_idx[j], rows_):
                raise ValueError(f"{d}: val set does not contain the reference rows")
            return order[j]

        # Both directions, as in S2.5: every molecule serves once as training and once as test.
        for direction, (hf, ht) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
            bf = bits[0] if direction == 0 else bits[1]
            bt = bits[1] if direction == 0 else bits[0]
            pf, pt = pos(hf), pos(ht)
            zf, zt = z_all[pf], z_all[pt]
            if args.scaling == "zscore":
                mu, sd = zf.mean(0), zf.std(0) + 1e-8
                zf, zt = (zf - mu) / sd, (zt - mu) / sd
            yf, yt = y[hf], y[ht]

            rng = np.random.default_rng(args.seed + direction)
            sub = rng.choice(len(zf), min(args.n_fit, len(zf)), replace=False)
            gp = fit_gp(zf[sub], yf[sub], args.seed)
            mean, std = gp.predict(zt, return_std=True)

            err = np.abs(yt - mean)
            nov = tanimoto_novelty(bt, bf[sub])
            predext = np.abs(pred_all[pt] - np.median(pred_all[pf][sub]))
            r = {"cell": d.parent.name, "seed": int(cfg["seed"]), "direction": direction,
                 "run": str(d), "w_vic": cfg.get("w_vic"), "scaling": args.scaling,
                 "calib_rho": rho(std, err),
                 "novelty_rho": rho(std, nov),
                 "predext_rho": rho(std, predext),
                 "embnn_rho": rho(std, nearest_distance(zt, zf[sub])),
                 "coverage95": float((err <= 1.96 * std).mean()),
                 "mean_std": float(std.mean()), "gp_spearman": rho(mean, yt),
                 "kernel": str(gp.kernel_)}
            if not args.no_acquisition:
                h = acquisition_sim(zf, yf, zt, yt, gp.kernel_, args.seed + direction)
                r.update({f"hits_{k}": v[-1] for k, v in h.items()})
                r["n_pool_positives"] = int((yt >= POSITIVE_EDGE).sum())
            rows.append(r)
            print(f"{r['cell']:>12} seed{r['seed']} dir{direction}  calib={r['calib_rho']:+.3f} "
                  f"novelty={r['novelty_rho']:+.3f} predext={r['predext_rho']:+.3f}"
                  + ("" if args.no_acquisition else
                     f"  ucb={r.get('hits_ucb')} rand={r.get('hits_random')} "
                     f"greedy={r.get('hits_greedy')}/{r.get('n_pool_positives')}"))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"script": "src/uncertainty_probe.py", "argv": sys.argv[1:], "fold": args.fold,
         "n_per_half": int(len(halves[0])), "n_fit": args.n_fit, "scaling": args.scaling,
         "seed": args.seed, "n_runs": int(df["run"].nunique()) if len(df) else 0}, indent=2))
    print(f"\nwrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
