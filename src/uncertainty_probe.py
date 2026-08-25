"""Does the exported geometry buy usable *uncertainty*, not just usable predictions?

    .venv/bin/python src/uncertainty_probe.py -o reports/uncertainty_probe_v2.csv

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

-------------------------------------------------------------------------------------------
D1 -- THE 2026-08-24 REPAIR (`reports/featurizer_design.md` S4, D1)

The first version of this probe reported the TOTAL predictive std, and S3(a) of the design
document showed that quantity is **91-99% a constant noise floor** -- 0.8% epistemic at
`A_base` rising to 8.9% at `D_w3`. V1, V2 and V3 were therefore all computed on a number that
barely varies across the pool, so their failures were properties of this probe's configuration
rather than evidence about the embedding. Three changes, in the design document's order:

1. REPORT THE LATENT STD. `*_lat` columns are the predictive std with the fitted white-noise
   term excluded, and `epistemic_share` is `mean(var_latent) / mean(var_total)`. The totals are
   kept beside them so the two readings stay comparable and the old CSV stays interpretable.
   **This propagates into `acquisition_sim`**: UCB scored `mu + 2*sd` on a std inflated by a
   constant floor, which is exactly why V3 could not steer away from greedy. Fixing the
   reported columns and leaving the scorer would have shipped a probe whose V3 was still broken.

   Free regression check: `maxvar` ranks on `sd` alone, and removing a constant from inside a
   square root is monotone, so **`hits_maxvar` must come back bit-identical to the pre-repair
   CSV while `hits_ucb` moves.** A shift in `maxvar` means the change touched something else.

2. THE `WhiteKernel` IS DELIBERATELY NOT CAPPED, and this is the counter-intuitive half. The
   fitted noise std (0.415-0.426) is essentially the model's own residual std on this fold
   (0.430, from `val/mse` = 0.1852), so a cap at the irreducible error would change nothing and
   a cap below it would force the GP to misreport. The noise floor is a true statement about
   the surrogate at n_fit = 1000, not a fitting artefact. What was wrong was reporting the
   total. The bounds below are unchanged from the pre-repair version on purpose.

3. THE ARMS. `--arms` selects which model consumes the embedding:
     `plain`  isotropic Matern + white. The honest round-1 model; kept as the reference.
     `ard`    one length-scale per dimension. A DIAGNOSTIC, not a request to the DKL side:
              does the embedding contain relevant information a reweighting model can find?
              Its caveat has to be read next to its number -- fitting 32 length-scales by
              marginal likelihood at n ~ 100 is a documented overfitting failure and can score
              WORSE than isotropic in exactly the low-label regime the deliverable is for,
              which is why `--n-fit-list` runs it at both regimes and `lml` is reported.
     `mlp`    a small supervised MLP's last hidden layer, then a GP on those features.

   `mlp` IS NOT DKL AND MUST NOT BE CALLED THAT. gpytorch is not installed (verified
   2026-08-24) and the link is ~0.5 MB/s, so nothing here can fit a feature extractor by
   marginal likelihood; this is a TWO-STAGE supervised-features GP. It still tests what S3(b)
   needs tested -- a supervised net can reweight and discard dimensions, and the isotropic
   arms cannot -- but the joint training that defines DKL is absent.

WHY A NEW DEFAULT OUTPUT PATH. `reports/uncertainty_probe.csv` is the sole source of S3(a),
the project's strongest single evidence claim. It is not overwritten.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Kernel, Matern, Sum,
                                              WhiteKernel)
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utility import load_reference                      # noqa: E402

POSITIVE_EDGE = 3.5

# The global pProp sd. Used ONLY to reproduce `featurizer_design.md` S3(a)'s stated
# approximation beside the exact figure -- that table stood in the global sd for each fit
# set's own, because `normalize_y` rescales internally and the per-fit value was not in the
# CSV. Emitting both makes S3(a) a checkable prediction rather than background.
GLOBAL_PPROP_SD = 0.8639


def _white_count(k):
    """How many `WhiteKernel`s appear anywhere in a kernel tree."""
    if isinstance(k, WhiteKernel):
        return 1
    return sum(_white_count(v) for v in vars(k).values() if isinstance(v, Kernel))


def split_noise(kernel):
    """Split a fitted kernel into (latent kernel, noise_level), locating white BY TYPE.

    Positional access -- `kernel.k1` -- would work for the one kernel this file used to
    build and silently return the WhiteKernel itself if any arm ever ordered its sum the
    other way round. Every latent variance would then be garbage that still looked
    plausible. Three arms now build their own kernels, so the lookup is by type and every
    unexpected shape raises.
    """
    n_white = _white_count(kernel)
    if n_white != 1:
        raise ValueError(f"expected exactly one WhiteKernel, found {n_white}: {kernel}")
    if not isinstance(kernel, Sum):
        raise ValueError(f"the WhiteKernel must be a top-level summand: {kernel}")
    if _white_count(kernel.k1) == 1 and _white_count(kernel.k2) == 0:
        white, latent = kernel.k1, kernel.k2
    elif _white_count(kernel.k2) == 1 and _white_count(kernel.k1) == 0:
        white, latent = kernel.k2, kernel.k1
    else:
        raise ValueError(f"WhiteKernel is nested inside a summand, not added to it: {kernel}")
    if not isinstance(white, WhiteKernel):
        raise ValueError(f"the noise summand is {type(white).__name__}, not WhiteKernel")
    return latent, float(white.noise_level)


def latent_std(gp, x_test, y_fit, std_total):
    """Predictive std with the fitted noise term excluded, computed two ways and checked.

    sklearn puts the WhiteKernel inside `kernel_`, so `kernel_.diag(X)` carries `noise_level`
    and `predict(return_std=True)` returns sqrt(latent + noise). `WhiteKernel(X, Y)` is
    exactly zero for Y is not None, so the cross-covariance block is identical between the
    full and latent kernels and the two routes must agree to float precision:

        route A   var_total - noise_level * y_std**2          (the identity)
        route B   latent_k.diag(X) - ||L^-1 k_*||^2, rescaled  (the decomposition)

    Route B is what is returned, because route A inherits sklearn's clip-at-zero. Asserting
    they agree is what catches a wrong decomposition -- the failure `split_noise` exists to
    prevent is one that produces plausible-looking numbers.

    `y_std` is recomputed here rather than read from `gp._y_train_std`: same value (numpy's
    ddof=0, which is what `normalize_y=True` uses), no private attribute.
    """
    latent_k, noise = split_noise(gp.kernel_)
    y_std = float(np.std(y_fit)) if gp.normalize_y else 1.0

    k_trans = latent_k(x_test, gp.X_train_)
    v = solve_triangular(gp.L_, k_trans.T, lower=True)
    var_b = (latent_k.diag(x_test) - np.einsum("ij,ji->i", v.T, v)) * y_std ** 2

    var_a = std_total ** 2 - noise * y_std ** 2
    live = std_total ** 2 > 0            # sklearn clips negative total variance to zero
    if live.any():
        gap = float(np.abs(var_a[live] - var_b[live]).max())
        scale = float(np.abs(var_b[live]).max()) + 1e-12
        if gap > 1e-6 * scale + 1e-10:
            raise AssertionError(f"latent-variance routes disagree by {gap:.3e} (scale {scale:.3e})")
    return np.sqrt(np.maximum(var_b, 0.0)), noise, y_std


def make_kernel(n_dim, ard=False):
    """Matern 5/2 with the length-scale fitted, not assumed.

    A fixed length-scale would silently favour whichever embedding happened to match it, and
    the cells differ in operating scale by 9x (`emb_trace` 4.0 at the control, 34.9 at
    gamma=1.0) -- which is exactly the axis S2 found to matter.
    """
    ls = np.full(n_dim, np.sqrt(n_dim)) if ard else np.sqrt(n_dim)
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e4), nu=2.5)
            + WhiteKernel(0.1, (1e-6, 1e1)))


def fit_gp(x, y, seed, ard=False):
    gp = GaussianProcessRegressor(kernel=make_kernel(x.shape[1], ard), normalize_y=True,
                                  n_restarts_optimizer=0, random_state=seed)
    gp.fit(x, y)
    return gp


def mlp_features(x_fit, y_fit, x_test, seed, hidden=(64, 16)):
    """Last-hidden-layer activations of a small supervised MLP, for both halves.

    Two-stage, NOT joint: the net is fitted to the same labels the GP then sees, which is the
    leak DKL also has, but the marginal likelihood never reaches the net's weights. See the
    module docstring on why this is the closest instrument available here.
    """
    ys = (y_fit - y_fit.mean()) / (y_fit.std() + 1e-12)
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3,
                       max_iter=2000, random_state=seed)
    net.fit(x_fit, ys)

    def forward(x):
        h = np.asarray(x, dtype=np.float64)
        for w, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
            h = np.maximum(h @ w + b, 0.0)
        return h

    return forward(x_fit), forward(x_test), net


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


def acquisition_sim(x_fit, y_fit, x_pool, y_pool, kernel, seed, n0=200, rounds=8, batch=50,
                    use_latent=True):
    """Batched active learning over the held-out pool, one curve per strategy.

    The kernel is FIXED to the one already fitted on the full training half rather than
    re-optimised each round: re-fitting hyperparameters on 200 points would make the early
    rounds a test of hyperparameter estimation rather than of the embedding.

    `use_latent` is the D1 repair. Scoring UCB on the total std adds a constant floor to every
    candidate, which shrinks the exploration bonus relative to the mean and is why V3 read the
    same as greedy in every cell. `maxvar` is invariant to it -- see the module docstring.
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
                if use_latent:
                    sd, _, _ = latent_std(gp, x_pool[avail], ty, sd)
                sc = {"greedy": mu, "ucb": mu + 2.0 * sd, "maxvar": sd}[strategy]
                pick = avail[np.argsort(-sc)[:batch]]
            taken[pick] = True
            tx = np.concatenate([tx, x_pool[pick]])
            ty = np.concatenate([ty, y_pool[pick]])
            found.append(int((y_pool[taken] >= POSITIVE_EDGE).sum()))
        hits[strategy] = found
    return hits


def score_arm(arm, zf, yf, zt, yt, bf, bt, sub, pred_f, pred_t, seed):
    """One (cell, direction, arm, n_fit) row's worth of measurements."""
    xf, xt = zf[sub], zt
    net_iters = np.nan
    if arm == "mlp":
        xf, xt, net = mlp_features(zf[sub], yf[sub], zt, seed)
        net_iters = int(net.n_iter_)
    gp = fit_gp(xf, yf[sub], seed, ard=(arm == "ard"))
    mean, std = gp.predict(xt, return_std=True)
    std_lat, noise, y_std = latent_std(gp, xt, yf[sub], std)

    err = np.abs(yt - mean)
    nov = tanimoto_novelty(bt, bf[sub])
    predext = np.abs(pred_t - np.median(pred_f[sub]))
    embnn = nearest_distance(xt, xf)
    var_tot, var_lat = float(np.mean(std ** 2)), float(np.mean(std_lat ** 2))
    return gp, {
        "arm": arm, "n_dim": int(xf.shape[1]),
        "calib_rho": rho(std, err), "calib_rho_lat": rho(std_lat, err),
        "novelty_rho": rho(std, nov), "novelty_rho_lat": rho(std_lat, nov),
        "predext_rho": rho(std, predext), "predext_rho_lat": rho(std_lat, predext),
        "embnn_rho": rho(std, embnn), "embnn_rho_lat": rho(std_lat, embnn),
        "coverage95": float((err <= 1.96 * std).mean()),
        "coverage95_lat": float((err <= 1.96 * std_lat).mean()),
        "mean_std": float(std.mean()), "mean_std_lat": float(std_lat.mean()),
        "std_lat_cv": float(std_lat.std() / (std_lat.mean() + 1e-12)),
        "noise_level": noise, "y_fit_std": y_std,
        "noise_std_yunits": float(np.sqrt(noise) * y_std),
        "epistemic_share": var_lat / var_tot if var_tot > 0 else np.nan,
        # S3(a) computed this with the GLOBAL pProp sd standing in for the fit set's own.
        # Emitted beside the exact value so the prediction is checkable, not reconstructed.
        "epistemic_share_s3a": (1.0 - noise * GLOBAL_PPROP_SD ** 2 / var_tot
                                if var_tot > 0 else np.nan),
        "gp_spearman": rho(mean, yt), "gp_mae": float(err.mean()),
        "lml": float(gp.log_marginal_likelihood_value_), "mlp_iters": net_iters,
        "kernel": str(gp.kernel_)}


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
    p.add_argument("--n-fit-list", nargs="+", type=int, default=[100, 1000],
                   help="label counts to fit at; 100 is D2(b)'s regime, 1000 the pre-repair one")
    p.add_argument("--arms", nargs="+", default=["plain", "ard", "mlp"],
                   choices=["plain", "ard", "mlp"])
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--scaling", choices=["raw", "zscore"], default="raw")
    p.add_argument("--acquisition-arm", default="plain",
                   help="arm to run V3 on; V3's own redesign is D2(b), not D1")
    p.add_argument("--acquisition-n-fit", type=int, default=1000,
                   help="n_fit at which V3 runs, matched to the pre-repair CSV for regression")
    p.add_argument("--ucb-std", choices=["latent", "total"], default="latent",
                   help="'total' reproduces the pre-repair V3")
    p.add_argument("--no-acquisition", action="store_true", help="skip V3")
    p.add_argument("-o", "--out", type=Path, default=Path("reports/uncertainty_probe_v2.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _, halves, y = load_reference(args.splits, args.csv_path, args.fold,
                                  args.n_per_half, args.seed)
    fp = np.load(args.splits / "fingerprints.npy")
    bits = [np.unpackbits(fp[h], axis=1).astype(np.float32) for h in halves]
    print(f"fold {args.fold}: halves of {len(halves[0])} / {len(halves[1])} by cluster; "
          f"n_fit={args.n_fit_list}, arms={args.arms}, scaling={args.scaling}")

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

            for n_fit in args.n_fit_list:
                rng = np.random.default_rng(args.seed + direction)
                sub = rng.choice(len(zf), min(n_fit, len(zf)), replace=False)
                for arm in args.arms:
                    gp, r = score_arm(arm, zf, yf, zt, yt, bf, bt, sub,
                                      pred_all[pf], pred_all[pt], args.seed)
                    r.update({"cell": d.parent.name, "seed": int(cfg["seed"]),
                              "direction": direction, "run": str(d), "w_vic": cfg.get("w_vic"),
                              "scaling": args.scaling, "n_fit": int(len(sub))})
                    do_v3 = (not args.no_acquisition and arm == args.acquisition_arm
                             and len(sub) == args.acquisition_n_fit)
                    if do_v3:
                        h = acquisition_sim(zf, yf, zt, yt, gp.kernel_, args.seed + direction,
                                            use_latent=(args.ucb_std == "latent"))
                        r.update({f"hits_{k}": v[-1] for k, v in h.items()})
                        r["n_pool_positives"] = int((yt >= POSITIVE_EDGE).sum())
                        r["ucb_std"] = args.ucb_std
                    rows.append(r)
                    print(f"{r['cell']:>12} seed{r['seed']} dir{direction} n{r['n_fit']:>4} "
                          f"{arm:>5}  calib={r['calib_rho_lat']:+.3f} "
                          f"nov={r['novelty_rho_lat']:+.3f} epi={r['epistemic_share']:.3f} "
                          f"lml={r['lml']:+.1f}"
                          + (f"  ucb={r.get('hits_ucb')} rand={r.get('hits_random')} "
                             f"greedy={r.get('hits_greedy')} maxvar={r.get('hits_maxvar')}"
                             f"/{r.get('n_pool_positives')}" if do_v3 else ""))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"script": "src/uncertainty_probe.py", "argv": sys.argv[1:], "fold": args.fold,
         "n_per_half": int(len(halves[0])), "n_fit_list": args.n_fit_list, "arms": args.arms,
         "scaling": args.scaling, "seed": args.seed, "ucb_std": args.ucb_std,
         "gpytorch_available": False,
         "n_runs": int(df["run"].nunique()) if len(df) else 0}, indent=2))
    print(f"\nwrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
