"""Validation metrics: binary AP, regression correlation/error, and tail enrichment.

Assembled from three places in `pProp_MLP`, because its ranking metrics and its live
metrics were never in the same file:

- `src/metrics.py` on `main`  -> `_weighted_pearson`, the MAE/Pearson pair, `mae_skill`.
- `src/metrics.py` on the dead `regression` branch -> `enrichment_factor`, the low/high
  group-split correlations and errors. These are the only ranking metrics either project
  ever had, and they are what a screening model actually needs.
- The objective's structure -> `src/objective.py` next door, which consumes this module.

EVERY METRIC IS COMPUTED UNDER BOTH WEIGHTINGS, AND SAYS WHICH
--------------------------------------------------------------
`uniform` and `balanced` (see `losses.build_sample_weights`) answer different questions:
`balanced` **up**-weights the potent tail 104x, `uniform` leaves the subset's own ~30x
enrichment in place. A key named `weighted_mae` would be ambiguous in the one place
ambiguity is most expensive, so keys are always suffixed -- `mae_uniform`, `mae_balanced` --
and there is no unsuffixed form to fall back on.

Rough guide to reading them:

- `*_uniform`  estimates performance on this 331k subset, which is ~30x tail-enriched by
               construction -- so it flatters the model rather than being neutral. Read it
               as a subset number, never as a library number.
- `*_balanced` is tail-emphasising, and matches what the loss optimises.

`ap_balanced` is the one worth flagging: forcing a 50/50 base rate makes average precision a
statement about the weighting rather than about the model. It is computed for symmetry and
should not be read as a performance number, which is why `objective.py` builds `AP*` from
`ap_uniform` alone.

A third flavour, `ipw`, was removed on 2026-08-12. It weighted each row by the number of
library molecules it stood for (`subset.py`'s per-bin subsampling rate), which made
`*_ipw` an estimate of performance on the full 10M library. That is a property of how the
dataset was assembled rather than a modelling choice, so it no longer weights any loss or
any reported metric; the column survives in `ampc_subset_331k.csv` and in the split
diagnostics as the record of the sampling. Worth knowing if you are reading older notes:
`ap_ipw` and `ap_uniform` measured *close* (0.4861 vs 0.4968 on synthetic predictions)
despite a 30x base-rate difference, because `--take-all-above 2.5` gave every near-threshold
molecule `ipw = 1.0` and those are exactly the false positives that dominate AP.

THE MAE SKILL SCORE
-------------------
MAE cannot be added to AP -- different units -- so it enters the objective as a skill score
against the constant-median predictor, `skill = 1 - MAE / MAE_ref`. 1 is perfect, 0 is no
better than a constant, negative is worse.

The baseline predictor is the same in every flavour: the constant **unweighted** median of
the targets. Only the *scoring* weights change, so `MAE_ref` for flavour f is the f-weighted
mean of `|y - median(y)|`. This is exactly pProp_MLP's construction and it is ported
unchanged -- the earlier reading of it here (a weighted-median baseline) was wrong.

That construction is what makes the flavours averageable, and the reason is worth stating
because it looks like a problem and is not. The weighted `MAE_ref` is larger than the
unweighted one -- measured, 1.660 vs 0.725 on this data, a factor of 2.29 -- so the weighted
baseline looks easier to beat. But the model's weighted MAE inflates alongside it, for the
same reason: both are averaging the same harder tail more heavily. Numerator and denominator
move together and the ratio largely survives. A model at MAE 0.45 unweighted / 0.90 balanced
scores 0.379 and 0.458 -- close enough to average honestly.

The same asymmetry exists in pProp_MLP (1.56x there, under its three-group scheme), so this
is inherent to the design rather than something two-group weighting introduces. Two groups
make it somewhat larger, not different in kind.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score

from losses import PPROP_EDGE, binary_labels

# Thresholds for the enrichment diagnostics. 3.5 is the classifier's own boundary; 4.0 and
# 5.0 look further into the tail than the binary task can. Per fold, 5.0 covers only 20
# molecules -- which is why these are diagnostics and `src/pool_oof.py` recomputes them
# across all five folds at once, where the count is 100.
EF_THRESHOLDS = (3.5, 4.0, 5.0)
EF_FRACTIONS = (0.001, 0.01)


# -- correlation primitives ---------------------------------------------------------

def _weighted_pearson(x, y, w):
    """Weighted Pearson correlation between x and y with per-point weights w.

    Returns np.nan if there are fewer than 2 points or either input is constant
    (correlation undefined). Invariant to the scale of w (weights are renormalized to sum 1
    internally).

    The constant check uses peak-to-peak rather than the weighted variance, and that choice
    is load-bearing: a constant input's variance is floating-point dust, not exactly zero,
    so it slips past a `denom == 0` guard and yields a spurious ~0.0 correlation instead of
    a nan. Constant predictions are a live possibility here -- 36% of rows sit below pProp
    1.0, so predicting the mean is a genuine local optimum in early epochs.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    w = w / wsum
    mx = np.sum(w * x)
    my = np.sum(w * y)
    dx = x - mx
    dy = y - my
    cov = np.sum(w * dx * dy)
    denom = np.sqrt(np.sum(w * dx * dx) * np.sum(w * dy * dy))
    if denom == 0:
        return np.nan
    r = cov / denom
    return float(r) if np.isfinite(r) else np.nan


def _weighted_spearman(x, y, w):
    """Weighted Spearman: the weighted Pearson of the rank-transformed values.

    `rankdata` uses the average method for ties, so the many exactly-equal pProp values in
    the bulk do not get an arbitrary ordering imposed on them.
    """
    return _weighted_pearson(rankdata(x), rankdata(y), w)


# -- baselines ----------------------------------------------------------------------

def median_baseline_abs(y_true_pprop):
    """`|y - median(y)|` -- the constant-median predictor's per-molecule error.

    Returned as the whole array, not a scalar, because each flavour averages it under its
    own weights to get its own `MAE_ref`. The *predictor* is the same constant everywhere;
    only the scoring changes. It depends only on the targets, so it is a fixed per-fold
    scale that nothing the model does can move.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    if y.size == 0:
        return y
    return np.abs(y - np.median(y))


# -- per-flavour metric block -------------------------------------------------------

def regression_block(y_true_pprop, pred_pprop, w, base_abs):
    """Correlation and error for one weighting, on the raw pProp scale.

    `base_abs` is the constant-median predictor's per-molecule error, computed once by
    `median_baseline_abs` and passed in so every flavour scores the *same baseline model*
    under its own weights -- which is what puts the resulting skill scores on a common
    footing.

    Returns unsuffixed keys -- `flavoured_metrics` adds the suffix, so this function never
    has to know which flavour it is computing.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    p = np.asarray(pred_pprop, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(y)
    if n == 0 or w.sum() <= 0:
        return {k: np.nan for k in
                ("pearson", "spearman", "mae", "mae_ref", "mae_skill", "r2")}

    mae = float(mean_absolute_error(y, p, sample_weight=w))
    ref = float((w * base_abs).sum() / w.sum())

    return {
        "pearson": _weighted_pearson(y, p, w),
        "spearman": _weighted_spearman(y, p, w),
        "mae": mae,
        "mae_ref": ref,
        "mae_skill": 1.0 - mae / ref if ref > 0 else np.nan,
        # r2 is undefined for <2 points or a constant target; guard rather than let
        # sklearn return a plausible-looking 0.0.
        "r2": (float(r2_score(y, p, sample_weight=w))
               if n >= 2 and np.ptp(y) != 0 else np.nan),
    }


def classification_block(y_binary, scores, w):
    """Average precision and base rate for one weighting.

    `scores` is the ranking signal -- the classifier logit, or the predicted pProp; both
    are monotone in "more potent" so either ranks the same way. AP needs both classes
    present, which is why the guard is on the label set rather than on n.
    """
    y_binary = np.asarray(y_binary, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pos = int(y_binary.sum())
    if len(y_binary) == 0 or pos == 0 or pos == len(y_binary) or w.sum() <= 0:
        return {"ap": np.nan, "base_rate": np.nan}
    return {
        "ap": float(average_precision_score(y_binary, scores, sample_weight=w)),
        "base_rate": float((w * y_binary).sum() / w.sum()),
    }


def flavoured_metrics(y_true_pprop, pred_pprop, scores, weight_vectors,
                      edge=PPROP_EDGE):
    """Every metric under every weighting, as one flat suffixed dict.

    Parameters
    ----------
    y_true_pprop   : (N,) true pProp, raw scale (denormalize before calling)
    pred_pprop     : (N,) predicted pProp, raw scale
    scores         : (N,) ranking signal for AP -- the classifier logit
    weight_vectors : {flavour_name: (N,) weights}, e.g. from `losses.build_sample_weights`
    edge           : pProp threshold defining the positive class

    Returns keys like `mae_uniform`, `pearson_balanced`, `ap_uniform`, plus each flavour's
    `mae_ref_*` and the positive count, so the numbers can be sanity-checked from the log
    alone without re-deriving the baseline.
    """
    y_binary = binary_labels(y_true_pprop, edge)
    base_abs = median_baseline_abs(y_true_pprop)

    out = {"n": int(len(y_true_pprop)), "n_positive": int(y_binary.sum())}
    for flavour, w in weight_vectors.items():
        for key, value in regression_block(y_true_pprop, pred_pprop, w, base_abs).items():
            out[f"{key}_{flavour}"] = value
        for key, value in classification_block(y_binary, scores, w).items():
            out[f"{key}_{flavour}"] = value
    return out


# -- tail diagnostics ---------------------------------------------------------------

def group_split_metrics(y_true_pprop, pred_pprop, group_edge=PPROP_EDGE):
    """Correlation and error computed *separately* within the low and high groups.

    Ported from pProp_MLP's `regression` branch, where the edge was 5.0; the default here
    is `PPROP_EDGE` so it lines up with the training partition, and `src/pool_oof.py` calls
    it again at 5.0 where there are enough molecules to mean anything.

    These matter because the whole-set numbers can look healthy while the model is only
    ordering the bulk. They also avoid the range-restriction artifact that makes narrow
    per-bin correlations unreliable -- each group spans a wide pProp range.

    `*_group_ge` on a single validation fold rests on ~630 molecules at edge 3.5 and only
    20 at edge 5.0. At 3.5 it is a real measurement; at 5.0 it is not, which is what
    pooling out-of-fold is for.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    p = np.asarray(pred_pprop, dtype=np.float64)
    ge = y >= group_edge
    out = {"group_edge": float(group_edge),
           "n_group_lt": int((~ge).sum()), "n_group_ge": int(ge.sum())}
    for tag, mask in (("lt", ~ge), ("ge", ge)):
        sub_y, sub_p = y[mask], p[mask]
        ones = np.ones(len(sub_y), dtype=np.float64)
        out[f"pearson_group_{tag}"] = _weighted_pearson(sub_y, sub_p, ones)
        out[f"spearman_group_{tag}"] = _weighted_spearman(sub_y, sub_p, ones)
        out[f"mae_group_{tag}"] = float(np.mean(np.abs(sub_p - sub_y))) if sub_y.size else np.nan
    return out


def embedding_metrics(embedding):
    """Is the exported 32-d bottleneck actually 32-dimensional, or a scalar in 32 slots?

    This is the only metric here that scores the *deliverable* rather than the predictions.
    `head.DualHead`'s bottleneck is what gets deployed, frozen, as the input featurization for
    a deep-kernel-learning GP; `goal_metric` says nothing about whether that vector is a
    usable one.

        emb_effective_rank  exp(entropy of the normalized covariance eigenvalues)
        emb_top1_share      fraction of total variance on the single largest eigenvalue
        emb_min_std         smallest per-dimension standard deviation

    HOW TO READ `emb_effective_rank`. Eigenvalues of the d x d covariance measure how far the
    molecules spread along each direction. Normalize them to sum to 1 and take exp(entropy):
    one dominant eigenvalue gives ~1, all-equal gives d. So it answers "how many dimensions
    are doing work" on the same scale as d itself, which plain matrix rank cannot -- a
    dimension carrying pure numerical noise counts as full rank but contributes nothing.

    For a 32-d export, roughly 15-25 is healthy. Around 2-4 is the predicted failure mode and
    means the GP is receiving a restatement of predicted pProp: see
    `losses.variance_covariance_loss`, which is the countermeasure and is disabled by default
    until this number says otherwise. `emb_top1_share` near 1.0 is the same finding stated
    more bluntly, and `emb_min_std` near 0 names a dimension that is simply dead.

    Unsuffixed by design. The `_uniform`/`_balanced` flavour convention does not apply --
    these describe the geometry of a representation, not performance under a reweighting of
    the molecules, and there is no sense in which a covariance eigenspectrum is "balanced".
    `group_split_metrics` and `enrichment_factor` set the precedent for self-describing keys.

    Returns plain floats: `train.log_prefixed` silently drops anything else.
    """
    z = np.asarray(embedding, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] == 0:
        return {"emb_effective_rank": np.nan, "emb_top1_share": np.nan,
                "emb_min_std": np.nan, "emb_dim": int(z.shape[1]) if z.ndim == 2 else 0}

    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    # eigvalsh, not eigvals: the covariance is symmetric, so this is both faster and free of
    # the tiny imaginary parts the general solver returns and which would poison the entropy.
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)     # symmetric solver can emit -1e-18
    total = eigenvalues.sum()

    if total <= 0:                                    # every dimension constant
        return {"emb_effective_rank": 1.0, "emb_top1_share": np.nan,
                "emb_min_std": 0.0, "emb_dim": int(z.shape[1])}

    p = eigenvalues / total
    nonzero = p[p > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return {"emb_effective_rank": float(np.exp(entropy)),
            "emb_top1_share": float(eigenvalues.max() / total),
            "emb_min_std": float(np.sqrt(np.diag(cov)).min()),
            "emb_dim": int(z.shape[1])}


def enrichment_factor(y_true_pprop, scores, thresholds=EF_THRESHOLDS,
                      fractions=EF_FRACTIONS):
    """
    Enrichment Factor at top fractions, for "active = true pProp >= threshold".

    Ranks molecules by `scores` descending (higher = predicted more potent), takes the top
    `fraction` of the list, and computes

        EF = (actives in top / size of top) / (total actives / N).

    EF = 1 is random selection; higher means the model concentrates actives near the top.
    The maximum possible EF at a given threshold is `1 / active_rate`, so a low threshold
    (where many molecules are active) has a low ceiling -- EF values are only comparable
    within a threshold, never across.

    Unweighted by construction. EF is a statement about a ranked list you would actually
    screen, and that list is the subset in front of you; reweighting it would describe a
    list nobody produces.

    Returns a flat dict keyed `ef_p{threshold}_top{fraction}` so it logs without further
    munging. np.nan when a threshold has no actives.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.argsort(-scores, kind="stable")   # descending; stable so ties are reproducible
    out = {}
    for thr in thresholds:
        active = y >= thr
        n_active = int(active.sum())
        ranked = active[order]
        base_rate = n_active / n if n else 0.0
        for frac in fractions:
            k = max(1, int(round(frac * n)))
            key = f"ef_p{thr}_top{frac}"
            out[key] = (float((int(ranked[:k].sum()) / k) / base_rate)
                        if n_active and n else np.nan)
    return out
