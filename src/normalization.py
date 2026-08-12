"""pProp target normalization for the regression head.

Ported verbatim from `pProp_MLP/src/normalization.py` — the arithmetic is unchanged, only
the docstrings are re-pointed at this repo. Keeping it byte-comparable matters: the swept
loss hyperparameters inherited from that project (`huber_delta`, `w_pair`, `w_std`) were
tuned against a **z-scored** target, so they are in z-units and only transfer if the
transform does.

The regression head can be trained to predict a *normalized* pProp rather than raw pProp,
selected by `--pprop-norm`. All statistics come from the **training fold only**, so the
validation fold never leaks into the transform, and they are recorded in `meta.json` so
predictions can be mapped back to raw pProp before metrics.

**Denormalize before metrics.** Pearson is scale-invariant, but MSE and MAE are not, and
`train.py` declares `val/mse` with `summary="min"`. Reporting a normalized MSE would make
that summary incomparable across `--pprop-norm` settings — silently, since every number
still looks plausible. `evaluate()` therefore denormalizes predictions first, and every
metric is on the raw pProp scale.

Strategies
----------
- "none"   : identity (raw pProp).
- "zscore" : (pprop - mean) / std
- "minmax" : (pprop - min) / max   (since min(pprop) ~= 0 on this data this coincides with
             the standard (x-min)/(max-min) and lands in [0, 1]).

All three functions are plain arithmetic on python-float scalars, so they apply
transparently to both numpy arrays and torch tensors (CPU or GPU) without any conversion —
the training loop passes tensors, eval/metrics pass numpy.
"""

NORM_STRATEGIES = ("none", "zscore", "minmax")


def compute_norm_stats(pprop_train, strategy):
    """Normalization statistics from the training fold's pProp values.

    Parameters
    ----------
    pprop_train : array-like of training-fold continuous pProp values.
    strategy    : one of NORM_STRATEGIES.

    Returns a dict {"strategy", "mean", "std", "min", "max"} (floats). The full set of
    stats is always recorded (even for "none") so `meta.json` captures the training
    distribution regardless of the chosen strategy.
    """
    import numpy as np

    if strategy not in NORM_STRATEGIES:
        raise ValueError(f"unknown pprop_norm strategy {strategy!r}; "
                         f"expected one of {NORM_STRATEGIES}")
    p = np.asarray(pprop_train, dtype=np.float64)
    std = float(p.std())  # population std (ddof=0)
    mx = float(p.max())
    if strategy == "zscore" and std == 0:
        raise ValueError("zscore normalization needs std(pprop_train) > 0")
    if strategy == "minmax" and mx == 0:
        raise ValueError("minmax normalization needs max(pprop_train) > 0")
    return {
        "strategy": strategy,
        "mean": float(p.mean()),
        "std": std,
        "min": float(p.min()),
        "max": mx,
    }


def normalize_pprop(pprop, stats):
    """Map raw pProp -> normalized target using training-fold `stats`."""
    s = stats["strategy"]
    if s == "none":
        return pprop
    if s == "zscore":
        return (pprop - stats["mean"]) / stats["std"]
    if s == "minmax":
        return (pprop - stats["min"]) / stats["max"]
    raise ValueError(f"unknown pprop_norm strategy {s!r}")


def denormalize_pprop(norm, stats):
    """Inverse of `normalize_pprop`: normalized prediction -> raw pProp."""
    s = stats["strategy"]
    if s == "none":
        return norm
    if s == "zscore":
        return norm * stats["std"] + stats["mean"]
    if s == "minmax":
        return norm * stats["max"] + stats["min"]
    raise ValueError(f"unknown pprop_norm strategy {s!r}")
