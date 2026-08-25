"""The combined loss, ported from `pProp_MLP/src/losses.py`.

    loss = w_cls * cls  +  huber  +  w_pair * pair  +  w_std * std

`huber` is **grounded at weight 1** — it is the only term anchoring the *absolute* pProp
level, so the other three are measured against it and only its `delta` is swept. `pair` and
`std` are "shape" terms added to fight the variance shrinkage that plain MSE/Huber induce.

WHAT CHANGED IN THE PORT, AND WHY
---------------------------------
Three deliberate deviations from the source. Each is a consequence of this repo's data
being drawn from a 10M library rather than a 1.468e9 one, which makes the potent tail 138x
thinner (3,153 molecules at pProp >= 3.5, of which 100 at >= 5.0).

1. **The classifier is binary at pProp >= 3.5, so `cls` is BCE, not 4-class CE** — and
   `weighted_bce_loss` normalises by `sample_weights.sum()`, *not* by N. This is not
   cosmetic. `nn.CrossEntropyLoss(weight=w)` divides by `sum(w_{y_i})` while
   `nn.BCEWithLogitsLoss(reduction="mean")` divides by `N`; at this class ratio (104x)
   those differ by a large constant, and `w_cls` is precisely the knob balancing this term
   against the weight-1 huber. Using the mean would silently move every inherited `w_cls`
   range out of the units it was swept in.

2. **`std_match_loss` takes optional sample weights, and the training loop passes them.**
   Measured on `ampc_subset_331k.csv`: the group-weighted target std is 1.455 against 0.864
   unweighted, a factor of 1.68. An unweighted std term therefore pulls prediction spread
   toward 0.864 while the weighted huber pulls toward a distribution of effective spread
   1.455 — the two terms fight. In pProp_MLP the same tension existed but was mild (its
   weights topped out at ~32x). `sample_weights=None` reproduces the source exactly, which
   is what the parity test uses.

3. **`pairwise_distance_loss` stays unweighted**, exactly as ported. It is shift-invariant
   and O(B^2); weighting it is a larger change with no equivalent argument behind it.

THE WEIGHTING SCHEME
--------------------
Two groups split at `PPROP_EDGE = 3.5`, the same threshold the classifier uses — one line
drives the classifier, the regression sample weights, and the metric weighting. Inverse
group frequency then gives each group equal total weight mass:
`[0.0190, 1.9810]`, a ratio of 104x.

`grouped_frequency_weights` is kept in its general form even though `WEIGHT_GROUPS = [0, 1]`
makes it identical to `inverse_frequency_weights` here. It cost nothing to port, and it is
what the code would need if the binning ever changes back.

**Why 3.5 and not 5.0.** Effective sample size (`effective_sample_size`, below) measured on
this data: edge 3.0 -> 38,711 (11.7% of N); **edge 3.5 -> 12,492 (3.8%)**; edge 4.0 ->
3,972 (1.2%); edge 5.0 -> **400 (0.12%)**. At 5.0 the scheme would train on an effective
400 molecules. 3.5 is the finest edge that survives, and it is independently the threshold
at which binders become possible.
"""

import numpy as np
import torch
import torch.nn.functional as F

# The single pProp threshold. The classifier's positive class, the regression sample
# weights and the metric weighting all derive from this one number -- see the module
# docstring. Changing it changes all three together, which is the point.
PPROP_EDGE = 3.5

GROUP_NAMES = (f"lt{PPROP_EDGE}", f"ge{PPROP_EDGE}")

# Class index -> group id. Identity for a binary task; kept explicit so the grouped
# machinery below reads the same way it does in pProp_MLP, where the map was [0, 1, 2, 0].
WEIGHT_GROUPS = [0, 1]

WEIGHT_KINDS = ("uniform", "balanced")


# -- labels -------------------------------------------------------------------------

def binary_labels(pprop, edge=PPROP_EDGE):
    """`pProp -> {0, 1}`, positive meaning `pprop >= edge`.

    Returns int64 so it indexes weight tables directly; cast to float for BCE at the call
    site rather than here, since the metric code wants integers.
    """
    return (np.asarray(pprop, dtype=np.float64) >= edge).astype(np.int64)


# -- class and sample weights -------------------------------------------------------

def inverse_frequency_weights(y_train, n_classes):
    """
    Per-class loss weights proportional to 1 / class frequency:
    w_c = N / (C * n_c), normalized so the present-class weights average 1
    (keeps the loss scale stable). Empty classes get weight 0.
    """
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = counts.sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def grouped_frequency_weights(y_train, n_classes, groups):
    """
    Inverse-frequency weights computed over *groups* of classes rather than over
    individual classes, then broadcast back to each member class.

    Classes in one group share a single weight `w_g = N / (G * n_g)` (n_g = total members
    of group g, G = number of present groups). Normalized so the present-class weights
    average 1, keeping the loss scale and the inherited sweep ranges stable. Empty classes
    get weight 0.

    With this repo's `WEIGHT_GROUPS = [0, 1]` the grouping is the identity and this reduces
    exactly to `inverse_frequency_weights`. It earns its place anyway: in pProp_MLP the map
    was `[0, 1, 2, 0]`, which existed to stop plain inverse *class* frequency from inverting
    on the rarest class. That class (pProp >= 7.5) cannot occur here -- pProp caps at 7.0 on
    a 10M library -- so the grouping has nothing left to do, but the shape of the argument
    is worth keeping.

    Parameters
    ----------
    y_train   : (N,) int array/tensor of class indices
    n_classes : int, number of classes C
    groups    : length-C sequence mapping class index -> group id (0..G-1)
    """
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    groups = np.asarray(groups)
    n_groups = int(groups.max()) + 1
    group_counts = np.bincount(groups, weights=counts, minlength=n_groups)
    present_groups = group_counts > 0
    gw = np.zeros(n_groups, dtype=np.float64)
    gw[present_groups] = group_counts.sum() / (present_groups.sum() * group_counts[present_groups])
    w = gw[groups]                       # broadcast group weight to member classes
    present = counts > 0
    w[~present] = 0.0                     # a class with no members gets weight 0
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def group_sample_weights(y_class, n_classes, groups):
    """
    Per-sample class-balancing weights based on inverse *group* frequency, the numpy
    analogue of `grouped_frequency_weights` for the reported/objective metrics. Sample i in
    class c gets weight `1 / n_group(group(c))`. Only relative magnitudes matter -- the
    normalization constant cancels in every weighted mean the metrics compute.

    Parameters
    ----------
    y_class   : (N,) int array of class indices
    n_classes : int, number of classes C
    groups    : length-C sequence mapping class index -> group id
    """
    y_class = np.asarray(y_class, dtype=np.int64)
    groups = np.asarray(groups)
    support = np.bincount(y_class, minlength=n_classes).astype(np.float64)
    group_counts = np.bincount(groups, weights=support, minlength=int(groups.max()) + 1)
    return 1.0 / group_counts[groups[y_class]]


def sample_weights_from_classes(y_class, class_weights):
    """Map a per-class weight tensor to a per-sample weight tensor."""
    return class_weights[y_class]


def balanced_sample_weights(pprop, edge=PPROP_EDGE):
    """Per-sample two-group inverse-frequency weights, mean-1 normalised over groups.

    The `balanced` flavour of `--weights`, and the vector that actually trains. Computed
    from the pProp values it is given, so a training fold gets the training fold's group
    sizes -- never the whole dataset's, which would leak the validation fold's composition
    into the loss.
    """
    y = binary_labels(pprop, edge)
    w = grouped_frequency_weights(y, len(GROUP_NAMES), WEIGHT_GROUPS).numpy()
    return w[y].astype(np.float64)


def build_sample_weights(kind, pprop, edge=PPROP_EDGE):
    """Dispatch `--weights` to a per-sample weight vector.

    The two flavours answer different questions, so nothing downstream may call a result
    `weighted_*` without saying which:

    - `uniform`  : all 1. The subset is already ~30x tail-enriched by construction, so this
                   is not neutral -- it estimates *subset* performance.
    - `balanced` : two-group inverse frequency at `edge`. **Up**-weights the tail 104x.
                   This is what trains.

    A third flavour, `ipw`, existed until 2026-08-12: the per-bin subsampling rate from
    `subset.py`, which down-weighted the tail ~30x to estimate performance on the full 10M
    library. It was removed as a modelling knob -- it describes how the dataset was
    assembled, not how the model should be trained or scored -- while remaining a column of
    `ampc_subset_331k.csv` and a row of the split diagnostics.
    """
    if kind not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight kind {kind!r}; choose from {WEIGHT_KINDS}")
    pprop = np.asarray(pprop, dtype=np.float64)
    if kind == "uniform":
        return np.ones(len(pprop), dtype=np.float64)
    return balanced_sample_weights(pprop, edge)


def effective_sample_size(w):
    """`(sum w)^2 / sum w^2` -- how many molecules the weighted gradient behaves like.

    Reported rather than merely reasoned about, because it is the number the choice of
    `PPROP_EDGE` rests on: 3.77% of N at 3.5, versus 0.12% at 5.0 and 18.5% for the scheme
    this was ported from. Logged per run so a weighting change shows up as a number instead
    of as unexplained variance.
    """
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    return float(total * total / (w * w).sum()) if total > 0 else 0.0


# -- loss terms ---------------------------------------------------------------------

def weighted_mse_loss(pred, target_pprop, sample_weights):
    """
    Weighted MSE normalized by the sum of sample weights (mirrors how
    nn.CrossEntropyLoss(weight=w) divides by sum(w_{y_i}) rather than batch count).

    Superseded as the training loss by `weighted_huber_loss`, but kept because it is the
    baseline the loss-parity test compares against: with the other three terms at weight 0
    and `huber_delta` large, the combined loss must reduce to exactly **half** of this.

    The factor of 2 is not slack in the test -- it is `F.huber_loss`'s definition, which is
    `0.5 * r**2` below the transition rather than `r**2`. Measured at `delta=1e6`: the ratio
    is 0.500000 to every digit float32 has. Inherited unchanged from pProp_MLP, which called
    the same function, so the loss scale that `w_cls`/`w_pair`/`w_std` were swept against is
    preserved.
    """
    sq_err = (pred.squeeze(-1) - target_pprop) ** 2
    return (sample_weights * sq_err).sum() / sample_weights.sum()


def weighted_huber_loss(pred, target_pprop, sample_weights, delta=1.0):
    """
    Weighted Huber (smooth-L1) loss, the robust analogue of weighted_mse_loss.

    Quadratic for residuals |r| <= delta, linear beyond, so large tail residuals contribute
    a bounded gradient. Normalized by the sum of sample weights (same convention as
    weighted_mse_loss / nn.CrossEntropyLoss(weight=w)).

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    sample_weights: (N,) float tensor of per-sample weights
    delta         : float, Huber transition point (L2 -> L1)
    """
    hub = F.huber_loss(pred.squeeze(-1), target_pprop, delta=delta, reduction="none")
    return (sample_weights * hub).sum() / sample_weights.sum()


def weighted_bce_loss(logits, target_binary, sample_weights):
    """
    Weighted binary cross-entropy on one logit, normalized by `sum(sample_weights)`.

    The normalization is the whole point of writing this out rather than passing
    `pos_weight` to `nn.BCEWithLogitsLoss`: that reduces with `mean`, i.e. divides by N,
    whereas `nn.CrossEntropyLoss(weight=w)` -- what pProp_MLP's `w_cls` range was swept
    against -- divides by `sum(w_{y_i})`. At a 104x class ratio the two differ by a large
    constant, so the mean form would silently rescale `w_cls` relative to the weight-1
    huber term.

    Written this way the term is exactly `CrossEntropyLoss(weight=class_weights)` on two
    classes, with `logits` standing for `z_1 - z_0`. `check_bce_matches_ce` in the test
    suite asserts that equality rather than leaving it as a claim.

    Parameters
    ----------
    logits        : (N,) or (N,1) float tensor, one logit per molecule
    target_binary : (N,) float tensor of 0.0 / 1.0 labels
    sample_weights: (N,) float tensor of per-sample weights
    """
    bce = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), target_binary, reduction="none"
    )
    return (sample_weights * bce).sum() / sample_weights.sum()


def pairwise_distance_loss(pred, target_pprop):
    """
    Distance-matching loss over all pairs (i, j) in a batch: the predicted pProp *gap*
    between two molecules should equal their true pProp gap. Encourages the model to
    reproduce relative differences (shift-invariant), countering the variance shrinkage
    that plain MSE/Huber induce.

        L = mean_{i,j} | (pred_i - pred_j) - (target_i - target_j) |

    Unweighted, as ported. Vectorized over the full B x B difference matrix (the diagonal
    contributes 0; the pair-count constant is absorbed by the caller's weight), so this is
    O(B^2) in time and memory -- ~1.4M pairs at batch 1200.

    Measured rather than assumed, forward+backward on an A6000: **0.307 ms at batch 1200**
    (0.291 at 256, 0.494 at 2048), i.e. 1.6x the huber term and well under 1% of a training
    step, which `reports/compute_profile.md` puts at 51 ms for batch 256 alone. The O(B^2)
    scaling is real but does not bite until far beyond the batch sizes in play here.
    """
    pred = pred.squeeze(-1)
    d_pred = pred[:, None] - pred[None, :]
    d_true = target_pprop[:, None] - target_pprop[None, :]
    return (d_pred - d_true).abs().mean()


def std_match_loss(pred, target_pprop, sample_weights=None):
    """
    Squared difference between the spread of predictions and of targets over the batch.
    Pushes prediction spread toward target spread, directly countering
    regression-to-the-mean.

        L = (std(pred) - std(target)) ** 2

    `sample_weights=None` gives the unweighted form ported from pProp_MLP. Passing weights
    computes both standard deviations under those weights instead, which is what this repo
    trains with -- see deviation 2 in the module docstring. Feeding the weighted huber and
    an unweighted std term together makes them pull toward spreads that differ by 1.68x
    here, so they would partly cancel.

    Both std's use the population (biased) estimator, weighted or not. The bias is
    identical on both sides of the subtraction and so cancels; using a ddof=1 correction on
    one side only would be the actual error.
    """
    pred = pred.squeeze(-1)
    if sample_weights is None:
        return (pred.std(unbiased=False) - target_pprop.std(unbiased=False)) ** 2

    w = sample_weights
    total = w.sum()

    def wstd(x):
        mean = (w * x).sum() / total
        return torch.sqrt((w * (x - mean) ** 2).sum() / total)

    return (wstd(pred) - wstd(target_pprop)) ** 2


def variance_covariance_loss(embedding, gamma=1.0, *, w_cov=1.0):
    """Keep the exported bottleneck from collapsing onto a handful of directions.

        L = mean_j relu(gamma - std(z_j))  +  w_cov * mean_{i != j} cov(z)_{ij} ** 2

    The variance and covariance halves of VICReg (Bardes, Ponce & LeCun, 2022), without its
    invariance term -- there is no second view here, so the only job is non-degeneracy.

    WHY THIS EXISTS. `head.DualHead`'s bottleneck is the artifact this repo ships, and the
    only gradient reaching it comes from two near-collinear outputs: regression on pProp, and
    classification at pProp >= 3.5, which is the same axis with its gradient concentrated at
    the boundary. So the *supervised* signal is close to rank-1 and nothing asks the remaining
    ~30 dimensions to carry anything. The predicted failure is an embedding of effective rank
    2-4 out of 32 -- a scalar in 32 slots. MEASURED 2026-08-18: 2.78 at the default, and as
    low as 1.32 on another seed. The prediction was right.

    An earlier version of this docstring named weight decay as what shrinks the unused
    dimensions. That is FALSIFIED -- AdamW's decoupled decay shrinks by `lr * wd` per step,
    so the head loses 1e-3 * 0.01 * 4,420 steps = 4.3% over a whole run and the trunk 0.33%,
    and a 2-4% shrink cannot erase 30 dimensions. The likelier shrinker is the LayerNorm
    immediately before the export point: it normalises across the 32 dims *within each row*,
    so one dominant pre-activation divides every other dim down -- which is exactly the
    measured baseline std profile (0.934 down to 0.017). See
    `reports/embedding_collapse_experiment.md` P6.3.

    That is fatal specifically for the downstream use. A deep-kernel-learning GP measures
    molecule similarity by distance in this space; if 30 dimensions are flat, distance
    degenerates into "difference in predicted pProp", so two structurally unrelated molecules
    with equal predicted score become indistinguishable and the posterior variance stops
    tracking genuine ignorance. Active learning is driven entirely by that variance.

    This term adds no target. It forbids dimensions from being flat or from duplicating each
    other, which lets the trunk's own chemical variation occupy them -- exactly the content a
    kernel needs and pProp cannot supply.

    Deliberately unweighted by `sample_weights`: this is a statement about the geometry of the
    batch, not about the target distribution, and the class-balancing weights (104x) would
    make the covariance estimate depend on a handful of tail molecules.

    Cost is O(B * d^2) -- 1200 x 32^2 is ~1.2M multiply-adds, negligible beside the O(B^2)
    pair term already in this loss.

    `gamma` is the target floor on each dimension's standard deviation. With LayerNorm
    immediately before the export point the natural scale is ~1, so 1.0 is the sensible
    default rather than a tuned value. Measured against a healthy bottleneck it is too high
    -- per-dim std is 0.585 post-LayerNorm+GELU, so the hinge fires even at effective rank
    30.78/32 -- which is why every run since 2026-08-18 passes `--vic-gamma 0.5` explicitly.
    Note gamma is a *scale target, not a strength knob*: `relu(gamma - std)` has slope -1 for
    every dim below gamma, so raising it moves the target without adding force.

    `w_cov` reweights the covariance half against the variance half. DEFAULT 1.0 reproduces
    the original term exactly. It is a knob rather than a constant because the two halves
    become binding at different times: measured at collapse they are comparable (ratio
    1.3-2.4) and the variance hinge is NOT saturated -- 28 of 32 dims sit below gamma even at
    `w_vic=0.3` -- so today the fix is more total force, not a re-balance. Once `w_vic` is
    large enough to drive every dim to gamma the hinge saturates, its gradient vanishes, and
    covariance is the only half still pushing. See `reports/embedding_collapse_experiment.md`
    P6.2.
    """
    z = embedding - embedding.mean(dim=0, keepdim=True)
    n = z.shape[0]
    if n < 2:                       # covariance is undefined; nothing to spread out
        return z.sum() * 0.0        # keeps the graph connected, contributes nothing

    std = torch.sqrt(z.var(dim=0, unbiased=True) + 1e-8)
    variance = torch.relu(gamma - std).mean()

    cov = (z.T @ z) / (n - 1)
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
    covariance = off_diagonal.pow(2).sum() / cov.shape[0]

    return variance + w_cov * covariance


def combined_loss(logits, pred, target_binary, target_pprop, sample_weights,
                  w_cls, w_pair, w_std, huber_delta, embedding=None, w_vic=0.0,
                  vic_gamma=1.0, w_cov=1.0):
    """The five-term loss, and the per-term breakdown for logging.

        loss = w_cls * cls + huber + w_pair * pair + w_std * std + w_vic * vic

    Returns `(loss, terms)` where `terms` holds the *unscaled* term values as floats.
    Logging them unscaled is deliberate: it is the only way to see whether a swept weight
    is doing anything, since a term that has collapsed and a term whose weight is tiny look
    identical once multiplied.

    `sample_weights` feeds `cls`, `huber` and `std`; `pair` and `vic` are unweighted by
    design, for different reasons -- see each function.

    `w_vic` DEFAULTS TO 0.0, so the term is inert until switched on. That is the point: the
    collapse it guards against is predicted but not yet measured here, and
    `val/emb_effective_rank` is what decides whether to enable it. Shipping it disabled makes
    that decision a sweep value rather than a code change plus a re-verification pass. It also
    keeps the loss-parity check in `verify_metrics.py` meaningful -- with the other weights at
    0 and a large `huber_delta`, this must still reduce to exactly half of `weighted_mse_loss`.
    """
    cls = weighted_bce_loss(logits, target_binary, sample_weights)
    huber = weighted_huber_loss(pred, target_pprop, sample_weights, delta=huber_delta)
    pair = pairwise_distance_loss(pred, target_pprop)
    std = std_match_loss(pred, target_pprop, sample_weights)

    loss = w_cls * cls + huber + w_pair * pair + w_std * std
    terms = {"cls": float(cls), "huber": float(huber),
             "pair": float(pair), "std": float(std)}

    # Skipped entirely when there is no embedding to regularise, so callers that only have
    # arrays (`train.loss_terms_from_arrays` on a split without saved embeddings) still work.
    if embedding is not None:
        vic = variance_covariance_loss(embedding, gamma=vic_gamma, w_cov=w_cov)
        loss = loss + w_vic * vic
        terms["vic"] = float(vic)

    return loss, terms
