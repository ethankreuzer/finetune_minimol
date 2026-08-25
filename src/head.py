"""Regression heads that sit on top of MiniMol's 512-d pooled embedding.

The trunk (`src/trunk.py`) is fixed by the pretrained checkpoint; everything after the
512-d embedding is ours to choose, and the current choice — 512 -> 1200 -> 32 -> 1 — is a
prototype, not a final architecture. So this module is built to be swapped rather than
edited: `MLPHead` takes its shape as a `hidden_dims` sequence, and `build_head` is a
registry so a genuinely different head (attention pooling, gated, ensemble) is a new class
plus one line, not a rewrite of `src/model.py`.

    from head import MLPHead, DualHead, build_head

    head = MLPHead()                                  # 512 -> 1200 -> 32 -> 1
    head = MLPHead(hidden_dims=(256,), dropout=0.1)   # 512 -> 256 -> 1
    head = MLPHead(hidden_dims=())                    # 512 -> 1, i.e. a linear probe
    head = build_head("mlp", hidden_dims=(1200, 32))  # same, by name

    head = DualHead()                                 # shared trunk -> (logit, pProp)
    head = build_head("dual", n_layers=4, hidden_dim=1740)

`DualHead` is what `train.py` uses: the task is a binary classification at pProp >= 3.5
*and* a regression on continuous pProp, trained jointly. It takes scalar `hidden_dim` /
`n_layers` rather than a `hidden_dims` sequence, because sweep drivers pass scalars.

Deliberately absent: target scaling and loss weighting. Those are properties of the loss
and the data pipeline, not of the head, and putting them here would make every head
re-implement them -- `src/normalization.py` and `src/losses.py` own them instead.
"""

from collections.abc import Sequence

import torch.nn as nn

# The MiniMol v1 embedding width: global max-pool over the final (16th) GNN layer, whose
# out_dim is 512 in the pretrained config. Kept as a named constant so a head built without
# a trunk in hand still defaults to the right input size.
EMBED_DIM = 512

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}

NORMS = {
    None: None,
    "none": None,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
}


def _make_activation(name):
    try:
        return ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}"
        ) from None


def _make_norm(name, dim):
    if name not in NORMS:
        raise ValueError(
            f"unknown norm {name!r}; choose from {sorted(k for k in NORMS if k)} or None"
        )
    cls = NORMS[name]
    return None if cls is None else cls(dim)


def _mlp_blocks(in_dim, hidden_dims, activation, dropout, norm, bias):
    """`[(Linear -> norm? -> activation -> dropout?)] * len(hidden_dims)`, and the width out.

    Shared by `MLPHead` and `DualHead` so the two cannot drift in what a "hidden block"
    means. Returns the layer list *without* a final output projection -- callers append
    their own, or none at all if they want the raw block stack (which is what `DualHead`'s
    shared trunk is).

    `hidden_dims=()` yields no layers and returns `in_dim` unchanged, so the degenerate
    depth-0 case needs no special-casing anywhere upstream.
    """
    layers = []
    prev = in_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width, bias=bias))
        layer_norm = _make_norm(norm, width)
        if layer_norm is not None:
            layers.append(layer_norm)
        layers.append(_make_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width
    return layers, prev


def _validate_dims(hidden_dims, dropout):
    """Normalise `hidden_dims` to a tuple of positive ints and range-check `dropout`."""
    if isinstance(hidden_dims, int):
        hidden_dims = (hidden_dims,)
    if not isinstance(hidden_dims, Sequence):
        raise TypeError(f"hidden_dims must be a sequence of ints, got {hidden_dims!r}")
    hidden_dims = tuple(int(d) for d in hidden_dims)
    if any(d <= 0 for d in hidden_dims):
        raise ValueError(f"hidden_dims must all be positive, got {hidden_dims}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    return hidden_dims


class MLPHead(nn.Module):
    """A plain MLP: `in_dim -> each of hidden_dims -> out_dim`.

    Each hidden block is `Linear -> norm? -> activation -> dropout?`. The output layer is a
    bare `Linear` — no activation, no dropout, no norm — because the target is an unbounded
    real number. Squashing it (or dropping units in it) would bound or bias the prediction.

    `hidden_dims=()` collapses to a single `Linear(in_dim, out_dim)`, which is exactly the
    linear-probe baseline of NOTES §7 Phase 3. That falls out of the general case rather
    than needing its own class, so the baseline and the model under test differ only in
    this argument and in whether the trunk's learning rate is zero.
    """

    def __init__(self, in_dim=EMBED_DIM, hidden_dims=(1200, 32), out_dim=1,
                 activation="gelu", dropout=0.0, norm=None, bias=True):
        super().__init__()

        hidden_dims = _validate_dims(hidden_dims, dropout)

        self.in_dim = int(in_dim)
        self.hidden_dims = hidden_dims
        self.out_dim = int(out_dim)

        layers, prev = _mlp_blocks(self.in_dim, hidden_dims, activation, dropout, norm, bias)
        layers.append(nn.Linear(prev, self.out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """`[B, in_dim] -> [B, out_dim]`. Squeezing to `[B]` is `MiniMolRegressor`'s job."""
        return self.net(x)

    def extra_repr(self):
        dims = " -> ".join(str(d) for d in (self.in_dim, *self.hidden_dims, self.out_dim))
        return f"dims={dims}"


class DualHead(nn.Module):
    """Shared MLP down to an `embed_dim` bottleneck, then a linear logit and a linear scalar.

    Ported in structure from `pProp_MLP.model.DualHeadMLP`, which is where the joint
    classify-and-regress design was validated. Two things about it are load-bearing here:

    - **The two heads sit on a *shared* trunk**, so the classification signal shapes the
      representation the regression head reads. That is the entire reason the classifier
      earns its keep -- the objective could be regression-only, but the auxiliary gradient
      is what makes "is this above pProp 3.5" inform "how far above".
    - **LayerNorm, never BatchNorm.** Batch size is a training decision, and the class
      ratio is 104:1, so per-batch statistics over ~11 positives in 1,200 rows would be
      unstable. `norm="batch"` remains reachable through the registry but is the wrong
      choice here.

    Sizes come in as scalars (`hidden_dim`, `n_layers`) rather than as a `hidden_dims`
    sequence, because sweep drivers pass scalars -- a list-valued hyperparameter has no
    natural representation in Optuna or a wandb sweep config. `n_layers=0` gives no shared
    stack above the bottleneck, which then reads the 512-d embedding directly.

    THE BOTTLENECK IS THE DELIVERABLE
    ---------------------------------
    `self.shared` ends at `embed_dim` (32), and that tensor -- not the predictions -- is what
    this repo exists to produce. It is deployed frozen into a generative + active-learning
    project, where a deep-kernel-learning GP trains its own network on top of it and MiniMol
    is never trained again. Three consequences that look like odd choices otherwise:

    - **The task heads are linear** (`cls_n_layers = reg_n_layers = 0`, which `MLPHead` turns
      into a bare `Linear`). This makes "pProp is linear in the exported embedding" literally
      true, which is the geometry an RBF/Matern kernel wants. Deeper branches would likely
      score better on `goal_metric` while letting the bottleneck encode the target in a shape
      a GP reads poorly -- and `goal_metric` is a proxy here, not the product.
    - **The export point is the OUTPUT of `self.shared`** -- post-norm, post-activation, the
      exact tensor the linear heads consume. Exporting the pre-activation instead would make
      the target linear-*after*-GELU, which is not the property above. Dropout is identity
      under `eval()`, so the export is unambiguous either way.
    - **`forward_with_embedding` exists** because the training loop needs `z` for the
      variance/covariance term (`losses.variance_covariance_loss`) and for
      `metrics.embedding_metrics`. `forward` delegates to it so the two cannot drift.

    The bottleneck is also where this architecture is most likely to fail, and the reason
    `--w-vic` exists. Regression and "pProp >= 3.5" are near-collinear -- the same axis, the
    second with its gradient concentrated at the boundary -- so the *supervised* signal
    arriving here is close to rank-1, so nothing asks the other ~30 dimensions to hold
    anything and the LayerNorm at the end of this very block divides them down -- it
    normalises across the 32 dims *within each row*, so one dominant pre-activation shrinks
    every other dim. (Earlier text here blamed weight decay. Falsified: at `lr * wd = 1e-5`
    the head shrinks 4.3% over a whole run and the trunk 0.33%, which cannot erase 30
    dimensions. See `reports/embedding_collapse_experiment.md` P6.3.) An embedding whose
    effective rank is 2-4 of 32 hands the GP a disguised scalar, making its distances a
    restatement of predicted pProp and its uncertainties meaningless on novel molecules.
    MEASURED 2026-08-18: effective rank 2.78 at the default and 1.32 on another seed, with
    rho(embedding distance, |delta predicted pProp|) = 0.9986 -- the failure, not the risk.
    `val/emb_effective_rank` is logged every epoch precisely so this stays observed.

    `forward` returns `(logits, pred)`, both shaped `[B]`. `MiniMolRegressor` passes tuples
    through untouched.
    """

    def __init__(self, in_dim=EMBED_DIM, hidden_dim=1024, n_layers=2, embed_dim=32,
                 cls_hidden_dim=256, cls_n_layers=0,
                 reg_hidden_dim=256, reg_n_layers=0,
                 activation="gelu", dropout=0.0, norm="layer", bias=True,
                 bottleneck_norm=None):
        super().__init__()

        _validate_dims((), dropout)          # range-checks dropout
        self.in_dim = int(in_dim)

        # The bottleneck goes through the same `Linear -> norm? -> activation -> dropout?`
        # block as every other width, so it needs no separate code path -- but it is built by
        # its own `_mlp_blocks` call so its NORM can differ from the hidden layers'.
        #
        # WHY THAT KNOB EXISTS. LayerNorm normalises across the `embed_dim` dimensions WITHIN
        # EACH ROW, so a single dominant pre-activation divides every other dimension down.
        # That is the mechanism `reports/embedding_collapse_experiment.md` P6.3 named as the
        # likely cause of the measured collapse (per-dim std 0.934 -> 0.017), after showing
        # weight decay is arithmetically inert here. `--w-vic` fights that normalisation with
        # a penalty; `bottleneck_norm="none"` removes the cause instead. `None` means "same as
        # `norm`", so the default is bit-identical to the pre-2026-08-19 architecture.
        #
        # The hidden layers keep their norm regardless: the collapse is a property of the
        # exported 32-d tensor, and destabilising a 1024-wide stack is not part of the test.
        self.bottleneck_norm = norm if bottleneck_norm is None else bottleneck_norm
        hidden = _validate_dims((int(hidden_dim),) * int(n_layers), dropout)
        layers, prev = _mlp_blocks(self.in_dim, hidden, activation, dropout, norm, bias)
        neck, trunk_out = _mlp_blocks(prev, _validate_dims((int(embed_dim),), dropout),
                                      activation, dropout, self.bottleneck_norm, bias)
        self.shared = nn.Sequential(*(layers + neck))
        self.shared_out_dim = trunk_out
        self.embed_dim = trunk_out

        head_kwargs = dict(in_dim=trunk_out, out_dim=1, activation=activation,
                           dropout=dropout, norm=norm, bias=bias)
        self.cls_head = MLPHead(hidden_dims=(int(cls_hidden_dim),) * int(cls_n_layers),
                                **head_kwargs)
        self.reg_head = MLPHead(hidden_dims=(int(reg_hidden_dim),) * int(reg_n_layers),
                                **head_kwargs)

    def forward_with_embedding(self, x):
        """`[B, in_dim] -> ([B], [B], [B, embed_dim])`, the third being the exported vector.

        The logit is raw -- no sigmoid. `losses.weighted_bce_loss` expects logits, and
        keeping the squashing out of the module means the same value doubles as the ranking
        score for average precision, which only cares about order.
        """
        z = self.shared(x)
        return self.cls_head(z).squeeze(-1), self.reg_head(z).squeeze(-1), z

    def forward(self, x):
        """`[B, in_dim] -> ([B], [B])` as `(classification logit, predicted pProp)`."""
        logits, pred, _ = self.forward_with_embedding(x)
        return logits, pred

    def extra_repr(self):
        return (f"in_dim={self.in_dim}, embed_dim={self.embed_dim}, "
                f"cls={self.cls_head.hidden_dims}, reg={self.reg_head.hidden_dims}")


HEADS = {
    "mlp": MLPHead,
    "dual": DualHead,
}


def build_head(name="mlp", **kwargs):
    """Construct a head by name, so architecture can come from a config file.

    Register a new head with `HEADS["my_head"] = MyHead`; nothing else needs to change.
    """
    try:
        cls = HEADS[name]
    except KeyError:
        raise ValueError(f"unknown head {name!r}; choose from {sorted(HEADS)}") from None
    return cls(**kwargs)
