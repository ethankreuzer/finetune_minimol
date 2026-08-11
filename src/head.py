"""Regression heads that sit on top of MiniMol's 512-d pooled embedding.

The trunk (`src/trunk.py`) is fixed by the pretrained checkpoint; everything after the
512-d embedding is ours to choose, and the current choice — 512 -> 1200 -> 32 -> 1 — is a
prototype, not a final architecture. So this module is built to be swapped rather than
edited: `MLPHead` takes its shape as a `hidden_dims` sequence, and `build_head` is a
registry so a genuinely different head (attention pooling, gated, ensemble) is a new class
plus one line, not a rewrite of `src/model.py`.

    from head import MLPHead, build_head

    head = MLPHead()                                  # 512 -> 1200 -> 32 -> 1
    head = MLPHead(hidden_dims=(256,), dropout=0.1)   # 512 -> 256 -> 1
    head = MLPHead(hidden_dims=())                    # 512 -> 1, i.e. a linear probe
    head = build_head("mlp", hidden_dims=(1200, 32))  # same, by name

Deliberately absent: target scaling and `ipw` weighting. Those are properties of the loss
and the data pipeline, not of the head, and putting them here would make every head
re-implement them. See NOTES.md §7 Phase 3.
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

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        if not isinstance(hidden_dims, Sequence):
            raise TypeError(f"hidden_dims must be a sequence of ints, got {hidden_dims!r}")
        hidden_dims = tuple(int(d) for d in hidden_dims)
        if any(d <= 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims must all be positive, got {hidden_dims}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_dim = int(in_dim)
        self.hidden_dims = hidden_dims
        self.out_dim = int(out_dim)

        layers = []
        prev = self.in_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width, bias=bias))
            layer_norm = _make_norm(norm, width)
            if layer_norm is not None:
                layers.append(layer_norm)
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, self.out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """`[B, in_dim] -> [B, out_dim]`. Squeezing to `[B]` is `MiniMolRegressor`'s job."""
        return self.net(x)

    def extra_repr(self):
        dims = " -> ".join(str(d) for d in (self.in_dim, *self.hidden_dims, self.out_dim))
        return f"dims={dims}"


HEADS = {
    "mlp": MLPHead,
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
