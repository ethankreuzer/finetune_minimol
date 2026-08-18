"""Compose a trainable MiniMol trunk with a regression head.

This module is deliberately thin. It exists for one reason that is not cosmetic: the
trunk and the head must be *addressable separately by the optimizer*. A freshly
initialised head wants a normal learning rate; the pretrained trunk wants one 10-100x
smaller, because a head-sized LR applied to 10M pretrained parameters destroys the
representation in the first few steps (NOTES.md §7 Phase 3). `param_groups` is where that
split is expressed, and it is the reason this file exists rather than a bare
`nn.Sequential`.

    from model import MiniMolRegressor
    from trunk import MiniMolTrunk
    from head import MLPHead

    model = MiniMolRegressor(MiniMolTrunk(), MLPHead())
    batch = model.trunk.featurize(["CCO", "c1ccccc1"])
    pred  = model(batch)                                  # [2]

    opt = torch.optim.AdamW(model.param_groups(trunk_lr=1e-5, head_lr=1e-3))

Setting `trunk_lr=0.0` reproduces the frozen-embedding baseline of NOTES §7 Phase 3
without a second code path — but note it still pays the trunk's forward and backward cost,
so it is the right way to run the *controlled* comparison, not the fast way to run a
linear probe.
"""

import torch.nn as nn

from head import MLPHead


class MiniMolRegressor(nn.Module):
    """`batch -> [B]` scalar predictions, differentiable all the way into the trunk.

    `squeeze_output` collapses the head's `[B, 1]` to `[B]` so it lines up with a `[B]`
    target without the caller reshaping. It is switched off automatically when the head
    predicts more than one value, since squeezing a `[B, k]` output would be wrong.
    """

    def __init__(self, trunk, head=None, squeeze_output=True):
        super().__init__()
        self.trunk = trunk
        self.head = head if head is not None else MLPHead()
        out_dim = getattr(self.head, "out_dim", 1)
        self.squeeze_output = bool(squeeze_output) and out_dim == 1

    def forward(self, batch):
        embedding = self.trunk(batch)          # [B, 512], carries grad_fn
        out = self.head(embedding)             # [B, out_dim], or a tuple from a dual head
        # A multi-output head (`head.DualHead`) returns an already-shaped tuple. Pass it
        # through rather than trying to squeeze it -- the dispatch is on the value, not on
        # a flag the head has to remember to set, so a new multi-head needs no change here.
        if isinstance(out, tuple):
            return out
        return out.squeeze(-1) if self.squeeze_output else out

    def forward_with_embedding(self, batch):
        """`(logits, pred, z)` where `z` is the head's `[B, embed_dim]` bottleneck.

        Distinct from `embed` on purpose, and the distinction matters: `embed` returns the
        trunk's **512-d** output, while `z` is the head's **32-d** bottleneck — the vector
        this repo exists to export (see `head.DualHead`). Confusing the two would ship the
        wrong artifact.

        `forward` is deliberately left as a 2-tuple rather than widened to carry `z`:
        `train.py` unpacks it as `logits, pred = model(batch)` in two places, and a 3-tuple
        would fail there with an unpacking error rather than anything self-explanatory.
        """
        return self.head.forward_with_embedding(self.trunk(batch))

    def embed(self, batch):
        """The 512-d trunk embedding alone — for caching, probing, or diagnostics."""
        return self.trunk(batch)

    def param_groups(self, trunk_lr, head_lr, weight_decay=None, trunk_weight_decay=None):
        """Two optimizer groups, tagged by name so wandb and checkpoints stay readable.

        A group with no parameters is dropped rather than passed on empty: some optimizers
        accept an empty group and others raise, and an empty group is never meaningful
        here anyway.
        """
        groups = []
        for name, module, lr, wd in (
            ("trunk", self.trunk, trunk_lr,
             trunk_weight_decay if trunk_weight_decay is not None else weight_decay),
            ("head", self.head, head_lr, weight_decay),
        ):
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                continue
            group = {"name": name, "params": params, "lr": lr}
            if wd is not None:
                group["weight_decay"] = wd
            groups.append(group)
        return groups

    def n_parameters(self):
        """`{"trunk": n, "head": n, "total": n}` over trainable parameters only."""
        counts = {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in (("trunk", self.trunk), ("head", self.head))
        }
        counts["total"] = counts["trunk"] + counts["head"]
        return counts
