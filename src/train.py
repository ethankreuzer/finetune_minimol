"""One training run: staged fine-tuning of MiniMol on the AmpC pProp task.

One fold, one seed. The schedule is the experiment, and it has two phases with independent
lengths: the head trains alone on the frozen embedding for `--freeze-epochs`, then the trunk
is unfrozen and both train together for `--unfrozen-epochs`. The head is given a chance to
stop being random before its gradients are allowed to reach 10M pretrained parameters.

The two phases are one trajectory, not two runs -- the weights carry straight over, and
phase 1's final weights *are* phase 2's initialization. What changes at the boundary is
which parameters are free, and the learning rate, which restarts. **Each phase anneals its
own cosine from its own peak down to `--eta-min`**; see `scheduled_lr`.

The task is joint (`head.DualHead`): a **binary classifier at pProp >= 3.5** and a
**regression on continuous pProp**, sharing one MLP over the 512-d embedding. 3.5 is the
threshold at which binders become possible; it is also the finest split at which the
positive class is still large enough to learn from -- 3,153 molecules, 619-643 per
validation fold, against only 100 in the whole dataset at pProp >= 5.0.

    python src/train.py --fold 0 --seed 0
    python src/train.py --freeze-epochs 1 --unfrozen-epochs 1 --subset 5000 --no-wandb
    python src/train.py --fold 0 --seed 0 --freeze-epochs 20 --unfrozen-epochs 0

The last form is the frozen-trunk baseline: `--unfrozen-epochs 0` never unfreezes, so it
reproduces the frozen-embedding workflow *on these splits*, which is the only honest
comparison for the fine-tuned arm. There is no `--epochs`; the total is
`--freeze-epochs + --unfrozen-epochs`, derived in `main` and stamped into `meta.json`.

THE LOSS
--------
`losses.combined_loss`: `w_cls * cls + huber + w_pair * pair + w_std * std + w_vic * vic`,
with huber grounded at weight 1 as the only term anchoring the absolute pProp level. See
`losses.py` for what changed in the port from `pProp_MLP` and why. `vic` regularises the
exported 32-d bottleneck rather than the predictions and is **off by default**.

`--weights` selects the per-sample weight vector, defaulting to `balanced` (two-group
inverse frequency at pProp 3.5, a 104x ratio). **Weights are derived per fold, from that
fold's own composition** -- the training fold never sees the validation fold's group sizes,
and vice versa.

TWO ORDERING TRAPS, BOTH ENFORCED IN CODE
-----------------------------------------
`MiniMolRegressor.param_groups` filters on `p.requires_grad` at *construction* time and
drops a group that comes back empty (model.py:74). So freezing the trunk before building
the optimizer yields a one-group optimizer, and unfreezing later sets `requires_grad = True`
on parameters the optimizer has never heard of -- the trunk would never train, while every
log line still looked healthy. The optimizer is therefore always built *before* the freeze,
and both groups are asserted to exist.

That failure is invisible in the loss curve, so `--assert-schedule` (on by default) also
measures it directly: the trunk must be bit-for-bit unchanged across the frozen phase
*while the head provably moves*, and must change once unfrozen. An unchanged trunk proves
nothing on its own -- it is equally what a broken training loop looks like -- which is why
the check is paired, following `verify_trunk.py::check_excluded_unreachable`.

METRICS ARE ON THE RAW pProp SCALE, ALWAYS
------------------------------------------
`--pprop-norm` changes what the regression head predicts, not what is reported: `evaluate`
denormalizes before computing anything. Pearson would survive either way (scale-invariant)
but MSE and MAE would not, and `val/mse` is declared with `summary="min"` -- so a normalized
report would silently make that summary incomparable across `--pprop-norm` settings.
"""

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_features                                          # noqa: E402
from head import DualHead                                                   # noqa: E402
from losses import (PPROP_EDGE, binary_labels, build_sample_weights,         # noqa: E402
                    combined_loss, effective_sample_size, pairwise_distance_loss,
                    std_match_loss, variance_covariance_loss, weighted_bce_loss,
                    weighted_huber_loss)
from metrics import (embedding_metrics, enrichment_factor, flavoured_metrics,  # noqa: E402
                     group_split_metrics)
from model import MiniMolRegressor                                          # noqa: E402
from normalization import (NORM_STRATEGIES, compute_norm_stats,              # noqa: E402
                           denormalize_pprop, normalize_pprop)
from objective import (OBJECTIVE_VERSION, REQUIRED_FLAVOURS,                 # noqa: E402
                       compute_objective)
from run_paths import run_dir                                               # noqa: E402
from splits import load_fold, load_meta                                     # noqa: E402
from trunk import MiniMolTrunk                                              # noqa: E402

# The tensor verify_trunk.py uses to prove an optimizer step moves the trunk. Reused here so
# both scripts are watching the same weight.
TRUNK_PROBE = "gnn.layers.0.model.nn.fully_connected.0.linear.weight"

# The pairwise term is O(N^2), so its *reported* value on a 66k validation fold is estimated
# on a fixed seeded subsample -- 66,296^2 is 4.4e9 pairs. Same constant and same reasoning as
# pProp_MLP. The loss scalar is an estimate; every metric beside it is exact.
PAIR_EVAL_K = 4096
PAIR_EVAL_SEED = 12345


def dashed(argv):
    """Rewrite `--head_lr=1e-3` to `--head-lr=1e-3`, leaving values untouched.

    A wandb sweep agent emits one `--<parameter>=<value>` per swept key, spelled exactly as
    the sweep yaml spells it -- and yaml keys are the config keys, which are underscored.
    argparse does *not* accept an underscore spelling of a dashed option, so without this
    every agent-launched run dies on `unrecognized arguments`. Normalising here beats adding
    a second option string to a dozen flags, and beats underscoring the CLI, which would
    break every command in the docs.

    Only the part before the first `=` is touched, so a value containing an underscore (a
    path, a tag) survives. A bare `--foo_bar value` pair is handled too: the value is a
    separate argv entry and never starts with `--`.
    """
    return [f"--{a[2:].split('=', 1)[0].replace('_', '-')}"
            + (f"={a.split('=', 1)[1]}" if "=" in a else "")
            if a.startswith("--") else a
            for a in argv]


def sweep_int(s):
    """`int`, tolerating `"3.0"`.

    wandb's `q_log_uniform_values` quantizes with `q * round(x / q)` and emits the result as
    a float, so an integer hyperparameter arrives as `--freeze_epochs=3.0` and plain
    `type=int` dies on it. Used for every integer that a sweep might touch.
    """
    return int(float(s))


def parse_args(argv=None):
    """Parse `train.py`'s CLI. See `build_parser` for the parser itself."""
    return build_parser().parse_args(dashed(sys.argv[1:] if argv is None else list(argv)))


def build_parser():
    """The parser, separately from parsing.

    `run_config.py` walks these actions to mirror every hyperparameter onto its own CLI, so
    a flag added here reaches the sweep with no second edit. That is the only reason this is
    split out -- returning a parser rather than a namespace is otherwise no better.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="head init, dropout and shuffling only -- never the partition")

    # Two phase LENGTHS, not a total and a cut point. A bayes sweep samples its parameters
    # independently, and (epochs, freeze_epochs) carries the cross-constraint
    # `freeze_epochs <= epochs` -- every violating draw would die on a SystemExit mid-sweep.
    # Two non-negative lengths have no such constraint. `args.epochs` is derived in `main`.
    p.add_argument("--freeze-epochs", type=sweep_int, default=5,
                   help="phase 1: epochs training the head alone on the frozen trunk")
    p.add_argument("--unfrozen-epochs", type=sweep_int, default=15,
                   help="phase 2: epochs training trunk and head together; "
                        "0 never unfreezes, which is the frozen-trunk baseline")

    # Head shape as scalars, not a list: sweep drivers pass scalars, and a list-valued
    # hyperparameter has no natural representation in Optuna or a wandb sweep config.
    p.add_argument("--n-layers", type=sweep_int, default=2, help="shared trunk depth (0 = none)")
    p.add_argument("--hidden-dim", type=sweep_int, default=1024)
    p.add_argument("--embed-dim", type=sweep_int, default=32,
                   help="the exported bottleneck width; 32 is fixed by the downstream GP")
    # 0 layers = a bare Linear on the bottleneck. Deliberate: it makes pProp linear in the
    # exported embedding, which is the geometry a GP kernel wants. See head.DualHead.
    p.add_argument("--cls-n-layers", type=sweep_int, default=0)
    p.add_argument("--cls-hidden-dim", type=sweep_int, default=256)
    p.add_argument("--reg-n-layers", type=sweep_int, default=0)
    p.add_argument("--reg-hidden-dim", type=sweep_int, default=256)
    p.add_argument("--head-norm", default="layer", help="'layer', 'batch' or 'none'")
    p.add_argument("--dropout", type=float, default=0.0, help="head dropout")

    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="head lr during phase 1 (the peak of the phase-1 cosine)")
    p.add_argument("--head-lr-unfrozen", type=float, default=None,
                   help="head lr during phase 2; defaults to --head-lr. The head is no "
                        "longer random by then, so this is usually lower")
    p.add_argument("--trunk-lr", type=float, default=1e-4,
                   help="trunk lr during phase 2 (it is 0 by construction in phase 1)")
    p.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine")
    p.add_argument("--eta-min", type=float, default=1e-8,
                   help="cosine floor; 1e-8 matches the schedule these defaults came from")
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Decoupled AdamW decay is `lr * wd` per step, so one `--weight-decay` means very
    # different things to the two groups: 4.3% total shrink on the head at lr 1e-3, 0.33%
    # on the trunk at 1e-4 (reports/embedding_collapse_experiment.md P6.3). This splits
    # them. None = follow --weight-decay, which is the historical behaviour exactly.
    p.add_argument("--trunk-weight-decay", type=float, default=None,
                   help="weight decay for the trunk group; defaults to --weight-decay")
    # Batch 1200 is doing double duty: throughput (reports/compute_profile.md measures the
    # fixed per-step overhead falling from 28% at 256 to ~9% at 1024) and tail variance
    # (~11.4 positives per batch instead of 2.4, so the up-weighted half of the loss is not
    # riding on two or three molecules). Both arguments point the same way.
    p.add_argument("--batch-size", type=sweep_int, default=1200)
    p.add_argument("--num-workers", type=int, default=16)

    p.add_argument("--weights", choices=("uniform", "balanced"), default="balanced",
                   help="per-sample loss weights (losses.build_sample_weights)")
    p.add_argument("--pprop-norm", choices=NORM_STRATEGIES, default="zscore",
                   help="regression target transform; stats from the TRAINING fold only")
    p.add_argument("--w-cls", type=float, default=0.4418)
    p.add_argument("--w-pair", type=float, default=7.486)
    p.add_argument("--w-std", type=float, default=0.7911)
    p.add_argument("--huber-delta", type=float, default=1.0513)
    # Guards the exported bottleneck against collapsing onto a couple of directions.
    # DEFAULT 0.0 = inert. `val/emb_effective_rank` is what decides whether to turn it on;
    # shipping it disabled makes that a sweep value rather than a code change.
    p.add_argument("--w-vic", type=float, default=0.0,
                   help="weight on the variance/covariance term (0 = off)")
    p.add_argument("--vic-gamma", type=float, default=1.0,
                   help="target floor on each embedding dimension's std")
    # 1.0 reproduces the term exactly as it was swept. Measured, the variance hinge is
    # the binding half today (28/32 dims below gamma even at --w-vic 0.3), so this is
    # insurance for the regime where enough force saturates the hinge and covariance
    # becomes the only half still pushing. Landed now because every new train.py flag
    # re-hashes every run_config `config_id`.
    p.add_argument("--w-cov", type=float, default=1.0,
                   help="weight on the covariance half of --w-vic's term")

    p.add_argument("--subset", type=int, default=None,
                   help="use only N train and N val rows (smoke test)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir; defaults to outputs/<sweep_id>/fold{fold}_seed{seed}")
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--sweep-id", default=None,
                   help="groups runs on disk; falls back to WANDB_SWEEP_ID then _no_sweep")
    p.add_argument("--wandb-project", default="finetune_minimol")
    p.add_argument("--wandb-entity", default="ethan_personal")
    p.add_argument("--wandb-group", default=None,
                   help="groups the runs of one grid into a single wandb view")
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-assert-schedule", dest="assert_schedule", action="store_false",
                   help="skip the freeze/unfreeze weight-delta assertions (not advised)")
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false",
                   help="skip final.pt (34 MB); predictions and meta.json are still written")
    return p


# -- data ---------------------------------------------------------------------------

class RowDataset(Dataset):
    """`(graph, y_norm, y_binary, weight, row_index)` for one row of the subset CSV.

    The row index is carried through the batch so validation predictions can be written back
    against the CSV rows they belong to -- which is what makes pooled out-of-fold tail
    metrics possible later (`src/pool_oof.py`) without re-running anything. It also means
    the *raw* target never has to travel in the batch: `y_raw = y[rows]` recovers it exactly
    after the fact.
    """

    def __init__(self, cache, y_norm, y_bin, w, indices):
        self.cache, self.y_norm, self.y_bin, self.w = cache, y_norm, y_bin, w
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = int(self.indices[i])
        return self.cache[row], self.y_norm[row], self.y_bin[row], self.w[row], row


def collate_batch(items):
    """Mirror of `MiniMolTrunk.collate(to_device=False)`, without needing the trunk.

    Deliberately not `trunk.collate`: this runs inside DataLoader workers, and binding it to
    the model would drag a CUDA-resident module into forked children. Nothing here touches
    CUDA -- the move to device happens in the training loop.
    """
    graphs, y_norm, y_bin, w, idx = zip(*items)
    batch = Batch.from_data_list(list(graphs))
    return ({"features": batch, "batch_indices": batch.batch},
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(y_bin, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long))


def fold_weights(kind, y, train_idx, val_idx):
    """A full-length weight vector whose folds are each weighted by their *own* composition.

    Building one vector over all 331,480 rows and slicing it would leak: `balanced` weights
    depend on the group sizes of whatever set they are computed over, so a global vector
    would encode the validation fold's tail fraction into the training loss. Each fold is
    therefore weighted independently and written into its own positions.

    Rows in neither fold keep weight 0; nothing reads them, and a zero is a louder failure
    than a plausible-looking 1 if that ever stops being true.
    """
    w = np.zeros(len(y), dtype=np.float64)
    for idx in (train_idx, val_idx):
        w[idx] = build_sample_weights(kind, y[idx])
    return w


# -- schedule -----------------------------------------------------------------------

def phase_bounds(epoch, cfg):
    """`(epochs_before_this_phase, phase_length)` for a 1-indexed epoch.

    Phase 1 is `[1, freeze_epochs]`, phase 2 is the rest. This is the only place the phase
    boundary is arithmetic rather than a comparison, so a schedule and a freeze can never
    disagree about which epoch belongs to which phase.
    """
    if epoch <= cfg.freeze_epochs:
        return 0, cfg.freeze_epochs
    return cfg.freeze_epochs, cfg.unfrozen_epochs


def scheduled_lr(base, epoch, cfg):
    """Cosine-annealed lr for a 1-indexed epoch, annealed **within its own phase**.

    Each phase gets its own cosine from `base` down to `eta_min`, so the head restarts at
    `--head-lr-unfrozen` when the trunk unfreezes rather than inheriting phase 1's decay.
    Before this, one cosine ran over the whole run and the trunk unfroze into it already
    partly annealed -- at freeze 5 / total 20 it began phase 2 at ~85% of
    `--trunk-lr` and fell from there, having "decayed" through five epochs in which its lr
    was pinned to 0. That was the head's schedule with a hole in it, not a trunk schedule.

    **The denominator is `length - 1`, deliberately unlike `CosineAnnealingLR(T_max=N)`.**
    Torch's form puts the last epoch at `t = (N-1)/N`, which never reaches `eta_min`: at
    N=20 that is ~0.6% of base and looks like convergence, but at N=3 it is 25% of base and
    is not. Phase lengths here are swept down to 1, so the torch form would leave short
    phases ending at a high lr. `t = (epoch - 1) / (length - 1)` lands the final epoch of
    every phase exactly on `eta_min`. A length-1 phase cannot anneal and returns `base`.

    That matters because selection is at the final epoch with no early stopping: annealed to
    `eta_min` the final epoch is a settled model, whereas at a live lr it is an arbitrary
    point on a still-moving trajectory. It applies to phase 1 too -- with
    `--unfrozen-epochs 0` (the frozen-trunk baseline) phase 1 *is* the terminal phase, and
    the baseline has to be as settled as the arm it is compared against.

    Computed in closed form rather than delegated to a torch scheduler because a scheduler
    writes `group["lr"]` every step and would silently fight the freeze, which writes the
    same field. Two authorities over one field is exactly the class of bug the module
    docstring's ordering traps are about, so there is one authority here: `apply_lrs`.
    """
    if cfg.lr_schedule == "none":
        return base
    start, length = phase_bounds(epoch, cfg)
    if length <= 1:
        return base
    t = (epoch - start - 1) / (length - 1)
    return cfg.eta_min + (base - cfg.eta_min) * (1 + math.cos(math.pi * t)) / 2


def set_trunk_trainable(model, optimizer, trainable, trunk_lr):
    """Flip the trunk between frozen and training.

    Sets `requires_grad` *and* the trunk group's lr. Either alone would suffice -- AdamW
    skips a parameter whose `.grad` is None -- but doing both makes the state legible in the
    logged learning rate and stops one of them being changed later in isolation.

    Every freeze transition in the run goes through this one function, including the ones
    `apply_lrs` performs each epoch. That is deliberate: `verify_metrics.py` neuters this
    name to prove the schedule assertion has teeth, and a second path around it would make
    that negative test pass while testing nothing.
    """
    for p in model.trunk.parameters():
        p.requires_grad_(trainable)
    for group in optimizer.param_groups:
        if group.get("name") == "trunk":
            group["lr"] = trunk_lr if trainable else 0.0


def apply_lrs(model, optimizer, epoch, cfg, trunk_trainable):
    """The single authority over both learning rates each epoch. Returns what it set.

    Delegates the trunk to `set_trunk_trainable` rather than writing its lr directly, so the
    freeze keeps exactly one implementation.

    The trunk's cosine is only *evaluated* once it is trainable: in phase 1 its schedule
    would be read off the phase-1 clock, which is not its clock. It is 0 there instead, which
    is also what `set_trunk_trainable` would write anyway.
    """
    trunk_lr = scheduled_lr(cfg.trunk_lr, epoch, cfg) if trunk_trainable else 0.0
    set_trunk_trainable(model, optimizer, trunk_trainable, trunk_lr)

    head_base = cfg.head_lr if epoch <= cfg.freeze_epochs else cfg.head_lr_unfrozen
    head_lr = scheduled_lr(head_base, epoch, cfg)
    for group in optimizer.param_groups:
        if group.get("name") == "head":
            group["lr"] = head_lr
    return {"head": head_lr, "trunk": trunk_lr}


def probe_weights(model):
    """Copies of one trunk tensor and one head tensor, for the schedule assertions."""
    trunk = dict(model.trunk.named_parameters())[TRUNK_PROBE].detach().clone()
    head = next(p for p in model.head.parameters() if p.dim() > 1).detach().clone()
    return trunk, head


# -- loops --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, cfg):
    """One pass over the training fold. Returns the mean of each loss term.

    Terms are accumulated weighted by batch size and reported *unscaled* -- before their
    `w_*` multipliers. That is the only way to see whether a swept weight is doing anything,
    since a term that has collapsed and a term whose weight is tiny look identical once
    multiplied.
    """
    model.train()
    totals = {"loss": 0.0, "cls": 0.0, "huber": 0.0, "pair": 0.0, "std": 0.0, "vic": 0.0}
    n_seen = 0
    for batch, y_norm, y_bin, w, _ in loader:
        batch = model.trunk.to_device(batch, device)
        y_norm, y_bin, w = y_norm.to(device), y_bin.to(device), w.to(device)

        # `forward_with_embedding`, not `forward`: the variance/covariance term regularises
        # the 32-d bottleneck, which `forward` discards. `z` is the exported artifact.
        logits, pred, z = model.forward_with_embedding(batch)
        loss, terms = combined_loss(logits, pred, y_bin, y_norm, w,
                                    w_cls=cfg.w_cls, w_pair=cfg.w_pair,
                                    w_std=cfg.w_std, huber_delta=cfg.huber_delta,
                                    embedding=z, w_vic=cfg.w_vic,
                                    vic_gamma=cfg.vic_gamma, w_cov=cfg.w_cov)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = len(y_norm)
        n_seen += bs
        totals["loss"] += float(loss) * bs
        for k, v in terms.items():
            totals[k] += v * bs

    return {k: v / max(n_seen, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    """Forward the whole loader.

    Returns normalized predictions, logits, row indices, and the `[N, embed_dim]` bottleneck.

    The embeddings come out of the same pass rather than a second one: the trunk dominates
    the cost, so re-running it just to collect `z` would roughly double evaluation time.
    `model.eval()` makes dropout the identity, so this is exactly the vector an exported
    model would emit for these molecules.
    """
    model.eval()
    preds, logits, rows, embeddings = [], [], [], []
    for batch, _, _, _, idx in loader:
        batch = model.trunk.to_device(batch, device)
        lo, pr, z = model.forward_with_embedding(batch)
        preds.append(pr.float().cpu().numpy())
        logits.append(lo.float().cpu().numpy())
        rows.append(idx.numpy())
        embeddings.append(z.float().cpu().numpy())
    return (np.concatenate(preds), np.concatenate(logits), np.concatenate(rows),
            np.concatenate(embeddings))


def loss_terms_from_arrays(pred_norm, logits, y_norm, y_bin, w, cfg, device="cpu",
                           embedding=None):
    """The loss terms recomputed on a whole split, for reporting.

    `cls`, `huber` and `std` are exact and O(N). `pair` is O(N^2) so it is estimated on a
    fixed seeded subsample of `PAIR_EVAL_K` points -- fixed and seeded so the estimate is
    comparable across epochs and runs rather than adding its own noise to the curve.

    `vic` is computed on the *same* subsample as `pair`, not on the whole split. That is not
    a cost concern -- it is O(N*d^2) and would be cheap -- but a definitional one: the term
    trains on batches of `--batch-size`, and its covariance estimate depends on the sample it
    is measured over, so evaluating it on 66k rows would report a number the optimiser never
    sees. Reusing the pair subsample keeps the reported value on a comparable footing.
    """
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    pred_t, logit_t, yn_t, yb_t, w_t = t(pred_norm), t(logits), t(y_norm), t(y_bin), t(w)

    rng = np.random.default_rng(PAIR_EVAL_SEED)
    k = min(PAIR_EVAL_K, len(pred_norm))
    sub = rng.choice(len(pred_norm), k, replace=False)

    with torch.no_grad():
        cls = float(weighted_bce_loss(logit_t, yb_t, w_t))
        huber = float(weighted_huber_loss(pred_t, yn_t, w_t, delta=cfg.huber_delta))
        pair = float(pairwise_distance_loss(pred_t[sub], yn_t[sub]))
        std = float(std_match_loss(pred_t, yn_t, w_t))
        vic = (float(variance_covariance_loss(t(embedding)[sub], gamma=cfg.vic_gamma,
                                              w_cov=cfg.w_cov))
               if embedding is not None else 0.0)

    out = {"cls": cls, "huber": huber, "pair": pair, "std": std,
           "loss": (cfg.w_cls * cls + huber + cfg.w_pair * pair + cfg.w_std * std
                    + cfg.w_vic * vic),
           "pair_eval_k": k}
    if embedding is not None:
        out["vic"] = vic
    return out


def score_split(pred_norm, logits, rows, y_raw, w_loss, y_norm, cfg, embedding=None):
    """Every reported number for one split, on the raw pProp scale.

    Predictions are denormalized first -- see the module docstring. The weighting flavours
    are rebuilt here from *this split's* rows, so `balanced` reflects this fold's own group
    sizes rather than the dataset's.

    `embedding` is the `[N, embed_dim]` bottleneck. It is optional so that `pool_oof.py` and
    any other array-only consumer can still score a split without it, but when present it
    contributes `emb_*` -- the only metrics here that score the exported artifact rather than
    the predictions.
    """
    pred_raw = denormalize_pprop(pred_norm, cfg._norm_stats)

    flavours = {f: build_sample_weights(f, y_raw) for f in REQUIRED_FLAVOURS}
    m = flavoured_metrics(y_raw, pred_raw, logits, flavours, edge=PPROP_EDGE)
    m.update(compute_objective(m))
    m.update(group_split_metrics(y_raw, pred_raw, group_edge=PPROP_EDGE))
    m.update(enrichment_factor(y_raw, logits))
    if embedding is not None:
        m.update(embedding_metrics(embedding))
    m.update(loss_terms_from_arrays(pred_norm, logits,
                                    y_norm[rows], binary_labels(y_raw, PPROP_EDGE),
                                    w_loss[rows], cfg, embedding=embedding))
    # Logged beside Pearson so a nan correlation is diagnosable rather than mysterious: a
    # model predicting a constant has zero variance and Pearson is undefined, which is a
    # real early-epoch possibility when 36% of rows sit below pProp 1.0.
    m["pred_std"] = float(np.std(pred_raw))
    m["mse"] = float(np.mean((pred_raw - y_raw) ** 2))
    return m, pred_raw


# -- provenance ---------------------------------------------------------------------

def hardware():
    """What the run actually executed on -- the cost model in reports/ is built from this."""
    import os
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpu_count": os.cpu_count()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu_name=props.name,
                    gpu_total_gb=round(props.total_memory / 1e9, 1),
                    gpu_capability=f"sm_{props.major}{props.minor}")
    return info


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:
        return None


def log_prefixed(prefix, metrics):
    """`{"val/ap_uniform": ...}` from `{"ap_uniform": ...}`, dropping non-scalar entries."""
    return {f"{prefix}/{k}": v for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating, np.integer))}


def main(argv=None):
    args = parse_args(argv)
    if args.freeze_epochs < 0 or args.unfrozen_epochs < 0:
        raise SystemExit("--freeze-epochs and --unfrozen-epochs must both be >= 0")
    # Derived, not supplied: see the note on the two phase lengths in `parse_args`. It is
    # stamped into the wandb config and meta.json like any other setting, so the total is
    # still recorded and still filterable.
    args.epochs = args.freeze_epochs + args.unfrozen_epochs
    if args.epochs < 1:
        raise SystemExit("--freeze-epochs + --unfrozen-epochs must be >= 1")
    frozen_baseline = args.unfrozen_epochs == 0
    # Resolved here rather than at use, so the value that ran is the value provenance
    # records -- a `null` in meta.json would be a config that cannot be replayed.
    if args.head_lr_unfrozen is None:
        args.head_lr_unfrozen = args.head_lr

    import os
    sweep_id = args.sweep_id or os.environ.get("WANDB_SWEEP_ID")
    out = (args.out if args.out is not None
           else run_dir(args.outputs_root, sweep_id, f"fold{args.fold}_seed{args.seed}"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Seeds govern head init, dropout and shuffling. The partition is loaded frozen from
    # disk and is identical across seeds by construction (CLAUDE.md).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | objective {OBJECTIVE_VERSION}")

    cache = load_features(args.features)
    y = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)
    if len(cache) != len(y):
        raise SystemExit(f"feature cache has {len(cache)} graphs but the CSV has {len(y)} "
                         "rows; they are aligned by row position only")

    train_idx, val_idx = load_fold(args.splits, fold=args.fold)
    if args.subset:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, min(args.subset, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, min(args.subset, len(val_idx)), replace=False)
    print(f"fold {args.fold}: {len(train_idx):,} train / {len(val_idx):,} val | "
          f"{int(binary_labels(y[train_idx]).sum()):,} / "
          f"{int(binary_labels(y[val_idx]).sum()):,} positive at pProp >= {PPROP_EDGE}")

    # Normalization statistics come from the TRAINING fold only, so the validation fold
    # never leaks into the transform.
    norm_stats = compute_norm_stats(y[train_idx], args.pprop_norm)
    args._norm_stats = norm_stats
    y_norm = normalize_pprop(y, norm_stats)
    y_bin = binary_labels(y, PPROP_EDGE).astype(np.float64)
    w = fold_weights(args.weights, y, train_idx, val_idx)
    ess = effective_sample_size(w[train_idx])
    print(f"weights={args.weights}: train ESS {ess:,.0f} "
          f"({100 * ess / len(train_idx):.2f}% of the fold) | norm={args.pprop_norm}")

    common = dict(batch_size=args.batch_size, collate_fn=collate_batch,
                  num_workers=args.num_workers,
                  persistent_workers=args.num_workers > 0)
    # shuffle=True is load-bearing, not hygiene: ampc_subset_331k.csv is sorted by the
    # target, so an unshuffled loader trains on target-sorted batches (CLAUDE.md footguns).
    train_loader = DataLoader(RowDataset(cache, y_norm, y_bin, w, train_idx), shuffle=True,
                              generator=generator, **common)
    # A second, unshuffled view of the training fold, used once at the final epoch. The
    # in-loop training numbers are computed from predictions made *while* the weights were
    # changing, so they are not comparable to validation; the train/val gap is the headline
    # result and it has to be measured with both sides scored the same way.
    train_eval_loader = DataLoader(RowDataset(cache, y_norm, y_bin, w, train_idx),
                                   shuffle=False, **common)
    val_loader = DataLoader(RowDataset(cache, y_norm, y_bin, w, val_idx),
                            shuffle=False, **common)

    model = MiniMolRegressor(
        MiniMolTrunk(),
        DualHead(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                 embed_dim=args.embed_dim,
                 cls_hidden_dim=args.cls_hidden_dim, cls_n_layers=args.cls_n_layers,
                 reg_hidden_dim=args.reg_hidden_dim, reg_n_layers=args.reg_n_layers,
                 dropout=args.dropout, norm=args.head_norm),
    ).to(device)
    counts = model.n_parameters()
    print(f"trunk {counts['trunk']:,} + head {counts['head']:,} = {counts['total']:,} params")

    # Build the optimizer BEFORE freezing -- see the module docstring. Both groups must
    # exist now, because param_groups drops an empty one and never adds it back.
    optimizer = torch.optim.AdamW(
        model.param_groups(trunk_lr=args.trunk_lr, head_lr=args.head_lr,
                           weight_decay=args.weight_decay,
                           trunk_weight_decay=args.trunk_weight_decay))
    names = {g.get("name") for g in optimizer.param_groups}
    if names != {"trunk", "head"}:
        raise SystemExit(f"optimizer has groups {names}, expected both 'trunk' and 'head'. "
                         "The trunk was frozen before the optimizer was built.")
    set_trunk_trainable(model, optimizer, trainable=False, trunk_lr=args.trunk_lr)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, tags=args.wandb_tags,
                         dir=str(out), config={**{k: v for k, v in vars(args).items()
                                                  if not k.startswith("_")},
                                               "device": device, "params": counts,
                                               "git_sha": git_sha(),
                                               "objective_version": OBJECTIVE_VERSION,
                                               "pprop_edge": PPROP_EDGE,
                                               "train_ess": ess, **hardware()},
                         settings=wandb.Settings(start_method="thread"))
        # These summaries are diagnostics -- the gap between `val/goal_metric.max` and the
        # `final/*` keys written after the loop is exactly what final-epoch selection costs
        # on this run. They are NOT the sweep objective: a bayes sweep optimising a `max`
        # summary would be selecting the best epoch of each trial, i.e. early stopping on
        # the validation fold it also reports, which is the bias the fixed epoch budget
        # exists to avoid. `sweeps/*.yaml` optimises `final/goal_metric`.
        wandb.define_metric("val/goal_metric", summary="max")
        wandb.define_metric("val/ap_uniform", summary="max")
        wandb.define_metric("val/mse", summary="min")

    trunk0, head0 = probe_weights(model)
    history, checks = [], {}
    started_all = time.time()
    val_metrics = {}

    for epoch in range(1, args.epochs + 1):
        trunk_trainable = epoch > args.freeze_epochs
        lrs = apply_lrs(model, optimizer, epoch, args, trunk_trainable)
        if epoch == args.freeze_epochs + 1:
            print(f"--- epoch {epoch}: unfreezing trunk (lr {lrs['trunk']:.2e}) ---")
        phase = "head_only" if epoch <= args.freeze_epochs else "full"
        final = epoch == args.epochs

        # Train and validation are timed separately: the report attributes cost per phase,
        # and validation is forward-only so it scales differently from training.
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, args)
        train_secs = time.time() - started

        started = time.time()
        val_pred_n, val_logits, val_rows, val_emb = predict(model, val_loader, device)
        val_metrics, val_pred_raw = score_split(val_pred_n, val_logits, val_rows,
                                                y[val_rows], w, y_norm, args,
                                                embedding=val_emb)
        val_secs = time.time() - started
        peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0

        row = {"epoch": epoch, "phase": phase, "seconds": round(train_secs + val_secs, 1),
               "train_seconds": round(train_secs, 2), "val_seconds": round(val_secs, 2),
               "peak_gpu_gb": round(peak_gb, 2),
               "lr/trunk": lrs["trunk"], "lr/head": lrs["head"],
               **{f"train/{k}": v for k, v in tr.items()},
               **log_prefixed("val", val_metrics)}

        # The train-set eval pass is the final epoch only: it costs ~4x a validation pass
        # (265k rows against 66k) and only the final-epoch value is comparable to the
        # selected model. Same throttle as pProp_MLP, for the same reason.
        if final:
            started = time.time()
            tr_pred_n, tr_logits, tr_rows, tr_emb = predict(model, train_eval_loader, device)
            train_metrics, _ = score_split(tr_pred_n, tr_logits, tr_rows,
                                           y[tr_rows], w, y_norm, args, embedding=tr_emb)
            row["train_eval_seconds"] = round(time.time() - started, 2)
            row.update(log_prefixed("train_eval", train_metrics))

        history.append(row)
        print(f"epoch {epoch} [{phase:9s}] {train_secs:5.1f}s+{val_secs:4.1f}s "
              f"{peak_gb:5.2f}GB  loss {tr['loss']:.4f}  |  "
              f"val goal {val_metrics['goal_metric']:.4f} "
              f"ap {val_metrics['ap_uniform']:.4f} "
              f"r {val_metrics['pearson_uniform']:.4f} "
              f"mae {val_metrics['mae_uniform']:.4f}", flush=True)
        if run is not None:
            run.log(row, step=epoch)

        # The schedule's central claim, measured. Paired: an unchanged trunk means nothing
        # unless the head provably moved in the same phase.
        if epoch == args.freeze_epochs and args.assert_schedule:
            t1, h1 = probe_weights(model)
            td = float((t1 - trunk0).abs().max())
            hd = float((h1 - head0).abs().max())
            checks["frozen_phase"] = {"trunk_max_delta": td, "head_max_delta": hd}
            print(f"  [freeze check] trunk Δ={td:.3e} (want 0)  head Δ={hd:.3e} (want >0)")
            if td != 0.0 or hd <= 0.0:
                raise SystemExit(
                    f"freeze violated: trunk moved by {td:.3e} (expected exactly 0) or the "
                    f"head did not move ({hd:.3e}). The schedule did not do what it claims.")

        if final and args.assert_schedule and not frozen_baseline:
            t2, _ = probe_weights(model)
            td = float((t2 - trunk0).abs().max())
            checks["after_unfreeze"] = {"trunk_max_delta": td}
            print(f"  [unfreeze check] trunk Δ={td:.3e} (want >0)")
            if td <= 0.0:
                raise SystemExit(
                    "trunk did not change after unfreezing. It is still effectively frozen "
                    "-- check that the optimizer holds a 'trunk' group.")

    # The selected model is the FINAL epoch, so the number a sweep optimises has to be the
    # final-epoch number. `run.summary` entries survive independently of the step-indexed
    # history, so `final/goal_metric` is unambiguous no matter how many epochs a trial ran --
    # which matters here because the epoch budget is itself swept.
    if run is not None:
        run.summary.update(log_prefixed("final", val_metrics))
        run.summary["final/epoch"] = args.epochs

    # 34 MB per model. A 250-trial sweep at 10 models a trial is ~85 GB of checkpoints that
    # nothing reads, so `run_config.py` turns this off unless asked to keep them. The
    # predictions below are what the pooled metrics actually need, and they are ~2 MB.
    if args.save_checkpoint:
        torch.save({"model_state": model.state_dict(),
                    "config": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items() if not k.startswith("_")},
                    "norm_stats": norm_stats,
                    "objective_version": OBJECTIVE_VERSION,
                    "pprop_edge": PPROP_EDGE,
                    "val_goal_metric": val_metrics.get("goal_metric")},
                   out / "final.pt")
    # The substrate for pooled out-of-fold tail metrics (`src/pool_oof.py`). Predictions are
    # raw-scale so they need no norm_stats to interpret; the logits ride along because AP is
    # the classifier's metric and reconstructing it from the regression output would be a
    # different measurement. Row indices tie both back to CSV rows.
    np.save(out / "val_predictions.npy", val_pred_raw)
    np.save(out / "val_logits.npy", val_logits)
    np.save(out / "val_indices.npy", val_rows)
    # The exported artifact itself, final-epoch, in the same row order as the three above.
    # ~8 MB at [66296, 32] float32. This is what any downstream analysis of the embedding
    # reads, and what makes `emb_effective_rank` checkable after the fact rather than only
    # in the training log.
    np.save(out / "val_embeddings.npy", val_emb.astype(np.float32))

    import graphium
    split_meta = load_meta(args.splits)
    meta = {
        "script": "src/train.py",
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if not k.startswith("_")},
        "device": device,
        "hardware": hardware(),
        "params": counts,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "frozen_baseline": frozen_baseline,
        # The provenance triple. pProp_MLP's runs/ became unreadable because an objective
        # revision and an in-place split regeneration both went unrecorded, so old
        # checkpoints scored new data and printed plausible wrong numbers. Filter on all
        # three before comparing any two runs.
        "objective_version": OBJECTIVE_VERSION,
        "split_sha256": split_meta.get("split_sha256"),
        "input_sha256": split_meta.get("input_sha256"),
        "pprop_edge": PPROP_EDGE,
        "norm_stats": norm_stats,
        "train_ess": ess,
        "history": history,
        "schedule_checks": checks,
        "total_minutes": round((time.time() - started_all) / 60, 2),
        "git_sha": git_sha(),
        "wandb_run": getattr(run, "id", None),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "graphium": graphium.__version__},
        "host": platform.node(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")

    print(f"\nfinal val goal_metric {val_metrics['goal_metric']:.4f} | wrote {out} | "
          f"{meta['total_minutes']:.1f} min")
    if run is not None:
        run.finish()
    # The run directory, not an exit code: `run_config.py` trains a whole grid by calling this
    # in-process and needs to know where each model landed. Failure is signalled by the
    # SystemExit raisings above, never by a return value.
    return out


if __name__ == "__main__":
    main()
