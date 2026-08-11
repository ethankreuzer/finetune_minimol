"""One training run: staged fine-tuning of MiniMol on AmpC pProp regression.

A prototype of the real thing, deliberately kept to one fold and one seed. The schedule is
the experiment: the head trains alone on top of the frozen embedding for `--freeze-epochs`,
then the trunk is unfrozen and both train together. The head is given a chance to stop being
random before its gradients are allowed to reach 10M pretrained parameters.

Target is `pprop` (NOTES §1, settled 2026-08-11). Objective is Pearson correlation on the
held-out fold; MSE is the loss and is tracked for both splits.

    python src/train.py --fold 0 --seed 0
    python src/train.py --epochs 2 --freeze-epochs 1 --subset 5000 --no-wandb   # smoke

THE ORDERING TRAP
-----------------
`MiniMolRegressor.param_groups` filters on `p.requires_grad` at *construction* time and
drops a group that comes back empty (model.py:69). So freezing the trunk before building
the optimizer yields a one-group optimizer, and unfreezing later sets `requires_grad = True`
on parameters the optimizer has never heard of -- the trunk would never train, while every
log line still looked healthy. The optimizer is therefore always built *before* the freeze,
and both groups are asserted to exist.

That failure is invisible in the loss curve, so `--assert-schedule` (on by default) also
measures it directly: the trunk must be bit-for-bit unchanged across the frozen phase
*while the head provably moves*, and must change once unfrozen. An unchanged trunk proves
nothing on its own -- it is equally what a broken training loop looks like -- which is why
the check is paired, following `verify_trunk.py::check_excluded_unreachable`.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_features          # noqa: E402
from head import MLPHead                    # noqa: E402
from model import MiniMolRegressor          # noqa: E402
from splits import load_fold                # noqa: E402
from trunk import MiniMolTrunk              # noqa: E402

# The tensor verify_trunk.py uses to prove an optimizer step moves the trunk. Reused here so
# both scripts are watching the same weight.
TRUNK_PROBE = "gnn.layers.0.model.nn.fully_connected.0.linear.weight"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="head init, dropout and shuffling only -- never the partition")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="epochs training the head alone before the trunk is unfrozen")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[1024, 32])
    p.add_argument("--dropout", type=float, default=0.0, help="head dropout")

    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--trunk-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--weights", choices=["uniform", "ipw"], default="uniform",
                   help="per-sample loss weights. 'uniform' is plain MSE; the hook exists "
                        "because a weighting scheme is planned but unchosen (NOTES §1)")
    p.add_argument("--subset", type=int, default=None,
                   help="use only N train and N val rows (smoke test)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir; defaults to outputs/fold{fold}_seed{seed}")
    p.add_argument("--wandb-project", default="finetune_minimol")
    p.add_argument("--wandb-entity", default="ethan_personal")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-assert-schedule", dest="assert_schedule", action="store_false",
                   help="skip the freeze/unfreeze weight-delta assertions (not advised)")
    return p.parse_args(argv)


# -- data ---------------------------------------------------------------------------

class RowDataset(Dataset):
    """`(graph, target, weight, row_index)` for one row of the subset CSV.

    The row index is carried through the batch so validation predictions can be written back
    against the CSV rows they belong to -- which is what makes pooled out-of-fold tail
    metrics possible later without re-running anything.
    """

    def __init__(self, cache, y, w, indices):
        self.cache, self.y, self.w, self.indices = cache, y, w, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = int(self.indices[i])
        return self.cache[row], self.y[row], self.w[row], row


def collate_batch(items):
    """Mirror of `MiniMolTrunk.collate(to_device=False)`, without needing the trunk.

    Deliberately not `trunk.collate`: this runs inside DataLoader workers, and binding it to
    the model would drag a CUDA-resident module into forked children. Nothing here touches
    CUDA -- the move to device happens in the training loop.
    """
    graphs, y, w, idx = zip(*items)
    batch = Batch.from_data_list(list(graphs))
    return ({"features": batch, "batch_indices": batch.batch},
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long))


def build_weights(kind, csv, n):
    if kind == "uniform":
        return np.ones(n, dtype=np.float64)
    ipw = pd.read_csv(csv, usecols=["ipw"])["ipw"].to_numpy(dtype=np.float64)
    if len(ipw) != n:
        raise SystemExit(f"ipw column has {len(ipw)} rows, expected {n}")
    return ipw


# -- loss and metrics ---------------------------------------------------------------

def weighted_mse(pred, target, w):
    """Plain MSE when `w` is uniform; the hook for a weighting scheme when it is not.

    Normalised by `w.sum()` rather than `len(w)`, so the loss scale does not move when a
    weighting scheme is swapped in -- otherwise the learning rate would silently need
    retuning alongside it.
    """
    return ((pred - target) ** 2 * w).sum() / w.sum()


def pearson(pred, target):
    """Pearson r, with the degenerate case reported rather than returned as a bare nan.

    A model predicting a constant has zero variance and Pearson is undefined. That is a real
    possibility here in early epochs -- 36% of rows sit below pProp 1.0, so predicting the
    mean is a genuine local optimum -- and a silent nan in wandb is far harder to diagnose
    than a logged standard deviation next to it.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    std = float(pred.std())
    if std == 0.0 or float(target.std()) == 0.0:
        return float("nan"), std
    return float(pearsonr(pred, target)[0]), std


# -- schedule -----------------------------------------------------------------------

def set_trunk_trainable(model, optimizer, trainable, trunk_lr):
    """Flip the trunk between frozen and training.

    Sets `requires_grad` *and* the trunk group's lr. Either alone would suffice -- AdamW
    skips a parameter whose `.grad` is None -- but doing both makes the state legible in the
    logged learning rate and stops one of them being changed later in isolation.
    """
    for p in model.trunk.parameters():
        p.requires_grad_(trainable)
    for group in optimizer.param_groups:
        if group.get("name") == "trunk":
            group["lr"] = trunk_lr if trainable else 0.0


def probe_weights(model):
    """Copies of one trunk tensor and one head tensor, for the schedule assertions."""
    trunk = dict(model.trunk.named_parameters())[TRUNK_PROBE].detach().clone()
    head = next(p for p in model.head.parameters() if p.dim() > 1).detach().clone()
    return trunk, head


# -- loops --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_w = 0.0
    total_se = 0.0
    preds, targets = [], []
    for batch, y, w, _ in loader:
        batch = model.trunk.to_device(batch, device)
        y, w = y.to(device), w.to(device)

        pred = model(batch)
        loss = weighted_mse(pred, y, w)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_se += float((((pred - y) ** 2) * w).sum())
            total_w += float(w.sum())
            preds.append(pred.detach().float().cpu().numpy())
            targets.append(y.detach().float().cpu().numpy())

    preds, targets = np.concatenate(preds), np.concatenate(targets)
    r, std = pearson(preds, targets)
    return {"mse": total_se / total_w, "pearson": r, "pred_std": std}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_w = 0.0
    total_se = 0.0
    preds, targets, rows = [], [], []
    for batch, y, w, idx in loader:
        batch = model.trunk.to_device(batch, device)
        y, w = y.to(device), w.to(device)
        pred = model(batch)
        total_se += float((((pred - y) ** 2) * w).sum())
        total_w += float(w.sum())
        preds.append(pred.float().cpu().numpy())
        targets.append(y.float().cpu().numpy())
        rows.append(idx.numpy())

    preds, targets, rows = (np.concatenate(x) for x in (preds, targets, rows))
    r, std = pearson(preds, targets)
    return {"mse": total_se / total_w, "pearson": r, "pred_std": std}, preds, rows


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    args = parse_args(argv)
    if args.freeze_epochs > args.epochs:
        raise SystemExit(f"--freeze-epochs {args.freeze_epochs} > --epochs {args.epochs}")

    out = args.out or Path("outputs") / f"fold{args.fold}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # Seeds govern head init, dropout and shuffling. The partition is loaded frozen from
    # disk and is identical across seeds by construction (CLAUDE.md).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cache = load_features(args.features)
    y = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)
    if len(cache) != len(y):
        raise SystemExit(f"feature cache has {len(cache)} graphs but the CSV has {len(y)} "
                         "rows; they are aligned by row position only")
    w = build_weights(args.weights, args.csv, len(y))
    train_idx, val_idx = load_fold(args.splits, fold=args.fold)
    if args.subset:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, min(args.subset, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, min(args.subset, len(val_idx)), replace=False)
    print(f"fold {args.fold}: {len(train_idx):,} train / {len(val_idx):,} val")

    common = dict(batch_size=args.batch_size, collate_fn=collate_batch,
                  num_workers=args.num_workers,
                  persistent_workers=args.num_workers > 0)
    # shuffle=True is load-bearing, not hygiene: ampc_subset_331k.csv is sorted by the
    # target, so an unshuffled loader trains on target-sorted batches (CLAUDE.md footguns).
    train_loader = DataLoader(RowDataset(cache, y, w, train_idx), shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(RowDataset(cache, y, w, val_idx), shuffle=False, **common)

    model = MiniMolRegressor(
        MiniMolTrunk(),
        MLPHead(hidden_dims=tuple(args.hidden_dims), dropout=args.dropout),
    ).to(device)
    counts = model.n_parameters()
    print(f"trunk {counts['trunk']:,} + head {counts['head']:,} = {counts['total']:,} params")

    # Build the optimizer BEFORE freezing -- see the module docstring. Both groups must
    # exist now, because param_groups drops an empty one and never adds it back.
    optimizer = torch.optim.AdamW(
        model.param_groups(trunk_lr=args.trunk_lr, head_lr=args.head_lr,
                           weight_decay=args.weight_decay))
    names = {g.get("name") for g in optimizer.param_groups}
    if names != {"trunk", "head"}:
        raise SystemExit(f"optimizer has groups {names}, expected both 'trunk' and 'head'. "
                         "The trunk was frozen before the optimizer was built.")
    set_trunk_trainable(model, optimizer, trainable=False, trunk_lr=args.trunk_lr)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         dir=str(out), config={**vars(args), "device": device,
                                               "params": counts, "git_sha": git_sha()},
                         settings=wandb.Settings(start_method="thread"))
        wandb.define_metric("val/pearson", summary="max")
        wandb.define_metric("val/mse", summary="min")

    trunk0, head0 = probe_weights(model)
    history, checks = [], {}
    started_all = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_trunk_trainable(model, optimizer, trainable=True, trunk_lr=args.trunk_lr)
            print(f"--- epoch {epoch}: unfreezing trunk (lr {args.trunk_lr}) ---")
        phase = "head_only" if epoch <= args.freeze_epochs else "full"

        started = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va, val_preds, val_rows = evaluate(model, val_loader, device)
        secs = time.time() - started

        lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        row = {"epoch": epoch, "phase": phase, "seconds": round(secs, 1),
               "train/mse": tr["mse"], "train/pearson": tr["pearson"],
               "val/mse": va["mse"], "val/pearson": va["pearson"],
               "val/pred_std": va["pred_std"],
               "lr/trunk": lrs["trunk"], "lr/head": lrs["head"]}
        history.append(row)
        print(f"epoch {epoch} [{phase:9s}] {secs:5.1f}s  "
              f"train mse {tr['mse']:.4f} r {tr['pearson']:.4f}  |  "
              f"val mse {va['mse']:.4f} r {va['pearson']:.4f}", flush=True)
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

        if epoch == args.epochs and args.assert_schedule and args.freeze_epochs < args.epochs:
            t2, _ = probe_weights(model)
            td = float((t2 - trunk0).abs().max())
            checks["after_unfreeze"] = {"trunk_max_delta": td}
            print(f"  [unfreeze check] trunk Δ={td:.3e} (want >0)")
            if td <= 0.0:
                raise SystemExit(
                    "trunk did not change after unfreezing. It is still effectively frozen "
                    "-- check that the optimizer holds a 'trunk' group.")

    torch.save(model.state_dict(), out / "final.pt")
    np.save(out / "val_predictions.npy", val_preds)
    np.save(out / "val_indices.npy", val_rows)

    import graphium
    meta = {
        "script": "src/train.py",
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": device,
        "params": counts,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "history": history,
        "schedule_checks": checks,
        "total_minutes": round((time.time() - started_all) / 60, 2),
        "git_sha": git_sha(),
        "wandb_run": getattr(run, "id", None),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "graphium": graphium.__version__},
        "host": platform.node(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    best = max((h["val/pearson"] for h in history if h["val/pearson"] == h["val/pearson"]),
               default=float("nan"))
    print(f"\nbest val pearson {best:.4f} | wrote {out} | "
          f"{meta['total_minutes']:.1f} min")
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
