"""Train a head on frozen Mol-JEPA embeddings. One fold, one seed.

The counterpart of `src/train.py`, for a foundation model this branch does **not** fine-tune.
Mol-JEPA runs once, offline, in `src/jepa_embed.py`; this script reads the resulting floats
off disk and trains a head on them. There is no trunk here, no freeze schedule, and no
`--trunk-lr` -- those have no referent when the encoder never moves.

WHY A SEPARATE FILE RATHER THAN A FROZEN TRUNK IN `train.py`

A zero-parameter trunk makes `model.param_groups` (model.py:88-90) drop the empty group,
which trips `train.py:761`'s assertion that the optimizer's groups are exactly
`{trunk, head}`. That assertion is not decoration: it is what catches a silently-untrained
trunk in the MiniMol arm, the failure mode with a healthy-looking loss curve. Widening it to
admit a head-only optimizer would weaken a live guard on code this branch does not change.

WHAT IS SHARED, AND HOW

The point of the branch is to compare encoders, which is only meaningful if nothing else
moves. So the loss, the metrics, the objective, the splits, the normalization and the LR
schedule are not reimplemented here -- they are **imported from the MiniMol arm** and called:

    from losses  import combined_loss, binary_labels, effective_sample_size, PPROP_EDGE
    from train   import RowDataset, fold_weights, score_split, scheduled_lr

`score_split` is the whole reported-metrics path, `combined_loss` is the whole loss, and
`scheduled_lr` is the cosine with the `length - 1` denominator that makes final-epoch
selection valid. If any of them changed, both arms would change together, which is the
property that makes "identical loss" a fact rather than a claim. `verify_metrics.py` still
passing 8/8 is the evidence.

Importing `train` pulls graphium in with it (train.py imports MiniMolTrunk at module scope),
costing a few seconds once per process. That is the price of the identity, and it is worth
it: the alternative is a second copy of `score_split` that can drift.

The schedule is a single cosine over `--epochs`, obtained by handing `scheduled_lr` a config
whose phase 1 is the whole run (`freeze_epochs = epochs`, `unfrozen_epochs = 0`). Same
function, same arithmetic, one phase.

Usage:
    python src/train_jepa.py --fold 0 --seed 0
    python src/train_jepa.py --fold 0 --seed 0 --readout emb:graph+emb:ecfp
    python src/train_jepa.py --epochs 2 --subset 5000 --no-wandb          # smoke
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head import DualHead, TokenTransformerHead                              # noqa: E402
from jepa_features import load_embeddings, load_meta, load_tokens           # noqa: E402
from losses import (PPROP_EDGE, binary_labels, combined_loss,                # noqa: E402
                    effective_sample_size)
from normalization import compute_norm_stats, normalize_pprop               # noqa: E402
from objective import OBJECTIVE_VERSION                                      # noqa: E402
from run_paths import run_dir                                                # noqa: E402
from splits import load_fold, load_meta as load_split_meta                   # noqa: E402
# Imported, not reimplemented -- see WHAT IS SHARED above.
from train import (RowDataset, dashed, fold_weights, git_sha, hardware,      # noqa: E402
                   scheduled_lr, score_split, sweep_int)

ENCODER = "moljepa-v1"


def build_parser():
    """Exposed so `run_config.mirror_train_arguments` can copy every flag onto its own CLI.

    Same contract as `train.build_parser`: a hyperparameter added here reaches the sweep with
    no second edit.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # -- what sits on top of the frozen encoder
    p.add_argument("--head", default="dual", choices=["dual", "token_transformer"],
                   help="`dual` flattens a --readout into one vector and trains an MLP. "
                        "`token_transformer` attends over the 13 tokens with "
                        "--token-source, then pools and runs the same MLP + task heads.")

    # -- data
    p.add_argument("--embeddings", type=Path, default=Path("data/embeddings/moljepa_v1"))
    p.add_argument("--token-source", default=None, choices=["raw", "projected"],
                   help="token_transformer only: `raw` is Mol-JEPA's transformer output, "
                        "`projected` is each token through its own modality_pred head. "
                        "Swept, because the fold-0 scan favoured raw but only for a "
                        "flattened linear readout.")
    p.add_argument("--n-blocks", type=sweep_int, default=2,
                   help="transformer encoder layers over the 13 tokens")
    p.add_argument("--n-heads", type=sweep_int, default=4,
                   help="attention heads per block; must divide the 512-d token width")
    p.add_argument("--dim-feedforward", type=sweep_int, default=2048)
    p.add_argument("--pooling", default="mean", choices=["mean", "max"],
                   help="how the 13 contextualised tokens collapse to one vector")
    p.add_argument("--readout", default="cls",
                   help="which part of Mol-JEPA is the encoder. 'cls' is a PLACEHOLDER that "
                        "lets the pipeline be smoke-tested, not a settled design choice -- "
                        "pin it explicitly in the sweep yaml. Grammar: cls, emb:<token>, "
                        "proj:<token>, emb:*, proj:*, joined with '+'. See jepa_features.py. "
                        "Ignored when --head is token_transformer.")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--subset", type=int, default=None,
                   help="train/validate on N rows each (smoke test)")

    # -- head shape. These are the axes the MiniMol sweep marked "blocked on the
    #    architecture question"; with no trunk to tune they are what is left to sweep.
    p.add_argument("--n-layers", type=int, default=0)
    p.add_argument("--hidden-dim", type=sweep_int, default=1024)
    p.add_argument("--embed-dim", type=sweep_int, default=1024)
    p.add_argument("--cls-hidden-dim", type=sweep_int, default=256)
    p.add_argument("--cls-n-layers", type=int, default=0)
    p.add_argument("--reg-hidden-dim", type=sweep_int, default=256)
    p.add_argument("--reg-n-layers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--head-norm", default="layer",
                   help="'layer', 'batch' or 'none'")
    p.add_argument("--bottleneck-norm", default=None)

    # -- optimisation
    p.add_argument("--epochs", type=sweep_int, default=20,
                   help="a single phase, so a single cosine -- there is no freeze boundary")
    p.add_argument("--batch-size", type=sweep_int, default=1200,
                   help="1200 as in the MiniMol arm: ~11.4 positives per batch at the "
                        "pProp 3.5 edge. Moving it would confound every LR axis.")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eta-min", type=float, default=1e-8)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "none"])

    # -- loss. Identical to train.py:218-221, and pinned in the sweep for the same reason.
    p.add_argument("--w-cls", type=float, default=0.4418)
    p.add_argument("--w-pair", type=float, default=7.486)
    p.add_argument("--w-std", type=float, default=0.7911)
    p.add_argument("--huber-delta", type=float, default=1.0513)
    p.add_argument("--w-vic", type=float, default=0.0)
    p.add_argument("--vic-gamma", type=float, default=1.0)
    p.add_argument("--w-cov", type=float, default=1.0)
    p.add_argument("--weights", default="balanced", choices=["uniform", "balanced"])
    p.add_argument("--pprop-norm", default="zscore")

    # -- plumbing
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 by default, unlike train.py: a batch here is a slice of an "
                        "in-memory float32 matrix, so a worker process would cost more to "
                        "feed than it saves")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--outputs-root", type=Path, default=Path("outputs/jepa_v1"))
    p.add_argument("--sweep-id", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="finetune_minimol")
    p.add_argument("--wandb-entity", default="ethan_personal")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint",
                   action="store_false", default=True)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p


def collate_rows(items):
    """`RowDataset` items -> tensors. The 'batch' is a slice of a float32 matrix.

    `train.py`'s `collate_batch` builds a PyG `Batch` because its cache holds graphs; here
    the cache is `[N, D]`, so this is the whole of it.
    """
    x = torch.from_numpy(np.stack([i[0] for i in items])).float()
    y_norm = torch.tensor([i[1] for i in items], dtype=torch.float32)
    y_bin = torch.tensor([i[2] for i in items], dtype=torch.float32)
    w = torch.tensor([i[3] for i in items], dtype=torch.float32)
    rows = torch.tensor([i[4] for i in items], dtype=torch.long)
    return x, y_norm, y_bin, w, rows


def train_one_epoch(head, loader, optimizer, device, cfg):
    """One pass over the training fold. Returns the mean of each loss term, unscaled.

    Unscaled for the same reason train.py reports them that way: a term that has collapsed
    and a term whose weight is tiny look identical once multiplied.
    """
    head.train()
    totals = {"loss": 0.0, "cls": 0.0, "huber": 0.0, "pair": 0.0, "std": 0.0, "vic": 0.0}
    n_seen = 0
    for x, y_norm, y_bin, w, _ in loader:
        x = x.to(device, non_blocking=True)
        y_norm, y_bin, w = y_norm.to(device), y_bin.to(device), w.to(device)
        logits, pred, z = head.forward_with_embedding(x)
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
def predict(head, loader, device):
    """Normalized predictions, logits, row indices, and the `[N, embed_dim]` bottleneck."""
    head.eval()
    preds, logits, rows, embs = [], [], [], []
    for x, _, _, _, idx in loader:
        lo, pr, z = head.forward_with_embedding(x.to(device, non_blocking=True))
        preds.append(pr.float().cpu().numpy())
        logits.append(lo.float().cpu().numpy())
        rows.append(idx.numpy())
        embs.append(z.float().cpu().numpy())
    return (np.concatenate(preds), np.concatenate(logits), np.concatenate(rows),
            np.concatenate(embs))


def main(argv=None):
    args = build_parser().parse_args(dashed(argv if argv is not None else sys.argv[1:]))
    started_all = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | objective {OBJECTIVE_VERSION} | encoder {ENCODER}")

    # The two heads read the cache in different shapes, and silently feeding one the other's
    # input is exactly the failure this repo spends its guards on -- a flattened 6656-vector
    # and a 13x512 sequence are both "the cache", and only one is what a given head means.
    # So the combination is validated up front rather than left to a shape error deep in a
    # forward pass, or worse, to a broadcast that happens to work.
    if args.head == "token_transformer" and args.token_source is None:
        raise SystemExit("--head token_transformer needs --token-source {raw,projected}")
    if args.head != "token_transformer" and args.token_source is not None:
        raise SystemExit(f"--token-source is meaningless for --head {args.head}: that head "
                         "consumes a flattened --readout. Pick one.")

    emb_meta = load_meta(args.embeddings)
    y = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)
    token_names = None
    if args.head == "token_transformer":
        X, token_names = load_tokens(args.embeddings, args.token_source)
        in_dim, n_tokens = X.shape[2], X.shape[1]
        print(f"tokens {args.token_source!r} -> [{len(X):,}, {n_tokens}, {in_dim}] "
              f"({X.nbytes / 1e9:.2f} GB in memory)")
        print(f"  {token_names}")
    else:
        X = load_embeddings(args.embeddings, args.readout)
        in_dim, n_tokens = X.shape[1], None
        print(f"readout {args.readout!r} -> [{len(X):,}, {in_dim}] "
              f"({X.nbytes / 1e9:.2f} GB in memory)")
    if len(X) != len(y):
        raise SystemExit(f"embedding cache has {len(X)} rows but the CSV has {len(y)}; "
                         "they are aligned by row position only")

    train_idx, val_idx = load_fold(args.splits, fold=args.fold)
    if args.subset:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, min(args.subset, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, min(args.subset, len(val_idx)), replace=False)
    print(f"fold {args.fold}: {len(train_idx):,} train / {len(val_idx):,} val | "
          f"{int(binary_labels(y[train_idx]).sum()):,} / "
          f"{int(binary_labels(y[val_idx]).sum()):,} positive at pProp >= {PPROP_EDGE}")

    # Training fold only, so the validation fold never leaks into the transform.
    norm_stats = compute_norm_stats(y[train_idx], args.pprop_norm)
    args._norm_stats = norm_stats
    y_norm = normalize_pprop(y, norm_stats)
    y_bin = binary_labels(y, PPROP_EDGE).astype(np.float64)
    w = fold_weights(args.weights, y, train_idx, val_idx)
    ess = effective_sample_size(w[train_idx])
    print(f"weights={args.weights}: train ESS {ess:,.0f} "
          f"({100 * ess / len(train_idx):.2f}% of the fold) | norm={args.pprop_norm}")

    common = dict(batch_size=args.batch_size, collate_fn=collate_rows,
                  num_workers=args.num_workers)
    # shuffle=True is load-bearing, not hygiene: ampc_subset_331k.csv is sorted by the
    # target, so an unshuffled loader trains on target-sorted batches (CLAUDE.md footguns).
    train_loader = DataLoader(RowDataset(X, y_norm, y_bin, w, train_idx), shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(RowDataset(X, y_norm, y_bin, w, val_idx), shuffle=False,
                            **common)

    shared = dict(in_dim=in_dim, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                  cls_hidden_dim=args.cls_hidden_dim, cls_n_layers=args.cls_n_layers,
                  reg_hidden_dim=args.reg_hidden_dim, reg_n_layers=args.reg_n_layers,
                  dropout=args.dropout, norm=args.head_norm,
                  bottleneck_norm=args.bottleneck_norm)
    if args.head == "token_transformer":
        head = TokenTransformerHead(n_tokens=n_tokens, n_blocks=args.n_blocks,
                                    n_heads=args.n_heads,
                                    dim_feedforward=args.dim_feedforward,
                                    pooling=args.pooling, **shared).to(device)
        shape = (f"{n_tokens}x{in_dim} -> {args.n_blocks} blocks x {args.n_heads} heads "
                 f"(ff {args.dim_feedforward}) -> {args.pooling} pool -> {args.embed_dim}")
    else:
        head = DualHead(n_layers=args.n_layers, **shared).to(device)
        shape = f"{in_dim} -> {args.embed_dim}"
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head [{args.head}]: {shape} -> 2 heads | {n_params:,} params")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr,
                                  weight_decay=args.weight_decay)

    # One phase, so one cosine: `scheduled_lr` is reused unchanged by telling it phase 1 is
    # the whole run. That keeps the `length - 1` denominator -- the detail that lands the
    # final epoch exactly on `eta_min` and makes final-epoch selection a settled model
    # rather than an arbitrary point on a moving trajectory.
    sched_cfg = SimpleNamespace(freeze_epochs=args.epochs, unfrozen_epochs=0,
                                eta_min=args.eta_min, lr_schedule=args.lr_schedule)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, tags=args.wandb_tags,
                         config={**{k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()
                                    if not k.startswith("_")},
                                 "encoder": ENCODER,
                                 "objective_version": OBJECTIVE_VERSION})
        wandb.define_metric("val/mse", summary="min")
        wandb.define_metric("val/goal_metric", summary="max")

    history = []
    for epoch in range(1, args.epochs + 1):
        lr = scheduled_lr(args.head_lr, epoch, sched_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        train_terms = train_one_epoch(head, train_loader, optimizer, device, args)
        val_pred, val_logits, val_rows, val_emb = predict(head, val_loader, device)
        val_metrics, val_pred_raw = score_split(val_pred, val_logits, val_rows,
                                                y[val_rows], w, y_norm, args,
                                                embedding=val_emb)
        row = {"epoch": epoch, "lr": lr, "seconds": round(time.time() - t0, 1),
               **{f"train/{k}": v for k, v in train_terms.items()},
               **{f"val/{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch {epoch:>3}/{args.epochs}  lr {lr:.2e}  "
              f"train loss {train_terms['loss']:.4f}  "
              f"val mse {val_metrics['mse']:.4f}  "
              f"val goal {val_metrics['goal_metric']:.4f}  "
              f"({row['seconds']:.1f}s)", flush=True)
        if run is not None:
            run.log(row, step=epoch)

    out = Path(args.out) if args.out else run_dir(args.outputs_root, args.sweep_id,
                                                  f"fold{args.fold}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "val_predictions.npy", val_pred_raw)
    np.save(out / "val_logits.npy", val_logits)
    np.save(out / "val_indices.npy", val_rows)
    np.save(out / "val_embeddings.npy", val_emb.astype(np.float32))
    if args.save_checkpoint:
        torch.save({"head_state": head.state_dict(), "in_dim": in_dim,
                    "head": args.head, "readout": args.readout,
                    "token_source": args.token_source, "encoder": ENCODER,
                    "hf_revision": emb_meta.get("hf_revision"),
                    "config": {k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items() if not k.startswith("_")},
                    "objective_version": OBJECTIVE_VERSION,
                    "val_goal_metric": val_metrics.get("goal_metric")},
                   out / "final.pt")

    split_meta = load_split_meta(args.splits)
    meta = {
        "script": "src/train_jepa.py",
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if not k.startswith("_")},
        "device": device,
        "hardware": hardware(),
        "params": {"head": n_params, "trunk": 0, "total": n_params},
        "n_train": len(train_idx), "n_val": len(val_idx),
        "val_is_in_sample": False,
        "frozen_baseline": True,
        # The provenance triple, plus the two fields that separate this arm from the MiniMol
        # one. The triple alone cannot: same CSV, same splits, same objective by design, so a
        # MiniMol run and a Mol-JEPA run agree on all three. `run_config.aggregate` extends
        # its check with `encoder` for exactly this reason.
        "objective_version": OBJECTIVE_VERSION,
        "split_sha256": split_meta.get("split_sha256"),
        "input_sha256": split_meta.get("input_sha256"),
        "encoder": ENCODER,
        "head": args.head,
        "readout": args.readout if args.head != "token_transformer" else None,
        "token_source": args.token_source,
        "token_names": token_names,
        "in_dim": in_dim,
        "hf_repo": emb_meta.get("hf_repo"),
        "hf_revision": emb_meta.get("hf_revision"),
        "embeddings_sha256": {k: v["sha256"] for k, v in emb_meta["arrays"].items()},
        "pprop_edge": PPROP_EDGE,
        "norm_stats": norm_stats,
        "train_ess": ess,
        "history": history,
        "total_minutes": round((time.time() - started_all) / 60, 2),
        "git_sha": git_sha(),
        "wandb_run": getattr(run, "id", None),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "host": platform.node(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")

    print(f"\nfinal val goal_metric {val_metrics['goal_metric']:.4f} | wrote {out} | "
          f"{meta['total_minutes']:.1f} min")
    if run is not None:
        run.finish()
    # The run directory, not an exit code -- run_config.py trains a grid by calling this
    # in-process and needs to know where each model landed.
    return out


if __name__ == "__main__":
    main()
