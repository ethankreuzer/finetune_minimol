"""One hyperparameter configuration = one wandb run, with the 5x2 models inside it.

    python src/run_config.py --fold-list 0 1 2 3 4 --seed-list 0 1        # the full grid
    python src/run_config.py --fold-list 0 --seed-list 0 --head-lr 3e-4   # one cheap trial
    python src/run_config.py --aggregate-only outputs/_no_sweep/a1b2c3d4  # log, do not train

WHY THIS EXISTS
---------------
`train.py` trains one model -- one fold, one seed -- and opens its own wandb run. The 5x2
design therefore produced **ten** wandb runs per configuration, separated only by tags, and
nothing on wandb answered the question the project actually asks: which configuration is
best? Comparing two configurations meant eyeballing ten rows against ten other rows, and the
sweep dodged the problem by pinning fold 0 / seed 0, steering the search on one noisy fold.

Here the *configuration* is the unit. One `wandb.init()` holds every model's curves, a table
of their finals, the across-model mean and standard deviation, and the pooled out-of-fold
tail metrics. The runs table gets one row per configuration, sortable by one number.

WHY THE MODELS GO THROUGH DISK
------------------------------
Each model is trained by calling `train.main()` in-process with `--no-wandb`, and its results
are then read back from the files it wrote. Passing them in memory would be shorter and
wrong: `pool_oof.py` scores runs from exactly those files, so an in-memory path would be a
second implementation that could drift from the offline one. Reading them back also makes
`--aggregate-only` fall out for free, which is what makes a 40-minute trial recoverable and
what lets a SLURM array train the ten models in parallel and aggregate afterwards.

Constructing ten trunks in one process is safe *because* `trunk.py` loads MiniMol's config
with `OmegaConf.load` rather than `hydra.initialize` -- see its module docstring. Hydra's
once-per-process singleton would otherwise make this script impossible.

THE OBJECTIVE IS A MEAN, NOT THE POOLED VALUE
---------------------------------------------
`final/goal_metric_mean` -- the mean over models of each model's final-epoch `goal_metric`.
The pooled out-of-fold score is the more honest tail estimate (100 potent molecules against
20 per fold), but it is *undefined* unless a seed has all five folds. The mean is defined for
any subset, so a one-model search trial and a full 5x2 confirmation write the same summary
key and land in the same sortable column. Pooled metrics are logged too, whenever they exist.

**`nan` propagates on purpose** -- `np.mean`, never `np.nanmean`. Pearson is `nan` when a
model's predictions are constant (CLAUDE.md), and that is a broken model, not a missing
measurement. Averaging it away would let four healthy folds carry a configuration that
collapsed on the fifth; letting it through makes the trial score `nan`, which is how a bayes
sweep learns the configuration failed.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train                                                              # noqa: E402
from objective import OBJECTIVE_VERSION                                   # noqa: E402
from pool_oof import load_runs, pool_from_runs                            # noqa: E402
from train import dashed, sweep_int                                       # noqa: E402

# `train.py` flags the driver does not expose at all: it sets `--fold`, `--seed` and `--out`
# per model, and owns the checkpoint decision through `--keep-checkpoints`.
NOT_EXPOSED = {"fold", "seed", "out", "save_checkpoint"}

# Exposed on this CLI but NOT forwarded: the driver is the only thing that talks to wandb,
# and it resolves the output bucket itself. Forwarding `--no-wandb` would be harmless (the
# children always get it) but forwarding the rest would either duplicate a run per model or
# send every model to the same directory.
NOT_FORWARDED = NOT_EXPOSED | {"outputs_root", "sweep_id", "no_wandb", "wandb_project",
                               "wandb_entity", "wandb_group", "wandb_tags"}

# This script's own flags, which are neither forwarded nor part of the configuration's
# identity: they say which slice of the grid to run and where, not what is being trained.
OWN_FLAGS = {"fold_list", "seed_list", "config_id", "keep_checkpoints", "force",
             "aggregate_only", "expect_folds"}

# Excluded from `config_id` on top of the above: these name *where* things live, or how the
# work is executed, not what is trained. Two runs differing only in `--outputs-root` or
# `--num-workers` are the same configuration and belong in the same bucket.
#
# `--subset` is deliberately NOT here. It changes what is trained -- a 3,000-row smoke model
# and a 265,184-row model are not interchangeable -- and since a bucket's models are *reused*
# rather than retrained, excluding it would let a full-data run silently inherit five smoke
# models. Those pool to 15,000 rows, `pool_group` refuses them, and the refusal reads like a
# bug in the pooling rather than a stale bucket. The provenance triple does not catch it
# either: subsetting changes neither `split_sha256` nor `input_sha256`.
#
# The path arguments *are* excluded because the provenance triple does catch them: a different
# `--splits` moves `split_sha256`, a different `--csv` moves `input_sha256`, and `--features`
# re-hashes the CSV on load.
ID_EXCLUDED = NOT_FORWARDED | OWN_FLAGS | {"features", "splits", "csv",
                                           "num_workers", "assert_schedule"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # `--fold-list` / `--seed-list`, not `--folds` / `--seeds`: `train.py` has `--fold` and
    # `--seed`, and argparse resolves unambiguous prefixes, so near-identical names across
    # the two CLIs would let a copy-pasted command mean something different in each.
    p.add_argument("--fold-list", nargs="+", default=["0", "1", "2", "3", "4"],
                   help="folds to train; the 5 validation folds tile the dataset exactly once")
    p.add_argument("--seed-list", nargs="+", default=["0", "1"],
                   help="model seeds; these govern head init, dropout and shuffling only")
    p.add_argument("--config-id", default=None,
                   help="override the derived hash; only for resuming a renamed bucket")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep each model's 34 MB final.pt (off by default: 10 per trial)")
    p.add_argument("--force", action="store_true",
                   help="retrain models that are already complete on disk")
    p.add_argument("--aggregate-only", type=Path, default=None,
                   help="skip training; log the wandb run from this bucket's existing runs")
    p.add_argument("--expect-folds", type=int, default=5,
                   help="folds required before a seed's predictions may be pooled")
    mirror_train_arguments(p)
    args = p.parse_args(dashed(sys.argv[1:] if argv is None else list(argv)))
    args.fold_list = int_list(args.fold_list)
    args.seed_list = int_list(args.seed_list)
    if not args.fold_list or not args.seed_list:
        raise SystemExit("--fold-list and --seed-list must each name at least one value")
    return args


def int_list(tokens):
    """Every integer in `tokens`, however the caller chose to spell the list.

    A wandb agent renders a list-valued sweep parameter into `${args}` as its repr, so
    `fold_list: [0, 1]` can arrive as `--fold_list=[0,` `1]` after shell splitting -- while a
    hand-typed command gives `--fold-list 0 1`. Both are the same list, so the tokens are
    joined and the integers read out of them rather than parsed positionally. This is why the
    argument is `nargs="+"` of plain strings and not `type=sweep_int`: a per-token int() dies
    on `[0,`.
    """
    joined = " ".join(str(t) for t in (tokens if isinstance(tokens, (list, tuple))
                                       else [tokens]))
    return [int(m) for m in re.findall(r"-?\d+", joined)]


def mirror_train_arguments(p):
    """Copy `train.py`'s flags onto this parser, defaults and all.

    Walks `train.build_parser()`'s actions rather than restating them, so a hyperparameter
    added to `train.py` reaches the sweep with no edit here -- which is the whole reason
    `build_parser` is split out over there.

    Defaults are copied rather than suppressed, so every forwarded value is explicit on the
    child command line and lands in that model's `meta.json["argv"]`. The cost is that
    `config_id` moves if a default in `train.py` moves; that is correct, because it is then
    a different configuration.
    """
    group = p.add_argument_group("train.py hyperparameters (forwarded verbatim)")
    for action in train.build_parser()._actions:
        if action.dest in ("help", *NOT_EXPOSED):
            continue
        kwargs = {"dest": action.dest, "default": action.default, "help": action.help}
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            kwargs["action"] = "store_true" if action.const else "store_false"
        else:
            kwargs.update(type=action.type, choices=action.choices, nargs=action.nargs)
        group.add_argument(*action.option_strings, **kwargs)


def config_id(args):
    """A stable 8-char hash of the hyperparameters, and nothing else.

    Same construction as `objective.OBJECTIVE_VERSION`: hash the thing itself, so the name
    re-stamps automatically when the thing changes and two identical configurations always
    name the same bucket -- whether they arrived from a sweep agent or a typed command.
    """
    payload = {k: str(v) for k, v in sorted(vars(args).items())
               if k not in ID_EXCLUDED and not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def boolean_flags():
    """`{dest: (option_string, const)}` for `train.py`'s store_true/store_false flags.

    A boolean's flag name is not derivable from its dest -- `assert_schedule` is set by
    `--no-assert-schedule` -- and the flag is emitted only when the value differs from the
    default, i.e. equals the action's `const`. Read off the parser so a boolean added to
    `train.py` needs no edit here either.
    """
    return {a.dest: (a.option_strings[0], a.const)
            for a in train.build_parser()._actions
            if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction))}


def child_argv(args, fold, seed, out):
    """The `train.py` command line for one model of this configuration."""
    argv = ["--fold", str(fold), "--seed", str(seed), "--out", str(out), "--no-wandb"]
    if not args.keep_checkpoints:
        argv.append("--no-save-checkpoint")
    bools = boolean_flags()
    for key, value in sorted(vars(args).items()):
        if key in NOT_FORWARDED or key in OWN_FLAGS or value is None:
            continue
        if key in bools:
            option, const = bools[key]
            if value == const:
                argv.append(option)
        else:
            argv += ["--" + key.replace("_", "-"), str(value)]
    return argv


def already_done(model_dir):
    """True when this directory holds a finished model that need not be retrained.

    A run is finished only if `meta.json` and all three prediction arrays are present -- a
    crash mid-write leaves the directory looking plausible, and retraining is cheap next to
    silently aggregating a truncated prediction vector.
    """
    needed = ["meta.json", "val_predictions.npy", "val_logits.npy", "val_indices.npy"]
    return all((model_dir / f).exists() for f in needed)


# -- aggregation ---------------------------------------------------------------------

def metric_keys(history_row):
    """The `val/*` metric names in a history row, without the prefix."""
    return sorted(k.split("/", 1)[1] for k in history_row if k.startswith("val/"))


def aggregate(runs_meta, expect_folds, y_all, runs_loaded):
    """Everything the wandb run reports, as plain data.

    Returns `(per_epoch, per_model, final, pooled, pooled_skipped)`. Written to
    `aggregate.json` as well as logged, so `WANDB_MODE=offline` loses nothing and the numbers
    can be checked without wandb in the loop.
    """
    provenance = {(m["objective_version"], m["split_sha256"], m["input_sha256"])
                  for m in runs_meta.values()}
    if len(provenance) > 1:
        raise SystemExit(
            "models disagree on the provenance triple (objective_version, split_sha256, "
            f"input_sha256): {sorted(provenance)}. Averaging them would compare "
            "incomparables -- see the provenance rule in CLAUDE.md.")

    # Per-model: the final epoch, which is the selected model (no early stopping).
    per_model = {}
    for label, meta in sorted(runs_meta.items()):
        last = meta["history"][-1]
        per_model[label] = {
            "fold": meta["config"]["fold"], "seed": meta["config"]["seed"],
            "epochs": last["epoch"], "minutes": meta.get("total_minutes"),
            "peak_gpu_gb": last.get("peak_gpu_gb"),
            **{k.split("/", 1)[1]: v for k, v in last.items() if k.startswith("val/")},
        }

    # Per-epoch mean and std across models. Only epochs every model reached are aggregated:
    # a shorter run would otherwise drag the tail of the curve without it being visible.
    n_epochs = min(len(m["history"]) for m in runs_meta.values())
    keys = metric_keys(runs_meta[next(iter(runs_meta))]["history"][0])
    per_epoch = []
    for i in range(n_epochs):
        rows = [m["history"][i] for m in runs_meta.values()]
        entry = {"epoch": rows[0]["epoch"], "n_models": len(rows)}
        for k in keys:
            vals = [r[f"val/{k}"] for r in rows if isinstance(r.get(f"val/{k}"), (int, float))]
            if not vals:
                continue
            entry[f"{k}_mean"] = float(np.mean(vals))
            entry[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        per_epoch.append(entry)

    # The summary. `goal_metric_mean` is the sweep objective; the spread beside it is what
    # says whether a winning configuration actually won or got a lucky fold.
    final = {}
    for k in keys:
        vals = [m[k] for m in per_model.values() if isinstance(m.get(k), (int, float))]
        if not vals:
            continue
        final[f"{k}_mean"] = float(np.mean(vals))
        final[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        final[f"{k}_min"], final[f"{k}_max"] = float(np.min(vals)), float(np.max(vals))
    final["n_models"] = len(per_model)

    # Pooled out-of-fold tail metrics, only where a seed tiles the dataset exactly once.
    pooled, pooled_skipped = pool_from_runs(runs_loaded, y_all, expect_folds)
    if pooled:
        for k in sorted({k for m in pooled.values() for k in m}):
            vals = [m[k] for m in pooled.values() if isinstance(m.get(k), (int, float))]
            if vals:
                final[f"pooled_{k}_mean"] = float(np.mean(vals))
    return per_epoch, per_model, final, pooled, pooled_skipped


def log_to_wandb(args, bucket, cid, histories, per_epoch, per_model, final,
                 pooled, pooled_skipped):
    """One run. Per-model curves, the aggregate curve, one table, and the summary."""
    import os
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=f"cfg-{cid}", group=args.wandb_group, tags=args.wandb_tags,
                     dir=str(bucket),
                     config={**{k: (str(v) if isinstance(v, Path) else v)
                                for k, v in vars(args).items()
                                if not k.startswith("_") and k not in OWN_FLAGS},
                             "config_id": cid,
                             "fold_list": args.fold_list, "seed_list": args.seed_list,
                             "sweep_id": os.environ.get("WANDB_SWEEP_ID"),
                             "objective_version": OBJECTIVE_VERSION,
                             "n_models": final["n_models"]})
    wandb.define_metric("agg/val/goal_metric_mean", summary="max")

    # One `log` call per epoch carrying every model's value *and* the aggregate, so all the
    # series share a step axis and a single chart can show ten curves against their mean.
    by_epoch = defaultdict(dict)
    for label, history in histories.items():
        for h in history:
            by_epoch[h["epoch"]].update(
                {f"models/{label}/{k}": v for k, v in h.items()
                 if k.startswith("val/") and isinstance(v, (int, float))})
    for entry in per_epoch:
        row = by_epoch[entry["epoch"]]
        row.update({f"agg/val/{k}": v for k, v in entry.items()
                    if k not in ("epoch", "n_models")})
        row["agg/n_models"] = entry["n_models"]
    for epoch in sorted(by_epoch):
        run.log(by_epoch[epoch], step=epoch)

    # Logged once and unstepped: a table logged against a step collides with the
    # step-indexed series above.
    columns = ["model", *sorted(next(iter(per_model.values())))]
    table = wandb.Table(columns=columns)
    for label, m in sorted(per_model.items()):
        table.add_data(label, *[m.get(c) for c in columns[1:]])
    run.log({"models/table": table})

    summary = {f"final/{k}": v for k, v in final.items()}
    for label, m in pooled.items():
        summary.update({f"pooled/{label}/{k}": v for k, v in m.items()
                        if isinstance(v, (int, float))})
    summary["pooled/available"] = bool(pooled)
    if pooled_skipped:
        summary["pooled/skipped"] = json.dumps(pooled_skipped)
    run.summary.update(summary)
    run.finish()
    return run.id


def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    if args.aggregate_only is not None:
        bucket = Path(args.aggregate_only)
        cid = args.config_id or bucket.name
        if not bucket.is_dir():
            raise SystemExit(f"--aggregate-only {bucket} is not a directory")
        print(f"aggregate-only: {bucket}")
    else:
        cid = args.config_id or config_id(args)
        import os
        sweep_id = args.sweep_id or os.environ.get("WANDB_SWEEP_ID")
        bucket = Path(args.outputs_root) / (sweep_id or "_no_sweep") / cid
        bucket.mkdir(parents=True, exist_ok=True)

        pairs = [(f, s) for s in args.seed_list for f in args.fold_list]
        print(f"config {cid}: {len(pairs)} models "
              f"(folds {args.fold_list} x seeds {args.seed_list}) -> {bucket}")
        for n, (fold, seed) in enumerate(pairs, 1):
            out = bucket / f"fold{fold}_seed{seed}"
            if already_done(out) and not args.force:
                print(f"[{n}/{len(pairs)}] fold {fold} seed {seed}: already complete, reusing")
                continue
            print(f"[{n}/{len(pairs)}] fold {fold} seed {seed}", flush=True)
            train.main(child_argv(args, fold, seed, out))

    # Read back what was written. The offline path (`pool_oof.py`) reads the same files, so
    # the two cannot report different numbers for the same directory.
    runs_loaded = load_runs(bucket)
    if not runs_loaded:
        raise SystemExit(f"no finished models under {bucket}")
    runs_meta, histories = {}, {}
    for r in runs_loaded:
        meta = json.loads((r["dir"] / "meta.json").read_text())
        label = f"f{meta['config']['fold']}s{meta['config']['seed']}"
        runs_meta[label] = meta
        histories[label] = meta["history"]

    y_all = pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64)
    per_epoch, per_model, final, pooled, pooled_skipped = aggregate(
        runs_meta, args.expect_folds, y_all, runs_loaded)

    payload = {"config_id": cid, "bucket": str(bucket),
               "objective_version": OBJECTIVE_VERSION,
               "argv": sys.argv, "n_models": len(runs_meta),
               "per_model": per_model, "per_epoch": per_epoch, "final": final,
               "pooled": pooled, "pooled_skipped": pooled_skipped,
               "minutes": round((time.time() - started) / 60, 2)}
    (bucket / "aggregate.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print(f"\n{len(runs_meta)} models | goal_metric "
          f"{final['goal_metric_mean']:.4f} +/- {final['goal_metric_std']:.4f} "
          f"(min {final['goal_metric_min']:.4f}, max {final['goal_metric_max']:.4f})")
    if pooled:
        for label, m in pooled.items():
            print(f"  pooled {label}: goal {m['goal_metric']:.4f} | "
                  f"tail spearman {m['spearman_group_ge_e50']:.4f} | "
                  f"EF@5.0/top1% {m['ef_p5.0_top0.01']:.1f}")
    for label, problems in pooled_skipped.items():
        print(f"  pooled {label}: NOT POOLED -- " + "; ".join(problems))

    if not args.no_wandb:
        run_id = log_to_wandb(args, bucket, cid, histories, per_epoch, per_model, final,
                              pooled, pooled_skipped)
        print(f"wandb run {run_id}")
    print(f"wrote {bucket / 'aggregate.json'}")
    return bucket


if __name__ == "__main__":
    main()
