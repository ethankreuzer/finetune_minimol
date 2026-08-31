"""Pull the best configurations out of a finished wandb sweep, as runnable `train.py` argv.

Until now, reading a sweep's result was a manual browser step: `INSTRUCTIONS.md` §I says "sort
the wandb runs table by `final/goal_metric_mean`", and nothing in `src/` or `scripts/` calls
`wandb.Api()` at all. That is fine once and hopeless as the front of a pipeline that is meant to
re-run against future sweeps. This script is that front.

One sweep trial is one *configuration* (`run_config.py` trains every (fold, seed) pair and logs
them as a single run), so ranking runs ranks configurations directly.

Two things it is careful about:

**nan is a real outcome, not a missing value.** `run_config.aggregate` guards metric values with
`isinstance(v, (int, float))`, and `nan` passes that test -- so one model with constant
predictions (Pearson is nan) makes the objective nan for its whole trial. Sorting without
dropping those puts nan wherever the comparison happens to land it.

**A `run_config.py` run's config is not `train.py`'s argv.** It carries the driver's own
`fold_list` / `seed_list`, every wandb and path argument, and -- because `config_id` hashes
`str(v)` over the whole namespace -- unset options stringified to the literal `"None"`.
Forwarding `--head-lr-unfrozen None` would set argparse to the *string*. So the config is
filtered against `train.build_parser()` itself rather than against a hand-written list, which
means a hyperparameter added to `train.py` reaches this script with no second edit.

Usage:
    python src/sweep_top.py --sweep models-mila5723/finetune_minimol/lyc0lh2d --top 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train                                                # noqa: E402

OBJECTIVE = "final/goal_metric_mean"
SPREAD = "final/goal_metric_std"

# Dropped from the reconstructed argv even though `train.py` accepts them. Everything here is
# either set per-model by the driver (fold, seed, bootstrap_seed, out), a machine-specific path,
# a wandb destination, or a knob that must not silently ride along from the sweep host.
EXCLUDED = {
    "fold", "seed", "bootstrap_seed", "out", "outputs_root", "sweep_id",
    "wandb_project", "wandb_entity", "wandb_group", "wandb_tags", "no_wandb",
    "features", "splits", "csv", "num_workers", "assert_schedule", "save_checkpoint",
    "subset",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True,
                   help="entity/project/sweep_id, e.g. models-mila5723/finetune_minimol/lyc0lh2d")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="default: outputs/vn_analysis/<sweep_id>/top_configs.json")
    return p.parse_args(argv)


def forwardable():
    """`train.py`'s value-taking options, as `{dest: "--long-option"}`.

    Boolean actions are excluded rather than handled: `store_true`/`store_false` cannot be
    expressed as `--key value`, and every one of them is a machine or bookkeeping concern
    (`--no-wandb`, `--no-save-checkpoint`, `--no-assert-schedule`) that the caller sets itself.
    """
    out = {}
    for a in train.build_parser()._actions:
        if not a.option_strings or a.dest in EXCLUDED:
            continue
        if a.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction", "_HelpAction"):
            continue
        longest = max(a.option_strings, key=len)
        out[a.dest] = longest
    return out


def to_argv(config, options):
    """A wandb run config -> a `train.py` argument list, plus what was dropped and why."""
    argv, dropped = [], {}
    for dest, flag in sorted(options.items()):
        if dest not in config:
            dropped[dest] = "absent from the run config"
            continue
        v = config[dest]
        # The literal string "None" is what an unset option looks like after `config_id`
        # stringified the namespace; a real None is the same thing before it. Both mean
        # "train.py resolves this internally" -- head_lr_unfrozen falls back to head_lr,
        # trunk_weight_decay to weight_decay, bottleneck_norm to head_norm.
        if v is None or v == "None":
            dropped[dest] = "unset; train.py resolves it"
            continue
        argv += [flag, str(v)]
    return argv, dropped


def main(argv=None):
    args = parse_args(argv)
    import wandb

    api = wandb.Api()
    sweep = api.sweep(args.sweep)
    sweep_id = args.sweep.rstrip("/").split("/")[-1]
    print(f"sweep {args.sweep} | state {sweep.state} | {len(sweep.runs)} runs")

    scored, skipped = [], {"unfinished": 0, "no objective": 0, "nan objective": 0}
    for r in sweep.runs:
        if r.state != "finished":
            skipped["unfinished"] += 1
            continue
        v = r.summary.get(OBJECTIVE)
        if v is None:
            skipped["no objective"] += 1
            continue
        v = float(v)
        if math.isnan(v):
            # Not a missing value: one constant-prediction model makes Pearson nan and poisons
            # the whole trial's mean. Worth counting out loud rather than silently sorting.
            skipped["nan objective"] += 1
            continue
        scored.append((v, r))

    print(f"{len(scored)} ranked | skipped: " +
          ", ".join(f"{k} {n}" for k, n in skipped.items() if n))
    if len(scored) < args.top:
        raise SystemExit(f"only {len(scored)} rankable runs, asked for the top {args.top}")

    scored.sort(key=lambda t: t[0], reverse=True)
    options = forwardable()

    ranks = []
    for rank, (score, r) in enumerate(scored[:args.top]):
        cfg = dict(r.config)
        run_argv, dropped = to_argv(cfg, options)
        if cfg.get("subset") not in (None, "None", 0):
            print(f"  !! rank {rank} ran with --subset {cfg['subset']}; it is NOT a full-data "
                  "configuration and retraining it at full size is a different experiment")
        ranks.append({
            "rank": rank,
            "wandb_run_id": r.id,
            "wandb_run_name": r.name,
            "goal_metric_mean": score,
            "goal_metric_std": (float(r.summary.get(SPREAD))
                                if r.summary.get(SPREAD) is not None else None),
            "n_models": r.summary.get("final/n_models"),
            "objective_version": cfg.get("objective_version") or r.summary.get("objective_version"),
            "argv": run_argv,
            "dropped": dropped,
            "config": cfg,
        })

    out = args.out or Path("outputs/vn_analysis") / sweep_id / "top_configs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sweep": args.sweep, "sweep_id": sweep_id,
                               "objective": OBJECTIVE, "skipped": skipped,
                               "n_ranked": len(scored), "ranks": ranks}, indent=2))

    print()
    for e in ranks:
        std = f" +/- {e['goal_metric_std']:.4f}" if e["goal_metric_std"] is not None else ""
        print(f"rank {e['rank']}  {OBJECTIVE} = {e['goal_metric_mean']:.4f}{std}  "
              f"({e['n_models']} models)  {e['wandb_run_name']}")
    # Printed in full so a bad filter is visible here rather than silently retrained 10 times.
    print(f"\nrank 0 argv:\n  {' '.join(ranks[0]['argv'])}")
    if ranks[0]["dropped"]:
        print("  dropped: " + ", ".join(f"{k} ({v})" for k, v in ranks[0]["dropped"].items()))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
