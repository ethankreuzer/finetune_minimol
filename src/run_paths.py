"""Resolve a run id to its directory under `outputs/`.

Ported from `pProp_MLP/src/run_paths.py`. Runs are stored nested by sweep
(`outputs/<sweep_id>/<run_id>/`) so an unwanted sweep is one `rm -rf` away, and so a
single flat directory does not accumulate thousands of entries -- that repo reached 19,600
saved runs, and the flat layout it started with had to be migrated afterwards.

**Never hardcode `outputs/<run_id>`.** Resolve through this function instead: it handles
both the nested layout and the legacy flat one, so a grid run from before the nesting
(`outputs/fold0_seed0/`) still resolves. Run ids are globally unique, so the nested glob
yields at most one match.
"""

from pathlib import Path

DEFAULT_ROOT = "outputs"


def resolve_run_dir(run_id, runs_dir=DEFAULT_ROOT):
    """Return the Path to `runs_dir`'s directory for `run_id`.

    Prefers a nested layout at any depth -- `<runs_dir>/<sweep_id>/<run_id>` and, since
    `run_config.py` buckets a whole grid under one hyperparameter configuration,
    `<runs_dir>/<sweep_id>/<config_id>/<run_id>`. Falls back to the flat `<runs_dir>/<run_id>`.
    Raises FileNotFoundError if none exists.
    """
    runs_dir = Path(runs_dir)
    hits = sorted(p for p in runs_dir.rglob(str(run_id)) if p.is_dir())
    if hits:
        return hits[0]
    flat = runs_dir / run_id                           # pre-nesting fallback
    if flat.is_dir():
        return flat
    raise FileNotFoundError(f"run {run_id!r} not found under {runs_dir}")


def run_dir(root, sweep_id, run_id):
    """The directory a *new* run should write to, created on demand.

    `sweep_id` falls back to `_no_sweep` for manual single runs, matching the source, so
    one-off runs never land at the top level and break the `*/<run_id>` glob above.
    """
    path = Path(root) / (sweep_id or "_no_sweep") / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path
