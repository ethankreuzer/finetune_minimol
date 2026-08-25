#!/bin/bash
# One sweep trial. `wandb agent` execs this once per configuration.
#
# WHY A WRAPPER EXISTS AT ALL. Two things a sweep yaml cannot express:
#
#   1. The interpreter path is machine-dependent. sweeps/bayes_v1.yaml used to hardcode
#      /home/ethan2/finetune_minimol/.venv/bin/python, which does not exist on TamIA. A sweep
#      is created once and then served by whatever agents attach to it, so the yaml has to be
#      machine-independent or the same sweep cannot be worked by two clusters.
#   2. The data paths are decided at JOB time, not at sweep-creation time. The sbatch stages
#      the 4.5 GB feature cache into $SLURM_TMPDIR, whose path contains the job id. Only the
#      running job knows it.
#
# ARGUMENT ORDER IS LOad-BEARING. Our flags go BEFORE "$@". The yaml appends
# `--wandb-tags sweep ...` after ${args}, and --wandb-tags is nargs="*", so it must remain the
# last option on the line -- anything after it is swallowed as a tag. Putting our flags first
# also means a sweep that ever did sweep one of them would win, since argparse takes the last
# occurrence.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$REPO/.venv/bin/python}"

[ -x "$VENV" ] || { echo "FATAL: no interpreter at $VENV" >&2; exit 1; }

cd "$REPO"

# Each is applied only if the environment set it, so running this by hand -- or from an agent
# on a machine that stages nothing -- falls back to run_config.py's own defaults.
# --wandb-entity must be passed as a FLAG. Exporting WANDB_ENTITY is NOT enough:
# run_config.py:351 calls wandb.init(entity=args.wandb_entity), and an explicit entity=
# argument overrides the environment variable (verified 2026-08-25). Since --wandb-entity
# still defaults to `ethan_personal`, the env var alone would have left every run trying to
# write into the OLD account -- a permissions failure under the new key.
exec "$VENV" src/run_config.py \
    ${MINIMOL_ENTITY:+--wandb-entity "$MINIMOL_ENTITY"} \
    ${MINIMOL_FEATURES:+--features "$MINIMOL_FEATURES"} \
    ${MINIMOL_SPLITS:+--splits "$MINIMOL_SPLITS"} \
    ${MINIMOL_CSV:+--csv "$MINIMOL_CSV"} \
    ${MINIMOL_WORKERS:+--num-workers "$MINIMOL_WORKERS"} \
    ${MINIMOL_OUTPUTS_ROOT:+--outputs-root "$MINIMOL_OUTPUTS_ROOT"} \
    "$@"
