#!/usr/bin/env bash
# The 5-fold x 2-seed grid as a SLURM array: 10 runs, task id -> (fold, seed).
#
#   sbatch scripts/run_grid.sbatch
#
# Two deliberate departures from /home/ethan2/job.sh:
#
#   --gres=gpu:1, not mps:20. This grid feeds a computational-demand report, and MPS-shared
#   GPUs would make every timing a measurement of contention rather than of the workload.
#   Three concurrent runs on three dedicated GPUs finish in four waves and stay clean.
#
#   The uv venv, not `conda activate my_conda_env`. This project's interpreter now lives at
#   .venv/bin/python (CLAUDE.md: call interpreters by absolute path, never conda activate).
#
# The PATH-sanitisation block below is kept verbatim from job.sh -- sbatch propagates the
# submitting shell's environment, which may carry an activated virtualenv.

#SBATCH --job-name=minimol_grid
#SBATCH --output=/home/ethan2/logs/grid_%A_%a.out
#SBATCH --error=/home/ethan2/logs/grid_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=1000000000:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%3

set -e

REPO=/home/ethan2/finetune_minimol
EPOCHS=${EPOCHS:-4}
FREEZE=${FREEZE:-2}
GROUP=${GROUP:-grid_5fold_2seed}

# 10 runs = folds 0..4 x seeds 0..1. Integer division and modulo keep the mapping obvious
# and make each task independently reproducible from its array index alone.
FOLD=$((SLURM_ARRAY_TASK_ID / 2))
SEED=$((SLURM_ARRAY_TASK_ID % 2))

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> fold=$FOLD seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/home/ethan2/boltz_docking/.venv' | tr '\n' ':' | sed 's/:$//')

cd "$REPO"
"$REPO/.venv/bin/python" src/train.py \
    --fold "$FOLD" --seed "$SEED" \
    --epochs "$EPOCHS" --freeze-epochs "$FREEZE" \
    --wandb-group "$GROUP" --wandb-tags grid fold"$FOLD" seed"$SEED"
