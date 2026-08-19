#!/usr/bin/env bash
# S2.7 of reports/embedding_collapse_experiment.md: remove the export-point LayerNorm.
#
# Two waves of three. `L_nonorm` runs FIRST because it is the test -- an interrupted run then
# still answers "does removing the norm fix the collapse on its own?", leaving only the
# diagnostic `M_nonorm_w3` (does --w-vic still do anything once the cause is gone?) unrun.
# Controls A_base and D_w3 at seeds 0-2 are reused from outputs/rank_v1, not re-run.
set -u
cd /home/ethan2/finetune_minimol
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
ROOT=outputs/rank_v3
LOGS=/home/ethan2/logs
mkdir -p "$LOGS"

WAVES=(
 "L_nonorm|--w-vic 0 --vic-gamma 0.5 --bottleneck-norm none"
 "M_nonorm_w3|--w-vic 3 --vic-gamma 0.5 --bottleneck-norm none"
)

echo "=== rank_v3 S2.7 started $(date -Is) ===" | tee -a "$LOGS/rank_v3_driver.log"
for W in "${WAVES[@]}"; do
  NAME=${W%%|*}; FLAGS=${W#*|}
  echo "--- $NAME starting $(date -Is) ---" | tee -a "$LOGS/rank_v3_driver.log"
  for S in 0 1 2; do
   ( OUT=$ROOT/$NAME/fold0_seed$S
     if [ -f "$OUT/val_embeddings.npy" ]; then
       echo "skip $NAME seed$S (done)" >> "$LOGS/rank_v3_driver.log"; exit 0
     fi
     echo "start $NAME seed$S gpu$S $(date -Is)" >> "$LOGS/rank_v3_driver.log"
     CUDA_VISIBLE_DEVICES=$S $VENV src/train.py --fold 0 --seed "$S" $FLAGS \
       --no-wandb --no-save-checkpoint --out "$OUT" >> "$LOGS/rank_v3_gpu$S.log" 2>&1
     echo "done  $NAME seed$S rc=$? $(date -Is)" >> "$LOGS/rank_v3_driver.log"
   ) &
  done
  wait
  echo "--- $NAME finished $(date -Is) ---" | tee -a "$LOGS/rank_v3_driver.log"
done
echo "=== rank_v3 S2.7 finished $(date -Is) ===" | tee -a "$LOGS/rank_v3_driver.log"
