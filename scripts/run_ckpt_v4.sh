#!/usr/bin/env bash
# S2.8: retrain WITH checkpoints so the fine-tuned trunk's 512-d output can be probed.
#
# Every one of the 45 experiment runs used --no-save-checkpoint, so no fine-tuned trunk
# weights exist on disk. Both configs are re-run because the `vic` term reaches the trunk
# too: each config's bottleneck must be compared against PCA of ITS OWN trunk, not a
# borrowed one. ~34 MB per run, 6 runs.
set -u
cd /home/ethan2/finetune_minimol
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
ROOT=outputs/ckpt_v4
LOGS=/home/ethan2/logs
mkdir -p "$LOGS"

WAVES=(
 "A_base|--w-vic 0 --vic-gamma 0.5"
 "D_w3|--w-vic 3 --vic-gamma 0.5"
)
echo "=== ckpt_v4 started $(date -Is) ===" | tee -a "$LOGS/ckpt_v4_driver.log"
for W in "${WAVES[@]}"; do
  NAME=${W%%|*}; FLAGS=${W#*|}
  echo "--- $NAME starting $(date -Is) ---" | tee -a "$LOGS/ckpt_v4_driver.log"
  for S in 0 1 2; do
   ( OUT=$ROOT/$NAME/fold0_seed$S
     [ -f "$OUT/final.pt" ] && { echo "skip $NAME seed$S" >> "$LOGS/ckpt_v4_driver.log"; exit 0; }
     echo "start $NAME seed$S gpu$S $(date -Is)" >> "$LOGS/ckpt_v4_driver.log"
     CUDA_VISIBLE_DEVICES=$S $VENV src/train.py --fold 0 --seed "$S" $FLAGS \
       --no-wandb --out "$OUT" >> "$LOGS/ckpt_v4_gpu$S.log" 2>&1
     echo "done  $NAME seed$S rc=$? $(date -Is)" >> "$LOGS/ckpt_v4_driver.log"
   ) &
  done
  wait
  echo "--- $NAME finished $(date -Is) ---" | tee -a "$LOGS/ckpt_v4_driver.log"
done
echo "=== ckpt_v4 finished $(date -Is) ===" | tee -a "$LOGS/ckpt_v4_driver.log"
