#!/usr/bin/env bash
# S1 of reports/embedding_collapse_experiment.md: the screen.
# Randomized complete block design -- 8 cells x seeds {0,1,2}, fold 0.
#
# Drives train.py directly, one process per GPU, three in flight. NOT run_config.py:
# that trains in-process sequentially, so it would serialize each config's seeds.
# Ordering is seed-major, so stopping early leaves a complete unreplicated block
# rather than three finished cells and five empty ones.
set -u
cd /home/ethan2/finetune_minimol

VENV=/home/ethan2/finetune_minimol/.venv/bin/python
ROOT=outputs/rank_v1
LOGS=/home/ethan2/logs
mkdir -p "$LOGS"

CELLS=(
 "A_base|--w-vic 0    --vic-gamma 0.5"
 "B_w0.3|--w-vic 0.3  --vic-gamma 0.5"
 "C_w1|--w-vic 1      --vic-gamma 0.5"
 "D_w3|--w-vic 3      --vic-gamma 0.5"
 "E_w10|--w-vic 10    --vic-gamma 0.5"
 "F_w3_g1|--w-vic 3   --vic-gamma 1.0"
 "G_w3_drop|--w-vic 3 --vic-gamma 0.5 --dropout 0.2"
 "H_drop|--w-vic 0    --vic-gamma 0.5 --dropout 0.2"
)

echo "=== rank_v1 S1 started $(date -Is) ===" | tee -a "$LOGS/rank_v1_driver.log"
for S in 0 1 2; do
  echo "--- seed block $S starting $(date -Is) ---" | tee -a "$LOGS/rank_v1_driver.log"
  for G in 0 1 2; do
   ( for i in "${!CELLS[@]}"; do
       [ $(( i % 3 )) -eq "$G" ] || continue
       NAME=${CELLS[$i]%%|*}; FLAGS=${CELLS[$i]#*|}
       OUT=$ROOT/$NAME/fold0_seed$S
       # Resume on the artifact, not on the directory: val_embeddings.npy is the last
       # thing written, so its presence means the run finished.
       [ -f "$OUT/val_embeddings.npy" ] && { echo "skip $NAME seed$S (done)" >> "$LOGS/rank_v1_driver.log"; continue; }
       echo "start $NAME seed$S gpu$G $(date -Is)" >> "$LOGS/rank_v1_driver.log"
       CUDA_VISIBLE_DEVICES=$G $VENV src/train.py --fold 0 --seed "$S" $FLAGS \
         --no-wandb --no-save-checkpoint --out "$OUT" >> "$LOGS/rank_v1_gpu$G.log" 2>&1
       echo "done  $NAME seed$S rc=$? $(date -Is)" >> "$LOGS/rank_v1_driver.log"
     done ) &
  done
  wait                                   # one complete seed block at a time
  echo "--- seed block $S finished $(date -Is) ---" | tee -a "$LOGS/rank_v1_driver.log"
done
echo "=== rank_v1 S1 finished $(date -Is) ===" | tee -a "$LOGS/rank_v1_driver.log"
