#!/usr/bin/env bash
# S2 of reports/embedding_collapse_experiment.md: replication, covariance, weight-decay null.
#
# 15 new runs in 5 waves of 3, one process per GPU. S1's 24 runs are REUSED, not re-run:
# blocks 2 and 3 pair against A_base/D_w3 at seeds 0-2, which already exist under
# outputs/rank_v1. Wave order puts block 1 first, so an interrupted run still leaves the
# pin decision (does C_w1 collapse again?) answered rather than half-answered.
#
# Same rationale as run_rank_v1.sh for driving train.py directly rather than run_config.py.
set -u
cd /home/ethan2/finetune_minimol

VENV=/home/ethan2/finetune_minimol/.venv/bin/python
ROOT=outputs/rank_v2
LOGS=/home/ethan2/logs
mkdir -p "$LOGS"

# "cell|seed|flags" -- five waves of three, in order.
WAVES=(
 # block 1: replication to n=5
 "A_base|3|--w-vic 0 --vic-gamma 0.5;C_w1|3|--w-vic 1 --vic-gamma 0.5;D_w3|3|--w-vic 3 --vic-gamma 0.5"
 "A_base|4|--w-vic 0 --vic-gamma 0.5;C_w1|4|--w-vic 1 --vic-gamma 0.5;D_w3|4|--w-vic 3 --vic-gamma 0.5"
 # blocks 2 + 3: covariance dose and the wd null, on the seeds S1 already covers
 "I_w3_cov4|0|--w-vic 3 --vic-gamma 0.5 --w-cov 4;J_w3_cov16|0|--w-vic 3 --vic-gamma 0.5 --w-cov 16;K_wd0.1|0|--w-vic 0 --vic-gamma 0.5 --weight-decay 0.1"
 "I_w3_cov4|1|--w-vic 3 --vic-gamma 0.5 --w-cov 4;J_w3_cov16|1|--w-vic 3 --vic-gamma 0.5 --w-cov 16;K_wd0.1|1|--w-vic 0 --vic-gamma 0.5 --weight-decay 0.1"
 "I_w3_cov4|2|--w-vic 3 --vic-gamma 0.5 --w-cov 4;J_w3_cov16|2|--w-vic 3 --vic-gamma 0.5 --w-cov 16;K_wd0.1|2|--w-vic 0 --vic-gamma 0.5 --weight-decay 0.1"
)

echo "=== rank_v2 S2 started $(date -Is) ===" | tee -a "$LOGS/rank_v2_driver.log"
for W in "${!WAVES[@]}"; do
  echo "--- wave $W starting $(date -Is) ---" | tee -a "$LOGS/rank_v2_driver.log"
  IFS=';' read -r -a JOBS <<< "${WAVES[$W]}"
  for G in "${!JOBS[@]}"; do
   ( JOB=${JOBS[$G]}
     NAME=${JOB%%|*}; REST=${JOB#*|}; S=${REST%%|*}; FLAGS=${REST#*|}
     OUT=$ROOT/$NAME/fold0_seed$S
     # Resume on the artifact, not the directory: val_embeddings.npy is written last.
     if [ -f "$OUT/val_embeddings.npy" ]; then
       echo "skip $NAME seed$S (done)" >> "$LOGS/rank_v2_driver.log"; exit 0
     fi
     echo "start $NAME seed$S gpu$G $(date -Is)" >> "$LOGS/rank_v2_driver.log"
     CUDA_VISIBLE_DEVICES=$G $VENV src/train.py --fold 0 --seed "$S" $FLAGS \
       --no-wandb --no-save-checkpoint --out "$OUT" >> "$LOGS/rank_v2_gpu$G.log" 2>&1
     echo "done  $NAME seed$S rc=$? $(date -Is)" >> "$LOGS/rank_v2_driver.log"
   ) &
  done
  wait
  echo "--- wave $W finished $(date -Is) ---" | tee -a "$LOGS/rank_v2_driver.log"
done
echo "=== rank_v2 S2 finished $(date -Is) ===" | tee -a "$LOGS/rank_v2_driver.log"
