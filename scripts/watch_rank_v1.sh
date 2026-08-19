#!/usr/bin/env bash
# Waits for the S1 driver (scripts/run_rank_v1.sh) to write its completion sentinel,
# then runs the readout and the pre-registered analysis. Deliberately does NOT launch
# the driver: a second concurrent driver would double-run whatever is in flight.
set -u
cd /home/ethan2/finetune_minimol
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
LOG=/home/ethan2/logs/rank_v1_watch.log
DRIVER_LOG=/home/ethan2/logs/rank_v1_driver.log

echo "=== watcher started $(date -Is) ===" >> "$LOG"
while ! grep -q '=== rank_v1 S1 finished' "$DRIVER_LOG"; do
  # bail out if the driver is gone without finishing -- otherwise this polls forever
  if ! pgrep -f 'bash scripts/run_rank_v1.sh' > /dev/null; then
    echo "driver gone without sentinel $(date -Is); analysing what is on disk" >> "$LOG"
    break
  fi
  sleep 120
done
echo "--- driver done, readout starting $(date -Is) ---" >> "$LOG"
$VENV src/emb_readout.py --runs outputs/rank_v1 -o reports/rank_v1_runs.csv >> "$LOG" 2>&1
echo "readout rc=$? $(date -Is)" >> "$LOG"
$VENV src/rank_v1_analysis.py --csv reports/rank_v1_runs.csv -o reports/rank_v1_results.md >> "$LOG" 2>&1
echo "analysis rc=$? $(date -Is)" >> "$LOG"
echo "=== watcher finished $(date -Is) ===" >> "$LOG"
