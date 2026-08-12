#!/usr/bin/env bash
# Sample GPU clocks, temperature and power to CSV while the grid runs.
#
#   scripts/sample_gpu.sh out.csv [interval_seconds]
#
# Purpose: turn "SXM cards cool better than blower workstation cards" from a spec-sheet
# assertion into a measurement. If these 300 W A6000s sag from boost toward base clocks
# under sustained three-GPU load, or assert a thermal/power throttle reason, that is real
# performance TamIA would not lose.
#
# CAVEAT, and it is the whole point of writing this down: the 4-epoch grid runs ~13 minutes,
# which is NOT long enough to conclude anything about sustained-load throttling. This trace
# is a baseline for the deferred longer test. Do not read a trend into 13 minutes, and do
# not let a spec-sheet assumption fill the gap in the report.

set -e
OUT=${1:-gpu_telemetry.csv}
INTERVAL=${2:-5}

echo "sampling every ${INTERVAL}s -> $OUT   (ctrl-C or kill to stop)"
nvidia-smi \
  --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,clocks_throttle_reasons.active \
  --format=csv,nounits \
  -l "$INTERVAL" > "$OUT"
