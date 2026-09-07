#!/usr/bin/env bash
# Run the immutable factorized-logit aggregate as soon as all three arm results exist.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
RESULTS="$ROOT/sua_exploration/results/sua_t4_factorized_logit_residual_v1"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
AGGREGATOR="$ROOT/sua_exploration/scripts/aggregate_sua_t4_factorized_logit.py"
LOG="$RESULTS/logs/finalizer.log"
ARMS=(aligned_logit shuffled_logit additive_control)

mkdir -p "$RESULTS/logs"
echo "[$(date -Is)] waiting for complete arm results" >>"$LOG"

while true; do
  ready=true
  for arm in "${ARMS[@]}"; do
    if [[ ! -s "$RESULTS/${arm}_s42.json" ]]; then
      ready=false
      break
    fi
  done
  if [[ "$ready" == true ]]; then
    break
  fi
  sleep 20
done

echo "[$(date -Is)] all arm results present; starting fail-closed aggregate" >>"$LOG"
"$PY" "$AGGREGATOR" >>"$LOG" 2>&1
echo "[$(date -Is)] aggregate complete" >>"$LOG"
