#!/usr/bin/env bash
# Wait for the strict SUA FP32 aggregate and immediately launch encoder INT8
# when both T4 contrasts are positive.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
FP32_SCREEN="${FP32_SCREEN:-sua_spint_t4_mainline_fp32_v1}"
INT8_SCREEN="${INT8_SCREEN:-sua_spint_t4_encoder_int8_v1}"
AGGREGATE="$ROOT/sua_exploration/results/$FP32_SCREEN/aggregate.json"
INT8_RESULTS="$ROOT/sua_exploration/results/$INT8_SCREEN"
WATCH_LOG="$INT8_RESULTS/logs/watcher.log"

mkdir -p "$INT8_RESULTS/logs"
if [[ -f "$INT8_RESULTS/pipeline_completed.env" ]]; then
  echo "INT8 pipeline already completed: $INT8_RESULTS" >&2
  exit 0
fi
while [[ ! -f "$AGGREGATE" ]]; do
  echo "[$(date -Is)] waiting for $AGGREGATE" >>"$WATCH_LOG"
  sleep 60
done
echo "[$(date -Is)] FP32 aggregate appeared; launching INT8 gate" >>"$WATCH_LOG"
exec env \
  FP32_SCREEN="$FP32_SCREEN" \
  INT8_SCREEN="$INT8_SCREEN" \
  "$ROOT/sua_exploration/scripts/run_t4_encoder_int8_after_positive.sh"
