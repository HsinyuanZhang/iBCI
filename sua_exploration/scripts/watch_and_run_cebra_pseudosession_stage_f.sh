#!/usr/bin/env bash
# Conditional seed-43/44 continuation for the frozen SUA pseudo-session study.
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
RESULT="$SUA/results/cebra_pseudosession_sua_v1"
STAGE_P="$RESULT/stage_p_seed42_aggregate.json"
SCHEDULE_PREFLIGHT="$SUA/scripts/preflight_cebra_pseudosession_sua_stage_f_seed.py"
EXECUTOR="$SUA/scripts/execute_cebra_pseudosession_sua_cell.py"
SCORER="$SUA/scripts/score_cebra_pseudosession_sua.py"
AGGREGATOR="$SUA/scripts/aggregate_cebra_pseudosession_sua_terminal.py"
LOG="$RESULT/stage_f_bridge.log"

export PYTHONNOUSERSITE=1
export PYTHONPATH=

if [[ "${1:-}" != "--execute" ]]; then
  "$PYTHON" - <<PY
import json
print(json.dumps({
  "status": "DRY_RUN__NO_DATA_NO_GPU",
  "waits_for": "$STAGE_P",
  "stage_f_seeds": [43, 44],
  "fresh_source_cells": 4,
  "score_receipts": 8,
  "terminal_aggregate": "$RESULT/terminal_three_seed_aggregate.json",
}, indent=2, sort_keys=True))
PY
  exit 0
fi

mkdir -p "$RESULT"
if [[ -e "$LOG" ]]; then
  echo "refusing to reuse Stage-F bridge log: $LOG" >&2
  exit 2
fi
exec > >(tee "$LOG") 2>&1

echo "waiting for immutable Stage-P aggregate: $STAGE_P"
while [[ ! -f "$STAGE_P" || ! -f "$STAGE_P.sha256" ]]; do
  sleep 60
done

# Each command below validates Stage-P's exact gates itself.  A negative or
# tampered Stage-P receipt therefore stops before source data or CUDA access.
for seed in 43 44; do
  "$PYTHON" "$SCHEDULE_PREFLIGHT" \
    --seed "$seed" \
    --output "$RESULT/stage_f_seed${seed}_source_schedule_preflight.json" \
    --execute
done

run_source_pair() {
  local seed="$1"
  "$PYTHON" "$EXECUTOR" --arm t4 --seed "$seed" --gpu 0 --execute &
  local pid_t4=$!
  "$PYTHON" "$EXECUTOR" --arm z4 --seed "$seed" --gpu 1 --execute &
  local pid_z4=$!
  wait "$pid_t4"
  wait "$pid_z4"
}

run_score_pair() {
  local seed="$1"
  (
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCORER" \
      --arm t4 --seed "$seed" --domain within_subject --device cuda:0 --execute
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCORER" \
      --arm t4 --seed "$seed" --domain external_subject_M --device cuda:0 --execute
  ) &
  local pid_t4=$!
  (
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCORER" \
      --arm z4 --seed "$seed" --domain within_subject --device cuda:0 --execute
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCORER" \
      --arm z4 --seed "$seed" --domain external_subject_M --device cuda:0 --execute
  ) &
  local pid_z4=$!
  wait "$pid_t4"
  wait "$pid_z4"
}

# Keep both 3090s on fresh source cells.  No result-dependent choice occurs
# between the two predeclared confirmatory seeds.
run_source_pair 43
run_source_pair 44

run_score_pair 43
run_score_pair 44

"$PYTHON" "$AGGREGATOR" \
  --output "$RESULT/terminal_three_seed_aggregate.json" \
  --execute

echo "CEBRA pseudo-session terminal chain complete"
