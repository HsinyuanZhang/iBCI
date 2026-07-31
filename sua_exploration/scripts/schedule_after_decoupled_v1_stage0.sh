#!/usr/bin/env bash
# Evidence-gated continuation after all five decoupled K/V v1 seed-42 arms.
#
# A positive v1 mechanism expands matched seeds 43/44.  A failed v1 screen
# launches the representation-preserving teacher-readin v2 diagnostic instead
# of blindly starting M15 shrinkage.  This scheduler never opens formal data,
# launches INT8, or starts M15.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-20}"
V1_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
V1_RUNNER="$ROOT/sua_exploration/scripts/run_sua_decoupled_kv_one_cell.sh"
V1_AGGREGATOR="$ROOT/sua_exploration/scripts/aggregate_sua_decoupled_kv.py"
V1_AGGREGATE="$V1_RESULTS/aggregate_seed42.json"
V2_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v2"
V2_RUNNER="$ROOT/sua_exploration/scripts/run_sua_decoupled_kv_v2_one_cell.sh"
LOG="$V1_RESULTS/logs/evidence_gated_post_v1_watcher.log"
V1_ARMS=(coupled_t4 kv_e_t4 kv_e_ts4 kv_e_only kv_x_only)

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    sleep "$POLL_SECONDS"
  done
}

run_v1_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in "${V1_ARMS[@]}"; do
    local result="$V1_RESULTS/${arm}_m50_s${seed}.json"
    if [[ -f "$result" ]]; then
      echo "[$(date -Is)] skip existing v1 result $result"
      continue
    fi
    echo "[$(date -Is)] launch v1 arm=$arm seed=$seed gpu=$gpu"
    env ARM="$arm" SEED="$seed" GPU="$gpu" "$V1_RUNNER" --launch
  done
}

run_v2_lane0() {
  for arm in kv2_e_t4 kv2_e_only; do
    local result="$V2_RESULTS/${arm}_m50_s42.json"
    if [[ -f "$result" ]]; then
      echo "[$(date -Is)] skip existing v2 result $result"
      continue
    fi
    echo "[$(date -Is)] launch v2 arm=$arm seed=42 gpu=0"
    env ARM="$arm" SEED=42 GPU=0 "$V2_RUNNER" --launch
  done
}

run_v2_lane1() {
  for arm in kv2_e_ts4 kv2_x_only; do
    local result="$V2_RESULTS/${arm}_m50_s42.json"
    if [[ -f "$result" ]]; then
      echo "[$(date -Is)] skip existing v2 result $result"
      continue
    fi
    echo "[$(date -Is)] launch v2 arm=$arm seed=42 gpu=1"
    env ARM="$arm" SEED=42 GPU=1 "$V2_RUNNER" --launch
  done
}

echo "[$(date -Is)] waiting for complete v1 seed-42 five-arm screen"
for arm in "${V1_ARMS[@]}"; do
  wait_for_file "$V1_RESULTS/${arm}_m50_s42.json"
done

"$PY" "$V1_AGGREGATOR" \
  --result-dir "$V1_RESULTS" \
  --seeds 42 \
  --out "$V1_AGGREGATE"
echo "[$(date -Is)] wrote strict v1 seed-42 aggregate"

if jq -e '.stage0_descriptive_mechanism_pass == true' \
  "$V1_AGGREGATE" >/dev/null; then
  echo "[$(date -Is)] v1 Stage-0 passed; expand matched seeds 43/44"
  run_v1_seed 43 0 &
  lane0=$!
  run_v1_seed 44 1 &
  lane1=$!
  wait "$lane0"
  wait "$lane1"
  echo "[$(date -Is)] v1 seed expansion complete; stop before formal/INT8"
else
  echo "[$(date -Is)] v1 Stage-0 failed; run isolated teacher-readin v2"
  run_v2_lane0 &
  lane0=$!
  run_v2_lane1 &
  lane1=$!
  wait "$lane0"
  wait "$lane1"
  echo "[$(date -Is)] v2 seed-42 diagnostic complete; stop before M15/formal/INT8"
fi
