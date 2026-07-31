#!/usr/bin/env bash
# Evidence-gated handoff from a failed exact-head oracle to the four-arm
# M_T4=15 label-efficiency experiment.  It expands to seeds 43/44 only after
# the seed-42 mechanism and non-inferiority gates pass.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-30}"
V1_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
ORACLE_RESULTS="$ROOT/sua_exploration/results/sua_t4_head_oracle_v1"
ORACLE_AGGREGATE="$ORACLE_RESULTS/aggregate_seed42.json"
ORACLE_GATE="$ROOT/sua_exploration/scripts/validate_head_oracle_failure_gate.py"
SHRINK_RESULTS="$ROOT/sua_exploration/results/sua_t4_shrinkage_m15_v1"
SHRINK_RUNNER="$ROOT/sua_exploration/scripts/run_sua_t4_shrinkage_one_cell.sh"
SHRINK_AGGREGATOR="$ROOT/sua_exploration/scripts/aggregate_sua_t4_shrinkage.py"
REFERENCE_RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
SHRINK_STAGE0="$SHRINK_RESULTS/aggregate_seed42.json"
SHRINK_3SEED="$SHRINK_RESULTS/aggregate_3seed.json"
LOG="$SHRINK_RESULTS/logs/evidence_gated_post_oracle_watcher.log"
ORACLE_ARMS=(oracle_e_t4 oracle_e_ts4)
SHRINK_ARMS=(t4_m15 ts4_m15 t4w3_m15 ts4w3_m15)

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

run_arm() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local result="$SHRINK_RESULTS/${arm}_s${seed}.json"
  if [[ -f "$result" ]]; then
    echo "[$(date -Is)] skip existing M15 result $result"
    return
  fi
  echo "[$(date -Is)] launch M15 arm=$arm seed=$seed gpu=$gpu"
  env ARM="$arm" SEED="$seed" GPU="$gpu" \
    bash "$SHRINK_RUNNER" --launch
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in "${SHRINK_ARMS[@]}"; do
    run_arm "$arm" "$seed" "$gpu"
  done
}

echo "[$(date -Is)] waiting for complete exact-head seed-42 aggregate"
while true; do
  complete=true
  [[ -f "$ORACLE_AGGREGATE" ]] || complete=false
  for arm in "${ORACLE_ARMS[@]}"; do
    [[ -f "$ORACLE_RESULTS/${arm}_m50_s42.json" ]] || complete=false
  done
  if [[ "$complete" == true ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

if jq -e '.diagnostic_stage0_gates.pass == true' \
  "$ORACLE_AGGREGATE" >/dev/null; then
  echo "[$(date -Is)] exact-head diagnostic passed; stop for head-preserving compression design"
  exit 0
fi

"$PY" "$ORACLE_GATE" \
  --aggregate "$ORACLE_AGGREGATE" \
  --result-dir "$ORACLE_RESULTS" \
  --v1-result-dir "$V1_RESULTS"
echo "[$(date -Is)] strict oracle failure confirmed; launch M15 Stage-0"

(
  run_arm t4_m15 42 0
  run_arm t4w3_m15 42 0
) &
lane0=$!
(
  run_arm ts4_m15 42 1
  run_arm ts4w3_m15 42 1
) &
lane1=$!
wait "$lane0"
wait "$lane1"

"$PY" "$SHRINK_AGGREGATOR" \
  --result-dir "$SHRINK_RESULTS" \
  --reference-dir "$REFERENCE_RESULTS" \
  --seeds 42 \
  --out "$SHRINK_STAGE0"
echo "[$(date -Is)] M15 seed-42 aggregate complete"

if ! jq -e \
  '.stage0_descriptive_mechanism_and_label_reduction_pass == true' \
  "$SHRINK_STAGE0" >/dev/null; then
  echo "[$(date -Is)] M15 Stage-0 failed; stop for result analysis and one optimized round"
  exit 0
fi

echo "[$(date -Is)] M15 Stage-0 passed; expand paired seeds 43/44"
run_seed 43 0 &
seed43=$!
run_seed 44 1 &
seed44=$!
wait "$seed43"
wait "$seed44"

"$PY" "$SHRINK_AGGREGATOR" \
  --result-dir "$SHRINK_RESULTS" \
  --reference-dir "$REFERENCE_RESULTS" \
  --seeds 42,43,44 \
  --out "$SHRINK_3SEED"
echo "[$(date -Is)] M15 three-seed aggregate complete"
echo "[$(date -Is)] stop before formal held-out or INT8"
