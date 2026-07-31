#!/usr/bin/env bash
# Evidence-gated handoff from a failed v2 seed-42 screen to the exact-head
# two-arm topology oracle.  A passing v2 exits and leaves its own replication
# scheduler in control.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
V1_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
V2_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v2"
V2_AGGREGATE="$V2_RESULTS/aggregate_seed42.json"
V2_GATE="$ROOT/sua_exploration/scripts/validate_v2_decoupled_failure_gate.py"
ORACLE_RESULTS="$ROOT/sua_exploration/results/sua_t4_head_oracle_v1"
ORACLE_RUNNER="$ROOT/sua_exploration/scripts/run_sua_head_oracle_one_cell.sh"
ORACLE_AGGREGATOR="$ROOT/sua_exploration/scripts/aggregate_sua_head_oracle.py"
ORACLE_AGGREGATE="$ORACLE_RESULTS/aggregate_seed42.json"
LOG="$ORACLE_RESULTS/logs/evidence_gated_post_v2_watcher.log"
V2_ARMS=(kv2_e_t4 kv2_e_ts4 kv2_e_only kv2_x_only)

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] waiting for complete v2 seed-42 screen"
while true; do
  complete=true
  [[ -f "$V2_AGGREGATE" ]] || complete=false
  for arm in "${V2_ARMS[@]}"; do
    [[ -f "$V2_RESULTS/${arm}_m50_s42.json" ]] || complete=false
  done
  if [[ "$complete" == true ]]; then
    break
  fi
  sleep 30
done

if jq -e \
  '.stage0_descriptive_candidate_pass | to_entries | any(.value == true)' \
  "$V2_AGGREGATE" >/dev/null; then
  echo "[$(date -Is)] v2 candidate passed; leave replication to v2 scheduler"
  exit 0
fi

"$PY" "$V2_GATE" \
  --aggregate "$V2_AGGREGATE" \
  --result-dir "$V2_RESULTS" \
  --v1-result-dir "$V1_RESULTS"
echo "[$(date -Is)] strict v2 failure confirmed; launch exact-head oracle"

ARM=oracle_e_t4 SEED=42 GPU=0 \
  bash "$ORACLE_RUNNER" --launch &
lane0=$!
ARM=oracle_e_ts4 SEED=42 GPU=1 \
  bash "$ORACLE_RUNNER" --launch &
lane1=$!
wait "$lane0"
wait "$lane1"

"$PY" "$ORACLE_AGGREGATOR" \
  --result-dir "$ORACLE_RESULTS" \
  --v1-result-dir "$V1_RESULTS" \
  --seeds 42 \
  --out "$ORACLE_AGGREGATE"
echo "[$(date -Is)] exact-head seed-42 aggregate complete"
echo "[$(date -Is)] stop before compression optimization, M15, formal or INT8"
