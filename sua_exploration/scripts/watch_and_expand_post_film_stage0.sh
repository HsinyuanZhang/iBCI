#!/usr/bin/env bash
# Aggregate/expand FiLM, then run decoupled K/V, then use B3T as fallback.
#
# Ordering is intentional:
#   1. user-requested confidence-FiLM;
#   2. user-requested decoupled K/V;
#   3. B3T+T4 efficiency optimization only if neither accuracy mechanism is effective.
#
# This watcher never launches INT8 or formal-test evaluation.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-20}"
FILM_RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
FILM_STAGE0="$FILM_RESULTS/aggregate_m50_seed42.json"
FILM_3SEED="$FILM_RESULTS/aggregate_m50_3seed.json"
KV_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
KV_RUNNER="$ROOT/sua_exploration/scripts/run_sua_decoupled_kv_one_cell.sh"
KV_AGG="$ROOT/sua_exploration/scripts/aggregate_sua_decoupled_kv.py"
B3T_RESULTS="$ROOT/sua_exploration/results/sua_b3t_t4_efficiency_v1"
B3T_RUNNER="$ROOT/sua_exploration/scripts/run_sua_b3t_t4_one_cell.sh"
LOG="$FILM_RESULTS/logs/post_stage0_aggregate_watcher.log"
KV_ARMS=(coupled_t4 kv_e_t4 kv_e_ts4 kv_e_only kv_x_only)

mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    sleep "$POLL_SECONDS"
  done
}

wait_for_kv_seed() {
  local seed="$1"
  for arm in "${KV_ARMS[@]}"; do
    wait_for_file "$KV_RESULTS/${arm}_m50_s${seed}.json"
  done
}

run_kv_arm() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local result="$KV_RESULTS/${arm}_m50_s${seed}.json"
  if [[ -f "$result" ]]; then
    echo "[$(date -Is)] skip existing K/V result $result"
    return
  fi
  echo "[$(date -Is)] launch K/V arm=$arm seed=$seed gpu=$gpu"
  env ARM="$arm" SEED="$seed" GPU="$gpu" "$KV_RUNNER" --launch
}

run_kv_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in "${KV_ARMS[@]}"; do
    run_kv_arm "$arm" "$seed" "$gpu"
  done
}

run_kv_stage0_two_lanes() {
  (
    for arm in coupled_t4 kv_e_t4 kv_e_only; do
      run_kv_arm "$arm" 42 0
    done
  ) &
  local lane0=$!
  (
    for arm in kv_e_ts4 kv_x_only; do
      run_kv_arm "$arm" 42 1
    done
  ) &
  local lane1=$!
  wait "$lane0"
  wait "$lane1"
}

echo "[$(date -Is)] waiting for FiLM Stage-0 aggregate"
wait_for_file "$FILM_STAGE0"

film_effective=false
if jq -e '.budgets["50"].stage0_descriptive_mechanism_pass == true' \
  "$FILM_STAGE0" >/dev/null; then
  echo "[$(date -Is)] FiLM Stage-0 passed; waiting for complete three-seed matrix"
  for seed in 42 43 44; do
    for arm in film t4_continuation confidence_shuffle nofilm_match film_ts4; do
      wait_for_file "$FILM_RESULTS/${arm}_m50_s${seed}.json"
    done
  done
  "$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_confidence_film_t4_budget.py" \
    --result-dir "$FILM_RESULTS" \
    --budgets 50 \
    --seeds 42,43,44 \
    --out "$FILM_3SEED"
  if jq -e '.budgets["50"].formal_effectiveness_pass == true' \
    "$FILM_3SEED" >/dev/null; then
    film_effective=true
  fi
  echo "[$(date -Is)] FiLM three-seed effective=$film_effective"
  # The seed-expansion lanes are now free. Run the independently requested
  # K/V Stage-0 before any quantization decision.
  run_kv_stage0_two_lanes
else
  echo "[$(date -Is)] FiLM Stage-0 failed; K/V Stage-0 is owned by the two lane schedulers"
fi

wait_for_kv_seed 42
"$PY" "$KV_AGG" \
  --result-dir "$KV_RESULTS" \
  --seeds 42 \
  --out "$KV_RESULTS/aggregate_seed42.json"
echo "[$(date -Is)] wrote seed-42 decoupled K/V aggregate"

kv_effective=false
if jq -e '.stage0_descriptive_mechanism_pass == true' \
  "$KV_RESULTS/aggregate_seed42.json" >/dev/null; then
  echo "[$(date -Is)] K/V Stage-0 passed; expanding seeds 43/44"
  run_kv_seed 43 0 &
  pid43=$!
  run_kv_seed 44 1 &
  pid44=$!
  wait "$pid43"
  wait "$pid44"
  "$PY" "$KV_AGG" \
    --result-dir "$KV_RESULTS" \
    --seeds 42,43,44 \
    --out "$KV_RESULTS/aggregate_3seed.json"
  if jq -e '.formal_effectiveness_pass == true' \
    "$KV_RESULTS/aggregate_3seed.json" >/dev/null; then
    kv_effective=true
  fi
  echo "[$(date -Is)] K/V three-seed effective=$kv_effective"
fi

if [[ "$film_effective" == "true" || "$kv_effective" == "true" ]]; then
  echo "[$(date -Is)] verified accuracy mechanism exists; stop before INT8/formal"
  exit 0
fi

echo "[$(date -Is)] no verified accuracy mechanism; running B3T+T4 Stage-0"
(
  for arm in b3t_t4 t4; do
    if [[ ! -f "$B3T_RESULTS/${arm}_s42.json" ]]; then
      env ARM="$arm" SEED=42 GPU=0 "$B3T_RUNNER" --launch
    fi
  done
) &
b3t_lane0=$!
(
  if [[ ! -f "$B3T_RESULTS/b3t_ts4_s42.json" ]]; then
    env ARM=b3t_ts4 SEED=42 GPU=1 "$B3T_RUNNER" --launch
  fi
) &
b3t_lane1=$!
wait "$b3t_lane0"
wait "$b3t_lane1"

for arm in t4 b3t_t4 b3t_ts4; do
  wait_for_file "$B3T_RESULTS/${arm}_s42.json"
done
"$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_b3t_t4_efficiency.py" \
  --result-dir "$B3T_RESULTS" \
  --seeds 42 \
  --out "$B3T_RESULTS/aggregate_seed42.json"
echo "[$(date -Is)] wrote seed-42 B3T+T4 aggregate"

if ! jq -e '.stage0_candidate_for_multiseed_expansion == true' \
  "$B3T_RESULTS/aggregate_seed42.json" >/dev/null; then
  echo "[$(date -Is)] B3T+T4 Stage-0 failed; stop before INT8/formal"
  exit 0
fi

run_b3t_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in b3t_t4 t4 b3t_ts4; do
    if [[ ! -f "$B3T_RESULTS/${arm}_s${seed}.json" ]]; then
      env ARM="$arm" SEED="$seed" GPU="$gpu" "$B3T_RUNNER" --launch
    fi
  done
}
run_b3t_seed 43 0 &
pid43=$!
run_b3t_seed 44 1 &
pid44=$!
wait "$pid43"
wait "$pid44"

"$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_b3t_t4_efficiency.py" \
  --result-dir "$B3T_RESULTS" \
  --seeds 42,43,44 \
  --out "$B3T_RESULTS/aggregate_3seed.json"
echo "[$(date -Is)] wrote complete three-seed B3T+T4 aggregate; stop before INT8/formal"
