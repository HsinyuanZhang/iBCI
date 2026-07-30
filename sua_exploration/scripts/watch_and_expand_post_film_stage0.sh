#!/usr/bin/env bash
# Aggregate the conditional post-FiLM branch and expand only a positive pilot.
#
# This script never launches INT8 or formal-test evaluation.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-20}"
FILM_RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
FILM_STAGE0="$FILM_RESULTS/aggregate_m50_seed42.json"
B3T_RESULTS="$ROOT/sua_exploration/results/sua_b3t_t4_efficiency_v1"
B3T_RUNNER="$ROOT/sua_exploration/scripts/run_sua_b3t_t4_one_cell.sh"
LOG="$FILM_RESULTS/logs/post_stage0_aggregate_watcher.log"

mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1
echo "[$(date -Is)] waiting for FiLM Stage-0 aggregate"
while [[ ! -f "$FILM_STAGE0" ]]; do
  sleep "$POLL_SECONDS"
done

if jq -e '.budgets["50"].stage0_descriptive_mechanism_pass == true' \
  "$FILM_STAGE0" >/dev/null; then
  echo "[$(date -Is)] waiting for complete three-seed FiLM matrix"
  for seed in 42 43 44; do
    for arm in film t4_continuation confidence_shuffle nofilm_match film_ts4; do
      while [[ ! -f "$FILM_RESULTS/${arm}_m50_s${seed}.json" ]]; do
        sleep "$POLL_SECONDS"
      done
    done
  done
  "$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_confidence_film_t4_budget.py" \
    --result-dir "$FILM_RESULTS" \
    --budgets 50 \
    --seeds 42,43,44 \
    --out "$FILM_RESULTS/aggregate_m50_3seed.json"
  echo "[$(date -Is)] wrote complete three-seed FiLM aggregate"
  exit 0
fi

echo "[$(date -Is)] FiLM pilot failed; waiting for seed-42 B3T+T4 matrix"
for arm in t4 b3t_t4 b3t_ts4; do
  while [[ ! -f "$B3T_RESULTS/${arm}_s42.json" ]]; do
    sleep "$POLL_SECONDS"
  done
done
"$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_b3t_t4_efficiency.py" \
  --result-dir "$B3T_RESULTS" \
  --seeds 42 \
  --out "$B3T_RESULTS/aggregate_seed42.json"
echo "[$(date -Is)] wrote seed-42 B3T+T4 aggregate"

if ! jq -e '.stage0_candidate_for_multiseed_expansion == true' \
  "$B3T_RESULTS/aggregate_seed42.json" >/dev/null; then
  echo "[$(date -Is)] B3T+T4 pilot failed; stopping branch without extra seeds"
  exit 0
fi

echo "[$(date -Is)] B3T+T4 pilot passed; expanding seeds 43/44"
run_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in b3t_t4 t4 b3t_ts4; do
    env ARM="$arm" SEED="$seed" GPU="$gpu" "$B3T_RUNNER" --launch
  done
}
run_seed 43 0 &
pid43=$!
run_seed 44 1 &
pid44=$!
wait "$pid43"
wait "$pid44"

"$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_b3t_t4_efficiency.py" \
  --result-dir "$B3T_RESULTS" \
  --seeds 42,43,44 \
  --out "$B3T_RESULTS/aggregate_3seed.json"
echo "[$(date -Is)] wrote complete three-seed B3T+T4 aggregate"
