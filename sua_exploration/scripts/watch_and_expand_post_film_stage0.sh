#!/usr/bin/env bash
# Aggregate/expand FiLM, run decoupled K/V, then one evidence-driven
# train-audit-selected low-label shrinkage and residual-only optimization
# before the B3TStream+T4 fallback.
#
# Ordering is intentional:
#   1. user-requested confidence-FiLM;
#   2. user-requested decoupled K/V;
#   3. M_T4=15 uncertainty-shrunk T4 if both requested screens are ineffective;
#   4. residual-only FiLM if the evidence-selected shrinkage is ineffective;
#   5. B3TStream+T4 efficiency only if the optimized rounds are ineffective.
#
# This watcher never launches INT8 or formal-test evaluation.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-20}"
FILM_RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
FILM_STAGE0="$FILM_RESULTS/aggregate_m50_seed42.json"
FILM_3SEED="$FILM_RESULTS/aggregate_m50_3seed.json"
FILM_RUNNER="$ROOT/sua_exploration/scripts/run_sua_confidence_film_one_cell.sh"
KV_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
KV_RUNNER="$ROOT/sua_exploration/scripts/run_sua_decoupled_kv_one_cell.sh"
KV_AGG="$ROOT/sua_exploration/scripts/aggregate_sua_decoupled_kv.py"
SHRINK_RESULTS="$ROOT/sua_exploration/results/sua_t4_shrinkage_m15_v1"
SHRINK_RUNNER="$ROOT/sua_exploration/scripts/run_sua_t4_shrinkage_one_cell.sh"
SHRINK_AGG="$ROOT/sua_exploration/scripts/aggregate_sua_t4_shrinkage.py"
SHRINK_STAGE0="$SHRINK_RESULTS/aggregate_seed42.json"
SHRINK_3SEED="$SHRINK_RESULTS/aggregate_3seed.json"
RESIDUAL_AGG="$ROOT/sua_exploration/scripts/aggregate_sua_residual_film.py"
RESIDUAL_STAGE0="$FILM_RESULTS/aggregate_residual_seed42.json"
RESIDUAL_3SEED="$FILM_RESULTS/aggregate_residual_3seed.json"
B3T_RESULTS="$ROOT/sua_exploration/results/sua_b3t_t4_efficiency_v1"
B3T_RUNNER="$ROOT/sua_exploration/scripts/run_sua_b3t_t4_one_cell.sh"
LOG="$FILM_RESULTS/logs/post_stage0_aggregate_watcher.log"
KV_ARMS=(coupled_t4 kv_e_t4 kv_e_ts4 kv_e_only kv_x_only)
SHRINK_ARMS=(t4_m15 t4w3_m15 ts4w3_m15)

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

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

run_shrink_arm() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local result="$SHRINK_RESULTS/${arm}_s${seed}.json"
  if [[ -f "$result" ]]; then
    echo "[$(date -Is)] skip existing shrinkage result $result"
    return
  fi
  echo "[$(date -Is)] launch shrinkage arm=$arm seed=$seed gpu=$gpu"
  env ARM="$arm" SEED="$seed" GPU="$gpu" "$SHRINK_RUNNER" --launch
}

run_shrink_seed() {
  local seed="$1"
  local gpu="$2"
  for arm in "${SHRINK_ARMS[@]}"; do
    run_shrink_arm "$arm" "$seed" "$gpu"
  done
}

run_film_arm() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local result="$FILM_RESULTS/${arm}_m50_s${seed}.json"
  if [[ -f "$result" ]]; then
    echo "[$(date -Is)] skip existing FiLM result $result"
    return
  fi
  local anchor="$ROOT/sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s${seed}/epoch_ckpts/epoch_011.ckpt"
  local anchor_result="$FILM_RESULTS/t4m50_s${seed}.json"
  wait_for_file "$anchor"
  wait_for_file "$anchor_result"
  echo "[$(date -Is)] launch FiLM arm=$arm seed=$seed gpu=$gpu"
  env ARM="$arm" SEED="$seed" GPU="$gpu" M_T4=50 ANCHOR="$anchor" \
    "$FILM_RUNNER" --launch
}

run_residual_seed() {
  local seed="$1"
  local gpu="$2"
  # Full-confidence FiLM and T4 continuation are required references. They are
  # skipped when already produced by the preceding screen.
  for arm in film t4_continuation residual_film residual_shuffle residual_nofilm; do
    run_film_arm "$arm" "$seed" "$gpu"
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

echo "[$(date -Is)] no verified FiLM/KV mechanism; running M_T4=15 shrinkage Stage-0"
(
  run_shrink_arm t4_m15 42 0
  run_shrink_arm t4w3_m15 42 0
) &
shrink_lane0=$!
(
  run_shrink_arm ts4w3_m15 42 1
) &
shrink_lane1=$!
wait "$shrink_lane0"
wait "$shrink_lane1"

"$PY" "$SHRINK_AGG" \
  --result-dir "$SHRINK_RESULTS" \
  --reference-dir "$FILM_RESULTS" \
  --seeds 42 \
  --out "$SHRINK_STAGE0"
echo "[$(date -Is)] wrote seed-42 M_T4=15 shrinkage aggregate"

shrink_effective=false
if jq -e '.stage0_descriptive_mechanism_and_label_reduction_pass == true' \
  "$SHRINK_STAGE0" >/dev/null; then
  echo "[$(date -Is)] shrinkage Stage-0 passed; expanding seeds 43/44"
  run_shrink_seed 43 0 &
  shrink_pid43=$!
  run_shrink_seed 44 1 &
  shrink_pid44=$!
  wait "$shrink_pid43"
  wait "$shrink_pid44"
  "$PY" "$SHRINK_AGG" \
    --result-dir "$SHRINK_RESULTS" \
    --reference-dir "$FILM_RESULTS" \
    --seeds 42,43,44 \
    --out "$SHRINK_3SEED"
  if jq -e '.formal_effectiveness_pass == true' \
    "$SHRINK_3SEED" >/dev/null; then
    shrink_effective=true
  fi
  echo "[$(date -Is)] shrinkage three-seed effective=$shrink_effective"
fi

if [[ "$shrink_effective" == "true" ]]; then
  echo "[$(date -Is)] verified low-label shrinkage; stop before INT8/formal"
  exit 0
fi

echo "[$(date -Is)] no verified FiLM/KV/shrinkage mechanism; running residual-only FiLM optimization"
(
  run_film_arm residual_film 42 0
  run_film_arm residual_nofilm 42 0
) &
residual_lane0=$!
(
  run_film_arm residual_shuffle 42 1
) &
residual_lane1=$!
wait "$residual_lane0"
wait "$residual_lane1"

"$PY" "$RESIDUAL_AGG" \
  --result-dir "$FILM_RESULTS" \
  --seeds 42 \
  --out "$RESIDUAL_STAGE0"
echo "[$(date -Is)] wrote seed-42 residual-only FiLM aggregate"

residual_effective=false
if jq -e '.stage0_descriptive_mechanism_pass == true' \
  "$RESIDUAL_STAGE0" >/dev/null; then
  echo "[$(date -Is)] residual-only FiLM Stage-0 passed; expanding seeds 43/44"
  run_residual_seed 43 0 &
  residual_pid43=$!
  run_residual_seed 44 1 &
  residual_pid44=$!
  wait "$residual_pid43"
  wait "$residual_pid44"
  "$PY" "$RESIDUAL_AGG" \
    --result-dir "$FILM_RESULTS" \
    --seeds 42,43,44 \
    --out "$RESIDUAL_3SEED"
  if jq -e '.formal_effectiveness_pass == true' \
    "$RESIDUAL_3SEED" >/dev/null; then
    residual_effective=true
  fi
  echo "[$(date -Is)] residual-only FiLM three-seed effective=$residual_effective"
fi

if [[ "$residual_effective" == "true" ]]; then
  echo "[$(date -Is)] verified residual-only FiLM; stop before INT8/formal"
  exit 0
fi

echo "[$(date -Is)] no verified residual mechanism; running B3TStream+T4 Stage-0"
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
