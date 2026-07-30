#!/usr/bin/env bash
# Matched M2 FP32 mainline: full local SPINT teacher versus all-held-in T4.
#
# The fixed SPINT reference is outputs/streaming_calibration/b0_baseline. Candidate
# encoders reuse that teacher's decoder, train on all seven held-in M2 sessions, and
# receive exactly the chronological first 33 trials at calibration. T4 may use target
# labels from those same 33 trials; TS4 is the unit-permuted label control.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
MUA_ROOT="$ROOT/streaming_calibration_exp"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-m2_spint_t4_mainline_fp32_v1}"
GPU="${GPU:-0}"
MODE="${1:---dry-run}"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
BASELINE="$MUA_ROOT/outputs/streaming_calibration/b0_baseline/metrics_per_session.csv"
SEEDS=(42 43 44)
ARM_NAMES=(t4 ts4)

if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: $(basename "$0") [--dry-run|--launch]" >&2
  exit 2
fi
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$BASELINE" ]] || { echo "Missing local SPINT baseline: $BASELINE" >&2; exit 1; }
[[ -d "$ROOT/SPINT-main/data/000953" ]] || { echo "Missing FALCON M2 data" >&2; exit 1; }

experiment_for() {
  case "$1" in
    t4) echo "b3s_t4_m2_loso_internal" ;;
    ts4) echo "b3s_ts4_m2_loso_internal" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local group="$1" seed="$2"
  local experiment run_id log_file
  experiment="$(experiment_for "$group")"
  run_id="${SCREEN_ID}_${group}_m2"
  log_file="$LOGS/${group}_s${seed}.log"
  local command=(
    "$PY" -u src/train.py
    "experiment=$experiment"
    "run_id=$run_id"
    "seed=$seed"
    "data.validation_protocol=minival"
    "data.loso_fold=null"
    "data.include_heldout_in_fit=false"
    "data.include_heldout_in_test=true"
    "data.random_calibration=false"
    "data.calibration_n_trials=33"
    "trainer.max_epochs=12"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
    "baseline_metrics_path=$BASELINE"
    "require_baseline_validation=true"
  )
  if [[ "$MODE" == "--dry-run" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${command[@]}"
    echo
    return
  fi
  {
    echo "[$(date -Is)] START group=$group seed=$seed gpu=$GPU"
    echo "protocol=M2 all-held-in fit; first-33 chronological calibration; held-out test"
    (
      cd "$MUA_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" \
        MPLCONFIGDIR="/tmp/${SCREEN_ID}_gpu${GPU}" \
        "${command[@]}"
    )
    echo "[$(date -Is)] DONE group=$group seed=$seed"
  } >"$log_file" 2>&1
}

if [[ "$MODE" == "--launch" ]]; then
  mkdir -p "$LOGS"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader >/dev/null
  for seed in "${SEEDS[@]}"; do
    for group in "${ARM_NAMES[@]}"; do
      pattern="$MUA_ROOT/outputs/streaming_calibration/${SCREEN_ID}_${group}_m2_s${seed}_*"
      compgen -G "$pattern" >/dev/null && {
        echo "Refusing to overwrite existing artifact: $pattern" >&2
        exit 1
      }
    done
  done
  {
    echo "screen_id=$SCREEN_ID"
    echo "started_at=$(date -Is)"
    echo "gpu=$GPU"
    echo "baseline=$BASELINE"
    echo "seeds=${SEEDS[*]}"
    echo "groups=${ARM_NAMES[*]}"
    echo "calibration_selection=chronological_first_33"
    echo "training_sessions=all_seven_M2_held_in"
  } >"$RESULTS/manifest.env"
fi

# First complete T4 replication before shuffled controls.
for seed in "${SEEDS[@]}"; do
  run_one t4 "$seed"
done
for seed in "${SEEDS[@]}"; do
  run_one ts4 "$seed"
done

if [[ "$MODE" == "--launch" ]]; then
  (
    cd "$MUA_ROOT"
    "$PY" scripts/aggregate_m2_spint_t4_mainline.py \
      --screen_id "$SCREEN_ID" \
      --baseline "$BASELINE" \
      --out "$RESULTS/aggregate.json"
  ) >"$LOGS/aggregate.log" 2>&1
  {
    echo "status=completed"
    echo "completed_at=$(date -Is)"
    echo "aggregate=$RESULTS/aggregate.json"
  } >"$RESULTS/worker_status.env"
fi
