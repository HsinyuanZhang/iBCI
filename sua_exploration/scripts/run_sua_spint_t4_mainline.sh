#!/usr/bin/env bash
# Matched SUA FP32 mainline: original SPINT ID encoder (B0) versus T4 and TS4.
#
# Every arm uses the same 27/6 validation-only split, the same chronological first
# 30 calibration trials in training and evaluation, and scores only windows after
# trial 30. Formal test files are never opened.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_spint_t4_mainline_fp32_v1}"
GPU="${GPU:-1}"
MODE="${1:---dry-run}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
BURN_IN="${BURN_IN:-4}"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
SEEDS=(42 43 44)
ARM_NAMES=(b0 t4 ts4)

if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: $(basename "$0") [--dry-run|--launch]" >&2
  exit 2
fi
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -d "$DATA" ]] || { echo "Missing SUA data: $DATA" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "Missing teacher: $TEACHER" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing strict manifest: $MANIFEST" >&2; exit 1; }

group_config() {
  case "$1" in
    b0) echo "B0 none" ;;
    t4) echo "B3S t4" ;;
    ts4) echo "B3S ts4" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local group="$1" seed="$2"
  local variant side name log_file
  read -r variant side <<<"$(group_config "$group")"
  name="${SCREEN_ID}_${group}_dandi688_co_s${seed}"
  log_file="$LOGS/${group}_s${seed}.log"
  local train_command=(
    "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$TEACHER"
    --variant "$variant"
    --side_features "$side"
    --side_feature_pool_size 30
    --calibration_n_trials 30
    --out_name "$name"
    --data_dir "$DATA"
    --cache_dir "$CACHE"
    --train_val_manifest "$MANIFEST"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --max_epochs "$MAX_EPOCHS"
    --no_early_stopping
    --checkpoint_every_epoch
    --lr 1e-4
    --batch_size 32
    --num_workers 4
    --seed "$seed"
    --loss_mode task_only
    --identity_mode calibrated
    --require_gpu
    --disable_progress_bar
  )
  local eval_command=(
    "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py"
    --run_dir "$ROOT/sua_exploration/checkpoints/$name"
    --teacher_ckpt "$TEACHER"
    --data_dir "$DATA"
    --cache_dir "$CACHE"
    --train_val_manifest "$MANIFEST"
    --total_epochs "$MAX_EPOCHS"
    --burn_in "$BURN_IN"
    --calibration_n 30
    --pool_size 30
    --out_path "$RESULTS/${group}_s${seed}.json"
  )
  if [[ "$MODE" == "--dry-run" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${train_command[@]}"
    echo
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${eval_command[@]}"
    echo
    return
  fi
  {
    echo "[$(date -Is)] START group=$group seed=$seed variant=$variant gpu=$GPU"
    echo "protocol=SUA strict 27/6; train first-30; eval first-30/pool-30; no test files"
    CUDA_VISIBLE_DEVICES="$GPU" "${train_command[@]}"
    echo "[$(date -Is)] EVAL group=$group seed=$seed"
    CUDA_VISIBLE_DEVICES="$GPU" "${eval_command[@]}"
    echo "[$(date -Is)] DONE group=$group seed=$seed"
  } >"$log_file" 2>&1
}

if [[ "$MODE" == "--launch" ]]; then
  mkdir -p "$LOGS"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader >/dev/null
  for seed in "${SEEDS[@]}"; do
    for group in "${ARM_NAMES[@]}"; do
      out="$ROOT/sua_exploration/checkpoints/${SCREEN_ID}_${group}_dandi688_co_s${seed}"
      [[ -e "$out" ]] && { echo "Refusing to overwrite: $out" >&2; exit 1; }
    done
  done
  {
    echo "screen_id=$SCREEN_ID"
    echo "started_at=$(date -Is)"
    echo "gpu=$GPU"
    echo "seeds=${SEEDS[*]}"
    echo "groups=${ARM_NAMES[*]}"
    echo "training_activity_calibration_n=30"
    echo "side_feature_pool_size=30"
    echo "evaluation_forward_calibration_n=30"
    echo "evaluation_pool_size=30"
    echo "formal_test_evaluated=false"
  } >"$RESULTS/manifest.env"
fi

# Produce a complete B0/T4/TS4 paired seed before moving to the next seed.
for seed in "${SEEDS[@]}"; do
  for group in "${ARM_NAMES[@]}"; do
    run_one "$group" "$seed"
  done
done

if [[ "$MODE" == "--launch" ]]; then
  "$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_spint_t4_mainline.py" \
    --result_dir "$RESULTS" \
    --out "$RESULTS/aggregate.json" >"$LOGS/aggregate.log" 2>&1
  {
    echo "status=completed"
    echo "completed_at=$(date -Is)"
    echo "aggregate=$RESULTS/aggregate.json"
  } >"$RESULTS/worker_status.env"
fi
