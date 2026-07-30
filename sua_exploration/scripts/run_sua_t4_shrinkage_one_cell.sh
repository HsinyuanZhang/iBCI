#!/usr/bin/env bash
# One-cell validation-only runner for the train-audit-selected M_T4=15 shrinkage pilot.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_shrinkage_m15_v1}"
ARM="${ARM:?ARM is required: t4_m15, t4w3_m15, or ts4w3_m15}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
MODE="${1:---dry-run}"

M_ACTIVITY=30
M_T4=15
EVAL_START=50
TOTAL_EPOCHS=12
BURN_IN=4
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"

case "$ARM" in
  t4_m15)
    SIDE_FEATURES="t4"
    ;;
  t4w3_m15)
    SIDE_FEATURES="t4w3"
    ;;
  ts4w3_m15)
    SIDE_FEATURES="ts4w3"
    ;;
  *)
    echo "Unsupported ARM=$ARM" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: ARM=<arm> SEED=<seed> GPU=<index> $0 [--dry-run|--launch]" >&2
  exit 2
fi

NAME="${SCREEN_ID}_${ARM}_dandi688_co_s${SEED}"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${ARM}_s${SEED}.json"
LOG="$LOGS/${ARM}_s${SEED}.log"

TRAIN_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features "$SIDE_FEATURES"
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --out_name "$NAME"
  --data_dir "$DATA"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
  --signal_view sua
  --task CO
  --split_counts 27,6,6
  --max_units_exclusive 100
  --max_epochs "$TOTAL_EPOCHS"
  --no_early_stopping
  --checkpoint_every_epoch
  --lr 1e-4
  --batch_size 32
  --num_workers 4
  --seed "$SEED"
  --loss_mode task_only
  --identity_mode calibrated
  --require_gpu
  --disable_progress_bar
)

EVAL_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py"
  --run_dir "$RUN_DIR"
  --teacher_ckpt "$TEACHER"
  --data_dir "$DATA"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
  --total_epochs "$TOTAL_EPOCHS"
  --burn_in "$BURN_IN"
  --calibration_n "$M_ACTIVITY"
  --pool_size "$EVAL_START"
  --out_path "$RESULT"
)

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  echo
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${EVAL_COMMAND[@]}"
  echo
  exit 0
fi

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "Missing teacher checkpoint: $TEACHER" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing strict manifest: $MANIFEST" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to reuse run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || { echo "Refusing to overwrite result: $RESULT" >&2; exit 1; }

mkdir -p "$LOGS"
{
  echo "[$(date -Is)] START arm=$ARM seed=$SEED gpu=$GPU"
  echo "protocol=M_activity=$M_ACTIVITY; M_T4=$M_T4; evaluation=trials[$EVAL_START:]; strict 27/6; no formal test"
  echo "shrinkage=wiener_strength_3_selected_train_only_nested_loso"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL arm=$ARM seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE arm=$ARM seed=$SEED"
} >"$LOG" 2>&1
