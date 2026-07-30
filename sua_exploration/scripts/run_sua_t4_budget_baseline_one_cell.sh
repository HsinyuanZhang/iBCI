#!/usr/bin/env bash
# One-cell runner for the ordinary concat-T4 calibration-budget anchor.
#
# This deliberately separates activity support (first 30) from the labeled T4
# fit budget (first M_T4) and uses a fixed trial-50 evaluation boundary.  It is
# validation development only; the strict manifest keeps formal-test files
# sealed.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_confidence_film_v1}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
M_T4="${M_T4:-50}"
MODE="${1:---dry-run}"

M_ACTIVITY=30
EVAL_START=50
TOTAL_EPOCHS=12
BURN_IN=4
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
GROUP="t4m${M_T4}"
NAME="${SCREEN_ID}_${GROUP}_dandi688_co_s${SEED}"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${GROUP}_s${SEED}.json"
LOG="$LOGS/${GROUP}_s${SEED}.log"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: SEED=<seed> GPU=<index> M_T4=<budget> $0 [--dry-run|--launch]" >&2
  exit 2
fi
if [[ "$M_T4" -le 0 || "$M_T4" -gt "$EVAL_START" ]]; then
  echo "M_T4 must be in 1..$EVAL_START, got $M_T4" >&2
  exit 2
fi

TRAIN_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features t4
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --out_name "$NAME"
  --data_dir "$DATA"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
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
  echo "[$(date -Is)] START group=$GROUP seed=$SEED gpu=$GPU"
  echo "protocol=M_activity=$M_ACTIVITY; M_T4=$M_T4; evaluation=trials[$EVAL_START:]; strict 27/6; no formal test"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL group=$GROUP seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE group=$GROUP seed=$SEED"
} >"$LOG" 2>&1
