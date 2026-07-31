#!/usr/bin/env bash
# One-cell strict validation runner for the exact-head K/V oracle.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_head_oracle_v1}"
ARM="${ARM:?ARM is required}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
MODE="${1:---dry-run}"
M_ACTIVITY=30
M_T4=50
EVAL_START=50
TOTAL_EPOCHS=12
BURN_IN=4

TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
V1_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v1"
V2_RESULTS="$ROOT/sua_exploration/results/sua_t4_decoupled_kv_v2"
V2_AGGREGATE="$V2_RESULTS/aggregate_seed42.json"
V2_GATE="$ROOT/sua_exploration/scripts/validate_v2_decoupled_failure_gate.py"
V2_ARMS=(kv2_e_t4 kv2_e_ts4 kv2_e_only kv2_x_only)

case "$ARM" in
  oracle_e_t4)
    KEY_MODE=e_t4
    PERMUTATION_ARGS=()
    ;;
  oracle_e_ts4)
    KEY_MODE=e_ts4
    PERMUTATION_ARGS=(--oracle_key_permutation_seed "$SEED")
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

NAME="${SCREEN_ID}_${ARM}_m50_dandi688_co_s${SEED}"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${ARM}_m50_s${SEED}.json"
LOG="$LOGS/${ARM}_m50_s${SEED}.log"

TRAIN_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688_head_oracle.py"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features t4
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --decoder_mode coupled
  --oracle_key_mode "$KEY_MODE"
  "${PERMUTATION_ARGS[@]}"
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
  "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_head_oracle_dandi688.py"
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

[[ "$SEED" == "42" ]] || {
  echo "Initial oracle launch is bounded to seed 42" >&2
  exit 1
}
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$TEACHER" ]] || {
  echo "Missing teacher checkpoint: $TEACHER" >&2
  exit 1
}
[[ -f "$MANIFEST" ]] || {
  echo "Missing strict manifest: $MANIFEST" >&2
  exit 1
}
[[ -f "$V2_AGGREGATE" ]] || {
  echo "Refusing oracle launch before v2 seed-42 aggregate" >&2
  exit 1
}
for v2_arm in "${V2_ARMS[@]}"; do
  [[ -f "$V2_RESULTS/${v2_arm}_m50_s42.json" ]] || {
    echo "Refusing oracle launch before v2 result: $v2_arm" >&2
    exit 1
  }
done
"$PY" "$V2_GATE" \
  --aggregate "$V2_AGGREGATE" \
  --result-dir "$V2_RESULTS" \
  --v1-result-dir "$V1_RESULTS" >/dev/null
[[ ! -e "$RUN_DIR" ]] || {
  echo "Refusing to reuse run directory: $RUN_DIR" >&2
  exit 1
}
[[ ! -e "$RESULT" ]] || {
  echo "Refusing to overwrite result: $RESULT" >&2
  exit 1
}

mkdir -p "$LOGS"
{
  echo "[$(date -Is)] START arm=$ARM seed=$SEED gpu=$GPU"
  echo "protocol=exact-head oracle; M_activity=30; M_T4=50; evaluation=trials[50:]; strict 27/6; no formal test"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL arm=$ARM seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE arm=$ARM seed=$SEED"
} >"$LOG" 2>&1
