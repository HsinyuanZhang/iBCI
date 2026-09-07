#!/usr/bin/env bash
# Inert one-cell runner for A2 matched 2×2 correspondence experiment.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
CONTRACT="$SUA/docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"
SCREEN_ID="a2_matched_correspondence_v1"
AUTH_ENV="A2_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_A2_MATCHED_CORRESPONDENCE_GPU"

CELL="${CELL:?CELL required: within_z4|within_t4|cross_z4|cross_t4}"
SEED="${SEED:?SEED required}"
GPU="${GPU:?GPU index required}"
MODE="${1:---dry-run}"

M_ACTIVITY=30
M_T4=30
EVAL_START=30
TOTAL_EPOCHS=12
BURN_IN=4
RESULTS="$SUA/results/$SCREEN_ID"
DATA_SUBC="$SUA/data/dandi_000688/sub-C"
CACHE="$SUA/cache/dandi688_subc_co_v1"
TEACHER="$SUA/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$SUA/configs/subc_co_27_6_strict_train_val_manifest.json"
EXTERNAL_SCORER="$SUA/scripts/a2_matched_correspondence_score_external.py"

case "$CELL" in
  within_z4|cross_z4) SIDE="z4" ;;
  within_t4|cross_t4) SIDE="t4" ;;
  *) echo "Unsupported CELL=$CELL" >&2; exit 2 ;;
esac

CONTRACT_SHA="$("$PY" - "$CONTRACT" <<'PY'
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
print(h.hexdigest())
PY
)"

NAME="${SCREEN_ID}_${CELL}_dandi688_co_s${SEED}"
RUN_DIR="$SUA/checkpoints/$NAME"
RESULT="$RESULTS/${CELL}_s${SEED}.json"
LOG="$RESULTS/logs/${CELL}_s${SEED}.log"

TRAIN_COMMAND=(
  "$PY" -u "$SUA/scripts/train_variant_dandi688.py"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features "$SIDE"
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --out_name "$NAME"
  --data_dir "$DATA_SUBC"
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

if [[ "$CELL" == within_* ]]; then
  EVAL_COMMAND=(
    "$PY" -u "$SUA/scripts/eval_epoch_window_generic_dandi688.py"
    --run_dir "$RUN_DIR"
    --teacher_ckpt "$TEACHER"
    --data_dir "$DATA_SUBC"
    --cache_dir "$CACHE"
    --train_val_manifest "$MANIFEST"
    --total_epochs "$TOTAL_EPOCHS"
    --burn_in "$BURN_IN"
    --calibration_n "$M_ACTIVITY"
    --pool_size "$EVAL_START"
    --out_path "$RESULT"
  )
else
  EVAL_COMMAND=(
    "$PY" -u "$EXTERNAL_SCORER"
    --run-dir "$RUN_DIR"
    --cell "$CELL"
    --seed "$SEED"
    --out-path "$RESULT"
    --launch
  )
fi

echo "CONTRACT=$CONTRACT"
echo "CONTRACT_SHA256=$CONTRACT_SHA"
echo "SCREEN_ID=$SCREEN_ID"
echo "CELL=$CELL SEED=$SEED GPU=$GPU"
echo "EXPECTED_GPU_CELLS=12 (4 cells x 3 seeds)"
echo "AUTHORIZATION_REQUIRED=${AUTH_ENV}=${AUTH_VALUE}"

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  echo
  if [[ "$CELL" == cross_* ]]; then
    "$PY" "$EXTERNAL_SCORER" --run-dir "$RUN_DIR" --cell "$CELL" --seed "$SEED" --out-path "$RESULT"
  else
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${EVAL_COMMAND[@]}"
    echo
  fi
  exit 0
fi

if [[ "${A2_GPU_AUTHORIZATION:-}" != "$AUTH_VALUE" ]]; then
  echo "Refusing GPU launch: set ${AUTH_ENV}=${AUTH_VALUE}" >&2
  echo "This runner does not have GPU authorization." >&2
  exit 3
fi

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$CONTRACT" ]] || { echo "Missing contract: $CONTRACT" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing existing run dir: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || { echo "Refusing existing result: $RESULT" >&2; exit 1; }
mkdir -p "$RESULTS/logs"
{
  echo "[$(date -Is)] START cell=$CELL seed=$SEED gpu=$GPU contract_sha=$CONTRACT_SHA"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE"
} >"$LOG" 2>&1
