#!/usr/bin/env bash
# Inert one-cell runner for A10 matched no-backprop cost experiment.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
CONTRACT="$SUA/docs/A10_NO_BACKPROP_COST_CONTRACT_20260812.md"
SCREEN_ID="a10_no_backprop_cost_v1"
AUTH_ENV="A10_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_A10_NO_BACKPROP_COST_GPU"

ARM="${ARM:?ARM required: gradient_free|readout_probe|full_finetune}"
SEED="${SEED:?SEED required}"
GPU="${GPU:?GPU index required}"
MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" ]]; then
  echo "Refusing: A10 v1 is superseded; its arms do not use matched supervision." >&2
  echo "No authorization environment variable can revive this historical runner." >&2
  exit 4
fi

M_CALIB=30
EVAL_START=30
TOTAL_EPOCHS=12
BURN_IN=4
RESULTS="$SUA/results/$SCREEN_ID"
SOURCE_RUN="$SUA/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s${SEED}"
MAINLINE_RESULT="$SUA/results/sua_spint_t4_mainline_fp32_v1/t4_s${SEED}.json"
DATA="$SUA/data/dandi_000688/sub-C"
CACHE="$SUA/cache/dandi688_subc_co_v1"
TEACHER="$SUA/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$SUA/configs/subc_co_27_6_strict_train_val_manifest.json"
OUT="$RESULTS/${ARM}_s${SEED}.json"
LOG="$RESULTS/logs/${ARM}_s${SEED}.log"
EVAL_CELL="$SUA/scripts/a10_no_backprop_cost_eval_cell.py"

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

echo "CONTRACT=$CONTRACT"
echo "CONTRACT_SHA256=$CONTRACT_SHA"
echo "SCREEN_ID=$SCREEN_ID"
echo "ARM=$ARM SEED=$SEED GPU=$GPU"
echo "EXPECTED_GPU_CELLS=9 (3 arms x 3 seeds)"
echo "AUTHORIZATION_REQUIRED=${AUTH_ENV}=${AUTH_VALUE}"

if [[ "$ARM" == "gradient_free" ]]; then
  EVAL_COMMAND=(
    "$PY" -u "$SUA/scripts/eval_epoch_window_generic_dandi688.py"
    --run_dir "$SOURCE_RUN"
    --teacher_ckpt "$TEACHER"
    --data_dir "$DATA"
    --cache_dir "$CACHE"
    --train_val_manifest "$MANIFEST"
    --total_epochs "$TOTAL_EPOCHS"
    --burn_in "$BURN_IN"
    --calibration_n "$M_CALIB"
    --pool_size "$EVAL_START"
    --out_path "$OUT"
  )
else
  EVAL_COMMAND=(
    "$PY" -u "$EVAL_CELL"
    --arm "$ARM"
    --seed "$SEED"
    --source-run-dir "$SOURCE_RUN"
    --out-path "$OUT"
    --launch
  )
fi

if [[ "$MODE" == "--dry-run" ]]; then
  if [[ "$ARM" == "gradient_free" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${EVAL_COMMAND[@]}"
    echo
  else
    "$PY" "$EVAL_CELL" --arm "$ARM" --seed "$SEED" --source-run-dir "$SOURCE_RUN" --out-path "$OUT"
  fi
  exit 0
fi

echo "Unreachable superseded A10 runner path" >&2
exit 4
