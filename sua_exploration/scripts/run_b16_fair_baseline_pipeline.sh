#!/usr/bin/env bash
set -euo pipefail

VARIANT="B16"
TASK="CO"
SPLIT_COUNTS="27,6,6"
MAX_UNITS_EXCLUSIVE=100
SEED=42
LR=1e-4
EPOCHS=40
POOL_SIZE=50
TEACHER_CKPT="sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
OUT_NAME="b16_dandi688_co_learnedprior_s42"
RESULT_PREFIX="p3_no_calibration_validation_b16"

CKPT_DIR="sua_exploration/checkpoints/${OUT_NAME}"
TRAIN_CMD=(
  python3
  sua_exploration/scripts/train_variant_dandi688.py
  --variant
  "${VARIANT}"
  --identity_mode
  learned_prior
  --out_name
  "${OUT_NAME}"
  --task
  "${TASK}"
  --split_counts
  "${SPLIT_COUNTS}"
  --max_units_exclusive
  "${MAX_UNITS_EXCLUSIVE}"
  --seed
  "${SEED}"
  --max_epochs
  "${EPOCHS}"
  --lr
  "${LR}"
  --teacher_ckpt
  "${TEACHER_CKPT}"
)

echo "[1/3] training learned-prior no-calibration baseline"
"${TRAIN_CMD[@]}"
BEST_CKPT="$(python3 - <<PY
import json
from pathlib import Path
meta = json.loads(Path("${CKPT_DIR}/run_metadata.json").read_text())
print(meta["best_checkpoint"])
PY
)"
echo "best ckpt = ${BEST_CKPT}"

echo "[2/3] validation-only learned-prior control (no calibration)"
python3 sua_exploration/scripts/eval_no_calibration_validation_dandi688.py \
  --ckpt "${BEST_CKPT}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --variant "${VARIANT}" \
  --task "${TASK}" \
  --split_counts "${SPLIT_COUNTS}" \
  --max_units_exclusive "${MAX_UNITS_EXCLUSIVE}" \
  --pool_size "${POOL_SIZE}" \
  --seed "${SEED}" \
  --control_mode learned_prior

echo "[3/3] validation-only zero-identity control (for paired baseline)"
python3 sua_exploration/scripts/eval_no_calibration_validation_dandi688.py \
  --ckpt "${BEST_CKPT}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --variant "${VARIANT}" \
  --task "${TASK}" \
  --split_counts "${SPLIT_COUNTS}" \
  --max_units_exclusive "${MAX_UNITS_EXCLUSIVE}" \
  --pool_size "${POOL_SIZE}" \
  --seed "${SEED}" \
  --control_mode zero_identity

echo "Expected result files:"
ls -1 sua_exploration/results/"${RESULT_PREFIX}"*"_s${SEED}.json" | sort
