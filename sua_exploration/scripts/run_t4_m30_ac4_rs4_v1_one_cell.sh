#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
SCREEN="sua_t4_m30_ac4_rs4_v1"

case "$SEED" in 42|43|44) ;; *) echo "invalid seed: $SEED" >&2; exit 2 ;; esac

"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_ac4_rs4_v1.py" --require-claim

OUT="$ROOT/sua_exploration/checkpoints/${SCREEN}_ac4_rs4_dandi688_co_s${SEED}"
RESULT="$ROOT/sua_exploration/results/${SCREEN}/ac4_rs4_s${SEED}.json"
P3="$ROOT/sua_exploration/results/p3_${SCREEN}_ac4_rs4_dandi688_co_s${SEED}_seed${SEED}.json"
LOCK="${OUT}.lock"
if [[ -e "$OUT" || -e "$RESULT" || -e "$P3" || -e "$LOCK" ]]; then
    echo "refusing AC4-RS4 output collision for seed ${SEED}" >&2
    exit 1
fi
mkdir "$LOCK"
trap 'rmdir "$LOCK"' EXIT

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u \
    "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" \
    --variant B3S \
    --side_features ac4rs4 \
    --side_feature_pool_size 30 \
    --calibration_n_trials 30 \
    --out_name "${SCREEN}_ac4_rs4_dandi688_co_s${SEED}" \
    --data_dir "$ROOT/sua_exploration/data/dandi_000688/sub-C" \
    --train_val_manifest "$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json" \
    --task CO \
    --split_counts 27,6,6 \
    --max_units_exclusive 100 \
    --max_epochs 12 \
    --no_early_stopping \
    --checkpoint_every_epoch \
    --lr 1e-4 \
    --batch_size 32 \
    --num_workers 4 \
    --cache_dir "$ROOT/sua_exploration/cache/t4_m30_ac4_rs4_v1" \
    --seed "$SEED" \
    --loss_mode task_only \
    --identity_mode calibrated \
    --t4_logit_residual_mode none \
    --require_gpu \
    --disable_progress_bar

CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
    "$ROOT/sua_exploration/scripts/eval_t4_m30_experiment_a.py" \
    --run_dir "$OUT" \
    --out "$RESULT"
