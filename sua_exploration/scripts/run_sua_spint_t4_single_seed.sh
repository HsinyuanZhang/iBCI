#!/usr/bin/env bash
# One-seed worker for the matched SUA B0/T4/TS4 FP32 mainline.
# Intended to parallelize the frozen matrix across GPUs without sharing run dirs.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_spint_t4_mainline_fp32_v1}"
GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"

group_config() {
  case "$1" in
    b0) echo "B0 none" ;;
    t4) echo "B3S t4" ;;
    ts4) echo "B3S ts4" ;;
    *) return 2 ;;
  esac
}

mkdir -p "$LOGS"
for group in b0 t4 ts4; do
  target="$ROOT/sua_exploration/checkpoints/${SCREEN_ID}_${group}_dandi688_co_s${SEED}"
  [[ ! -e "$target" ]] || {
    echo "Refusing to reuse existing run directory: $target" >&2
    exit 1
  }
done

for group in b0 t4 ts4; do
  read -r variant side <<<"$(group_config "$group")"
  name="${SCREEN_ID}_${group}_dandi688_co_s${SEED}"
  log="$LOGS/${group}_s${SEED}.log"
  {
    echo "[$(date -Is)] START group=$group seed=$SEED variant=$variant gpu=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u \
      "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
      --teacher_ckpt "$TEACHER" \
      --variant "$variant" \
      --side_features "$side" \
      --side_feature_pool_size 30 \
      --calibration_n_trials 30 \
      --out_name "$name" \
      --data_dir "$DATA" \
      --cache_dir "$CACHE" \
      --train_val_manifest "$MANIFEST" \
      --task CO \
      --split_counts 27,6,6 \
      --max_units_exclusive 100 \
      --max_epochs 12 \
      --no_early_stopping \
      --checkpoint_every_epoch \
      --lr 1e-4 \
      --batch_size 32 \
      --num_workers 4 \
      --seed "$SEED" \
      --loss_mode task_only \
      --identity_mode calibrated \
      --require_gpu \
      --disable_progress_bar
    echo "[$(date -Is)] EVAL group=$group seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u \
      "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
      --run_dir "$ROOT/sua_exploration/checkpoints/$name" \
      --teacher_ckpt "$TEACHER" \
      --data_dir "$DATA" \
      --cache_dir "$CACHE" \
      --train_val_manifest "$MANIFEST" \
      --total_epochs 12 \
      --burn_in 4 \
      --calibration_n 30 \
      --pool_size 30 \
      --out_path "$RESULTS/${group}_s${SEED}.json"
    echo "[$(date -Is)] DONE group=$group seed=$SEED"
  } >"$log" 2>&1
done

{
  echo "status=completed"
  echo "seed=$SEED"
  echo "gpu=$GPU"
  echo "completed_at=$(date -Is)"
} >"$RESULTS/seed${SEED}_worker_status.env"

# The last finishing seed may safely produce the strict aggregate.
complete=1
for seed in 42 43 44; do
  for group in b0 t4 ts4; do
    [[ -f "$RESULTS/${group}_s${seed}.json" ]] || complete=0
  done
done
if [[ "$complete" == 1 ]]; then
  "$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_spint_t4_mainline.py" \
    --result_dir "$RESULTS" \
    --out "$RESULTS/aggregate.json" >"$LOGS/aggregate.log" 2>&1
fi
