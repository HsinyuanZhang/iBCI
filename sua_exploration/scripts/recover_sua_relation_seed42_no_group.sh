#!/usr/bin/env bash
# Recover only the seed-42 REL-NG cell after the original queue fail-closed
# before training because t4rel_nogroup was missing from base_feature_group().
#
# Successful T4/REL/REL-MS artifacts are reused read-only. The failed,
# initialized-only checkpoint directory is not reused; this run gets a fresh
# output token. No formal-test evaluator is invoked.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
RESULTS="$ROOT/sua_exploration/results/sua_electrode_relation_pilot_v2"
OUT_NAME="sua_electrode_relation_pilot_v2_no_group_recovery1_dandi688_co_s42"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$OUT_NAME"
LOG="$RESULTS/logs/no_group_s42_recovery1.log"
LAUNCH=0

if [ "${1:-}" = "--launch" ]; then
  LAUNCH=1
fi
if [ "$LAUNCH" -ne 1 ]; then
  echo "Refusing to recover without --launch." >&2
  exit 2
fi
for required in \
  "$RESULTS/t4_s42.json" \
  "$RESULTS/relation_s42.json" \
  "$RESULTS/membership_shuffle_s42.json"; do
  if [ ! -f "$required" ]; then
    echo "Required completed seed-42 artifact is missing: $required" >&2
    exit 2
  fi
done
for absent in \
  "$RUN_DIR" \
  "$RESULTS/no_group_s42.json" \
  "$RESULTS/seed42_strict_aggregate.json" \
  "$RESULTS/README.txt"; do
  if [ -e "$absent" ]; then
    echo "Refusing to overwrite recovery target: $absent" >&2
    exit 2
  fi
done

echo "[$(date --iso-8601=seconds)] start seed42 no-group recovery" | tee "$LOG"
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
  --teacher_ckpt "$TEACHER" --variant B3SERN --side_features t4rel_nogroup \
  --side_feature_pool_size 50 --out_name "$OUT_NAME" \
  --data_dir "$DATA" --cache_dir "$CACHE" --train_val_manifest "$MANIFEST" \
  --signal_view sua --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
  --max_epochs 12 --no_early_stopping --checkpoint_every_epoch \
  --lr 1e-4 --batch_size 32 --num_workers 4 --seed 42 \
  --loss_mode task_only --identity_mode calibrated --require_gpu \
  --disable_progress_bar >> "$LOG" 2>&1

CUDA_VISIBLE_DEVICES=1 "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
  --run_dir "$RUN_DIR" --teacher_ckpt "$TEACHER" \
  --data_dir "$DATA" --cache_dir "$CACHE" --total_epochs 12 --burn_in 4 \
  --train_val_manifest "$MANIFEST" \
  --out_path "$RESULTS/no_group_s42.json" >> "$LOG" 2>&1

"$PY" -u "$ROOT/sua_exploration/scripts/aggregate_sua_electrode_relation_pilot.py" \
  --seed 42 \
  --t4 "$RESULTS/t4_s42.json" \
  --relation "$RESULTS/relation_s42.json" \
  --membership-shuffle "$RESULTS/membership_shuffle_s42.json" \
  --no-group "$RESULTS/no_group_s42.json" \
  --out "$RESULTS/seed42_strict_aggregate.json" >> "$LOG" 2>&1

printf '%s\n' \
  "Validation-only seed 42 completed after fresh REL-NG recovery; formal test was not evaluated." \
  > "$RESULTS/README.txt"
echo "[$(date --iso-8601=seconds)] done seed42 no-group recovery" | tee -a "$LOG"
