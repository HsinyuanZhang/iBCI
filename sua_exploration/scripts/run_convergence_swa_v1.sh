#!/usr/bin/env bash
# E1/E2 precursor runs (ROADMAP.md "当前实验计划（2026-07-26 起）"): B3 (the F0 baseline
# config from side_feature_ablation_v2) x seeds {42,43,44}, 40 epochs each, with
# --no_early_stopping --checkpoint_every_epoch so every epoch's checkpoint is available for
# (E2) the epoch-5..40 convergence curve and (E1) post-hoc SWA weight averaging.
#
# Every hyperparameter other than max_epochs/out_name is copied verbatim from the F0 group
# of run_side_feature_ablation_v2.sh (equivalently: from
# checkpoints/side_feature_ablation_v2_f0_dandi688_co_s42/run_metadata.json), so this batch
# differs from that screen's F0 arm only in training for 40 instead of 12 epochs. Validation-
# only. Never loads test-session spikes/behavior/trials (see run_metadata.json
# session_splits.test / MEASUREMENT_PROTOCOL_V4.md section 6).
#
# Only 3 runs on 2 GPUs: seed 42 (GPU0) and seed 43 (GPU1) launched concurrently first (this
# is also the M1 real-concurrency check -- two runs started in the same second must resolve
# to two distinct run directories), then seed 44 once a GPU frees up.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
RES="$ROOT/sua_exploration/results/convergence_swa_v1"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

run_one() {
  local gpu="$1" seed="$2"
  local name="convergence_swa_v1_b3_dandi688_co_s${seed}"
  local log="$RES/logs/train_s${seed}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] GPU$gpu START seed=$seed name=$name" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$TEACHER" --variant B3 --side_features none \
    --out_name "$name" --data_dir "$DATA" --cache_dir "$CACHE" \
    --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
    --max_epochs 40 --no_early_stopping --checkpoint_every_epoch \
    --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$seed" \
    --loss_mode task_only --identity_mode calibrated \
    --require_gpu --disable_progress_bar >>"$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] GPU$gpu DONE seed=$seed rc=$rc" | tee -a "$log"
  return $rc
}

echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] Launching seed 42 (GPU0) and seed 43 (GPU1) concurrently"
run_one 0 42 & p0=$!
run_one 1 43 & p1=$!
wait $p0; rc0=$?
wait $p1; rc1=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] seed42 rc=$rc0  seed43 rc=$rc1"

if [ $rc0 -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] seed42 FAILED (rc=$rc0) -- see $RES/logs/train_s42.log"
fi
if [ $rc1 -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] seed43 FAILED (rc=$rc1) -- see $RES/logs/train_s43.log"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] Launching seed 44 (GPU0)"
run_one 0 44
rc2=$?
if [ $rc2 -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] seed44 FAILED (rc=$rc2) -- see $RES/logs/train_s44.log"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S.%N')] ALL 3 CONVERGENCE/SWA TRAINING RUNS FINISHED (rc: seed42=$rc0 seed43=$rc1 seed44=$rc2)"
printf 'rc42=%s rc43=%s rc44=%s\n' "$rc0" "$rc1" "$rc2" > "$RES/TRAINING_DONE"
