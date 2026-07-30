#!/usr/bin/env bash
# Side-feature ablation under MEASUREMENT_PROTOCOL_V4.
#   15 runs = {F0,F1,F2,FS1,FS2} x seeds {42,43,44}
#   M2: --no_early_stopping (fixed 12-epoch budget, equal draws for every group)
#   M3: --checkpoint_every_epoch, scored by eval_epoch_window_dandi688.py (epochs 5-12)
# Hyperparameters other than the M1/M2/M3 additions are copied verbatim from the v3 SUA
# arm (attention_arch_screen_v3_b3_dandi688_co_s42/run_metadata.json) so that any v4-vs-v3
# difference is attributable only to the measurement fixes.
# Validation-only. Never loads test-session spikes/behavior/trials.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
RES="$ROOT/sua_exploration/results/side_feature_ablation_v2"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

# group -> "variant side_features"
declare -A GROUP=( [f0]="B3 none" [f1]="B3S f1" [f2]="B3S f2" [fs1]="B3S fs1" [fs2]="B3S fs2" )

run_one() {
  local gpu="$1" group="$2" seed="$3"
  read -r variant side <<<"${GROUP[$group]}"
  local name="side_feature_ablation_v2_${group}_dandi688_co_s${seed}"
  local log="$RES/logs/${group}_s${seed}.log"

  echo "[$(date +%H:%M:%S)] GPU$gpu START $group seed=$seed ($variant/$side)" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$TEACHER" --variant "$variant" --side_features "$side" \
    --out_name "$name" --data_dir "$DATA" --cache_dir "$CACHE" \
    --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
    --max_epochs 12 --no_early_stopping --checkpoint_every_epoch \
    --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$seed" \
    --loss_mode task_only --identity_mode calibrated \
    --require_gpu --disable_progress_bar >>"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] GPU$gpu TRAIN-FAIL $group s$seed rc=$rc" | tee -a "$log"
    return $rc
  fi

  echo "[$(date +%H:%M:%S)] GPU$gpu EVAL $group seed=$seed" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_dandi688.py" \
    --run_dir "$ROOT/sua_exploration/checkpoints/$name" \
    --teacher_ckpt "$TEACHER" --data_dir "$DATA" --cache_dir "$CACHE" \
    --out_path "$RES/${group}_s${seed}.json" >>"$log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE $group s$seed eval_rc=$rc" | tee -a "$log"
  return $rc
}

# Build the 15-job queue, then drain it over 2 GPUs.
JOBS=()
for seed in 42 43 44; do for g in f0 f1 f2 fs1 fs2; do JOBS+=("$g $seed"); done; done

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  read -r g0 s0 <<<"${JOBS[$i]}"; run_one 0 "$g0" "$s0" & p0=$!
  j=$((i+1)); p1=""
  if [ $j -lt ${#JOBS[@]} ]; then
    read -r g1 s1 <<<"${JOBS[$j]}"; run_one 1 "$g1" "$s1" & p1=$!
  fi
  wait $p0; [ -n "$p1" ] && wait $p1
  i=$((i+2))
done

echo "[$(date +%H:%M:%S)] ALL 15 RUNS FINISHED"
ls -1 "$RES"/*.json 2>/dev/null | wc -l
