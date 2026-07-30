#!/usr/bin/env bash
# E1/E2 analysis driver for convergence_swa_v1. Waits for the training orchestrator
# (run_convergence_swa_v1.sh, tmux session spint_conv_swa) to finish, then:
#   E2: eval_convergence_curve_dandi688.py for each seed (epochs 5-40), then
#       aggregate_convergence_swa_v1.py -> results/convergence_swa_v1/convergence.json
#   E1: eval_swa_dandi688.py for each seed (windows last-5/10/20), then
#       aggregate_swa_windows_v1.py -> results/convergence_swa_v1/swa.json
#
# This script only ever performs the MECHANICAL evaluation + aggregation (all of it reuses
# evaluate_fixed_protocol_over_validation_sessions, never reimplements R2). It does not write
# CONVERGENCE_AND_SWA.md -- the epoch-budget recommendation and the SWA verdict require
# judgment and are written up separately after this script's JSON outputs are inspected.
#
# If any seed's training did not complete successfully, this script stops after logging which
# seed(s) failed and does NOT run the aggregators (which hard-require all 3 seeds) -- per task
# constraints, a crashed run must be recorded and reported, not silently retried or papered
# over by aggregating only the seeds that happened to succeed.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
RES="$ROOT/sua_exploration/results/convergence_swa_v1"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
SCRIPTS="$ROOT/sua_exploration/scripts"

mkdir -p "$RES/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Waiting for $RES/TRAINING_DONE ..."
while [ ! -f "$RES/TRAINING_DONE" ]; do
  sleep 60
done
log "Training marker found: $(cat "$RES/TRAINING_DONE")"

FAILED=0
for seed in 42 43 44; do
  run_dir="$ROOT/sua_exploration/checkpoints/convergence_swa_v1_b3_dandi688_co_s${seed}"
  status=$("$PY" -c "import json; print(json.load(open('$run_dir/run_metadata.json')).get('status'))" 2>/dev/null || echo "MISSING")
  n_ckpts=$(ls "$run_dir/epoch_ckpts" 2>/dev/null | wc -l)
  log "seed $seed: run_metadata status=$status, epoch checkpoints on disk=$n_ckpts"
  if [ "$status" != "completed" ] || [ "$n_ckpts" -ne 40 ]; then
    log "seed $seed did NOT complete cleanly (status=$status, n_ckpts=$n_ckpts, expected 40) -- see $RES/logs/train_s${seed}.log"
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  log "At least one seed failed to complete 40 epochs cleanly. Stopping WITHOUT running the E1/E2 evaluation or aggregation pipeline. This must be reported, not silently retried."
  touch "$RES/ANALYSIS_ABORTED_TRAINING_INCOMPLETE"
  exit 1
fi

# --- E2: convergence curve, epochs 5-40, per seed. 2 GPUs concurrently, then the 3rd. ---
run_curve() {
  local gpu="$1" seed="$2"
  local run_dir="$ROOT/sua_exploration/checkpoints/convergence_swa_v1_b3_dandi688_co_s${seed}"
  local log="$RES/logs/curve_s${seed}.log"
  log_inner() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$log"; }
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU$gpu START curve seed=$seed" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$SCRIPTS/eval_convergence_curve_dandi688.py" \
    --run_dir "$run_dir" --teacher_ckpt "$TEACHER" --data_dir "$DATA" --cache_dir "$CACHE" \
    --epoch_start 5 --epoch_end 40 --expected_max_epochs 40 \
    --out_path "$RES/curve_s${seed}.json" >>"$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU$gpu DONE curve seed=$seed rc=$rc" | tee -a "$log"
  return $rc
}

log "E2: launching convergence-curve eval for seed 42 (GPU0) and seed 43 (GPU1)"
run_curve 0 42 & c0=$!
run_curve 1 43 & c1=$!
wait $c0; rc_c0=$?
wait $c1; rc_c1=$?
log "E2: seed42 rc=$rc_c0  seed43 rc=$rc_c1"
log "E2: launching convergence-curve eval for seed 44 (GPU0)"
run_curve 0 44
rc_c2=$?
log "E2: seed44 rc=$rc_c2"

if [ "$rc_c0" -ne 0 ] || [ "$rc_c1" -ne 0 ] || [ "$rc_c2" -ne 0 ]; then
  log "E2 curve evaluation failed for at least one seed (rc: $rc_c0 $rc_c1 $rc_c2). Stopping before aggregation."
  touch "$RES/ANALYSIS_ABORTED_CURVE_EVAL_FAILED"
  exit 1
fi

log "E2: aggregating convergence.json"
"$PY" -u "$SCRIPTS/aggregate_convergence_swa_v1.py" 2>&1 | tee "$RES/logs/aggregate_convergence.log"
rc_agg1=${PIPESTATUS[0]}
if [ "$rc_agg1" -ne 0 ]; then
  log "convergence.json aggregation FAILED (rc=$rc_agg1)"
  touch "$RES/ANALYSIS_ABORTED_CONVERGENCE_AGGREGATION_FAILED"
  exit 1
fi

# --- E1: SWA windows 5/10/20, per seed. 2 GPUs concurrently, then the 3rd. ---
run_swa() {
  local gpu="$1" seed="$2"
  local run_dir="$ROOT/sua_exploration/checkpoints/convergence_swa_v1_b3_dandi688_co_s${seed}"
  local log="$RES/logs/swa_s${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU$gpu START swa seed=$seed" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$SCRIPTS/eval_swa_dandi688.py" \
    --run_dir "$run_dir" --teacher_ckpt "$TEACHER" --data_dir "$DATA" --cache_dir "$CACHE" \
    --max_epoch 40 --windows 5,10,20 \
    --out_path "$RES/swa_s${seed}.json" >>"$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU$gpu DONE swa seed=$seed rc=$rc" | tee -a "$log"
  return $rc
}

log "E1: launching SWA eval for seed 42 (GPU0) and seed 43 (GPU1)"
run_swa 0 42 & s0=$!
run_swa 1 43 & s1=$!
wait $s0; rc_s0=$?
wait $s1; rc_s1=$?
log "E1: seed42 rc=$rc_s0  seed43 rc=$rc_s1"
log "E1: launching SWA eval for seed 44 (GPU0)"
run_swa 0 44
rc_s2=$?
log "E1: seed44 rc=$rc_s2"

if [ "$rc_s0" -ne 0 ] || [ "$rc_s1" -ne 0 ] || [ "$rc_s2" -ne 0 ]; then
  log "E1 SWA evaluation failed for at least one seed (rc: $rc_s0 $rc_s1 $rc_s2). Stopping before aggregation."
  touch "$RES/ANALYSIS_ABORTED_SWA_EVAL_FAILED"
  exit 1
fi

log "E1: aggregating swa.json"
"$PY" -u "$SCRIPTS/aggregate_swa_windows_v1.py" 2>&1 | tee "$RES/logs/aggregate_swa.log"
rc_agg2=${PIPESTATUS[0]}
if [ "$rc_agg2" -ne 0 ]; then
  log "swa.json aggregation FAILED (rc=$rc_agg2)"
  touch "$RES/ANALYSIS_ABORTED_SWA_AGGREGATION_FAILED"
  exit 1
fi

log "E1/E2 ANALYSIS PIPELINE COMPLETE. convergence.json and swa.json are ready."
touch "$RES/ANALYSIS_DONE"
