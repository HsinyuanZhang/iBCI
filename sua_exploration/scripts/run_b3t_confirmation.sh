#!/usr/bin/env bash
# B3T confirmation run: more seeds on the single strongest result so far -- B3T+SWA vs B3 =
# +0.0579 over 3 seeds, positive on all 3 (CURRENT_RESULTS.md sections J.3/J.3b/J.3c). The
# limiting factor on every conclusion in this project is sigma_seed and the seed count, so
# the highest-value next run is more seeds on exactly this comparison.
#
# Trains ONLY B3 and B3T (no B3A, no side features) on THREE NEW seeds -- 45, 46, 47 (42/43/44
# already exist in results/e4_encoder_variants/{b3,b3t}_s{42,43,44}.json) -- at the same
# --max_epochs 12 --burn_in 4 M2/M3 budget as E4, so the new seeds are poolable with the
# existing e4_encoder_variants artifacts by aggregate_e4_encoder_variants.py. Results land in
# a SEPARATE out_name prefix / results dir (b3t_confirmation_*, results/b3t_confirmation/), so
# this never overwrites, races with, or otherwise touches results/e4_encoder_variants/.
# Pooling the two result sets (if wanted later) is a separate step, not performed here.
#
# Hyperparameters other than --variant/--seed/--out_name are copied VERBATIM from
# run_e4_encoder_variants.sh's run_one(): --teacher_ckpt, --data_dir, --cache_dir, --task CO,
# --split_counts 27,6,6, --max_units_exclusive 100, --no_early_stopping,
# --checkpoint_every_epoch, --lr 1e-4, --batch_size 32, --num_workers 4, --loss_mode
# task_only, --identity_mode calibrated, --require_gpu, --disable_progress_bar. Same for the
# eval_epoch_window_generic_dandi688.py invocation (--teacher_ckpt/--data_dir/--cache_dir/
# --total_epochs/--burn_in/--out_path). Diffed by hand against run_e4_encoder_variants.sh at
# authoring time; the only differences are TEACHER/DATA/CACHE resolved from the same fixed
# paths, GROUP restricted to {b3, b3t}, SEEDS replaced, and RES/out_name prefix changed.
#
# Scheduling (2026-07-27 fix): unlike run_side_feature_ablation_v2.sh /
# run_e3_tuning_ablation.sh / run_e4_encoder_variants.sh -- which drain their job queue in
# LOCKSTEP PAIRS (launch 2, `wait` for BOTH before launching the next 2), so a GPU that
# finishes first sits idle until its pair partner finishes -- this script uses a proper
# shared work queue: two GPU worker loops each claim the next not-yet-started job the instant
# they are free, via an atomic mkdir-lock-protected counter. A GPU is only ever idle when the
# queue itself is empty. Run with --self-test to prove this against fake sleep-based jobs of
# deliberately unequal duration (no GPU, no training, no eval touched) -- see
# run_one_selftest()/SELFTEST_DURATION below. This does NOT retrofit the currently-running
# E3 runner (run_e3_tuning_ablation.sh) or any other existing script; those are left alone.
#
# Validation-only; never loads test-session spikes/behavior/trials (train_variant_dandi688.py
# and eval_epoch_window_generic_dandi688.py only ever touch validation-session spike data;
# test sessions are only ever used for their names and NWB unit-table row counts).
#
# Do not launch this script directly -- only through chain_after_e3.sh, after E3 has
# finished and both GPUs are confirmed free.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"

MAX_EPOCHS=12
BURN_IN=4
SEEDS=(45 46 47)
SELF_TEST=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--self-test]

Trains B3 and B3T (no B3A, no side features) on seeds 45/46/47 at --max_epochs $MAX_EPOCHS
--burn_in $BURN_IN (matching E4), 2 concurrent runs across 2 GPUs drained by a proper shared
work queue (no lockstep idling -- see the scheduling comment at the top of this file).
Writes to $ROOT/sua_exploration/results/b3t_confirmation/.

  --self-test   Exercise ONLY the work-queue scheduler, against fake sleep-based jobs of
                deliberately unequal duration -- no GPU, no training, no eval. Prints each
                worker's start/done timestamps, compares actual wall time against what the
                OLD lockstep-pairs scheduler would have taken on the same jobs, and reports
                PASS/FAIL on whether either worker was ever idle while jobs remained queued.
                Use this to verify the scheduler before trusting it with real runs.
  -h, --help    Show this message and exit.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

RES="$ROOT/sua_exploration/results/b3t_confirmation"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

# group -> "variant side_features" (side_features is always none -- no side-feature arm in
# this confirmation run, charter section 4 logic: B3/B3T are architecture-only variants).
declare -A GROUP=( [b3]="B3 none" [b3t]="B3T none" )
GROUPS_ORDER=(b3 b3t)

# ---------------------------------------------------------------------------------------
# Shared work queue. JOBS is built once (seed-major, group-minor, matching
# run_e4_encoder_variants.sh's ordering convention) and drained by 2 GPU worker loops.
# claim_next_job_index() is the only critical section, protected by an atomic mkdir lock
# (mkdir is a single atomic syscall on a local filesystem: exactly one concurrent caller can
# ever succeed) -- everything else in each worker loop runs unlocked and in parallel.
# ---------------------------------------------------------------------------------------
JOBS=()
for seed in "${SEEDS[@]}"; do for g in "${GROUPS_ORDER[@]}"; do JOBS+=("$g $seed"); done; done

QUEUE_STATE_DIR="$RES/.queue_state_$$"
mkdir -p "$QUEUE_STATE_DIR"
NEXT_INDEX_FILE="$QUEUE_STATE_DIR/next_index"
LOCK_DIR="$QUEUE_STATE_DIR/lock"
SELFTEST_LOG="$QUEUE_STATE_DIR/selftest_log"
echo 0 > "$NEXT_INDEX_FILE"
cleanup_queue_state() { rm -rf "$QUEUE_STATE_DIR"; }
trap cleanup_queue_state EXIT

# Prints the claimed job index on stdout, or an empty line if the queue is exhausted.
claim_next_job_index() {
  local idx=""
  while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      local current
      current=$(cat "$NEXT_INDEX_FILE")
      if [ "$current" -lt "${#JOBS[@]}" ]; then
        idx="$current"
        echo $((current + 1)) > "$NEXT_INDEX_FILE"
      fi
      rmdir "$LOCK_DIR"
      echo "$idx"
      return 0
    fi
    sleep 0.05
  done
}

# gpu_worker: pulls jobs from the shared queue until it is empty, running each one via
# $runner (a function name: run_one_real for real runs, run_one_selftest for --self-test).
# Only ever idle once claim_next_job_index() returns empty -- i.e. once the queue is drained.
gpu_worker() {
  local gpu="$1" runner="$2"
  while true; do
    local idx
    idx=$(claim_next_job_index)
    if [ -z "$idx" ]; then
      break
    fi
    read -r g s <<<"${JOBS[$idx]}"
    "$runner" "$gpu" "$g" "$s"
  done
}

run_one_real() {
  local gpu="$1" group="$2" seed="$3"
  read -r variant side <<<"${GROUP[$group]}"
  local name="b3t_confirmation_${group}_dandi688_co_s${seed}"
  local log="$RES/logs/${group}_s${seed}.log"

  echo "[$(date +%H:%M:%S)] GPU$gpu START $group seed=$seed ($variant/$side) max_epochs=$MAX_EPOCHS" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$TEACHER" --variant "$variant" --side_features "$side" \
    --out_name "$name" --data_dir "$DATA" --cache_dir "$CACHE" \
    --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
    --max_epochs "$MAX_EPOCHS" --no_early_stopping --checkpoint_every_epoch \
    --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$seed" \
    --loss_mode task_only --identity_mode calibrated \
    --require_gpu --disable_progress_bar >>"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] GPU$gpu TRAIN-FAIL $group s$seed rc=$rc" | tee -a "$log"
    return $rc
  fi

  echo "[$(date +%H:%M:%S)] GPU$gpu EVAL $group seed=$seed" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
    --run_dir "$ROOT/sua_exploration/checkpoints/$name" \
    --teacher_ckpt "$TEACHER" --data_dir "$DATA" --cache_dir "$CACHE" \
    --total_epochs "$MAX_EPOCHS" --burn_in "$BURN_IN" \
    --out_path "$RES/${group}_s${seed}.json" >>"$log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE $group s$seed eval_rc=$rc" | tee -a "$log"
  return $rc
}

# --self-test job bodies: deliberately unequal sleep durations in place of real training, to
# prove the scheduler never idles a worker while jobs remain. One long job first, then five
# short ones -- lockstep pairing pays for that long job TWICE (once in its own pair, once
# forcing its pair-partner to wait idle); a proper queue pays for it once.
declare -A SELFTEST_DURATION=(
  ["b3 45"]=5 ["b3t 45"]=1 ["b3 46"]=1 ["b3t 46"]=1 ["b3 47"]=1 ["b3t 47"]=1
)

run_one_selftest() {
  local gpu="$1" group="$2" seed="$3"
  local dur="${SELFTEST_DURATION["$group $seed"]}"
  echo "$(date +%s.%N) $gpu START ${group}_s${seed} dur=${dur}" >> "$SELFTEST_LOG"
  sleep "$dur"
  echo "$(date +%s.%N) $gpu DONE ${group}_s${seed} dur=${dur}" >> "$SELFTEST_LOG"
}

if [ "$SELF_TEST" -eq 1 ]; then
  echo "[$(date +%H:%M:%S)] SELF-TEST: work-queue scheduler only, ${#JOBS[@]} fake jobs, 2 workers (no GPU/training/eval touched)"
  : > "$SELFTEST_LOG"

  gpu_worker 0 run_one_selftest & p0=$!
  gpu_worker 1 run_one_selftest & p1=$!
  wait $p0
  wait $p1

  echo "--- self-test log (start/done timestamps, unsorted-arrival order) ---"
  sort -n "$SELFTEST_LOG"

  # Lockstep-pairs equivalent: sum over consecutive JOBS pairs of max(pair duration) -- what
  # run_e4_encoder_variants.sh's scheduler (`wait $p0; wait $p1` after launching both) would
  # have paid on these exact same jobs in this exact same order.
  durations=()
  for seed in "${SEEDS[@]}"; do for g in "${GROUPS_ORDER[@]}"; do durations+=("${SELFTEST_DURATION["$g $seed"]}"); done; done
  lockstep_total=0
  i=0
  while [ $i -lt ${#durations[@]} ]; do
    a="${durations[$i]}"; b="${durations[$((i+1))]:-0}"
    pair_max=$a; [ "$b" -gt "$a" ] && pair_max=$b
    lockstep_total=$((lockstep_total + pair_max))
    i=$((i+2))
  done

  echo "--- verdict ---"
  "$PY" - "$SELFTEST_LOG" "$lockstep_total" <<'PYEOF'
import sys
from collections import defaultdict

log_path, lockstep_total = sys.argv[1], float(sys.argv[2])
events = defaultdict(list)
with open(log_path) as f:
    for line in f:
        parts = line.split()
        ts, gpu, kind = float(parts[0]), parts[1], parts[2]
        events[gpu].append((ts, kind))

ok = True
max_gap = 0.0
overall_start, overall_end = None, None
for gpu, evs in sorted(events.items()):
    evs.sort()
    intervals, stack = [], []
    for ts, kind in evs:
        if kind == "START":
            stack.append(ts)
        else:
            intervals.append((stack.pop(), ts))
    intervals.sort()
    for s, e in intervals:
        overall_start = s if overall_start is None else min(overall_start, s)
        overall_end = e if overall_end is None else max(overall_end, e)
    for (s0, e0), (s1, e1) in zip(intervals, intervals[1:]):
        gap = s1 - e0
        max_gap = max(max_gap, gap)
        if gap > 0.3:  # generous tolerance for process-launch overhead
            ok = False
            print(f"  GPU{gpu}: idle gap of {gap:.3f}s between jobs while the queue was non-empty")

wall_total = overall_end - overall_start
faster = wall_total < lockstep_total
print(f"actual queue wall time:          {wall_total:.2f}s")
print(f"lockstep-pairs equivalent:       {lockstep_total:.2f}s  (same jobs, old scheduler pattern)")
print(f"max inter-job idle gap observed: {max_gap:.3f}s (tolerance 0.3s for process-launch overhead)")
print("wall time strictly less than lockstep equivalent: " + ("YES" if faster else "NO"))
verdict_ok = ok and faster
print("SELF-TEST " + ("PASS" if verdict_ok else "FAIL")
      + ": workers " + ("never" if ok else "WERE") + " idle while jobs remained queued"
      + ("" if faster else "; wall time was NOT faster than lockstep"))
sys.exit(0 if verdict_ok else 1)
PYEOF
  exit $?
fi

echo "[$(date +%H:%M:%S)] b3t_confirmation: max_epochs=$MAX_EPOCHS burn_in=$BURN_IN seeds=${SEEDS[*]} groups=${GROUPS_ORDER[*]}"
echo "[$(date +%H:%M:%S)] Total runs: ${#JOBS[@]}"

gpu_worker 0 run_one_real & p0=$!
gpu_worker 1 run_one_real & p1=$!
wait $p0
wait $p1

echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS FINISHED"
ls -1 "$RES"/*.json 2>/dev/null | wc -l
