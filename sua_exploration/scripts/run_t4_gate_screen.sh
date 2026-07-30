#!/usr/bin/env bash
# T4-gate screen (design D, docs/ELECTRODE_ANCHOR_DESIGNS.md): does a per-electrode
# reliability gate on top of T4 add anything T4 does not already supply?
#
#   groups = {T4, T4GATE, T4GATE_SHUFFLED} x seeds (from --seeds)
#     T4              -> variant B3S,   --side_features t4               (substrate; T4 is
#                          trained FRESH by this screen, not reused from e3_tuning_ablation,
#                          so the whole 3-way comparison is self-contained -- see
#                          docs/ELECTRODE_ANCHOR_DESIGNS.md section 5)
#     T4GATE          -> variant B3SEG, --side_features t4gate            (design D: E_i <-
#                          E_i * (1 + tanh(g[electrode(i)])), g zero-init -> exactly T4 at
#                          step 0)
#     T4GATE_SHUFFLED -> variant B3SEG, --side_features t4gate_shuffled   (electrode ids
#                          permuted along the unit axis, fixed seed = --seed; T4's own tuning
#                          values are untouched -- the dimension/parameter-matched control)
#
# M2: --no_early_stopping (fixed --max_epochs budget, equal draws for every group).
# M3: --checkpoint_every_epoch, scored by eval_epoch_window_generic_dandi688.py over the
#     window burn_in+1..max_epochs (this script does not modify that script; see section 0
#     of E3_E4_ENCODER_PROGRAM.md for why it takes an explicit --total_epochs/--burn_in
#     instead of the frozen 12-epoch script).
#
# Hyperparameters other than --variant/--side_features/--max_epochs/--seed are copied
# verbatim from run_e3_tuning_ablation.sh / run_electrode_ablation_f3.sh so any T4-gate-vs-T4
# difference is attributable only to the gate mechanism.
#
# Scheduling: a proper shared work queue (claim-next-job-from-queue over 2 GPU worker loops),
# copied from run_b3t_confirmation.sh -- NOT the lockstep-pairs pattern
# run_e3_tuning_ablation.sh / run_e4_encoder_variants.sh / run_electrode_ablation_f3.sh use
# (launch 2, `wait` for BOTH before launching the next 2), which idles a GPU whenever one job
# in a pair finishes first. Run with --self-test to prove the scheduler against fake
# sleep-based jobs of deliberately unequal duration (no GPU, no training, no eval touched).
#
# Validation-only. Never loads test-session spikes/behavior/trials (train_variant_dandi688.py
# and eval_epoch_window_generic_dandi688.py only ever touch validation-session spike data;
# test sessions are only ever used for their names and NWB unit-table row counts).
#
# STAGING (docs/ELECTRODE_ANCHOR_DESIGNS.md section 5): this is the ONLY one of the three
# T4-substrate electrode designs (D/C/A) wired into a runnable screen. Design C (B3SEA,
# t4anchor/t4anchor_shuffled) and design A (B3S, t4e/t4e_shuffled) are implemented and tested
# but have no runner script yet -- they stay implemented-but-unrun until this screen reports.
#
# Do NOT launch this script without explicit confirmation that both GPUs are free -- as of
# writing, b3t_confirmation seeds 46/47 are still running in tmux session spint_s4647.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
DOC="sua_exploration/docs/ELECTRODE_ANCHOR_DESIGNS.md"

usage() {
  cat <<EOF
Usage: $(basename "$0") --max_epochs N --seeds S1,S2,... [--burn_in B] [--self-test]

Required (NO DEFAULTS -- E3_E4_ENCODER_PROGRAM.md section 0's rule: never silently guess an
epoch budget or seed count):
  --max_epochs N       Training epoch budget for every group/seed run.
  --seeds S1,S2,...    Comma-separated seed list, e.g. 42,43,44.

Optional:
  --burn_in B     Epochs excluded from the trailing-average scoring window; the scored window
                  is epochs (B+1)..max_epochs. Default: 4 (matches E3/E4/F3's M3 estimator).
  --self-test     Exercise ONLY the work-queue scheduler, against fake sleep-based jobs of
                  deliberately unequal duration -- no GPU, no training, no eval. Prints each
                  worker's start/done timestamps and reports PASS/FAIL on whether either
                  worker was ever idle while jobs remained queued. Use this to verify the
                  scheduler before trusting it with real runs.
  -h, --help      Show this message and exit.

Trains T4 / T4GATE / T4GATE_SHUFFLED (see $DOC section 5) on the given seeds, 2 concurrent
runs across 2 GPUs drained by a shared work queue (no lockstep idling). Writes to
$ROOT/sua_exploration/results/t4_gate_screen/.
EOF
}

MAX_EPOCHS=""
SEEDS=""
BURN_IN=4
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --max_epochs) MAX_EPOCHS="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --burn_in) BURN_IN="${2:-}"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ "$SELF_TEST" -eq 0 ] && { [ -z "$MAX_EPOCHS" ] || [ -z "$SEEDS" ]; }; then
  echo "ERROR: --max_epochs and --seeds are required and have no default." >&2
  echo "See $DOC section 5 and E3_E4_ENCODER_PROGRAM.md section 0: this project has twice" >&2
  echo "set a screen's gate below its own measured noise floor by guessing an epoch budget" >&2
  echo "or seed count instead of measuring it first. Supply both explicitly (same values" >&2
  echo "used for E3/E4/F3, unless a new E1/E2 measurement says otherwise)." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [ "$SELF_TEST" -eq 0 ]; then
  if ! [[ "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCHS" -lt 1 ]; then
    echo "ERROR: --max_epochs must be a positive integer, got: $MAX_EPOCHS" >&2
    exit 1
  fi
  if ! [[ "$BURN_IN" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --burn_in must be a non-negative integer, got: $BURN_IN" >&2
    exit 1
  fi
  if [ "$BURN_IN" -ge "$MAX_EPOCHS" ]; then
    echo "ERROR: --burn_in ($BURN_IN) must be strictly less than --max_epochs ($MAX_EPOCHS)." >&2
    exit 1
  fi
  if ! [[ "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --seeds must be a comma-separated list of non-negative integers, got: $SEEDS" >&2
    exit 1
  fi
fi
IFS=',' read -ra SEED_ARR <<< "${SEEDS:-42,43,44}"

RES="$ROOT/sua_exploration/results/t4_gate_screen"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

# group -> "variant side_features"
declare -A GROUP=(
  [t4]="B3S t4"
  [t4gate]="B3SEG t4gate"
  [t4gate_shuffled]="B3SEG t4gate_shuffled"
)
GROUPS_ORDER=(t4 t4gate t4gate_shuffled)

# ---------------------------------------------------------------------------------------
# Shared work queue (copied from run_b3t_confirmation.sh's 2026-07-27 scheduler fix). JOBS is
# built once (seed-major, group-minor) and drained by 2 GPU worker loops; claim_next_job_index
# is the only critical section, protected by an atomic mkdir lock. A GPU worker is only ever
# idle once the queue itself is empty.
# ---------------------------------------------------------------------------------------
JOBS=()
for seed in "${SEED_ARR[@]}"; do for g in "${GROUPS_ORDER[@]}"; do JOBS+=("$g $seed"); done; done

QUEUE_STATE_DIR="$RES/.queue_state_$$"
mkdir -p "$QUEUE_STATE_DIR"
NEXT_INDEX_FILE="$QUEUE_STATE_DIR/next_index"
LOCK_DIR="$QUEUE_STATE_DIR/lock"
SELFTEST_LOG="$QUEUE_STATE_DIR/selftest_log"
echo 0 > "$NEXT_INDEX_FILE"
cleanup_queue_state() { rm -rf "$QUEUE_STATE_DIR"; }
trap cleanup_queue_state EXIT

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
  local name="t4_gate_screen_${group}_dandi688_co_s${seed}"
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

# --self-test job bodies: deliberately unequal sleep durations, one long job first then five
# short ones -- lockstep pairing pays for the long job twice (its own pair, plus forcing its
# partner idle); a proper queue pays for it once. Mirrors run_b3t_confirmation.sh exactly.
declare -A SELFTEST_DURATION=(
  ["t4 42"]=5 ["t4gate 42"]=1 ["t4gate_shuffled 42"]=1
  ["t4 43"]=1 ["t4gate 43"]=1 ["t4gate_shuffled 43"]=1
  ["t4 44"]=1 ["t4gate 44"]=1 ["t4gate_shuffled 44"]=1
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

  durations=()
  for seed in "${SEED_ARR[@]}"; do for g in "${GROUPS_ORDER[@]}"; do durations+=("${SELFTEST_DURATION["$g $seed"]}"); done; done
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
        if gap > 0.3:
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

echo "[$(date +%H:%M:%S)] t4_gate_screen: max_epochs=$MAX_EPOCHS burn_in=$BURN_IN seeds=${SEED_ARR[*]} groups=${GROUPS_ORDER[*]}"
echo "[$(date +%H:%M:%S)] Total runs: ${#JOBS[@]}"

gpu_worker 0 run_one_real & p0=$!
gpu_worker 1 run_one_real & p1=$!
wait $p0
wait $p1

echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS FINISHED"
ls -1 "$RES"/*.json 2>/dev/null | wc -l
