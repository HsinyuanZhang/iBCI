#!/usr/bin/env bash
# Validation-only pseudo-MUA T4 bridge:
#   F0 / T4 / TS4 x seeds 42 / 43 / 44
#
# Formal launch:
#   run_pseudomua_t4_bridge.sh --launch --wait \
#     --screen-id pseudomua_t4_bridge_v1
#
# The two GPU workers claim jobs from one atomic queue.  A worker that finishes
# early immediately takes the next job; there is no lockstep pairing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"

MODE="dry-run"
WAIT_FOR_COMPLETION=0
SELF_TEST=0
ORCHESTRATOR_CHILD=0
SCREEN_ID="pseudomua_t4_bridge_v1"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --dry-run [--screen-id ID]
  $(basename "$0") --self-test
  $(basename "$0") --launch [--wait] [--screen-id ID]

Modes:
  --dry-run    Print the exact 9-job train/eval matrix without writing results.
  --self-test  Test the dynamic queue and failure propagation using fake sleep jobs.
  --launch     Start the formal validation-only screen.
  --wait       With --launch, run in the foreground.  Without --wait, background the
               orchestrator and print its PID/log path.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --self-test)
      SELF_TEST=1
      ;;
    --launch)
      MODE="launch"
      ;;
    --wait)
      WAIT_FOR_COMPLETION=1
      ;;
    --screen-id)
      [[ $# -ge 2 ]] || { echo "ERROR: --screen-id requires a value" >&2; exit 2; }
      SCREEN_ID="$2"
      shift
      ;;
    --orchestrator-child)
      # Private flag used only by the no---wait background wrapper.
      ORCHESTRATOR_CHILD=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! "$SCREEN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: --screen-id must match [A-Za-z0-9_.-]+, got $SCREEN_ID" >&2
  exit 2
fi
if [[ "$MODE" != "launch" && "$WAIT_FOR_COMPLETION" -eq 1 ]]; then
  echo "ERROR: --wait is meaningful only with --launch" >&2
  exit 2
fi

DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_pseudomua_t4_v1"
RESULTS_DIR="$ROOT/sua_exploration/results/$SCREEN_ID"
CHECKPOINT_ROOT="$ROOT/sua_exploration/checkpoints"
SUA_RESULTS_DIR="$ROOT/sua_exploration/results/e3_tuning_ablation"
MANIFEST="$RESULTS_DIR/manifest.jsonl"
ORCHESTRATOR_LOG="$RESULTS_DIR/orchestrator.log"
PID_FILE="$RESULTS_DIR/orchestrator.pid"
SUMMARY_PATH="$RESULTS_DIR/summary.json"

declare -A VARIANT_BY_GROUP=(
  [f0]="B3"
  [t4]="B3S"
  [ts4]="B3S"
)
declare -A SIDE_BY_GROUP=(
  [f0]="none"
  [t4]="t4"
  [ts4]="ts4"
)
GROUP_ORDER=(f0 t4 ts4)
SEEDS=(42 43 44)
JOBS=()
for seed in "${SEEDS[@]}"; do
  for group in "${GROUP_ORDER[@]}"; do
    JOBS+=("$group $seed")
  done
done

RUN_ID=""
QUEUE_DIR=""
QUEUE_NEXT_FILE=""
QUEUE_LOCK_FILE=""
EXECUTION_KIND="formal"

shell_join() {
  local joined
  printf -v joined "%q " "$@"
  printf "%s" "${joined% }"
}

emit_event() {
  local event="$1"
  local group="${2:-}"
  local seed="${3:-}"
  local gpu="${4:-}"
  local worker_pid="${5:-}"
  local status="${6:-}"
  local rc="${7:-}"
  local job_log="${8:-}"
  local train_command="${9:-}"
  local eval_command="${10:-}"
  local message="${11:-}"
  "$PY" - \
    "$MANIFEST" "$RUN_ID" "$event" "$group" "$seed" "$gpu" "$worker_pid" \
    "$status" "$rc" "$job_log" "$train_command" "$eval_command" "$message" <<'PY'
import datetime
import json
import os
import sys

(
    manifest,
    run_id,
    event,
    group,
    seed,
    gpu,
    worker_pid,
    status,
    rc,
    job_log,
    train_command,
    eval_command,
    message,
) = sys.argv[1:]
payload = {
    "time": datetime.datetime.now().astimezone().isoformat(),
    "run_id": run_id,
    "event": event,
}
for key, value in {
    "group": group,
    "seed": seed,
    "gpu": gpu,
    "worker_pid": worker_pid,
    "status": status,
    "rc": rc,
    "job_log": job_log,
    "train_command": train_command,
    "eval_command": eval_command,
    "message": message,
}.items():
    if value != "":
        payload[key] = int(value) if key in {"seed", "gpu", "worker_pid", "rc"} else value
line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
try:
    os.write(fd, line)
finally:
    os.close(fd)
PY
}

build_job_commands() {
  local group="$1"
  local seed="$2"
  local variant="${VARIANT_BY_GROUP[$group]}"
  local side="${SIDE_BY_GROUP[$group]}"

  JOB_NAME="${SCREEN_ID}_${group}_dandi688_co_s${seed}"
  JOB_RESULT="$RESULTS_DIR/${group}_s${seed}.json"
  JOB_LOG="$RESULTS_DIR/logs/${group}_s${seed}.log"
  JOB_CHECKPOINT_DIR="$CHECKPOINT_ROOT/$JOB_NAME"
  TRAIN_CMD=(
    "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$TEACHER"
    --variant "$variant"
    --side_features "$side"
    --out_name "$JOB_NAME"
    --data_dir "$DATA"
    --cache_dir "$CACHE"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --signal_view pseudo_mua
    --max_epochs 12
    --no_early_stopping
    --checkpoint_every_epoch
    --lr 1e-4
    --batch_size 32
    --num_workers 4
    --seed "$seed"
    --loss_mode task_only
    --identity_mode calibrated
    --require_gpu
    --disable_progress_bar
  )
  EVAL_CMD=(
    "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py"
    --run_dir "$JOB_CHECKPOINT_DIR"
    --teacher_ckpt "$TEACHER"
    --data_dir "$DATA"
    --cache_dir "$CACHE"
    --total_epochs 12
    --burn_in 4
    --out_path "$JOB_RESULT"
  )
}

valid_artifact() {
  local path="$1"
  local group="$2"
  local seed="$3"
  local upper_group="${group^^}"
  PYTHONPATH="$ROOT/sua_exploration/scripts" "$PY" - \
    "$path" "$upper_group" "$seed" <<'PY'
import sys
from pathlib import Path

from aggregate_pseudomua_t4_bridge import load_artifact

path = Path(sys.argv[1])
load_artifact(path, group=sys.argv[2], seed=int(sys.argv[3]), view="pseudo_mua")
PY
}

claim_next_job() {
  exec 9>"$QUEUE_LOCK_FILE"
  flock 9
  local index
  index="$(<"$QUEUE_NEXT_FILE")"
  if ((index >= ${#JOBS[@]})); then
    flock -u 9
    exec 9>&-
    return 1
  fi
  printf "%d\n" "$((index + 1))" >"$QUEUE_NEXT_FILE"
  printf "%s\n" "${JOBS[index]}"
  flock -u 9
  exec 9>&-
}

run_formal_job() {
  local gpu="$1"
  local group="$2"
  local seed="$3"
  local worker_pid="$BASHPID"
  local train_display
  local eval_display
  local status
  local rc

  build_job_commands "$group" "$seed"
  train_display="CUDA_VISIBLE_DEVICES=$gpu $(shell_join "${TRAIN_CMD[@]}")"
  eval_display="CUDA_VISIBLE_DEVICES=$gpu $(shell_join "${EVAL_CMD[@]}")"
  emit_event \
    "job_start" "$group" "$seed" "$gpu" "$worker_pid" "" "" "$JOB_LOG" \
    "$train_display" "$eval_display" ""

  if valid_artifact "$JOB_RESULT" "$group" "$seed" >/dev/null 2>&1; then
    status="skipped_valid"
    rc=0
  elif [[ -e "$JOB_RESULT" || -e "$JOB_CHECKPOINT_DIR" ]]; then
    status="failed_partial_artifact"
    rc=1
    printf "Refusing to overwrite partial result/checkpoint for %s seed %s\n" \
      "$group" "$seed" >"$JOB_LOG"
  else
    status="completed"
    rc=0
    printf "[%s] GPU%s TRAIN %s seed=%s\n" \
      "$(date -Is)" "$gpu" "$group" "$seed" >"$JOB_LOG"
    if CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_CMD[@]}" >>"$JOB_LOG" 2>&1; then
      :
    else
      rc=$?
      status="failed_train"
    fi
    if [[ "$rc" -eq 0 ]]; then
      if CUDA_VISIBLE_DEVICES="$gpu" "${EVAL_CMD[@]}" >>"$JOB_LOG" 2>&1; then
        :
      else
        rc=$?
        status="failed_eval"
      fi
    fi
    if [[ "$rc" -eq 0 ]]; then
      if valid_artifact "$JOB_RESULT" "$group" "$seed" >>"$JOB_LOG" 2>&1; then
        :
      else
        rc=$?
        status="failed_validation"
      fi
    fi
  fi

  emit_event \
    "job_done" "$group" "$seed" "$gpu" "$worker_pid" "$status" "$rc" "$JOB_LOG" \
    "$train_display" "$eval_display" ""
  return "$rc"
}

run_selftest_job() {
  local gpu="$1"
  local group="$2"
  local seed="$3"
  local worker_pid="$BASHPID"
  local duration="0.05"
  local rc=0
  [[ "$group" == "slow" ]] && duration="0.40"

  emit_event \
    "job_start" "$group" "$seed" "$gpu" "$worker_pid" "" "" "" \
    "sleep $duration" "" "queue self-test"
  sleep "$duration"
  [[ "$group" == "fail" ]] && rc=7
  emit_event \
    "job_done" "$group" "$seed" "$gpu" "$worker_pid" \
    "$([[ "$rc" -eq 0 ]] && printf completed || printf failed_expected)" \
    "$rc" "" "sleep $duration" "" "queue self-test"
  return "$rc"
}

worker_loop() {
  local gpu="$1"
  local failed=0
  local job
  local group
  local seed
  while job="$(claim_next_job)"; do
    read -r group seed <<<"$job"
    if [[ "$EXECUTION_KIND" == "selftest" ]]; then
      if ! run_selftest_job "$gpu" "$group" "$seed"; then
        failed=1
      fi
    elif ! run_formal_job "$gpu" "$group" "$seed"; then
      failed=1
    fi
  done
  return "$failed"
}

run_worker_pool() {
  local worker0
  local worker1
  local rc0
  local rc1
  worker_loop 0 &
  worker0=$!
  worker_loop 1 &
  worker1=$!
  set +e
  wait "$worker0"
  rc0=$?
  wait "$worker1"
  rc1=$?
  set -e
  if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
    return 1
  fi
  return 0
}

initialize_queue() {
  QUEUE_DIR="$(mktemp -d "$RESULTS_DIR/.queue_state.XXXXXX")"
  QUEUE_NEXT_FILE="$QUEUE_DIR/next"
  QUEUE_LOCK_FILE="$QUEUE_DIR/claim.lock"
  printf "0\n" >"$QUEUE_NEXT_FILE"
}

cleanup_queue() {
  if [[ -n "$QUEUE_DIR" && -d "$QUEUE_DIR" ]]; then
    rm -f "$QUEUE_NEXT_FILE" "$QUEUE_LOCK_FILE"
    rmdir "$QUEUE_DIR" 2>/dev/null || true
  fi
}

pid_file_is_live() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

cleanup_orchestrator() {
  cleanup_queue
  if [[ -f "$PID_FILE" ]]; then
    local recorded_pid
    recorded_pid="$(<"$PID_FILE")"
    if [[ "$recorded_pid" == "$$" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
}

formal_preflight() {
  [[ -x "$PY" ]] || { echo "ERROR: missing Python executable $PY" >&2; return 1; }
  [[ -d "$DATA" ]] || { echo "ERROR: missing data directory $DATA" >&2; return 1; }
  [[ -f "$TEACHER" ]] || { echo "ERROR: missing teacher checkpoint $TEACHER" >&2; return 1; }
  [[ -d "$SUA_RESULTS_DIR" ]] || {
    echo "ERROR: missing SUA E3 reference directory $SUA_RESULTS_DIR" >&2
    return 1
  }
  command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi is unavailable" >&2
    return 1
  }
  local gpu_ids
  gpu_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
  grep -qx "0" <<<"$gpu_ids" && grep -qx "1" <<<"$gpu_ids" || {
    echo "ERROR: GPU 0 and GPU 1 are both required" >&2
    return 1
  }
  if nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq "[0-9]"; then
    echo "ERROR: GPU 0 or GPU 1 already has a compute process" >&2
    return 1
  fi
  PYTHONPATH="$ROOT/sua_exploration:$ROOT/sua_exploration/scripts" \
    "$PY" -m pytest -q \
      "$ROOT/sua_exploration/tests/test_pseudomua_t4_bridge.py" \
      "$ROOT/sua_exploration/tests/test_aggregate_pseudomua_t4_bridge.py"
}

run_aggregation() {
  local aggregator=(
    "$PY" -u "$ROOT/sua_exploration/scripts/aggregate_pseudomua_t4_bridge.py"
    --pseudomua_results_dir "$RESULTS_DIR"
    --sua_results_dir "$SUA_RESULTS_DIR"
    --out_path "$SUMMARY_PATH"
    --effective_mean_delta 0.03
    --interaction_tolerance 0.03
  )
  emit_event \
    "aggregation_start" "" "" "" "$$" "" "" "$ORCHESTRATOR_LOG" "" \
    "$(shell_join "${aggregator[@]}")" ""
  local rc=0
  if PYTHONPATH="$ROOT/sua_exploration/scripts" "${aggregator[@]}"; then
    :
  else
    rc=$?
  fi
  emit_event \
    "aggregation_done" "" "" "" "$$" \
    "$([[ "$rc" -eq 0 ]] && printf completed || printf failed)" \
    "$rc" "$ORCHESTRATOR_LOG" "" "$(shell_join "${aggregator[@]}")" ""
  return "$rc"
}

run_self_test() {
  local selftest_dir
  local pool_rc
  selftest_dir="$(mktemp -d)"
  RESULTS_DIR="$selftest_dir"
  MANIFEST="$RESULTS_DIR/manifest.jsonl"
  RUN_ID="queue_selftest_$(date +%Y%m%dT%H%M%S)_$$"
  EXECUTION_KIND="selftest"
  JOBS=("slow 1" "fast 2" "mid 3" "fail 4" "tail 5")
  initialize_queue
  : >"$MANIFEST"
  if run_worker_pool; then
    pool_rc=0
  else
    pool_rc=$?
  fi
  if [[ "$pool_rc" -ne 1 ]]; then
    echo "ERROR: self-test expected worker-pool failure propagation, got rc=$pool_rc" >&2
    return 1
  fi
  "$PY" - "$MANIFEST" <<'PY'
import datetime
import json
import sys

events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
starts = {f"{e['group']}:{e['seed']}": e for e in events if e["event"] == "job_start"}
dones = {f"{e['group']}:{e['seed']}": e for e in events if e["event"] == "job_done"}
expected = {"slow:1", "fast:2", "mid:3", "fail:4", "tail:5"}
assert set(starts) == expected
assert set(dones) == expected
assert dones["fail:4"]["rc"] == 7
assert dones["tail:5"]["status"] == "completed"
parse = datetime.datetime.fromisoformat
# A later queued job must begin before the deliberately slow first job ends.
# This proves the free worker claimed more work instead of waiting in lockstep.
assert parse(starts["mid:3"]["time"]) < parse(dones["slow:1"]["time"])
print("SELF-TEST PASS: dynamic claim, queue drain after failure, failure propagation")
PY
  cleanup_queue
  rm -f "$MANIFEST"
  rmdir "$selftest_dir"
}

if [[ "$SELF_TEST" -eq 1 ]]; then
  run_self_test
  exit 0
fi

if [[ "$MODE" == "dry-run" ]]; then
  echo "screen_id=$SCREEN_ID jobs=${#JOBS[@]} protocol=validation_only_27_6_6_epochs_5_12"
  for job in "${JOBS[@]}"; do
    read -r group seed <<<"$job"
    build_job_commands "$group" "$seed"
    echo "[$group seed=$seed]"
    echo "  CUDA_VISIBLE_DEVICES=<worker> $(shell_join "${TRAIN_CMD[@]}")"
    echo "  CUDA_VISIBLE_DEVICES=<worker> $(shell_join "${EVAL_CMD[@]}")"
  done
  exit 0
fi

mkdir -p "$RESULTS_DIR/logs"

if [[ "$WAIT_FOR_COMPLETION" -eq 0 ]]; then
  if [[ -e "$PID_FILE" ]]; then
    if pid_file_is_live; then
      echo "ERROR: orchestrator already running with PID $(<"$PID_FILE")" >&2
    else
      echo "ERROR: stale/invalid PID file exists at $PID_FILE; inspect it before retrying" >&2
    fi
    exit 1
  fi
  nohup "$0" \
    --launch --wait --orchestrator-child --screen-id "$SCREEN_ID" \
    >"$ORCHESTRATOR_LOG" 2>&1 &
  child_pid=$!
  printf "%s\n" "$child_pid" >"$PID_FILE"
  echo "orchestrator PID $child_pid"
  echo "log $ORCHESTRATOR_LOG"
  exit 0
fi

if [[ "$ORCHESTRATOR_CHILD" -eq 1 ]]; then
  # The backgrounding parent owns PID-file creation.  Wait briefly for that
  # atomic hand-off and verify that the file refers to this process.
  for _ in $(seq 1 100); do
    [[ -f "$PID_FILE" ]] && break
    sleep 0.01
  done
  [[ -f "$PID_FILE" && "$(<"$PID_FILE")" == "$$" ]] || {
    echo "ERROR: background PID hand-off failed" >&2
    exit 1
  }
else
  if [[ -e "$PID_FILE" ]]; then
    if pid_file_is_live; then
      echo "ERROR: orchestrator already running with PID $(<"$PID_FILE")" >&2
    else
      echo "ERROR: stale/invalid PID file exists at $PID_FILE; inspect it before retrying" >&2
    fi
    exit 1
  fi
  printf "%s\n" "$$" >"$PID_FILE"
fi

trap cleanup_orchestrator EXIT INT TERM
formal_preflight

RUN_ID="${SCREEN_ID}_$(date +%Y%m%dT%H%M%S)_$$"
initialize_queue
emit_event \
  "screen_start" "" "" "" "$$" "running" "0" "$ORCHESTRATOR_LOG" "" "" \
  "F0/T4/TS4 x seeds 42/43/44; validation only; 27/6/6; pMUA; epochs 5-12"

pool_rc=0
if ! run_worker_pool; then
  pool_rc=1
fi
if [[ "$pool_rc" -ne 0 ]]; then
  emit_event \
    "screen_done" "" "" "" "$$" "failed" "$pool_rc" "$ORCHESTRATOR_LOG" "" "" \
    "At least one group/seed job failed; aggregation was not run"
  exit "$pool_rc"
fi

aggregation_rc=0
if ! run_aggregation; then
  aggregation_rc=1
fi
if [[ "$aggregation_rc" -ne 0 ]]; then
  emit_event \
    "screen_done" "" "" "" "$$" "failed_aggregation" "$aggregation_rc" \
    "$ORCHESTRATOR_LOG" "" "" "All jobs finished but strict aggregation failed"
  exit "$aggregation_rc"
fi

emit_event \
  "screen_done" "" "" "" "$$" "completed" "0" "$ORCHESTRATOR_LOG" "" "" \
  "All 9 jobs and strict aggregation completed"
echo "COMPLETED: $SUMMARY_PATH"
