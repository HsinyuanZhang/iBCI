#!/usr/bin/env bash
# Continue the frozen three-seed SUA relation matrix without leaving a GPU idle
# after the already-running native-MUA M1 and SUA seed-42 queues finish.
#
# This scheduler is specific to the 2026-07-29 handoff:
#   GPU1: running seed 42 -> seed 44 {t4, relation}
#   GPU0: running final native-MUA M1 -> seed 43 full -> seed 44 {MS, NG}
#
# It never invokes a formal-test evaluator. All child jobs use the strict
# train/validation manifest enforced by run_sua_electrode_relation_pilot.sh.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
RUNNER="$ROOT/sua_exploration/scripts/run_sua_electrode_relation_pilot.sh"
AGG="$ROOT/sua_exploration/scripts/aggregate_sua_electrode_relation_pilot.py"
MULTI_AGG="$ROOT/sua_exploration/scripts/aggregate_sua_electrode_relation_multiseed.py"
STATE="$ROOT/sua_exploration/results/sua_electrode_relation_full_v1_scheduler"
MULTI_OUT="$STATE/multiseed_strict_aggregate.json"
MULTI_LOCK="$STATE/multiseed_strict_aggregate.lock"
M1_DONE="$ROOT/sua_exploration/results/native_mua_t4_v1/aggregate_m1.json"
S42_DONE="$ROOT/sua_exploration/results/sua_electrode_relation_pilot_v2/README.txt"
S43_ID="sua_electrode_relation_pilot_s43_v1"
S44_A_ID="sua_electrode_relation_pilot_s44_t4_rel_v1"
S44_B_ID="sua_electrode_relation_pilot_s44_controls_v1"
LAUNCH=0

if [ "${1:-}" = "--launch" ]; then
  LAUNCH=1
fi
if [ "$LAUNCH" -ne 1 ]; then
  echo "Refusing to schedule without --launch." >&2
  exit 2
fi
if [ -e "$STATE" ]; then
  echo "Refusing to reuse scheduler state: $STATE" >&2
  exit 2
fi
mkdir -p "$STATE"

wait_for_file() {
  local path="$1"
  local label="$2"
  local timeout_seconds="$3"
  local started="$SECONDS"
  while [ ! -f "$path" ]; do
    if [ $((SECONDS - started)) -ge "$timeout_seconds" ]; then
      echo "Timed out waiting for $label: $path" >&2
      return 1
    fi
    sleep 30
  done
}

(
  wait_for_file "$M1_DONE" "strict native-MUA M1 aggregate" 21600
  echo "[$(date --iso-8601=seconds)] GPU0 starting full seed 43"
  bash "$RUNNER" --launch --gpu 0 --seed 43 --screen-id "$S43_ID"
  echo "[$(date --iso-8601=seconds)] GPU0 starting seed 44 controls"
  bash "$RUNNER" --launch --gpu 0 --seed 44 \
    --arms membership_shuffle,no_group --screen-id "$S44_B_ID"
) > "$STATE/gpu0_queue.log" 2>&1 &
GPU0_QUEUE_PID=$!

(
  wait_for_file "$S42_DONE" "strict seed-42 queue completion" 43200
  echo "[$(date --iso-8601=seconds)] GPU1 starting seed 44 T4 and relation"
  bash "$RUNNER" --launch --gpu 1 --seed 44 \
    --arms t4,relation --screen-id "$S44_A_ID"
) > "$STATE/gpu1_queue.log" 2>&1 &
GPU1_QUEUE_PID=$!

printf '%s\n' "$GPU0_QUEUE_PID" > "$STATE/gpu0_queue.pid"
printf '%s\n' "$GPU1_QUEUE_PID" > "$STATE/gpu1_queue.pid"
wait "$GPU0_QUEUE_PID"
wait "$GPU1_QUEUE_PID"

"$PY" -u "$AGG" --seed 44 \
  --t4 "$ROOT/sua_exploration/results/$S44_A_ID/t4_s44.json" \
  --relation "$ROOT/sua_exploration/results/$S44_A_ID/relation_s44.json" \
  --membership-shuffle "$ROOT/sua_exploration/results/$S44_B_ID/membership_shuffle_s44.json" \
  --no-group "$ROOT/sua_exploration/results/$S44_B_ID/no_group_s44.json" \
  --out "$STATE/seed44_strict_aggregate.json"

exec 9>"$MULTI_LOCK"
flock 9
if [ ! -f "$MULTI_OUT" ]; then
  "$PY" -u "$MULTI_AGG" \
    --seed-aggregate "$ROOT/sua_exploration/results/sua_electrode_relation_pilot_v2/seed42_strict_aggregate.json" \
    --seed-aggregate "$ROOT/sua_exploration/results/$S43_ID/seed43_strict_aggregate.json" \
    --seed-aggregate "$STATE/seed44_strict_aggregate.json" \
    --expected-seeds 42,43,44 \
    --effective-mean-delta 0.03 \
    --out "$MULTI_OUT"
else
  echo "Strict multi-seed aggregate already finalized by the independent watcher: $MULTI_OUT"
fi
flock -u 9

printf '%s\n' \
  "Seeds 42, 43, and 44 validation-only queues completed; formal test was not evaluated." \
  > "$STATE/COMPLETE.txt"
