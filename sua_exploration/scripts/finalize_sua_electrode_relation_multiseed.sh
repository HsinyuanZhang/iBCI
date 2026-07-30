#!/usr/bin/env bash
# Wait for the three frozen single-seed relation aggregates and emit one strict
# cross-seed decision. This watcher reads JSON/metadata only; it never opens NWB.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
AGG="$ROOT/sua_exploration/scripts/aggregate_sua_electrode_relation_multiseed.py"
STATE="$ROOT/sua_exploration/results/sua_electrode_relation_full_v1_scheduler"
S42="$ROOT/sua_exploration/results/sua_electrode_relation_pilot_v2/seed42_strict_aggregate.json"
S43="$ROOT/sua_exploration/results/sua_electrode_relation_pilot_s43_v1/seed43_strict_aggregate.json"
S44="$STATE/seed44_strict_aggregate.json"
OUT="$STATE/multiseed_strict_aggregate.json"
LOCK="$STATE/multiseed_strict_aggregate.lock"

if [ "${1:-}" != "--launch" ]; then
  echo "Refusing to finalize without --launch." >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing multi-seed result: $OUT" >&2
  exit 2
fi

wait_for_file() {
  local path="$1"
  local timeout_seconds="$2"
  local started="$SECONDS"
  while [ ! -f "$path" ]; do
    if [ $((SECONDS - started)) -ge "$timeout_seconds" ]; then
      echo "Timed out waiting for strict seed aggregate: $path" >&2
      return 1
    fi
    sleep 30
  done
}

wait_for_file "$S42" 43200
wait_for_file "$S43" 43200
wait_for_file "$S44" 43200

exec 9>"$LOCK"
flock 9
if [ ! -f "$OUT" ]; then
  "$PY" -u "$AGG" \
    --seed-aggregate "$S42" \
    --seed-aggregate "$S43" \
    --seed-aggregate "$S44" \
    --expected-seeds 42,43,44 \
    --effective-mean-delta 0.03 \
    --out "$OUT"
else
  echo "Strict multi-seed aggregate already finalized by the scheduler: $OUT"
fi
flock -u 9
