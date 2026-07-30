#!/usr/bin/env bash
# Two-lane scheduler for the predeclared M_T4=50, seed-42 mechanism screen.
#
# Lane 0 starts as soon as the seed-42 ordinary-T4 anchor and strict SUA
# B0/T4/TS4 aggregate are complete. Lane 1 deliberately waits for the seed-43
# ordinary-T4 anchor job to release GPU 1, then runs the two remaining controls.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
LANE="${LANE:?LANE must be 0 or 1}"
GPU="${GPU:?GPU is required}"
POLL_SECONDS="${POLL_SECONDS:-15}"
RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
MAIN_AGGREGATE="$ROOT/sua_exploration/results/sua_spint_t4_mainline_fp32_v1/aggregate.json"
ANCHOR_RESULT="$RESULTS/t4m50_s42.json"
GPU1_RELEASE="$RESULTS/t4m50_s43.json"
ANCHOR="$ROOT/sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt"
RUNNER="$ROOT/sua_exploration/scripts/run_sua_confidence_film_one_cell.sh"
SCHEDULER_LOG="$RESULTS/logs/scheduler_lane${LANE}.log"

if [[ "$LANE" != "0" && "$LANE" != "1" ]]; then
  echo "LANE must be 0 or 1, got $LANE" >&2
  exit 2
fi

mkdir -p "$(dirname "$SCHEDULER_LOG")"
exec >"$SCHEDULER_LOG" 2>&1
echo "[$(date -Is)] waiting lane=$LANE gpu=$GPU"

while [[ ! -f "$MAIN_AGGREGATE" || ! -f "$ANCHOR_RESULT" ]]; do
  sleep "$POLL_SECONDS"
done

python3 - "$MAIN_AGGREGATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("main_claim_pass"):
    raise SystemExit(f"{path}: T4-vs-B0 strict gate did not pass")
if not payload.get("label_information_control_pass"):
    raise SystemExit(f"{path}: T4-vs-TS4 strict label-content gate did not pass")
if payload.get("formal_test_files_opened") is not False:
    raise SystemExit(f"{path}: formal-test seal is not intact")
PY

[[ -f "$ANCHOR" ]] || {
  echo "Missing final-epoch seed-42 T4 anchor: $ANCHOR" >&2
  exit 1
}

if [[ "$LANE" == "0" ]]; then
  ARMS=(film t4_continuation film_ts4)
else
  while [[ ! -f "$GPU1_RELEASE" ]]; do
    sleep "$POLL_SECONDS"
  done
  ARMS=(confidence_shuffle nofilm_match)
fi

for arm in "${ARMS[@]}"; do
  echo "[$(date -Is)] launch arm=$arm lane=$LANE gpu=$GPU"
  env ARM="$arm" SEED=42 GPU="$GPU" M_T4=50 ANCHOR="$ANCHOR" \
    "$RUNNER" --launch
  echo "[$(date -Is)] completed arm=$arm lane=$LANE gpu=$GPU"
done

if [[ "$LANE" == "1" ]]; then
  echo "[$(date -Is)] launch ordinary-T4 M50 seed=44 anchor preparation gpu=$GPU"
  env SEED=44 GPU="$GPU" M_T4=50 \
    "$ROOT/sua_exploration/scripts/run_sua_t4_budget_baseline_one_cell.sh" --launch
  echo "[$(date -Is)] completed ordinary-T4 M50 seed=44 anchor preparation gpu=$GPU"
fi

echo "[$(date -Is)] lane complete lane=$LANE gpu=$GPU"
