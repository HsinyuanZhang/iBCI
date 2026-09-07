#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python3.10
ORIGINAL=$ROOT/sua_exploration/xinyuan/M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json
CANONICAL=$ROOT/sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json
RECOVERY=$ROOT/sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_STATE_PATH_RECOVERY_v1.json
GATE=$ROOT/sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_GATE_v2.json
RECEIPT=$ROOT/sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_STAGED_RECEIPT_v2.json

for _ in $(seq 1 720); do
  [[ -f "$ORIGINAL" ]] && break
  sleep 30
done
[[ -f "$ORIGINAL" ]]

RUNNER_ARGV_JSON='["/home/xinyuan/miniconda3/envs/spint/bin/python3.10","sua_exploration/m1_compact_replication/fold2_runner_v2.py","--execute","--receipt","/home/xinyuan/Work_host/SPINT/sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_STAGED_RECEIPT_v2.json","--state","/home/xinyuan/Work_host/SPINT/sua_exploration/xinyuan/M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json","--allow-root-launch","FOLD2_APPROVED_BY_ROOT"]'
"$PY" "$ROOT/sua_exploration/m1_compact_replication/fold2_state_path_recovery.py" \
  --original "$ORIGINAL" --canonical "$CANONICAL" --receipt "$RECOVERY" \
  --runner-argv-json "$RUNNER_ARGV_JSON"

[[ "$(sha256sum "$ORIGINAL" | cut -d' ' -f1)" == "$(sha256sum "$CANONICAL" | cut -d' ' -f1)" ]]
CUDA_VISIBLE_DEVICES=-1 PYTHONNOUSITE=1 PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/sua_exploration/m1_compact_replication/fold2_evaluate.py" \
  --state "$CANONICAL" --output "$GATE" --receipt "$RECEIPT"
echo PATH_RECOVERY_AND_EVALUATION_COMPLETE
