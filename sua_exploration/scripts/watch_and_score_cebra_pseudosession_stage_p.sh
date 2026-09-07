#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
RESULT="$SUA/results/cebra_pseudosession_sua_v1"
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
SCORER="$SUA/scripts/score_cebra_pseudosession_sua.py"
AGGREGATOR="$SUA/scripts/aggregate_cebra_pseudosession_sua_stage_p.py"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUA"

completion_t4="$RESULT/cell_completion_mix_t4_s42.json"
completion_z4="$RESULT/cell_completion_mix_z4_s42.json"

while [[ ! -f "$completion_t4" || ! -f "$completion_z4" ]]; do
  for receipt in "$completion_t4" "$completion_z4"; do
    if [[ -f "$receipt" ]] && ! jq -e '.successful == true and .return_code == 0' "$receipt" >/dev/null; then
      echo "FAIL_CLOSED unsuccessful source completion: $receipt" >&2
      exit 2
    fi
  done
  sleep 60
done

jq -e '.successful == true and .return_code == 0' "$completion_t4" >/dev/null
jq -e '.successful == true and .return_code == 0' "$completion_z4" >/dev/null

score_arm() {
  local arm="$1"
  local gpu="$2"
  local log="$RESULT/logs/score_mix_${arm}_s42.log"
  test ! -e "$log"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$SCORER" \
    --arm "$arm" --seed 42 --domain within_subject --device cuda:0 --execute \
    >"$log" 2>&1
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$SCORER" \
    --arm "$arm" --seed 42 --domain external_subject_M --device cuda:0 --execute \
    >>"$log" 2>&1
}

score_arm t4 0 &
pid_t4=$!
score_arm z4 1 &
pid_z4=$!
wait "$pid_t4"
wait "$pid_z4"

"$PY" -u "$AGGREGATOR" --execute
