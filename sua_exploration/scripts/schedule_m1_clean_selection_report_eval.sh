#!/usr/bin/env bash
# Run all 27 report-window evaluations with at most one job per GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL="$ROOT/sua_exploration/scripts/run_m1_clean_selection_report_eval_one.sh"
LOG_DIR="$ROOT/sua_exploration/results/m1_clean_selection_v1/logs"
mkdir -p "$LOG_DIR"
chmod +x "$EVAL"

SOURCES=(clean_best fixed_last frozen_best)
ARMS=(f0 t4 ts4)
CELLS=(fold1_seed42 fold1_seed43 fold2_seed42)

jobs=()
for source in "${SOURCES[@]}"; do
  for arm in "${ARMS[@]}"; do
    for cell in "${CELLS[@]}"; do
      jobs+=("$source|$arm|$cell")
    done
  done
done

run_one() {
  IFS='|' read -r source arm cell <<<"$1"
  local gpu="$2"
  "$EVAL" --launch --source "$source" --group "$arm" --cell "$cell" --gpu "$gpu"
}

echo "[$(date -Is)] scheduling ${#jobs[@]} report-window evaluations (2-GPU queue)"
idx=0
fail=0
while (( idx < ${#jobs[@]} )); do
  if (( idx + 1 < ${#jobs[@]} )); then
    run_one "${jobs[$idx]}" 0 &
    pid0=$!
    run_one "${jobs[$((idx + 1))]}" 1 &
    pid1=$!
    wait "$pid0" || fail=1
    wait "$pid1" || fail=1
    idx=$((idx + 2))
  else
    run_one "${jobs[$idx]}" 0 || fail=1
    idx=$((idx + 1))
  fi
done

echo "[$(date -Is)] report eval jobs finished (fail=$fail)"
exit "$fail"
