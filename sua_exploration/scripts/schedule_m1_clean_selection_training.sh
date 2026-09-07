#!/usr/bin/env bash
# Schedule all nine M1 clean-selection training runs across two GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCREEN="m1_clean_selection_v1"
TRAIN="$ROOT/sua_exploration/scripts/run_m1_clean_selection_train_one_cell.sh"
PROTO="$ROOT/sua_exploration/results/$SCREEN/protocol_receipt.json"
LOG_DIR="$ROOT/sua_exploration/results/$SCREEN/logs"
mkdir -p "$LOG_DIR"

[[ -f "$PROTO" ]] || { echo "missing protocol receipt" >&2; exit 1; }
chmod +x "$TRAIN"

JOBS=(
  "f0 fold1_seed42 0"
  "f0 fold1_seed43 1"
  "f0 fold2_seed42 0"
  "t4 fold1_seed42 1"
  "t4 fold1_seed43 0"
  "t4 fold2_seed42 1"
  "ts4 fold1_seed42 0"
  "ts4 fold1_seed43 1"
  "ts4 fold2_seed42 0"
)

echo "[$(date -Is)] scheduling ${#JOBS[@]} clean-selection training jobs on 2 GPUs"
echo "protocol_sha=$(sha256sum "$PROTO" | awk '{print $1}')"

pids=()
for entry in "${JOBS[@]}"; do
  read -r group cell gpu <<<"$entry"
  "$TRAIN" --launch --group "$group" --cell "$cell" --gpu "$gpu" &
  pids+=($!)
  # Stagger launches slightly to avoid simultaneous data loading spikes.
  sleep 5
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "[$(date -Is)] all training jobs finished (fail=$fail)"
exit "$fail"
