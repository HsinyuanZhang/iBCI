#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: watch_attention_architecture_screen.sh --screen-id ID [--tmux-session NAME]

Waits for a completed primary attention screen, then advances only through
pre-registered aggregate gates: source-matched M2 B3 supplement, pseudo-MUA
bridge, and M1 internal-LOSO replication. It never launches formal held-out
evaluation.
EOF
}

SCREEN_ID=""
MAIN_TMUX_SESSION="${MAIN_TMUX_SESSION:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --screen-id) SCREEN_ID="$2"; shift 2 ;;
    --tmux-session) MAIN_TMUX_SESSION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SCREEN_ID" ]] || { echo "--screen-id is required" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
MAIN_TMUX_SESSION="${MAIN_TMUX_SESSION:-spint_${SCREEN_ID}}"
LOG_PATH="$RESULTS_DIR/pipeline_watcher.log"

mkdir -p "$RESULTS_DIR"
exec >"$LOG_PATH" 2>&1

read_gate() {
  local aggregate_path="$1" gate_name="$2"
  "$PYTHON_BIN" -c \
    'import json, sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["gates"][sys.argv[2]]).lower())' \
    "$aggregate_path" "$gate_name"
}

echo "[$(date -Is)] waiting for primary tmux session $MAIN_TMUX_SESSION"
while tmux has-session -t "=$MAIN_TMUX_SESSION" 2>/dev/null; do
  sleep 60
done
echo "[$(date -Is)] primary screen session ended"

sua_count="$(find "$RESULTS_DIR" -maxdepth 1 -type f -name 'sua_*_s*.json' | wc -l)"
mua_root="$ROOT_DIR/streaming_calibration_exp/outputs/streaming_calibration"
mua_count="$("$PYTHON_BIN" - "$mua_root" "$SCREEN_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
screen_id = sys.argv[2]
variants = {"B15P", "B15D", "B15"}
count = 0
for metadata_path in root.glob("*/run_metadata.json"):
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", ""))
    if run_id.startswith(f"{screen_id}_") and "_m2_" in run_id and payload.get("variant") in variants:
        count += 1
print(count)
PY
)"
echo "primary artifacts: sua=$sua_count, non-B3 M2=$mua_count"
if [[ "$sua_count" -ne 8 || "$mua_count" -ne 9 ]]; then
  echo "Primary prerequisites incomplete; stopping before B3 supplement." >&2
  exit 1
fi

echo "[$(date -Is)] running source-matched M2 B3 supplement"
bash "$ROOT_DIR/sua_exploration/scripts/run_current_mua_b3_baselines.sh" \
  --launch --screen-id "$SCREEN_ID"

echo "[$(date -Is)] aggregating primary screen"
"$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/aggregate_attention_architecture_screen.py" \
  --screen-id "$SCREEN_ID"
primary_aggregate="$RESULTS_DIR/aggregate.json"
advance="$(read_gate "$primary_aggregate" advance_to_paired_pilot)"
echo "advance_to_paired_pilot=$advance"
if [[ "$advance" != true ]]; then
  echo "Primary gate did not pass; pseudo-MUA and M1 were intentionally not launched."
  exit 0
fi

pseudo_screen="${SCREEN_ID}_pseudomua"
echo "[$(date -Is)] running pseudo-MUA bridge: $pseudo_screen"
bash "$ROOT_DIR/sua_exploration/scripts/run_pseudo_mua_attention_pilot.sh" \
  --launch --wait --screen-id "$pseudo_screen"
echo "[$(date -Is)] aggregating pseudo-MUA bridge"
"$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/aggregate_pseudo_mua_attention_pilot.py" \
  --parent-screen-id "$SCREEN_ID" --screen-id "$pseudo_screen"
pseudo_aggregate="$ROOT_DIR/sua_exploration/results/$pseudo_screen/aggregate.json"
advance="$(read_gate "$pseudo_aggregate" advance_to_external_mua_replication)"
echo "advance_to_external_mua_replication=$advance"
if [[ "$advance" != true ]]; then
  echo "Pseudo-MUA gate did not pass; M1 was intentionally not launched."
  exit 0
fi

m1_screen="${SCREEN_ID}_m1"
echo "[$(date -Is)] running M1 internal-LOSO replication: $m1_screen"
bash "$ROOT_DIR/sua_exploration/scripts/run_m1_external_attention_replication.sh" \
  --launch --wait --screen-id "$m1_screen"
echo "[$(date -Is)] aggregating M1 replication"
"$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/aggregate_m1_external_attention_replication.py" \
  --parent-screen-id "$pseudo_screen" --screen-id "$m1_screen"
echo "[$(date -Is)] gated pipeline completed without formal held-out evaluation"
