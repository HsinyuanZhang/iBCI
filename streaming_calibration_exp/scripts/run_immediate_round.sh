#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX="${MATRIX:-outputs/streaming_calibration/gate2_revised_matrix.csv}"
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
LOG_DIR="logs/screen_runs"
mkdir -p "$LOG_DIR"

echo "== Phase R0 pre-run checks (revised plan) =="
bash scripts/pre_run_checks_revised.sh

echo
echo "== Register existing B3-D64 task+y+E anchor as loss-ablation reference =="
REFERENCE_ANCHOR="${REFERENCE_ANCHOR:-outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020}"
if [[ ! -d "$REFERENCE_ANCHOR" ]]; then
  echo "Reference anchor not found: $REFERENCE_ANCHOR" >&2
  echo "Set REFERENCE_ANCHOR to the completed B3-D64 task+y+E fold-0 artifact directory." >&2
  exit 1
fi
python scripts/update_gate2_matrix.py "$REFERENCE_ANCHOR" \
  --matrix "$MATRIX" \
  --comparison-role loss_ablation_reference \
  --notes "Reused anchor; no retrain per revised plan section 5.1"

run_and_register() {
  local experiment="$1"
  local notes="$2"
  shift 2
  echo
  echo "[$(date -Is)] Starting ${experiment} LOSO fold=${FOLD} seed=${SEED}"
  bash scripts/run_loso_fold.sh "$experiment" "$FOLD" "seed=${SEED}" "$@"
  local latest
  latest="$(ls -1dt outputs/streaming_calibration/${experiment}_f${FOLD}_s${SEED}_* 2>/dev/null | head -1)"
  if [[ -z "$latest" ]]; then
    latest="$(ls -1dt outputs/streaming_calibration/${experiment}_s${SEED}_* 2>/dev/null | head -1)"
  fi
  if [[ -z "$latest" ]]; then
    echo "Could not locate artifact directory for ${experiment}" >&2
    exit 1
  fi
  python scripts/update_gate2_matrix.py "$latest" --matrix "$MATRIX" --notes "$notes" --refresh-d512-deltas
  echo "[$(date -Is)] Registered ${latest}"
}

echo
echo "== Phase R1 immediate round (sequential, no GPU concurrency) =="
run_and_register b2_d512_protocol_control "R1 order 1: protocol control"
run_and_register b3_d64_task_only "R1 order 2: loss ablation task_only"
run_and_register b3_d64_task_plus_y "R1 order 3: loss ablation task_plus_y"

echo
echo "== Phase R1 decision summary (informational; exits non-zero until R1 complete) =="
set +e
python scripts/evaluate_r1_decisions.py --matrix "$MATRIX" --fold "$FOLD" --seed "$SEED" --emit-loss-overrides
decision_status=$?
set -e
echo "evaluate_r1_decisions.py exit code: ${decision_status}"
echo
echo "Immediate round complete. Matrix: $MATRIX"
echo "Review gate2_revised_matrix.csv before any conditional round."
exit 0
