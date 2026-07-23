#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX="${MATRIX:-outputs/streaming_calibration/gate2_revised_matrix.csv}"
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <winning_loss_mode> [extra hydra overrides...]" >&2
  echo "Example: $0 task_plus_y" >&2
  exit 1
fi

WINNING_LOSS="$1"
shift

echo "== Conditional round preflight: require formal R1 winner =="
DECISION_JSON="$(python scripts/evaluate_r1_decisions.py --matrix "$MATRIX" --fold "$FOLD" --seed "$SEED" --require-ready --require-winner --emit-loss-overrides)"
echo "$DECISION_JSON"

FORMAL_WINNER="$(python - <<'PY' "$DECISION_JSON"
import json, sys
print(json.loads(sys.argv[1])["winning_loss"])
PY
)"

if [[ "$WINNING_LOSS" != "$FORMAL_WINNER" ]]; then
  echo "CLI loss '${WINNING_LOSS}' does not match formal winner '${FORMAL_WINNER}'." >&2
  exit 1
fi

case "$WINNING_LOSS" in
  task_only)
    LOSS_OVERRIDES=(model.loss_mode=task_only model.lambda_y=0.0 model.lambda_E=0.0)
    ;;
  task_plus_y)
    LOSS_OVERRIDES=(model.loss_mode=task_plus_y model.lambda_y=1.0 model.lambda_E=0.0)
    ;;
  task_plus_y_plus_E)
    LOSS_OVERRIDES=(model.loss_mode=task_plus_y_plus_E model.lambda_y=1.0 model.lambda_E=0.1)
    ;;
  *)
    echo "Unsupported loss mode: $WINNING_LOSS" >&2
    exit 1
    ;;
esac

run_and_register() {
  local experiment="$1"
  local notes="$2"
  shift 2
  echo
  echo "[$(date -Is)] Starting ${experiment} LOSO fold=${FOLD} seed=${SEED}"
  bash scripts/run_loso_fold.sh "$experiment" "$FOLD" "seed=${SEED}" "${LOSS_OVERRIDES[@]}" "$@"
  local latest
  latest="$(ls -1dt outputs/streaming_calibration/${experiment}_f${FOLD}_s${SEED}_* 2>/dev/null | head -1)"
  if [[ -z "$latest" ]]; then
    latest="$(ls -1dt outputs/streaming_calibration/${experiment}_s${SEED}_* 2>/dev/null | head -1)"
  fi
  python scripts/update_gate2_matrix.py "$latest" --matrix "$MATRIX" --notes "$notes" --refresh-d512-deltas
}

run_and_register b3_d128_gate2 "R2: B3-D128 capacity recovery"
run_and_register b5_ema_r4_loso_probe "R4: B5 cubic hardware probe"
run_and_register b6_fir_r4_k5_loso_probe "R4: B6 cubic hardware probe"

echo
echo "Conditional round complete. Matrix: $MATRIX"
