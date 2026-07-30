#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_current_mua_b3_baselines.sh [--dry-run|--launch] --screen-id ID

Regenerates the three B3 M2 internal-LOSO cells under the current source tree.
These are the source-matched controls for B15P/B15D/B15 architecture screening.
It never includes FALCON external held-out sessions in fit or test.
EOF
}

MODE="dry-run"
SCREEN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --launch) MODE="launch"; shift ;;
    --screen-id) SCREEN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SCREEN_ID" ]] || { echo "--screen-id is required" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
MUA_ROOT="$ROOT_DIR/streaming_calibration_exp"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS_DIR/logs"

run_cell() {
  local gpu="$1" fold="$2" seed="$3"
  local run_name="${SCREEN_ID}_b3_m2"
  local log_file="$LOG_DIR/mua_b3_f${fold}_s${seed}.log"
  local command=(
    "$PYTHON_BIN" src/train.py
    experiment=b3_m2_loso_internal
    "data.loso_fold=$fold"
    "seed=$seed"
    "run_id=$run_name"
    data.include_heldout_in_fit=false
    data.include_heldout_in_test=false
    trainer.accelerator=gpu
    trainer.devices=1
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu MUA B3 fold $fold seed $seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start MUA B3 fold=$fold seed=$seed gpu=$gpu"
    (
      cd "$MUA_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mua_b3_${gpu}" "${command[@]}"
    )
    echo "[$(date -Is)] complete MUA B3 fold=$fold seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

gpu0_queue() {
  run_cell 0 1 42
  run_cell 0 2 42
}

gpu1_queue() {
  run_cell 1 1 43
}

if [[ "$MODE" == "dry-run" ]]; then
  gpu0_queue
  gpu1_queue
  exit 0
fi

mkdir -p "$LOG_DIR"
for fold_seed in 1:42 1:43 2:42; do
  fold="${fold_seed%%:*}"
  seed="${fold_seed##*:}"
  pattern="$MUA_ROOT/outputs/streaming_calibration/${SCREEN_ID}_b3_m2_f${fold}_s${seed}_*"
  if compgen -G "$pattern" > /dev/null; then
    echo "Refusing to overwrite existing current-source B3 artifact: $pattern" >&2
    exit 1
  fi
done

gpu1_queue &
gpu1_pid=$!
gpu0_queue
wait "$gpu1_pid"
