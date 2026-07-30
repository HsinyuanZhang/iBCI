#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: recover_attention_architecture_m2.sh [--dry-run|--launch] --screen-id ID

Regenerates all nine non-B3 M2 internal-LOSO cells after a runner-level
failure. Both fit and test keep FALCON held-out sessions disabled; the test
pass is held-in validation only.
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
GPU0_CELLS=("B15P:1:42" "B15D:1:42" "B15:1:42" "B15P:2:42")
GPU1_CELLS=("B15P:1:43" "B15D:1:43" "B15:1:43" "B15D:2:42" "B15:2:42")

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python executable: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$MUA_ROOT" ]] || { echo "Missing MUA root: $MUA_ROOT" >&2; exit 1; }

run_cell() {
  local gpu="$1" variant="$2" fold="$3" seed="$4"
  local lower_variant="${variant,,}"
  local run_name="${SCREEN_ID}_${lower_variant}_m2"
  local log_file="$LOG_DIR/mua_${lower_variant}_f${fold}_s${seed}.log"
  local command=(
    "$PYTHON_BIN" src/train.py
    "experiment=${lower_variant}_m2_loso_internal"
    "data.loso_fold=$fold"
    "seed=$seed"
    "run_id=$run_name"
    data.include_heldout_in_fit=false
    data.include_heldout_in_test=false
    trainer.accelerator=gpu
    trainer.devices=1
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu MUA $variant fold $fold seed $seed"
    printf '  %q ' "CUDA_VISIBLE_DEVICES=$gpu" "${command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start recovered MUA variant=$variant fold=$fold seed=$seed gpu=$gpu"
    (
      cd "$MUA_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mua_recovery_${gpu}" "${command[@]}"
    )
    echo "[$(date -Is)] complete recovered MUA variant=$variant fold=$fold seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

for cell in "${GPU0_CELLS[@]}" "${GPU1_CELLS[@]}"; do
  IFS=: read -r variant fold seed <<<"$cell"
  lower_variant="${variant,,}"
  pattern="$MUA_ROOT/outputs/streaming_calibration/${SCREEN_ID}_${lower_variant}_m2_f${fold}_s${seed}_*/run_metadata.json"
  if compgen -G "$pattern" > /dev/null; then
    echo "Refusing to overwrite completed M2 artifact: $pattern" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
run_queue() {
  local gpu="$1"
  shift
  for cell in "$@"; do
    IFS=: read -r variant fold seed <<<"$cell"
    run_cell "$gpu" "$variant" "$fold" "$seed"
  done
}

if [[ "$MODE" == "dry-run" ]]; then
  run_queue 0 "${GPU0_CELLS[@]}"
  run_queue 1 "${GPU1_CELLS[@]}"
  exit 0
fi

run_queue 1 "${GPU1_CELLS[@]}" &
gpu1_pid=$!
run_queue 0 "${GPU0_CELLS[@]}"
wait "$gpu1_pid"
