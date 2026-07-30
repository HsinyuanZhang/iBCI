#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m1_external_attention_replication.sh [--dry-run|--launch] [--wait] --screen-id ID

Runs a source-matched four-variant M1 internal-LOSO replication on FALCON 000941.
It excludes M1 held-out sessions from both fit and test, so it does not consume
the M1 formal held-out evaluation scope.
EOF
}

MODE="dry-run"
WAIT_FOR_WORKERS=false
SCREEN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --launch) MODE="launch"; shift ;;
    --wait) WAIT_FOR_WORKERS=true; shift ;;
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

[[ -d "$ROOT_DIR/SPINT-main/data/000941" ]] || { echo "Missing M1 data directory" >&2; exit 1; }

run_cell() {
  local gpu="$1" variant="$2" fold="$3" seed="$4"
  local lower_variant="${variant,,}"
  local experiment="${lower_variant}_m1_loso_internal"
  local run_name="${SCREEN_ID}_${lower_variant}_m1"
  local log_file="$LOG_DIR/m1_${lower_variant}_f${fold}_s${seed}.log"
  local command=(
    "$PYTHON_BIN" src/train.py
    "experiment=$experiment"
    "data.loso_fold=$fold"
    "seed=$seed"
    "run_id=$run_name"
    data.include_heldout_in_fit=false
    data.include_heldout_in_test=false
    trainer.accelerator=gpu
    trainer.devices=1
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu M1 $variant fold $fold seed $seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start M1 variant=$variant fold=$fold seed=$seed gpu=$gpu"
    (
      cd "$MUA_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_m1_${gpu}" "${command[@]}"
    )
    echo "[$(date -Is)] complete M1 variant=$variant fold=$fold seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

gpu0_queue() {
  run_cell 0 B3 0 42
  run_cell 0 B15P 0 42
  run_cell 0 B15D 0 42
  run_cell 0 B15 0 42
  run_cell 0 B3 1 42
  run_cell 0 B15P 1 42
}

gpu1_queue() {
  run_cell 1 B3 0 43
  run_cell 1 B15P 0 43
  run_cell 1 B15D 0 43
  run_cell 1 B15 0 43
  run_cell 1 B15D 1 42
  run_cell 1 B15 1 42
}

if [[ "$MODE" == "dry-run" ]]; then
  gpu0_queue
  gpu1_queue
  exit 0
fi

mkdir -p "$LOG_DIR"
nvidia-smi -L >"$RESULTS_DIR/gpu_inventory.txt"
{
  echo "screen_id=$SCREEN_ID"
  echo "started_at=$(date -Is)"
  echo "protocol=FALCON M1 000941 current-source internal LOSO; held-out excluded from fit and test"
} >"$RESULTS_DIR/manifest.env"

for variant in b3 b15p b15d b15; do
  for fold_seed in 0:42 0:43 1:42; do
    fold="${fold_seed%%:*}"
    seed="${fold_seed##*:}"
    pattern="$MUA_ROOT/outputs/streaming_calibration/${SCREEN_ID}_${variant}_m1_f${fold}_s${seed}_*"
    if compgen -G "$pattern" > /dev/null; then
      echo "Refusing to overwrite existing M1 artifact: $pattern" >&2
      exit 1
    fi
  done
done

gpu0_queue >"$LOG_DIR/worker_gpu0.log" 2>&1 &
pid0=$!
gpu1_queue >"$LOG_DIR/worker_gpu1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" >"$RESULTS_DIR/worker_gpu0.pid"
printf '%s\n' "$pid1" >"$RESULTS_DIR/worker_gpu1.pid"
echo "Launched M1 replication $SCREEN_ID with workers $pid0 and $pid1"

if [[ "$WAIT_FOR_WORKERS" == true ]]; then
  set +e
  wait "$pid0"
  worker0_status=$?
  wait "$pid1"
  worker1_status=$?
  set -e
  if [[ "$worker0_status" -ne 0 || "$worker1_status" -ne 0 ]]; then
    echo "M1 workers failed: GPU0=$worker0_status GPU1=$worker1_status" >&2
    exit 1
  fi
  echo "M1 replication $SCREEN_ID completed successfully"
fi
