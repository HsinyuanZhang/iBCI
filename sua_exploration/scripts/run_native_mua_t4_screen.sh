#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_native_mua_t4_screen.sh [--dry-run|--launch] --screen-id ID [--task m1|m2|both]

Runs matched native-FALCON MUA F0/T4/TS4 internal-LOSO cells.  T4 reads target
metadata only from held-in calibration NWBs; held-out FALCON sessions are excluded
from fit and test.  The frozen minimum matrix is folds 1/2 with seeds 42/43 as
available: (1,42), (1,43), (2,42), for each group and task.
EOF
}

MODE=dry-run
SCREEN_ID=""
TASK=both
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --launch) MODE=launch; shift ;;
    --screen-id) SCREEN_ID="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$SCREEN_ID" ]] || { echo "--screen-id is required" >&2; exit 2; }
[[ "$TASK" =~ ^(m1|m2|both)$ ]] || { echo "--task must be m1, m2, or both" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MUA_ROOT="$ROOT_DIR/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS_DIR/logs"
CELLS=("1:42" "1:43" "2:42")

tasks=()
[[ "$TASK" == m1 || "$TASK" == both ]] && tasks+=(m1)
[[ "$TASK" == m2 || "$TASK" == both ]] && tasks+=(m2)

for task in "${tasks[@]}"; do
  data_id=000941; [[ "$task" == m2 ]] && data_id=000953
  [[ -d "$ROOT_DIR/SPINT-main/data/$data_id" ]] || { echo "Missing FALCON $task data" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }

experiment_for() {
  local group="$1" task="$2"
  case "$group" in
    f0) echo "b3_native_mua_f0_${task}_loso_internal" ;;
    t4) echo "b3s_t4_${task}_loso_internal" ;;
    ts4) echo "b3s_ts4_${task}_loso_internal" ;;
    *) return 2 ;;
  esac
}

run_cell() {
  local gpu="$1" group="$2" task="$3" fold="$4" seed="$5"
  local experiment run_name log_file
  experiment="$(experiment_for "$group" "$task")"
  run_name="${SCREEN_ID}_${group}_${task}"
  log_file="$LOG_DIR/${task}_${group}_f${fold}_s${seed}.log"
  local command=(
    "$PYTHON_BIN" src/train.py "experiment=$experiment" "data.loso_fold=$fold"
    "seed=$seed" "run_id=$run_name" data.include_heldout_in_fit=false
    data.include_heldout_in_test=false trainer.accelerator=gpu trainer.devices=1
    require_baseline_validation=false
  )
  if [[ "$MODE" == dry-run ]]; then
    echo "GPU $gpu native-MUA $task $group fold=$fold seed=$seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start task=$task group=$group fold=$fold seed=$seed gpu=$gpu"
    (cd "$MUA_ROOT" && CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_${task}_${gpu}" "${command[@]}")
    echo "[$(date -Is)] complete task=$task group=$group fold=$fold seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

# Put the first complete paired contrast ahead of all replication cells.  With both
# tasks requested, GPU0 receives M1 and GPU1 M2, each beginning f0->t4->ts4 for
# fold1/seed42.  This produces the earliest diagnostic result without making any
# selection decision from that one cell; the remaining frozen cells still run later.
build_task_queue() {
  local task="$1" cell group fold seed
  task_queue=()
  for group in f0 t4 ts4; do
    task_queue+=("$task:$group:1:42")
  done
  for group in f0 t4 ts4; do
    for cell in "${CELLS[@]}"; do
      IFS=: read -r fold seed <<<"$cell"
      [[ "$fold:$seed" == "1:42" ]] && continue
      task_queue+=("$task:$group:$fold:$seed")
    done
  done
}

m1_cells=()
m2_cells=()
if [[ "$TASK" == m1 || "$TASK" == both ]]; then build_task_queue m1; m1_cells=("${task_queue[@]}"); fi
if [[ "$TASK" == m2 || "$TASK" == both ]]; then build_task_queue m2; m2_cells=("${task_queue[@]}"); fi
all_cells=("${m1_cells[@]}" "${m2_cells[@]}")

if [[ "$MODE" == launch ]]; then
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader >/dev/null || {
    echo "GPU preflight failed; refusing launch" >&2; exit 1;
  }
  process_pattern='src/train.py.*native_mua'
  if [[ "$TASK" != both ]]; then
    process_pattern="${process_pattern}.*_${TASK}"
  fi
  if pgrep -af "$process_pattern" >/dev/null; then
    echo "Existing native-MUA T4 process found for task=$TASK; refusing overlapping launch" >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  for cell in "${all_cells[@]}"; do
    IFS=: read -r task group fold seed <<<"$cell"
    pattern="$MUA_ROOT/outputs/streaming_calibration/${SCREEN_ID}_${group}_${task}_f${fold}_s${seed}_*/run_metadata.json"
    compgen -G "$pattern" >/dev/null && { echo "Refusing to overwrite $pattern" >&2; exit 1; }
  done
  {
    echo "screen_id=$SCREEN_ID"
    echo "started_at=$(date -Is)"
    echo "protocol=native FALCON M1/M2 T4; held-in calibration target labels only; internal LOSO; held-out excluded"
    printf 'cells=%s\n' "${all_cells[*]}"
  } >"$RESULTS_DIR/manifest.env"
fi

queue() {
  local gpu="$1"
  shift
  for cell in "$@"; do
    IFS=: read -r task group fold seed <<<"$cell"
    run_cell "$gpu" "$group" "$task" "$fold" "$seed"
  done
}

if [[ "$MODE" == dry-run ]]; then
  if [[ "$TASK" == both ]]; then
    queue 0 "${m1_cells[@]}"
    queue 1 "${m2_cells[@]}"
  elif [[ "$TASK" == m1 ]]; then
    queue 0 "${m1_cells[@]}"
  else
    queue 0 "${m2_cells[@]}"
  fi
  exit 0
fi

if [[ "$TASK" == both ]]; then
  queue 0 "${m1_cells[@]}" >"$LOG_DIR/worker_gpu0.log" 2>&1 & pid0=$!
  queue 1 "${m2_cells[@]}" >"$LOG_DIR/worker_gpu1.log" 2>&1 & pid1=$!
elif [[ "$TASK" == m1 ]]; then
  queue 0 "${m1_cells[@]}" >"$LOG_DIR/worker_gpu0.log" 2>&1 & pid0=$!
  pid1=""
else
  queue 0 "${m2_cells[@]}" >"$LOG_DIR/worker_gpu0.log" 2>&1 & pid0=$!
  pid1=""
fi
printf '%s\n' "$pid0" >"$RESULTS_DIR/worker_gpu0.pid"
if [[ -n "$pid1" ]]; then printf '%s\n' "$pid1" >"$RESULTS_DIR/worker_gpu1.pid"; fi
echo "Launched $SCREEN_ID: gpu0 PID=$pid0${pid1:+ gpu1 PID=$pid1}"

# Keep the supervisor alive until both queues finish.  A previous fire-and-forget
# version returned immediately; launchers that clean up a completed process group could
# then reap the workers before Python even initialized CUDA.  Waiting here also gives
# the detached supervisor a reliable final status artifact.
status0=0
status1=0
wait "$pid0" || status0=$?
if [[ -n "$pid1" ]]; then
  wait "$pid1" || status1=$?
fi
{
  echo "gpu0_status=$status0"
  echo "gpu1_status=$status1"
  echo "completed_at=$(date -Is)"
} >"$RESULTS_DIR/worker_status.env"
if (( status0 != 0 || status1 != 0 )); then
  exit 1
fi
