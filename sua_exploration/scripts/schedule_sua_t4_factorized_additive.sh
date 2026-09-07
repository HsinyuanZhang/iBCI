#!/usr/bin/env bash
# Launch the additive control on the first GPU whose primary arm finishes cleanly.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SCREEN="sua_t4_factorized_logit_residual_v1"
RESULTS="$ROOT/sua_exploration/results/$SCREEN"
RUNNER="$ROOT/sua_exploration/scripts/run_sua_t4_factorized_logit_one_arm.sh"
SCHEDULER_LOG="$RESULTS/logs/additive_scheduler.log"
ALIGNED_PID="${1:?aligned runner PID required}"
SHUFFLED_PID="${2:?shuffled runner PID required}"

is_running() {
  local pid="$1"
  local state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]]
}

arm_finished_cleanly() {
  local arm="$1"
  [[ -s "$RESULTS/${arm}_s42.json" ]] &&
    grep -Fq "DONE arm=$arm" "$RESULTS/logs/${arm}_s42.log"
}

launch_additive() {
  local gpu="$1"
  echo "[$(date -Is)] source arm complete; launching additive_control on GPU $gpu"
  exec env ARM=additive_control GPU="$gpu" "$RUNNER" --launch
}

mkdir -p "$RESULTS/logs"
echo "[$(date -Is)] scheduler start aligned_pid=$ALIGNED_PID shuffled_pid=$SHUFFLED_PID" >>"$SCHEDULER_LOG"

while true; do
  if ! is_running "$ALIGNED_PID" && arm_finished_cleanly aligned_logit; then
    launch_additive 0 >>"$SCHEDULER_LOG" 2>&1
  fi
  if ! is_running "$SHUFFLED_PID" && arm_finished_cleanly shuffled_logit; then
    launch_additive 1 >>"$SCHEDULER_LOG" 2>&1
  fi

  if ! is_running "$ALIGNED_PID" && ! is_running "$SHUFFLED_PID"; then
    echo "[$(date -Is)] both source arms stopped without a clean reusable result; additive not launched" >>"$SCHEDULER_LOG"
    exit 1
  fi
  sleep 15
done
