#!/usr/bin/env bash
# Polls until the E3 tuning-ablation screen has finished, runs the F3 electrode-ablation
# screen (F3/FS3 only -- F1/F2 skipped), then launches run_b3t_confirmation.sh.
#
# Pipeline: E3 complete -> F3 (electrode_ablation_f3) -> B3T confirmation.
#
# E3-finished detection (both must hold):
#   (a) sua_exploration/results/e3_tuning_ablation/ contains all 15 expected artifacts;
#   (b) no train_variant_dandi688.py process is running.
#
# F3-finished detection:
#   (a) sua_exploration/results/electrode_ablation_f3/ contains all 6 {f3,fs3}_s{42,43,44}.json;
#   (b) no train_variant_dandi688.py process is running.
#
# GPU-free verification before each launch stage (nvidia-smi --query-compute-apps).
#
# Idempotent launch markers:
#   f3_chain_state/launched.marker
#   b3t_confirmation_chain_state/launched.marker (existing B3T marker)
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
E3_RESULTS_DIR="${E3_RESULTS_DIR:-$ROOT/sua_exploration/results/e3_tuning_ablation}"
F3_RESULTS_DIR="${F3_RESULTS_DIR:-$ROOT/sua_exploration/results/electrode_ablation_f3}"
E3_GROUPS=(f0 t4 t8 ts4 ts8)
F3_GROUPS=(f3 fs3)
SEEDS=(42 43 44)

F3_CHAIN_STATE_DIR="$ROOT/sua_exploration/results/f3_chain_state"
F3_LAUNCHED_MARKER="$F3_CHAIN_STATE_DIR/launched.marker"
F3_ABORT_MARKER="$F3_CHAIN_STATE_DIR/gpu_not_free.abort"
F3_LAUNCH_CLAIM_LOCK="$F3_CHAIN_STATE_DIR/launch_claim.lock"

B3T_CHAIN_STATE_DIR="$ROOT/sua_exploration/results/b3t_confirmation_chain_state"
B3T_LAUNCHED_MARKER="$B3T_CHAIN_STATE_DIR/launched.marker"
B3T_ABORT_MARKER="$B3T_CHAIN_STATE_DIR/gpu_not_free.abort"
B3T_LAUNCH_CLAIM_LOCK="$B3T_CHAIN_STATE_DIR/launch_claim.lock"

POLL_LOG="$F3_CHAIN_STATE_DIR/poll.log"
POLL_INTERVAL_SECONDS=120

# Match E3 launch defaults (run_e3_tuning_ablation.sh invocation in spint_e3).
F3_MAX_EPOCHS="${F3_MAX_EPOCHS:-12}"
F3_BURN_IN="${F3_BURN_IN:-4}"
F3_SEEDS_CSV="${F3_SEEDS_CSV:-42,43,44}"

CHECK_ONCE=0
usage() {
  cat <<EOF
Usage: $(basename "$0") [--check-once]

Polls until E3 has finished, runs run_electrode_ablation_f3.sh, waits for F3 to finish,
verifies GPUs are free, then launches run_b3t_confirmation.sh (once each stage).

  --check-once  Print E3/F3 readiness once and exit (0=ready for next stage, 1=not ready).
  -h, --help    Show this message and exit.
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --check-once) CHECK_ONCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$F3_CHAIN_STATE_DIR" "$B3T_CHAIN_STATE_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$POLL_LOG"
}

missing_artifacts() {
  local results_dir="$1"
  shift
  local -a groups=("$@")
  for g in "${groups[@]}"; do
    for s in "${SEEDS[@]}"; do
      local p="$results_dir/${g}_s${s}.json"
      [ -f "$p" ] || echo "$p"
    done
  done
}

train_running() {
  pgrep -f "train_variant_dandi688.py" >/dev/null 2>&1
}

e3_finished() {
  local missing
  missing=$(missing_artifacts "$E3_RESULTS_DIR" "${E3_GROUPS[@]}")
  [ -z "$missing" ] && ! train_running
}

f3_finished() {
  local missing
  missing=$(missing_artifacts "$F3_RESULTS_DIR" "${F3_GROUPS[@]}")
  [ -z "$missing" ] && ! train_running
}

gpus_free() {
  local n_busy
  n_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n_busy" -eq 0 ]
}

if [ "$CHECK_ONCE" -eq 1 ]; then
  echo "E3_RESULTS_DIR=$E3_RESULTS_DIR"
  echo "F3_RESULTS_DIR=$F3_RESULTS_DIR"
  e3_miss=$(missing_artifacts "$E3_RESULTS_DIR" "${E3_GROUPS[@]}")
  f3_miss=$(missing_artifacts "$F3_RESULTS_DIR" "${F3_GROUPS[@]}")
  echo "E3 missing: $(echo "$e3_miss" | grep -c . || true) / 15"
  echo "F3 missing: $(echo "$f3_miss" | grep -c . || true) / 6"
  if train_running; then echo "train_variant_dandi688.py: RUNNING"; else echo "train_variant_dandi688.py: not running"; fi
  if [ -f "$F3_LAUNCHED_MARKER" ]; then echo "F3 launched marker: present"; else echo "F3 launched marker: absent"; fi
  if [ -f "$B3T_LAUNCHED_MARKER" ]; then echo "B3T launched marker: present"; else echo "B3T launched marker: absent"; fi
  if ! e3_finished; then echo "RESULT: E3 NOT READY"; exit 1; fi
  if [ ! -f "$F3_LAUNCHED_MARKER" ]; then echo "RESULT: E3 READY, F3 not launched yet"; exit 1; fi
  if ! f3_finished; then echo "RESULT: F3 IN PROGRESS or incomplete"; exit 1; fi
  if [ -f "$B3T_LAUNCHED_MARKER" ]; then echo "RESULT: F3 DONE, B3T already launched"; exit 0; fi
  echo "RESULT: F3 DONE, B3T not launched yet"
  exit 0
fi

log "chain_after_e3 starting (PID $$). Pipeline: E3 -> F3 -> B3T"

# ---------------------------------------------------------------------------------------
# Stage 0: wait for E3
# ---------------------------------------------------------------------------------------
while true; do
  if e3_finished; then
    log "E3 detection satisfied."
    break
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done

# ---------------------------------------------------------------------------------------
# Stage 1: launch F3 once
# ---------------------------------------------------------------------------------------
if [ -f "$F3_LAUNCHED_MARKER" ]; then
  log "F3 already launched at $(cat "$F3_LAUNCHED_MARKER"); skipping F3 launch."
else
  if ! mkdir "$F3_LAUNCH_CLAIM_LOCK" 2>/dev/null; then
    log "Another process claimed F3 launch; waiting for F3 artifacts."
  else
    if [ -f "$F3_LAUNCHED_MARKER" ]; then
      log "F3 launched marker appeared after claim; skipping."
    else
      log "Verifying GPUs free before F3 launch..."
      if ! gpus_free; then
        log "ABORT: GPUs not free before F3."
        date '+%Y-%m-%d %H:%M:%S' > "$F3_ABORT_MARKER"
        exit 1
      fi
      log "Launching run_electrode_ablation_f3.sh (max_epochs=$F3_MAX_EPOCHS seeds=$F3_SEEDS_CSV)."
      date '+%Y-%m-%d %H:%M:%S' > "$F3_LAUNCHED_MARKER"
      "$ROOT/sua_exploration/scripts/run_electrode_ablation_f3.sh" \
        --max_epochs "$F3_MAX_EPOCHS" --burn_in "$F3_BURN_IN" --seeds "$F3_SEEDS_CSV" \
        >> "$ROOT/sua_exploration/results/electrode_ablation_f3_chain.log" 2>&1
      f3_rc=$?
      log "run_electrode_ablation_f3.sh exited with code $f3_rc."
      if [ $f3_rc -ne 0 ]; then
        exit $f3_rc
      fi
    fi
  fi
fi

# Wait for F3 artifacts if another process launched F3 or we just finished a long run.
while true; do
  if f3_finished; then
    log "F3 detection satisfied."
    break
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done

# ---------------------------------------------------------------------------------------
# Stage 2: launch B3T once (existing marker semantics)
# ---------------------------------------------------------------------------------------
if [ -f "$B3T_LAUNCHED_MARKER" ]; then
  log "B3T already launched at $(cat "$B3T_LAUNCHED_MARKER"); exiting."
  exit 0
fi

if ! mkdir "$B3T_LAUNCH_CLAIM_LOCK" 2>/dev/null; then
  log "Another process claimed B3T launch; exiting."
  exit 0
fi
if [ -f "$B3T_LAUNCHED_MARKER" ]; then
  log "B3T launched marker appeared after claim; exiting."
  exit 0
fi

log "Verifying GPUs free before B3T launch..."
if ! gpus_free; then
  log "ABORT: GPUs not free before B3T."
  {
    date '+%Y-%m-%d %H:%M:%S'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
  } > "$B3T_ABORT_MARKER"
  exit 1
fi

log "Launching run_b3t_confirmation.sh."
date '+%Y-%m-%d %H:%M:%S' > "$B3T_LAUNCHED_MARKER"
"$ROOT/sua_exploration/scripts/run_b3t_confirmation.sh" >> "$ROOT/sua_exploration/results/b3t_confirmation_chain.log" 2>&1
b3t_rc=$?
log "run_b3t_confirmation.sh exited with code $b3t_rc."
exit $b3t_rc
