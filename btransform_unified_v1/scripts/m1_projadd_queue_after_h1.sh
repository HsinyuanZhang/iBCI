#!/usr/bin/env bash
# Start M1 P16 then P32 on GPU1 only after H1 stage-2 +8 has a winner and GPU1 is clean.
# Does not touch GPU0. Does not register EvalAI. Does not rerun H1 stage-1.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
PAIR="${1:-$(cat "$ROOT/results/h1_stage2_full13/LATEST_PAIR.txt")}"
H1_DONE="$PAIR/extend8/winner.json"
LOG="$ROOT/results/m1_projadd_series/queue_after_h1.log"
GPU1_UUID="GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"

mkdir -p "$ROOT/results/m1_projadd_series"
echo "queued utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for=$H1_DONE" | tee -a "$LOG"

gpu1_clean() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  # exit 0 = clean (no GPU1 pid above 500 MiB, including "no apps")
  echo "$apps" | awk -F',' -v u="$GPU1_UUID" '
    $1 ~ u && ($3+0) > 500 { dirty=1 }
    END { exit dirty+0 }
  '
}

while [ ! -f "$H1_DONE" ]; do
  sleep 60
done
echo "H1 extend8 winner present utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

while ! gpu1_clean; do
  echo "GPU1 not clean; sleep 60 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  sleep 60
done

if pgrep -f 'scripts/m1_projadd_series.py' >/dev/null; then
  echo "REFUSED: m1_projadd_series.py already running" | tee -a "$LOG"
  exit 2
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1
export BTRANSFORM_M1_PROJADD_TRAIN=1

for P in 16 32; do
  DEST="$ROOT/results/m1_projadd_series/P${P}_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "=== start P$P dest=$DEST utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  set +e
  "$PY" "$ROOT/scripts/m1_projadd_series.py" --proj-dim "$P" --stage all --dest "$DEST" \
    >>"$DEST.log" 2>&1
  rc=$?
  set -e
  echo "=== P$P rc=$rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "stop series after P$P failure" | tee -a "$LOG"
    exit "$rc"
  fi
  if ! gpu1_clean; then
    echo "GPU1 dirty after P$P; not starting next cell" | tee -a "$LOG"
    exit 2
  fi
done
echo "=== M1 series controller done utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
