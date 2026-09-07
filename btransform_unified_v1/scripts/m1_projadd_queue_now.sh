#!/usr/bin/env bash
# Start M1 P16 then P32 on GPU1 now. User lifted the H1-extend8 wait at 2026-09-06 23:07 +0800.
# Does not touch GPU0. Does not register EvalAI. Does not wait for H1.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
LOG="$ROOT/results/m1_projadd_series/queue_now.log"
GPU1_UUID="GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"

mkdir -p "$ROOT/results/m1_projadd_series"
echo "start_now utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) reason=user_lifted_h1_wait" | tee -a "$LOG"

gpu1_clean() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  echo "$apps" | awk -F',' -v u="$GPU1_UUID" '
    $1 ~ u && ($3+0) > 500 { dirty=1 }
    END { exit dirty+0 }
  '
}

if ! gpu1_clean; then
  echo "REFUSED: GPU1 not clean" | tee -a "$LOG"
  exit 2
fi

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
