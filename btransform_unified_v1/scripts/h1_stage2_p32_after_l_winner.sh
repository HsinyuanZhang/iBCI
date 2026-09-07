#!/usr/bin/env bash
# Queue H1 P32 on GPU0 after the L=200/250 +8 winner exists. Only the winning L.
# Does not touch GPU1. Does not register EvalAI. Does not rerun stage-1 LODO.
# Does not +8 P32 (24-ep endpoint24 EMA only). Does not pack unless asked later.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
PAIR="${1:-$(cat "$ROOT/results/h1_stage2_full13/LATEST_PAIR.txt")}"
WINNER="$PAIR/extend8/winner.json"
LOG="$PAIR/p32_after_l_winner.log"
GPU0_UUID="GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"

mkdir -p "$PAIR"
echo "queued utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for=$WINNER" | tee -a "$LOG"

gpu0_clean() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  echo "$apps" | awk -F',' -v u="$GPU0_UUID" '
    $1 ~ u && ($3+0) > 500 { dirty=1 }
    END { exit dirty+0 }
  '
}

while [ ! -f "$WINNER" ]; do
  sleep 60
done
echo "extend8 winner present utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

L=$(PYTHONNOUSERSITE=1 "$PY" - <<PY
import json
from pathlib import Path
p = Path("$WINNER")
payload = json.loads(p.read_text(encoding="utf-8"))
winner = payload.get("winner")
if winner not in {"L200", "L250"}:
    raise SystemExit(f"no L winner in {p}: {winner!r}")
window = int(payload["arms"][winner]["window"])
print(window)
PY
)
echo "winning_L=$L utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

while ! gpu0_clean; do
  echo "GPU0 not clean; sleep 60 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  sleep 60
done

if pgrep -f 'h1_stage2_full13_proj_add.py --proj-dim 32' >/dev/null; then
  echo "REFUSED: P32 stage-2 already running" | tee -a "$LOG"
  exit 2
fi

DEST="$PAIR/p32/L${L}_P32_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
echo "=== start P32 L=$L dest=$DEST utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export BTRANSFORM_H1_STAGE2_TRAIN=1

set +e
"$PY" "$ROOT/scripts/h1_stage2_full13_proj_add.py" \
  --window "$L" --proj-dim 32 --stage all --dest "$DEST" --peak-lr 1e-4 \
  >>"$DEST.log" 2>&1
rc=$?
set -e
echo "=== P32 L=$L rc=$rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "$DEST" > "$PAIR/p32/LATEST_DEST.txt"
exit "$rc"
