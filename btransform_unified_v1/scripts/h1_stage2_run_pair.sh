#!/usr/bin/env bash
# Sequential H1 stage-2 pair on GPU0: L=250 then L=200, then pick winner.
# Does not touch GPU1. Does not register EvalAI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PAIR_ROOT="${PAIR_ROOT:-$ROOT/results/h1_stage2_full13/pair_${STAMP}}"
L250_DEST="${L250_DEST:-$PAIR_ROOT/L250_${STAMP}}"
L200_DEST="${L200_DEST:-$PAIR_ROOT/L200_${STAMP}}"
LOG="$PAIR_ROOT/pair.log"

mkdir -p "$PAIR_ROOT"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) pair_root=$PAIR_ROOT"
  echo "gpu0_only CUDA_VISIBLE_DEVICES=0; GPU1 never touched"
} | tee -a "$LOG"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export BTRANSFORM_H1_STAGE2_TRAIN=1

run_one() {
  local L="$1"
  local DEST="$2"
  echo "=== start L=$L dest=$DEST utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  set +e
  "$PY" "$ROOT/scripts/h1_stage2_full13_proj_add.py" \
    --window "$L" --stage all --dest "$DEST" --peak-lr 1e-4 \
    >>"$DEST.log" 2>&1
  local rc=$?
  set -e
  echo "=== done L=$L rc=$rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  cp -f "$DEST.log" "$PAIR_ROOT/L${L}.log" 2>/dev/null || true
  return 0
}

mkdir -p "$L250_DEST" "$L200_DEST"
run_one 250 "$L250_DEST"
run_one 200 "$L200_DEST"

echo "=== pick winner utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
set +e
PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/h1_stage2_pick_winner.py" \
  --l250 "$L250_DEST" --l200 "$L200_DEST" --dest "$PAIR_ROOT" \
  >>"$LOG" 2>&1
pick_rc=$?
echo "=== pick rc=$pick_rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
if [ "$pick_rc" -eq 0 ]; then
  echo "=== pack exact-E (register=false) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  PYTHONNOUSERSITE=1 H1_STAGE2_WINNER_JSON="$PAIR_ROOT/winner.json" \
    "$PY" "$ROOT/scripts/pack_h1_stage2_exacte_v1.py" >>"$LOG" 2>&1
  echo "=== pack rc=$? utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
fi
echo "=== pair finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
