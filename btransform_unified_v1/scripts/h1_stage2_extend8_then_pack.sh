#!/usr/bin/env bash
# After the 24-ep pair writes winner.json: +8 each completed STAGE-2 arm,
# re-pick, pack exact-E. No stage-1 / LODO F250 rerun. GPU0 only. register=false.
set -euo pipefail
PAIR_ROOT="${1:-$(cat /home/xinyuan/Work_host/SPINT/btransform_unified_v1/results/h1_stage2_full13/LATEST_PAIR.txt)}"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
LOG="$PAIR_ROOT/extend8_then_pack.log"
WINNER24="$PAIR_ROOT/winner.json"
echo "watching $WINNER24" | tee -a "$LOG"
while [ ! -f "$WINNER24" ]; do
  sleep 60
done
echo "24-ep winner present $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
sleep 8

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export BTRANSFORM_H1_EXTEND8_TRAIN=1

L250_SRC=$(find "$PAIR_ROOT" -maxdepth 1 -type d -name 'L250_*' | head -n 1)
L200_SRC=$(find "$PAIR_ROOT" -maxdepth 1 -type d -name 'L200_*' | head -n 1)
EXT_ROOT="$PAIR_ROOT/extend8"
mkdir -p "$EXT_ROOT"
L250_DEST="$EXT_ROOT/$(basename "$L250_SRC")_ext8"
L200_DEST="$EXT_ROOT/$(basename "$L200_SRC")_ext8"

run_ext() {
  local KIND="$1" SRC="$2" DEST="$3" L="$4"
  if [ ! -f "$SRC/cell_receipt.json" ]; then
    echo "skip extend $KIND: no 24-ep receipt" | tee -a "$LOG"
    return 0
  fi
  echo "=== extend8 $KIND L=$L utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  "$PY" "$ROOT/scripts/h1_extend8_proj_add.py" \
    --kind "$KIND" --source "$SRC" --dest "$DEST" --window "$L" \
    >>"$LOG" 2>&1
  echo "=== extend8 $KIND rc=$? utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
}

run_ext stage2 "$L250_SRC" "$L250_DEST" 250
run_ext stage2 "$L200_SRC" "$L200_DEST" 200

PICK_L250="$L250_DEST"
PICK_L200="$L200_DEST"
[ -f "$L250_DEST/cell_receipt.json" ] || PICK_L250="$L250_SRC"
[ -f "$L200_DEST/cell_receipt.json" ] || PICK_L200="$L200_SRC"

echo "=== re-pick after extend8 ===" | tee -a "$LOG"
"$PY" "$ROOT/scripts/h1_stage2_pick_winner.py" \
  --l250 "$PICK_L250" --l200 "$PICK_L200" --dest "$EXT_ROOT" >>"$LOG" 2>&1

if [ -f "$EXT_ROOT/winner.json" ]; then
  echo "=== pack exact-E from extend8 winner ===" | tee -a "$LOG"
  PYTHONNOUSERSITE=1 H1_STAGE2_WINNER_JSON="$EXT_ROOT/winner.json" \
    "$PY" "$ROOT/scripts/pack_h1_stage2_exacte_v1.py" >>"$LOG" 2>&1
  echo "=== pack rc=$? ===" | tee -a "$LOG"
fi
echo "=== controller done (stage-2 only; no LODO F250 rerun) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
