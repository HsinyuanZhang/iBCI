#!/usr/bin/env bash
# After M1 P16 scores: pack exact-E if diagnostic minival pooled >= 0.70.
# Does not register. Does not touch GPU0 / H1.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
DEST="${1:-$ROOT/results/m1_projadd_series/P16_20260906T150917Z}"
SCORE="$DEST/score_receipt.json"
LOG="$ROOT/results/m1_projadd_series/pack_if_hopeful.log"
HOPEFUL=0.70

mkdir -p "$ROOT/results/m1_projadd_series"
echo "watching $SCORE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

while [ ! -f "$SCORE" ]; do
  sleep 60
done
echo "score present utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

MINIVAL=$(PYTHONNOUSERSITE=1 "$PY" - <<PY
import json
from pathlib import Path
s=json.loads(Path("$SCORE").read_text(encoding="utf-8"))
print(s["acceptance_accounting"]["ours_minival_pooled"])
PY
)
echo "diagnostic_minival_pooled=$MINIVAL" | tee -a "$LOG"

ok=$(PYTHONNOUSERSITE=1 "$PY" -c "import sys; print(int(float('$MINIVAL')>=$HOPEFUL))")
if [ "$ok" != "1" ]; then
  echo "SKIP_NOT_HOPEFUL minival=$MINIVAL threshold=$HOPEFUL" | tee -a "$LOG"
  exit 3
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=
echo "=== pack P16 exact-E utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
set +e
"$PY" "$ROOT/scripts/pack_m1_projadd_exacte_v1.py" >>"$LOG" 2>&1
rc=$?
set -e
echo "=== pack rc=$rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
exit "$rc"
