#!/usr/bin/env bash
# Wait for the live pair winner.json, then pack exact-E (register=false).
set -euo pipefail
PAIR_ROOT="${1:-$(cat /home/xinyuan/Work_host/SPINT/btransform_unified_v1/results/h1_stage2_full13/LATEST_PAIR.txt)}"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
WINNER="$PAIR_ROOT/winner.json"
LOG="$PAIR_ROOT/pack_watch.log"
echo "watching $WINNER" | tee -a "$LOG"
while [ ! -f "$WINNER" ]; do
  sleep 60
done
echo "winner present $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
# wait until pair script finishes writing / sealing
sleep 5
export PYTHONNOUSERSITE=1
export H1_STAGE2_WINNER_JSON="$WINNER"
if [ -f /home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1/artifacts/evalai_candidate.json ]; then
  echo "already packed; skip" | tee -a "$LOG"
  exit 0
fi
"$PY" /home/xinyuan/Work_host/SPINT/btransform_unified_v1/scripts/pack_h1_stage2_exacte_v1.py >>"$LOG" 2>&1
echo "pack rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
