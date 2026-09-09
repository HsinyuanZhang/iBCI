#!/usr/bin/env bash
# Start M2 u1 → vstate4 on GPU0 only, after 688 releases the card.
# Does not touch GPU1.
set -euo pipefail
WS=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
M2TRAIN="$WS/btransform_unified_v2/scripts/carrier_v3_m2/train_m2_variant.py"
CACHE_U1="$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1/cache/carrier_normalizer.json"
CACHE_V4="$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4/cache/carrier_normalizer.json"
TRAIN_U1="$WS/btransform_unified_v2/results/carrier_v3_m2/train_u1"
TRAIN_V4="$WS/btransform_unified_v2/results/carrier_v3_m2/train_vstate4"
LOG="$WS/btransform_unified_v2/results/carrier_v3_wave/m2_gpu0.log"

gpu0_busy() {
  nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

wave_alive() {
  pgrep -f 'scripts/carrier_v3_m2/run_wave_after_t4.sh' >/dev/null
}

echo "=== M2 GPU0 waiter $(date -Is) ===" | tee -a "$LOG"
while wave_alive || gpu0_busy; do
  echo "GPU0/688 still busy $(date -Is)" | tee -a "$LOG"
  sleep 30
done

if [[ ! -f "$CACHE_U1" || ! -f "$CACHE_V4" ]]; then
  echo "M2 cache missing after GPU0 idle; refusing to train $(date -Is)" | tee -a "$LOG"
  exit 2
fi

if [[ ! -f "$TRAIN_U1/train_receipt.json" ]]; then
  echo "=== M2 u1 train+score $(date -Is) ===" | tee -a "$LOG"
  "$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1" \
    --dest "$TRAIN_U1" --stage all --device cuda:0 --cpu-threads 4 \
    >>"$LOG" 2>&1
else
  echo "M2 u1 already trained" | tee -a "$LOG"
fi

if [[ ! -f "$TRAIN_V4/train_receipt.json" ]]; then
  echo "=== M2 vstate4 train+score $(date -Is) ===" | tee -a "$LOG"
  "$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4" \
    --dest "$TRAIN_V4" --stage all --device cuda:0 --cpu-threads 4 \
    >>"$LOG" 2>&1
else
  echo "M2 vstate4 already trained" | tee -a "$LOG"
fi

echo "=== M2 GPU0 COMPLETE $(date -Is) ===" | tee -a "$LOG"
