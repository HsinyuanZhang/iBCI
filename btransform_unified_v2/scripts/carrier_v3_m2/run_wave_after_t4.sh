#!/usr/bin/env bash
# GPU0 only. t4 is already running; f0/z0 are launched as siblings to fill
# unused memory. This script scores t4 when ready, waits for the packed arms,
# then runs later variants two-at-a-time on GPU0.
set -euo pipefail
WS=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
BENCH="$WS/btransform_unified_v2/dandi688_bench_v1"
RUN688="$BENCH/scripts/run_688_bench.py"
M2TRAIN="$WS/btransform_unified_v2/scripts/carrier_v3_m2/train_m2_variant.py"
FROZEN="$WS/btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_contract_v2"
LOG="$WS/btransform_unified_v2/results/carrier_v3_wave"

wait_for() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    echo "waiting for $path $(date -Is)"
    sleep 20
  done
}

run688() {
  local arm="$1" cache="$2" dest="$3"
  echo "=== 688 $arm train $(date -Is) ==="
  "$PY" "$RUN688" --stage train --arm "$arm" --protocol exp1_narrow \
    --prepared-cache "$cache" --dest "$dest" --device cuda:0 --cpu-threads 2
  echo "=== 688 $arm score $(date -Is) ==="
  "$PY" "$RUN688" --stage score --arm "$arm" --protocol exp1_narrow \
    --prepared-cache "$cache" --dest "$dest" --device cuda:0 --cpu-threads 2
}

score688() {
  local arm="$1" cache="$2" dest="$3"
  if [[ ! -f "$dest/score_receipt_${arm}.json" ]]; then
    echo "=== 688 $arm score $(date -Is) ==="
    "$PY" "$RUN688" --stage score --arm "$arm" --protocol exp1_narrow \
      --prepared-cache "$cache" --dest "$dest" --device cuda:0 --cpu-threads 2
  fi
}

wait_for "$BENCH/results/train_exp1_narrow_t4/train_receipt.json"
score688 t4 "$FROZEN" "$BENCH/results/train_exp1_narrow_t4"

wait_for "$BENCH/results/train_exp1_narrow_f0/train_receipt.json"
score688 f0 "$FROZEN" "$BENCH/results/train_exp1_narrow_f0"

wait_for "$BENCH/results/train_exp1_narrow_z0/train_receipt.json"
score688 z0 "$FROZEN" "$BENCH/results/train_exp1_narrow_z0"

wait_for "$BENCH/results/cache_u1_m10/prepared_contract.json"
wait_for "$BENCH/results/cache_u1_m30/prepared_contract.json"
echo "=== 688 u1_m10 + u1_m30 packed on GPU0 $(date -Is) ==="
run688 t4 "$BENCH/results/cache_u1_m10" "$BENCH/results/train_exp1_narrow_u1_m10" \
  > "$LOG/u1_m10.log" 2>&1 &
pid_a=$!
run688 t4 "$BENCH/results/cache_u1_m30" "$BENCH/results/train_exp1_narrow_u1_m30" \
  > "$LOG/u1_m30.log" 2>&1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

if [[ -f "$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1/cache/carrier_normalizer.json" ]]; then
  echo "=== M2 u1 train+score $(date -Is) ==="
  "$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1" \
    --dest "$WS/btransform_unified_v2/results/carrier_v3_m2/train_u1" --stage all --device cuda:0
else
  echo "=== M2 u1 skipped: cache not built $(date -Is) ==="
fi
if [[ -f "$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4/cache/carrier_normalizer.json" ]]; then
  echo "=== M2 vstate4 train+score $(date -Is) ==="
  "$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4" \
    --dest "$WS/btransform_unified_v2/results/carrier_v3_m2/train_vstate4" --stage all --device cuda:0
else
  echo "=== M2 vstate4 skipped: cache not built $(date -Is) ==="
fi

echo "=== WAVE COMPLETE $(date -Is) ==="
