#!/usr/bin/env bash
# Sequential single-GPU wave: 688 exp1_narrow t4 → u1_m10 → u1_m30, then M2 u1 → vstate4.
set -euo pipefail
WS=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
BENCH="$WS/btransform_unified_v2/dandi688_bench_v1"
RUN688="$BENCH/scripts/run_688_bench.py"
M2TRAIN="$WS/btransform_unified_v2/scripts/carrier_v3_m2/train_m2_variant.py"
LOGROOT="$WS/btransform_unified_v2/results/carrier_v3_wave"
mkdir -p "$LOGROOT"

run688() {
  local arm="$1" cache="$2" dest="$3"
  echo "=== 688 $arm train $(date -Is) ==="
  "$PY" "$RUN688" --stage train --arm t4 --protocol exp1_narrow \
    --prepared-cache "$cache" --dest "$dest" --device cuda:0
  echo "=== 688 $arm score $(date -Is) ==="
  "$PY" "$RUN688" --stage score --arm t4 --protocol exp1_narrow \
    --prepared-cache "$cache" --dest "$dest" --device cuda:0
}

wait_for() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    echo "waiting for $path $(date -Is)"
    sleep 30
  done
}

FROZEN="$WS/btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_contract_v2"
run688 t4 "$FROZEN" "$BENCH/results/train_exp1_narrow_t4"

wait_for "$BENCH/results/cache_u1_m10/prepared_contract.json"
run688 u1_m10 "$BENCH/results/cache_u1_m10" "$BENCH/results/train_exp1_narrow_u1_m10"

wait_for "$BENCH/results/cache_u1_m30/prepared_contract.json"
run688 u1_m30 "$BENCH/results/cache_u1_m30" "$BENCH/results/train_exp1_narrow_u1_m30"

wait_for "$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1/cache/carrier_normalizer.json"
echo "=== M2 u1 train+score $(date -Is) ==="
"$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_u1" \
  --dest "$WS/btransform_unified_v2/results/carrier_v3_m2/train_u1" --stage all --device cuda:0

wait_for "$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4/cache/carrier_normalizer.json"
echo "=== M2 vstate4 train+score $(date -Is) ==="
"$PY" "$M2TRAIN" --run-root "$WS/btransform_unified_v2/results/carrier_v3_m2/run_vstate4" \
  --dest "$WS/btransform_unified_v2/results/carrier_v3_m2/train_vstate4" --stage all --device cuda:0

echo "=== WAVE COMPLETE $(date -Is) ==="
