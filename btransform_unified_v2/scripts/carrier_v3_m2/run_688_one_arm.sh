#!/usr/bin/env bash
set -euo pipefail
ARM="$1"
CACHE="$2"
DEST="$3"
WS=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
RUN688="$WS/btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py"
echo "=== 688 $ARM train $(date -Is) ==="
"$PY" "$RUN688" --stage train --arm "$ARM" --protocol exp1_narrow \
  --prepared-cache "$CACHE" --dest "$DEST" --device cuda:0 --cpu-threads 2
echo "=== 688 $ARM score $(date -Is) ==="
"$PY" "$RUN688" --stage score --arm "$ARM" --protocol exp1_narrow \
  --prepared-cache "$CACHE" --dest "$DEST" --device cuda:0 --cpu-threads 2
echo "=== 688 $ARM done $(date -Is) ==="
