#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/btransform_unified_v1/results
mkdir -p "$OUT/history_mask_v1" "$OUT/m1_loso_outer20120924_v1" "$OUT/logs"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export PYTHONNOUSERSITE=1

# 1a. M2 Original history mask inside as-shipped image (source_minival + ext4)
for surface in source_minival ext4; do
  nohup docker run --rm --network none --cpus=2 \
    -e CUDA_VISIBLE_DEVICES= \
    -e OMP_NUM_THREADS=2 \
    -v "$ROOT/btransform_unified_v1/scripts/inner_original_history_mask.py:/in/run.py:ro" \
    -v "$ROOT/tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache:/in/cache:ro" \
    -v "$OUT/history_mask_v1:/out" \
    --entrypoint python \
    spint-m2:e8-epoch027-76f0fb2 \
    /in/run.py --task m2 --cache /in/cache --surface "$surface" \
    --out "/out/m2_original_history_mask_${surface}.json" --batch 32 \
    > "$OUT/logs/m2_history_mask_${surface}.log" 2>&1 &
  echo "m2_history_mask_${surface} pid=$!"
done

# 1b. H1 Original history mask inside as-shipped image
nohup docker run --rm --network none --cpus=2 \
  -e CUDA_VISIBLE_DEVICES= \
  -e OMP_NUM_THREADS=2 \
  -v "$ROOT/btransform_unified_v1/scripts/inner_original_history_mask.py:/in/run.py:ro" \
  -v "$ROOT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt:/in/cache.pt:ro" \
  -v "$OUT/history_mask_v1:/out" \
  --entrypoint python \
  spint-h1-released-code-lr-5e-5:epoch-049 \
  /in/run.py --task h1 --cache /in/cache.pt --out /out/h1_original_history_mask.json --batch 32 \
  > "$OUT/logs/h1_history_mask.log" 2>&1 &
echo "h1_history_mask pid=$!"

# 1c. M1 Original history mask inside as-shipped image
nohup docker run --rm --network none --cpus=2 \
  -e CUDA_VISIBLE_DEVICES= \
  -e OMP_NUM_THREADS=2 \
  -v "$ROOT/btransform_unified_v1/scripts/inner_original_history_mask.py:/in/run.py:ro" \
  -v "$ROOT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/m1_optimized_v2_source_runtime_cache.npz:/in/cache.npz:ro" \
  -v "$ROOT/tfpd_exploration/results/family_runtime_v1/original_m1_frozen_same31252_v1/original_m1_frozen_selected_source_dev_fp32.npz:/in/archive.npz:ro" \
  -v "$OUT/history_mask_v1:/out" \
  --entrypoint python \
  spint-original-m1:e9-epoch019-052e9ea \
  /in/run.py --task m1 --cache /in/cache.npz --archive /in/archive.npz --out /out/m1_original_history_mask.json --batch 32 \
  > "$OUT/logs/m1_history_mask.log" 2>&1 &
echo "m1_history_mask pid=$!"

# 2. M1 family LOSO on 20120924 (host CPU; scores Original in docker at the end)
nohup env CUDA_VISIBLE_DEVICES= PYTHONPATH="$ROOT:$ROOT/streaming_calibration_exp:${PYTHONPATH:-}" \
  "$PY" "$ROOT/btransform_unified_v1/scripts/m1_family_loso_outer20120924.py" \
  > "$OUT/logs/m1_loso.log" 2>&1 &
echo "m1_loso pid=$!"

# 3. Five-arm waiter is a separate long-lived process; do not duplicate it here.
echo "launched (five-arm waiter not relaunched)"
