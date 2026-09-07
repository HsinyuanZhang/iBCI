#!/bin/bash
set -e
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp

eval_folds="${1:-0 1 2}"
arms="${2:-k4 ks4}"

for arm in $arms; do
  config="p2_general_carrier_m2_${arm}"
  for fold in $eval_folds; do
    # Find the complete training run (has periodic epoch_009)
    ckpt=$(find logs/${config}/runs/*_f${fold}_s42/ -name "epoch_009.ckpt" -path "*/periodic_ckpt/*" 2>/dev/null | sort -r | head -1)
    if [ -z "$ckpt" ]; then
      echo "SKIP ${arm} fold-${fold}: no complete checkpoint"
      continue
    fi
    # Use the best_ckpt/last.ckpt from the same run directory
    run_dir=$(dirname $(dirname "$ckpt"))
    best_ckpt="${run_dir}/best_ckpt/last.ckpt"
    if [ ! -f "$best_ckpt" ]; then
      best_ckpt="${run_dir}/best_ckpt/epoch_011.ckpt"
    fi
    if [ ! -f "$best_ckpt" ]; then
      best_ckpt=$(ls "${run_dir}/best_ckpt/"epoch_*.ckpt 2>/dev/null | sort -r | head -1)
    fi
    if [ ! -f "$best_ckpt" ]; then
      echo "SKIP ${arm} fold-${fold}: no best checkpoint in $(ls ${run_dir}/best_ckpt/ 2>/dev/null)"
      continue
    fi
    echo "=== EVAL ${arm} fold-${fold} ckpt=${best_ckpt} ==="
    CUDA_VISIBLE_DEVICES=0 /home/xinyuan/miniconda3/envs/spint/bin/python src/train.py \
      experiment=${config} \
      test=true train=false \
      ckpt_path="${best_ckpt}" \
      data.loso_fold=${fold} \
      data.include_heldout_in_test=true \
      data.query_start_trial=33 \
      data.calibration_n_trials=33 \
      data.allow_empty_heldout_query=true \
      trainer=gpu 2>&1 | tee -a /tmp/eval_${arm}_heldout.log | grep -E "test_heldout|r2_mean|Error|error|SKIP|EVAL"
    echo "---"
  done
done
echo "=== ALL EVALS DONE ==="
