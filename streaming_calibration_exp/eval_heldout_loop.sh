#!/bin/bash
# Continuously evaluate completed folds until all 7 are done for both k4 and ks4
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
MAX_WAIT=7200  # 2 hours max
START_TIME=$(date +%s)

while true; do
  all_done=true
  for arm in k4 ks4; do
    config="p2_general_carrier_m2_${arm}"
    for fold in 0 1 2 3 4 5 6; do
      # Check if already evaluated
      marker="/tmp/eval_done_${arm}_f${fold}"
      if [ -f "$marker" ]; then
        continue
      fi
      # Check if training is complete (has periodic epoch_009)
      ckpt=$(find logs/${config}/runs/*_f${fold}_s42/ -name "epoch_009.ckpt" -path "*/periodic_ckpt/*" 2>/dev/null | head -1)
      if [ -z "$ckpt" ]; then
        all_done=false
        continue
      fi
      # Find best checkpoint from the same run
      run_dir=$(dirname $(dirname "$ckpt"))
      best_ckpt="${run_dir}/best_ckpt/last.ckpt"
      if [ ! -f "$best_ckpt" ]; then
        best_ckpt=$(ls "${run_dir}/best_ckpt/"epoch_*.ckpt 2>/dev/null | sort -r | head -1)
      fi
      if [ ! -f "$best_ckpt" ]; then
        echo "$(date): SKIP ${arm} fold-${fold} - no best ckpt"
        all_done=false
        continue
      fi
      echo "$(date): EVAL ${arm} fold-${fold}"
      CUDA_VISIBLE_DEVICES=0 $PYTHON src/train.py \
        experiment=${config} \
        test=true train=false \
        ckpt_path="${best_ckpt}" \
        data.loso_fold=${fold} \
        data.include_heldout_in_test=true \
        data.query_start_trial=33 \
        data.calibration_n_trials=33 \
        data.allow_empty_heldout_query=true \
        trainer=gpu 2>&1 | grep -E "test_heldout/r2_mean" | head -2 | tee -a /tmp/eval_${arm}_f${fold}_r2.txt
      touch "$marker"
      echo "$(date): DONE ${arm} fold-${fold}"
    done
  done

  if [ "$all_done" = true ]; then
    echo "$(date): ALL 14 EVALS COMPLETE"
    break
  fi

  ELAPSED=$(($(date +%s) - START_TIME))
  if [ $ELAPSED -gt $MAX_WAIT ]; then
    echo "$(date): TIMEOUT after ${ELAPSED}s"
    break
  fi
  echo "$(date): Waiting 60s for more folds to complete..."
  sleep 60
done
