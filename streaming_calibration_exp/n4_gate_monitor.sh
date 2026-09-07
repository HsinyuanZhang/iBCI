#!/bin/bash
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python

while true; do
  n4_positive=0
  n4_total=0
  results=""
  for fold in 0 1 2 3 4 5 6; do
    n4_log=$(ls -t logs/n4_neural_only_carrier_m2/runs/*_f${fold}_s42/*.log 2>/dev/null | head -1)
    ns4_log=$(ls -t logs/ns4_shuffled_neural_only_carrier_m2/runs/*_f${fold}_s42/*.log 2>/dev/null | head -1)
    n4_r2=""; ns4_r2=""
    if [ -f "$n4_log" ]; then n4_r2=$(grep "Retrieved metric" "$n4_log" | tail -1 | grep -oP 'r2_mean=\K[0-9.]+'); fi
    if [ -f "$ns4_log" ]; then ns4_r2=$(grep "Retrieved metric" "$ns4_log" | tail -1 | grep -oP 'r2_mean=\K[0-9.]+'); fi
    if [ -n "$n4_r2" ] && [ -n "$ns4_r2" ]; then
      n4_total=$((n4_total + 1))
      delta=$(python3 -c "print(f'{${n4_r2} - ${ns4_r2}:+.4f}')")
      results="${results}f${fold}: N4=${n4_r2} NS4=${ns4_r2} Δ=${delta}\n"
      if python3 -c "exit(0 if ${n4_r2} > ${ns4_r2} else 1)" 2>/dev/null; then
        n4_positive=$((n4_positive + 1))
      fi
    fi
  done

  echo "$(date +%H:%M:%S): ${n4_total}/7 collected, ${n4_positive} positive"
  echo -e "$results"

  if [ $n4_positive -ge 5 ] && [ $n4_total -ge 5 ]; then
    echo "GATE PASSED: >=5/7 positive. Launching held-out."
    tmux new-session -d -s n4_heldout "CUDA_VISIBLE_DEVICES=0 $PYTHON src/train.py experiment=n4_neural_only_carrier_m2 seed=42 data.calibration_n_trials=24 data.include_heldout_in_test=true data.query_start_trial=24 data.allow_empty_heldout_query=true trainer.max_epochs=12 2>&1 | tee /tmp/n4_heldout.log"
    sleep 5
    tmux new-session -d -s ns4_heldout "CUDA_VISIBLE_DEVICES=1 $PYTHON src/train.py experiment=ns4_shuffled_neural_only_carrier_m2 seed=42 data.calibration_n_trials=24 data.include_heldout_in_test=true data.query_start_trial=24 data.allow_empty_heldout_query=true trainer.max_epochs=12 2>&1 | tee /tmp/ns4_heldout.log"
    echo "Held-out sessions launched."
    break
  fi

  remaining=$((7 - n4_total))
  if [ $n4_total -ge 5 ] && [ $n4_positive -lt 5 ]; then
    impossible=$((5 - n4_positive - remaining))
    if [ $impossible -gt 0 ]; then
      echo "GATE FAILED: cannot reach 5 positive (have ${n4_positive}, ${remaining} remaining)"
      break
    fi
  fi

  sleep 60
done
