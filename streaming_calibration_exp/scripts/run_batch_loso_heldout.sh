#!/usr/bin/env bash
# Run all variants with PROPER LOSO + held-out evaluation.
# Uses: validation_protocol=loso, loso_fold=0, include_heldout_in_test=true
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH

EPOCHS=20
PATIENCE=5
FOLD=0
SEED=42
COMMON="data.validation_protocol=loso data.loso_fold=${FOLD} data.include_heldout_in_test=true seed=${SEED} trainer.max_epochs=${EPOCHS} callbacks.early_stopping.patience=${PATIENCE} train=true test=true trainer.accelerator=gpu trainer.devices=1"

EXPERIMENTS=(
  "b3_d64"            # baseline reference (re-run with proper protocol)
  "b7_count_cond"
  "b8_randproj"
  "b9_hash"
  "b11_hybrid"
  "b13_ensemble"
  "b10_popstats"
  "b3_dropout_mild"
  "b3_dropout_standard"
)

mkdir -p batch_logs
for exp in "${EXPERIMENTS[@]}"; do
  echo "[$(date +%H:%M:%S)] === Starting ${exp} (LOSO+heldout) ==="
  python src/train.py "experiment=${exp}" ${COMMON} \
    > "batch_logs/loso_${exp}.log" 2>&1
  status=$?
  if [ ${status} -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] === DONE ${exp} ==="
  else
    echo "[$(date +%H:%M:%S)] === FAIL ${exp} (exit ${status}) ==="
  fi
done
echo "[$(date +%H:%M:%S)] === ALL LOSO EXPERIMENTS COMPLETE ==="
