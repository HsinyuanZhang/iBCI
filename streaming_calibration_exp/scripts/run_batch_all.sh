#!/usr/bin/env bash
# Batch training for B7-B14 baselines + B3 dropout variants.
# All runs: LOSO fold 0, seed 42, 20 epochs, early stop patience=5.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH

EPOCHS=20
PATIENCE=5
FOLD=0
SEED=42
COMMON="data.loso_fold=${FOLD} seed=${SEED} trainer.max_epochs=${EPOCHS} callbacks.early_stopping.patience=${PATIENCE} train=true test=true"

# Ordered by expected information value
EXPERIMENTS=(
  "b7_count_cond"
  "b8_randproj"
  "b9_hash"
  "b14_ternarized"
  "b11_hybrid"
  "b12_streaming_hash"
  "b13_ensemble"
  "b10_popstats"
  "b3_dropout_mild"
  "b3_dropout_standard"
)

mkdir -p batch_logs
for exp in "${EXPERIMENTS[@]}"; do
  echo "[$(date +%H:%M:%S)] === Starting ${exp} ==="
  python src/train.py "experiment=${exp}" ${COMMON} \
    trainer.accelerator=gpu trainer.devices=1 \
    > "batch_logs/${exp}.log" 2>&1
  status=$?
  if [ ${status} -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] === DONE ${exp} ==="
  else
    echo "[$(date +%H:%M:%S)] === FAIL ${exp} (exit ${status}) ==="
  fi
done
echo "[$(date +%H:%M:%S)] === ALL EXPERIMENTS COMPLETE ==="
