#!/usr/bin/env bash
# Run B3 + dropout experiments (training strategy, not architecture change).
# Plus re-evaluate B0 teacher on heldout for reference ceiling.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH

EPOCHS=20
PATIENCE=5
FOLD=0
SEED=42
COMMON="data.validation_protocol=loso data.loso_fold=${FOLD} data.include_heldout_in_test=true seed=${SEED} trainer.max_epochs=${EPOCHS} callbacks.early_stopping.patience=${PATIENCE} train=true test=true trainer.accelerator=gpu trainer.devices=1"

EXPERIMENTS=(
  "b3_dropout_mild"       # uniform p in [0, 0.15]
  "b3_dropout_standard"   # uniform p in [0, 0.30]
  "b3_dropout_curriculum" # curriculum 0 -> 0.30 over 10 epochs
)

mkdir -p batch_logs
for exp in "${EXPERIMENTS[@]}"; do
  echo "[$(date +%H:%M:%S)] === Starting ${exp} (LOSO+heldout) ==="
  python src/train.py "experiment=${exp}" ${COMMON} \
    > "batch_logs/dropout_${exp}.log" 2>&1
  status=$?
  if [ ${status} -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] === DONE ${exp} ==="
  else
    echo "[$(date +%H:%M:%S)] === FAIL ${exp} (exit ${status}) ==="
  fi
done
echo "[$(date +%H:%M:%S)] === ALL DROPOUT EXPERIMENTS COMPLETE ==="
