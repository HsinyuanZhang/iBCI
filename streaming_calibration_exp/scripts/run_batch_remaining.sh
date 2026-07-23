#!/usr/bin/env bash
# Restart batch for remaining experiments after vectorization fix.
# B7, B8, B9, B14 already completed. B11 needs vectorized forward (now fixed).
# B12 still too slow — skip. B10 now vectorized. B13 was always OK.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH

EPOCHS=20
PATIENCE=5
FOLD=0
SEED=42
COMMON="data.loso_fold=${FOLD} seed=${SEED} trainer.max_epochs=${EPOCHS} callbacks.early_stopping.patience=${PATIENCE} train=true test=true trainer.accelerator=gpu trainer.devices=1"

# Remaining experiments (B12 skipped — needs CUDA kernel for practical speed)
EXPERIMENTS=(
  "b11_hybrid"
  "b13_ensemble"
  "b10_popstats"
  "b3_dropout_mild"
  "b3_dropout_standard"
)

mkdir -p batch_logs
for exp in "${EXPERIMENTS[@]}"; do
  echo "[$(date +%H:%M:%S)] === Starting ${exp} ==="
  python src/train.py "experiment=${exp}" ${COMMON} \
    > "batch_logs/${exp}_v2.log" 2>&1
  status=$?
  if [ ${status} -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] === DONE ${exp} ==="
  else
    echo "[$(date +%H:%M:%S)] === FAIL ${exp} (exit ${status}) ==="
  fi
done
echo "[$(date +%H:%M:%S)] === ALL REMAINING EXPERIMENTS COMPLETE ==="
