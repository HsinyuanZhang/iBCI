#!/usr/bin/env bash
# Wait for M1 teacher to finish, then immediately run B3 student with LOSO+heldout.
# Uses inotify-style polling on the expected checkpoint path.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH

TEACHER_DIR="/home/xinyuan/Work_host/SPINT/SPINT-main/logs/train/runs/2026-07-21-19-11-01"

echo "[$(date +%H:%M:%S)] Waiting for M1 teacher (PID 1149227) to finish"
echo "  Teacher dir: ${TEACHER_DIR}"

# Wait until the specific training process exits
while kill -0 1149227 2>/dev/null; do
  sleep 300
  LATEST_CKPT=$(ls -t "${TEACHER_DIR}/checkpoints/best_ckpt/"*.ckpt 2>/dev/null | head -1)
  echo "[$(date +%H:%M:%S)] Teacher running. Latest: $(basename "$LATEST_CKPT" 2>/dev/null || echo none)"
done

echo "[$(date +%H:%M:%S)] Teacher training process exited."
sleep 10

# Find actual checkpoint
CKPT=$(ls -t "${TEACHER_DIR}/checkpoints/best_ckpt/"*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t "${TEACHER_DIR}/checkpoints/periodic_ckpt/"*.ckpt 2>/dev/null | head -1)
fi
echo "[$(date +%H:%M:%S)] Using checkpoint: $CKPT"

# Update the m1_teacher_ckpt_path with actual checkpoint
# (Hydra will read from paths.default.yaml, but we override here for safety)
echo "[$(date +%H:%M:%S)] Starting B3 M1 student training (LOSO+heldout)"
python src/train.py experiment=b3_m1 \
  data.validation_protocol=loso data.loso_fold=0 \
  data.include_heldout_in_test=true \
  model.teacher_ckpt_path="$CKPT" \
  seed=42 trainer.max_epochs=20 \
  callbacks.early_stopping.patience=5 \
  train=true test=true \
  trainer.accelerator=gpu trainer.devices=1 \
  require_baseline_validation=false \
  run_id=b3_m1_loso_f0_s42 \
  2>&1 | tee batch_logs/b3_m1_student.log

echo "[$(date +%H:%M:%S)] B3 M1 student done"
