#!/usr/bin/env bash
set -uo pipefail

# Fresh, fixed-order queue for the isolated M1 Version-B source-only pilot.
# The output root is deliberately unique so the v2 exporter incident cannot
# be mistaken for a valid arm or included in the aggregator.
REPO=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
LOG_DIR="$REPO/sua_exploration/results/m1_version_b_5070ti_queue_f0_s42_v3"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1
printf 'QUEUE_START=%s\n' "$(date -Is)"
printf 'HOST=%s GPU=%s\n' "$(hostname)" "${CUDA_VISIBLE_DEVICES:-0}"
for arm in c0 c hs; do
  start=$(date -Is)
  log="$LOG_DIR/${arm}.log"
  printf 'ARM_START arm=%s time=%s log=%s\n' "$arm" "$start" "$log"
  set +e
  PYTHONNOUSERSITE=1 PYTHON_BIN="$PY" CUDA_VISIBLE_DEVICES=0 \
    bash "$REPO/sua_exploration/scripts/run_m1_version_b_5070ti.sh" "$arm" --run 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  end=$(date -Is)
  printf 'ARM_END arm=%s time=%s status=%s\n' "$arm" "$end" "$status"
  if [[ "$status" -ne 0 ]]; then
    printf 'QUEUE_ABORT arm=%s time=%s\n' "$arm" "$end"
    exit "$status"
  fi
done
printf 'QUEUE_END=%s status=0\n' "$(date -Is)"
