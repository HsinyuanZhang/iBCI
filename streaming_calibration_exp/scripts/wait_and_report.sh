#!/usr/bin/env bash
# Wait for the batch to complete, then generate the final scoring report.
set -uo pipefail
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp

echo "[$(date +%H:%M:%S)] Waiting for batch completion..."

# Wait until all 10 experiments are done
while true; do
  DONE=$(grep -c "DONE\|FAIL" batch_logs/master.log 2>/dev/null || echo 0)
  TOTAL=10
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "[$(date +%H:%M:%S)] All $TOTAL experiments completed!"
    break
  fi
  # Check if the master process is still alive
  MASTER_PID=$(cat batch_logs/master.pid 2>/dev/null || echo "")
  if [ -n "$MASTER_PID" ] && ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Master process $MASTER_PID exited (done=$DONE)"
    break
  fi
  sleep 60
done

echo ""
echo "=== Final master log ==="
cat batch_logs/master.log
echo ""

# Generate the scoring report
export PATH=/home/xinyuan/miniconda3/envs/ks4/bin:$PATH
python scripts/aggregate_scores.py 2>&1 | tee batch_logs/final_scoring.txt

echo ""
echo "[$(date +%H:%M:%S)] Report generation complete."
