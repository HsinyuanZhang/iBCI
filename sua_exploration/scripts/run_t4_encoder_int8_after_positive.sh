#!/usr/bin/env bash
# Two-GPU automatic PTQ -> QAT pipeline for a positive SUA T4 FP32 screen.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
FP32_SCREEN="${FP32_SCREEN:-sua_spint_t4_mainline_fp32_v1}"
INT8_SCREEN="${INT8_SCREEN:-sua_spint_t4_encoder_int8_v1}"
FP32_RESULTS="$ROOT/sua_exploration/results/$FP32_SCREEN"
INT8_RESULTS="$ROOT/sua_exploration/results/$INT8_SCREEN"
AGGREGATE="$FP32_RESULTS/aggregate.json"
LOGS="$INT8_RESULTS/logs"

mkdir -p "$LOGS"

"$PY" - "$AGGREGATE" "$INT8_RESULTS/trigger_receipt.json" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

source_path = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
source = json.loads(source_path.read_text())
if source.get("formal_test_files_opened") is not False:
    raise SystemExit("FP32 aggregate did not preserve the sealed formal test")
contrasts = source.get("contrasts") or {}
d0 = float(contrasts["t4_vs_original_spint_b0"]["mean_paired_delta_r2"])
d1 = float(contrasts["t4_vs_shuffled_label_ts4"]["mean_paired_delta_r2"])
if not (d0 > 0.0 and d1 > 0.0):
    raise SystemExit(f"INT8 not triggered: T4 deltas are b0={d0}, ts4={d1}")
protocol = source.get("protocol") or {}
if (
    protocol.get("same_trial_count_and_prefix_for_all_arms") is not True
    or protocol.get("evaluation_backward_gradients") is not False
    or protocol.get("scored_epoch_window") != list(range(5, 13))
):
    raise SystemExit("INT8 not triggered: FP32 protocol audit failed")
payload = {
    "status": "triggered",
    "triggered_at": datetime.now().astimezone().isoformat(),
    "source_fp32_aggregate": str(source_path),
    "t4_minus_b0": d0,
    "t4_minus_ts4": d1,
    "condition": "both strict paired mean deltas > 0 and protocol audit passed",
    "quant_scope": "T4/B3S encoder W8A8 + FP32 decoder",
    "ptq_delta_r2_gate": -0.01,
    "ptq_failure_action": "automatic encoder QAT",
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt_path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, receipt_path)
print(json.dumps(payload, indent=2))
PY

worker() {
  local seed="$1"
  local gpu="$2"
  local seed_dir="$INT8_RESULTS/seed${seed}"
  local run_dir="$ROOT/sua_exploration/checkpoints/${FP32_SCREEN}_t4_dandi688_co_s${seed}"
  mkdir -p "$seed_dir/ptq" "$seed_dir/qat"
  if [[ -f "$seed_dir/worker_completed.env" ]]; then
    echo "seed=$seed already completed; refusing to overwrite" >&2
    return 0
  fi
  if [[ ! -f "$run_dir/epoch_ckpts/epoch_011.ckpt" ]]; then
    echo "missing final T4 checkpoint for seed=$seed: $run_dir" >&2
    return 2
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    "$ROOT/sua_exploration/scripts/eval_t4_encoder_int8_dandi688.py" \
    --run_dir "$run_dir" \
    --out_dir "$seed_dir/ptq" \
    --device cuda
  local ptq_rc=$?
  set -e
  local method
  if [[ "$ptq_rc" == 0 ]]; then
    method="ptq"
  elif [[ "$ptq_rc" == 10 ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
      "$ROOT/sua_exploration/scripts/train_t4_encoder_qat_dandi688.py" \
      --run_dir "$run_dir" \
      --ptq_report "$seed_dir/ptq/ptq_report.json" \
      --out_dir "$seed_dir/qat" \
      --epochs 8 \
      --device cuda
    method="qat"
  else
    echo "seed=$seed PTQ crashed with rc=$ptq_rc" >&2
    return "$ptq_rc"
  fi
  {
    echo "status=completed"
    echo "seed=$seed"
    echo "gpu=$gpu"
    echo "method=$method"
    echo "completed_at=$(date -Is)"
  } >"$seed_dir/worker_completed.env"
}

# Keep both GPUs occupied: start 42/43, then put 44 on whichever GPU becomes
# free first.  No two workers ever share a GPU.
worker 42 0 >"$LOGS/seed42.log" 2>&1 &
pid42=$!
worker 43 1 >"$LOGS/seed43.log" 2>&1 &
pid43=$!

first_pid=""
first_rc=0
set +e
wait -n -p first_pid "$pid42" "$pid43"
first_rc=$?
set -e
if [[ "$first_rc" != 0 ]]; then
  echo "first INT8 worker failed with rc=$first_rc pid=$first_pid" >&2
  wait "$pid42" 2>/dev/null || true
  wait "$pid43" 2>/dev/null || true
  exit "$first_rc"
fi
if [[ "$first_pid" == "$pid42" ]]; then
  free_gpu=0
  remaining_pid="$pid43"
else
  free_gpu=1
  remaining_pid="$pid42"
fi
worker 44 "$free_gpu" >"$LOGS/seed44.log" 2>&1 &
pid44=$!
wait "$remaining_pid"
wait "$pid44"

"$PY" "$ROOT/sua_exploration/scripts/aggregate_t4_encoder_int8.py" \
  --source_fp32_aggregate "$AGGREGATE" \
  --result_dir "$INT8_RESULTS" \
  --out "$INT8_RESULTS/aggregate.json" >"$LOGS/aggregate.log" 2>&1

{
  echo "status=completed"
  echo "completed_at=$(date -Is)"
  echo "aggregate=$INT8_RESULTS/aggregate.json"
} >"$INT8_RESULTS/pipeline_completed.env"
