#!/usr/bin/env bash
# Two-GPU automatic PTQ -> QAT pipeline for a positive SUA T4 FP32 screen.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
FP32_SCREEN="${FP32_SCREEN:-sua_spint_t4_mainline_fp32_v1}"
INT8_SCREEN="${INT8_SCREEN:-sua_t4_encoder_int8_m50_v1}"
FP32_RESULTS="$ROOT/sua_exploration/results/$FP32_SCREEN"
INT8_RESULTS="$ROOT/sua_exploration/results/$INT8_SCREEN"
AGGREGATE="$FP32_RESULTS/aggregate.json"
SELECTION="${SELECTION:-$ROOT/sua_exploration/manifests/sua_t4_final_architecture_selection_v1.json}"
LOGS="$INT8_RESULTS/logs"

mkdir -p "$LOGS"

"$PY" - "$AGGREGATE" "$INT8_RESULTS/trigger_receipt.json" "$SELECTION" "$ROOT/sua_exploration/scripts" <<'PY'
import json
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

source_path = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
selection_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, sys.argv[4])
from t4_encoder_int8_protocol import validate_selection

selection, _validated_source, _validated_deltas = validate_selection(
    selection_path, source_fp32_path=source_path
)
if receipt_path.exists():
    raise SystemExit(f"refusing to overwrite immutable trigger receipt: {receipt_path}")
source_bytes = source_path.read_bytes()
source = json.loads(source_bytes)
if source.get("formal_test_files_opened") is not False:
    raise SystemExit("FP32 aggregate did not preserve the sealed formal test")
contrasts = source.get("contrasts") or {}
contrast_rows = {
    "t4_minus_b0": contrasts["t4_vs_original_spint_b0"],
    "t4_minus_ts4": contrasts["t4_vs_shuffled_label_ts4"],
}
d0 = float(contrast_rows["t4_minus_b0"]["mean_paired_delta_r2"])
d1 = float(contrast_rows["t4_minus_ts4"]["mean_paired_delta_r2"])
for name, row in contrast_rows.items():
    if row.get("passes_all_gates") is not True:
        raise SystemExit(f"INT8 not triggered: {name} did not pass all FP32 gates")
    gates = row.get("gates") or {}
    if not gates or not all(value is True for value in gates.values()):
        raise SystemExit(f"INT8 not triggered: {name} has an incomplete/failed gate receipt")
if not (d0 >= 0.03 and d1 >= 0.03):
    raise SystemExit(f"INT8 not triggered: T4 deltas are b0={d0}, ts4={d1}")
protocol = source.get("protocol") or {}
if (
    protocol.get("same_trial_count_and_prefix_for_all_arms") is not True
    or protocol.get("evaluation_backward_gradients") is not False
    or protocol.get("scored_epoch_window") != list(range(5, 13))
    or protocol.get("seeds") != [42, 43, 44]
    or len(protocol.get("sessions") or []) != 6
):
    raise SystemExit("INT8 not triggered: FP32 protocol audit failed")
payload = {
    "status": "triggered",
    "triggered_at": datetime.now().astimezone().isoformat(),
    "source_fp32_aggregate": str(source_path),
    "source_fp32_aggregate_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "final_architecture_selection": str(selection_path),
    "final_architecture_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "selected_architecture": selection["selected_architecture"],
    "activity_calibration_n": selection["activity_calibration_n"],
    "t4_label_feature_pool_n": selection["t4_label_feature_pool_n"],
    "evaluation_start_trial": selection["evaluation_start_trial"],
    "t4_minus_b0": d0,
    "t4_minus_ts4": d1,
    "condition": "both FP32 contrasts passed every frozen gate and protocol audit passed",
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
  local run_dir="$ROOT/sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s${seed}"
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
    --selection "$SELECTION" \
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
      --selection "$SELECTION" \
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
  --selection "$SELECTION" \
  --result_dir "$INT8_RESULTS" \
  --out "$INT8_RESULTS/aggregate.json" >"$LOGS/aggregate.log" 2>&1

{
  echo "status=completed"
  echo "completed_at=$(date -Is)"
  echo "aggregate=$INT8_RESULTS/aggregate.json"
} >"$INT8_RESULTS/pipeline_completed.env"
