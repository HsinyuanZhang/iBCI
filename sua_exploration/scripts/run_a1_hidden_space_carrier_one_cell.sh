#!/usr/bin/env bash
# One-cell runner for the only fresh A1 family: H/T4 seed 42.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="a1_hidden_space_carrier_v2"
RESULTS="${A1_RESULT_ROOT:-$SUA/results/$SCREEN}"
PREFLIGHT="${A1_PREFLIGHT:-$RESULTS/official_cpu_preflight.json}"
RUN_DIR="${A1_RUN_DIR:-$SUA/checkpoints/${SCREEN}_h_t4_dandi688_co_s42}"
LAUNCH="$RESULTS/h_t4_s42.pretraining.json"
SCORE="$RESULTS/h_t4_s42.score.json"
AGGREGATE="$RESULTS/pilot_aggregate.json"
LOG="$RESULTS/h_t4_s42.log"
MODE="${1:---dry-run}"
GPU="${GPU:-0}"
CELL="${CELL:-H/T4}"
SEED="${SEED:-42}"
AUTH_ENV="A1_HIDDEN_CARRIER_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_A1_HIDDEN_CARRIER_PILOT_GPU"
ROOT_GO_ENV="A1_HIDDEN_CARRIER_ROOT_GO"
ROOT_GO_VALUE="I_SIGN_A1_HIDDEN_CARRIER_PILOT_ROOT_GO"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$SUA${PYTHONPATH:+:$PYTHONPATH}"

[[ "$CELL" == "H/T4" ]] || { echo "Only fresh family H/T4 is runnable" >&2; exit 2; }
[[ "$SEED" == "42" ]] || { echo "Pilot accepts only seed 42" >&2; exit 2; }
case "$MODE" in --dry-run|--launch) ;; *) echo "Expected --dry-run or --launch" >&2; exit 2 ;; esac

TRAIN=("$PY" -m a1_hidden_carrier.trainer
  --official-preflight "$PREFLIGHT" --launch-receipt "$LAUNCH"
  --run-dir "$RUN_DIR" --seed "$SEED" --accelerator gpu
  --batch-size 32 --num-workers 4)
EVAL=("$PY" -m a1_hidden_carrier.scorer
  --run-dir "$RUN_DIR" --official-preflight "$PREFLIGHT"
  --launch-receipt "$LAUNCH" --out "$SCORE" --device cuda:0
  --permutation-seed 20260813 --launch)
AGG=("$PY" "$SUA/scripts/aggregate_a1_hidden_space_carrier.py"
  --preflight "$PREFLIGHT" --fresh-score "$SCORE" --out "$AGGREGATE")

echo "SCREEN_ID=$SCREEN"
echo "CELL=$CELL SEED=$SEED GPU=$GPU"
echo "LOGICAL_MATRIX=W/Z4,W/T4,H/Z4,H/T4"
echo "SEALED_A2_REUSE=W/Z4,W/T4"
echo "STRUCTURAL_ALIAS=H/Z4->W/Z4"
echo "FRESH_TRAINING_FAMILIES=H/T4"
echo "ATTACHMENT_CONTROL=H/TS4 same H/T4 checkpoint, no training"
echo "FORMAL_SUBC_TEST_NWB_OPENED=false"

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${TRAIN[@]}"; echo
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${EVAL[@]}"; echo
  printf '%q ' "${AGG[@]}"; echo
  echo "NO_COMMAND_EXECUTED=true"
  exit 0
fi

[[ "${!ROOT_GO_ENV:-}" == "$ROOT_GO_VALUE" ]] || { echo "Missing explicit root go" >&2; exit 3; }
[[ "${!AUTH_ENV:-}" == "$AUTH_VALUE" ]] || { echo "Missing explicit GPU authorization" >&2; exit 3; }
[[ -f "$PREFLIGHT" && -f "$PREFLIGHT.sha256" ]] || { echo "Missing immutable A1 preflight" >&2; exit 1; }
for target in "$RUN_DIR" "$LAUNCH" "$LAUNCH.sha256" "$SCORE" "$SCORE.sha256" "$AGGREGATE" "$AGGREGATE.sha256"; do
  [[ ! -e "$target" ]] || { echo "Refusing existing A1 target: $target" >&2; exit 1; }
done
mkdir -p "$RESULTS"

# Validate preflight, CUDA runtime, and exact candidate freshness, then write a
# non-reusable immutable launch receipt before training begins.
CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$PREFLIGHT" "$LAUNCH" "$GPU" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from a1_hidden_carrier.artifacts import write_immutable_json
from a1_hidden_carrier.contract import SCREEN_ID, PILOT_SEED
from a1_hidden_carrier.evidence import load_verified_preflight

preflight, digest = load_verified_preflight(Path(sys.argv[1]))
if not torch.cuda.is_available():
    raise SystemExit("A1 launch requires CUDA")
payload = {
    "schema_version": 2,
    "receipt_kind": "a1_hidden_space_carrier_pretraining_launch",
    "screen_id": SCREEN_ID,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "ROOT_GO_AND_GPU_AUTHORIZED_PRETRAINING_ENVIRONMENT_VERIFIED",
    "cell": "H/T4", "seed": PILOT_SEED,
    "official_preflight_sha256": digest,
    "implementation_bindings_sha256": preflight["implementation_bindings_sha256"],
    "cuda_visible_devices": sys.argv[3],
    "torch_version": torch.__version__,
    "device_name": torch.cuda.get_device_name(0),
    "training_started": False,
    "formal_subc_test_nwb_opened": False,
    "no_test_files_evaluated": True,
}
write_immutable_json(Path(sys.argv[2]), payload)
PY

{
  echo "[$(date -Is)] START A1 H/T4 seed42 GPU=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN[@]}"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL[@]}"
  "${AGG[@]}"
  echo "[$(date -Is)] DONE"
} >"$LOG" 2>&1
