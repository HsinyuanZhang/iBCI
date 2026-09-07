#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_ssc_t4_clean_teacher_one_arm.sh --arm ordinary|ssc --gpu INDEX \
  --teacher-checkpoint PATH --teacher-receipt PATH [--launch|--dry-run]

The supplied teacher is never accepted by path alone: this runner invokes the
fail-closed receipt validator before composing Hydra.  The legacy epoch_034
teacher and all unreceipted checkpoints are rejected.
EOF
}

ARM=""
GPU=""
TEACHER=""
RECEIPT=""
MODE="dry-run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --teacher-checkpoint) TEACHER="$2"; shift 2 ;;
    --teacher-receipt) RECEIPT="$2"; shift 2 ;;
    --launch) MODE="launch"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$ARM" == "ordinary" || "$ARM" == "ssc" ]] || { usage >&2; exit 2; }
[[ -n "$GPU" && -n "$TEACHER" && -n "$RECEIPT" ]] || { usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python"
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" \
  --checkpoint "$TEACHER" --receipt "$RECEIPT"

if [[ "$ARM" == "ordinary" ]]; then
  EXPERIMENT="b3s_t4_m2_m24_clean_teacher_loso_internal"
  RUN_ID="m2_ssc_t4_v1_ordinary"
else
  EXPERIMENT="ssc_t4_m2_m24_clean_teacher_loso_internal"
  RUN_ID="m2_ssc_t4_v1_ssc"
fi
COMMAND=("$PYTHON" src/train.py "experiment=$EXPERIMENT" "run_id=$RUN_ID"
  "paths.root_dir=$ROOT/streaming_calibration_exp"
  "model.teacher_ckpt_path=$(realpath "$TEACHER")"
  "model.teacher_receipt_path=$(realpath "$RECEIPT")"
  # Source qualification is not complete without the held-in test metric that
  # aggregate_m2_ssc_t4_source.py consumes.  State this explicitly rather than
  # relying on train.yaml's inherited default.
  "test=true" "data.include_heldout_in_test=false")

printf 'validated command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "$MODE" == "launch" ]]; then
  cd "$ROOT/streaming_calibration_exp"
  CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${RUN_ID}_gpu${GPU}" "${COMMAND[@]}"
fi
