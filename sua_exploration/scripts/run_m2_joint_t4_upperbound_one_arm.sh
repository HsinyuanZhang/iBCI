#!/usr/bin/env bash
# One source-only arm of the independent M2 joint decoder upper-bound screen.
set -euo pipefail
usage(){ echo "Usage: $0 --arm zero4|t4|ts4 --gpu INDEX --teacher-checkpoint PATH --teacher-receipt PATH [--launch|--dry-run]" >&2; }
ARM=""; GPU=""; TEACHER=""; RECEIPT=""; MODE=dry-run
while [[ $# -gt 0 ]]; do case "$1" in
 --arm) ARM="$2"; shift 2;; --gpu) GPU="$2"; shift 2;; --teacher-checkpoint) TEACHER="$2"; shift 2;; --teacher-receipt) RECEIPT="$2"; shift 2;; --launch) MODE=launch; shift;; --dry-run) shift;; *) usage; exit 2;; esac; done
[[ "$ARM" == zero4 || "$ARM" == t4 || "$ARM" == ts4 ]] && [[ "$GPU" =~ ^[0-9]+$ && -f "$TEACHER" && -f "$RECEIPT" ]] || { usage; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
if [[ "$MODE" == launch ]]; then
  BUSY="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print $1}' | paste -sd, -)"
  [[ -z "$BUSY" ]] || { echo "GPU $GPU is busy with compute PID(s): $BUSY; refusing competing launch" >&2; exit 1; }
fi
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$RECEIPT" >/dev/null
RID="m2_joint_t4_upperbound_v2_${ARM}"; OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
compgen -G "$OUT/${RID}_f1_s42_*" >/dev/null && { echo "refusing existing artifact for $ARM" >&2; exit 1; }
CMD=("$PYTHON" src/train.py "experiment=m2_joint_upperbound_${ARM}_m24_loso" "run_id=$RID" "paths.root_dir=$ROOT/streaming_calibration_exp" "model.teacher_ckpt_path=$(realpath "$TEACHER")" "model.teacher_receipt_path=$(realpath "$RECEIPT")" test=false)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$MODE" == launch ]] || exit 0
cd "$ROOT/streaming_calibration_exp"; CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${RID}_gpu${GPU}" "${CMD[@]}"
