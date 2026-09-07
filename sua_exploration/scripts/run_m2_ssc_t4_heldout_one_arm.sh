#!/usr/bin/env bash
# One frozen test-only local-heldout replay arm for SSC-T4 M24.
set -euo pipefail

usage() { cat <<'EOF'
Usage: run_m2_ssc_t4_heldout_one_arm.sh --arm ordinary|ssc --gpu INDEX --teacher-checkpoint PATH --teacher-receipt PATH --source-gate PATH [--launch|--dry-run]
EOF
}

ARM=""; GPU=""; TEACHER=""; RECEIPT=""; SOURCE_GATE=""; MODE=dry-run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;; --gpu) GPU="$2"; shift 2;;
    --teacher-checkpoint) TEACHER="$2"; shift 2;; --teacher-receipt) RECEIPT="$2"; shift 2;;
    --source-gate) SOURCE_GATE="$2"; shift 2;; --launch) MODE=launch; shift;; --dry-run) shift;;
    -h|--help) usage; exit 0;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$ARM" == ordinary || "$ARM" == ssc ]] || { usage >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ && -n "$TEACHER" && -n "$RECEIPT" && -n "$SOURCE_GATE" ]] || { usage >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
PROTOCOL="$ROOT/sua_exploration/results/m2_ssc_t4_v1/heldout_protocol_receipt.json"
PROTOCOL_SHA=c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc
[[ "$(sha256sum "$PROTOCOL" | awk '{print $1}')" == "$PROTOCOL_SHA" ]] || { echo 'protocol receipt SHA drift' >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$RECEIPT" >/dev/null
SOURCE_INFO=$("$PYTHON" - "$SOURCE_GATE" "$ARM" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1])); arm='ordinary_t4' if sys.argv[2]=='ordinary' else 'ssc_t4'
if p.get('catastrophic_stop') is not False or p.get('next_action')!='one_frozen_local_heldout_replay_required': raise SystemExit('source gate forbids held-out replay')
c=p.get('arms',{}).get(arm,{}).get('checkpoint',{}); path=c.get('path'); want=c.get('sha256')
if not path or not want: raise SystemExit('source gate lacks selected checkpoint')
h=hashlib.sha256(open(path,'rb').read()).hexdigest()
if h!=want: raise SystemExit('source checkpoint SHA drift')
print(path,want)
PY
)
read -r SOURCE_CKPT SOURCE_SHA <<<"$SOURCE_INFO"
[[ -f "$SOURCE_CKPT" ]] || { echo 'source checkpoint missing' >&2; exit 1; }
if [[ "$ARM" == ordinary ]]; then EXP=b3s_t4_m2_m24_clean_teacher_loso_internal; RUN_ID=m2_ssc_t4_v1_heldout_ordinary; else EXP=ssc_t4_m2_m24_clean_teacher_loso_internal; RUN_ID=m2_ssc_t4_v1_heldout_ssc; fi
OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
compgen -G "$OUT/${RUN_ID}_f1_s42_*" >/dev/null && { echo "refusing existing heldout artifact for $RUN_ID" >&2; exit 1; }
CMD=("$PYTHON" src/train.py "experiment=$EXP" "run_id=$RUN_ID" "paths.root_dir=$ROOT/streaming_calibration_exp"
  "model.teacher_ckpt_path=$(realpath "$TEACHER")" "model.teacher_receipt_path=$(realpath "$RECEIPT")"
  "train=false" "test=true" "ckpt_path=$(realpath "$SOURCE_CKPT")" "require_baseline_validation=false"
  # A test-only process has no validation loop, hence no val_heldin metric to
  # optimize.  Null this training-only field so completed metric export exits
  # zero; checkpoint choice remains bound to the frozen source gate above.
  "optimized_metric=null"
  "trainer.accelerator=gpu" "trainer.devices=1"
  "data.include_heldout_in_fit=false" "data.include_heldout_in_test=true" "data.query_start_trial=24" "data.calibration_n_trials=24" "data.random_calibration=false")
if [[ "$MODE" == dry-run ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
(cd "$ROOT/streaming_calibration_exp" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${RUN_ID}_gpu${GPU}" "${CMD[@]}")
