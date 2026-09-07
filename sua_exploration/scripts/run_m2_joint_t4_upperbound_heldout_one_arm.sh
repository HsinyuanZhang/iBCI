#!/usr/bin/env bash
# Locked local held-out replay. A separate signed/protocol GO file is mandatory.
set -euo pipefail
usage(){ echo "Usage: $0 --arm zero4|t4|ts4 --gpu INDEX --source-receipt PATH --protocol-go PATH --teacher-checkpoint PATH --teacher-receipt PATH [--launch|--dry-run]" >&2; }
ARM="";GPU="";SRC="";GO="";TEACHER="";RECEIPT="";MODE=dry-run
while [[ $# -gt 0 ]]; do case "$1" in --arm)ARM="$2";shift 2;;--gpu)GPU="$2";shift 2;;--source-receipt)SRC="$2";shift 2;;--protocol-go)GO="$2";shift 2;;--teacher-checkpoint)TEACHER="$2";shift 2;;--teacher-receipt)RECEIPT="$2";shift 2;;--launch)MODE=launch;shift;;--dry-run)shift;;*)usage;exit 2;;esac;done
[[ "$ARM" =~ ^(zero4|t4|ts4)$ && "$GPU" =~ ^[0-9]+$ && -f "$SRC" && -f "$GO" && -f "$TEACHER" && -f "$RECEIPT" ]] || { usage;exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
CKPT=$("$PYTHON" - "$SRC" "$GO" "$ARM" <<'PY'
import hashlib,json,sys
s=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2])); a=sys.argv[3]
if s.get('screen')!='m2_joint_t4_upperbound_v2' or s.get('source_heldout_opened') is not False: raise SystemExit('invalid source receipt')
if g.get('screen')!='m2_joint_t4_upperbound_v2' or g.get('allow_local_heldout_replay') is not True: raise SystemExit('independent protocol GO is absent')
if g.get('source_receipt_sha256')!=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(): raise SystemExit('GO/source receipt binding drift')
p=s['arms'][a]['checkpoint']['path']; print(p)
PY
)
BUSY="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null|awk 'NF{print $1}'|paste -sd, -)";[[ -z "$BUSY" ]]||{ echo "GPU busy: $BUSY" >&2;exit 1;}
RID="m2_joint_t4_upperbound_v2_${ARM}_heldout"; OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"; compgen -G "$OUT/${RID}_f1_s42_*" >/dev/null&&{ echo "existing heldout artifact" >&2;exit 1;}
CMD=("$PYTHON" src/train.py "experiment=m2_joint_upperbound_${ARM}_m24_loso" "run_id=$RID" "paths.root_dir=$ROOT/streaming_calibration_exp" "model.teacher_ckpt_path=$(realpath "$TEACHER")" "model.teacher_receipt_path=$(realpath "$RECEIPT")" train=false test=true "ckpt_path=$CKPT" optimized_metric=null data.include_heldout_in_fit=false data.include_heldout_in_test=true data.query_start_trial=24)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU";printf '%q ' "${CMD[@]}";printf '\n';[[ "$MODE" == launch ]]||exit 0
cd "$ROOT/streaming_calibration_exp";CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${RID}_gpu${GPU}" "${CMD[@]}"
