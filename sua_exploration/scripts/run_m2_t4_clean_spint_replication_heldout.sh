#!/usr/bin/env bash
# Exactly one authorized test-only ordinary-T4 replay for replication seed 43/44.
set -euo pipefail
usage() { echo "Usage: $0 --seed 43|44 --gpu INDEX --source-receipt PATH --teacher-checkpoint PATH --teacher-receipt PATH [--launch|--dry-run]" >&2; }
SEED=""; GPU=""; SOURCE=""; TEACHER=""; RECEIPT=""; MODE=dry-run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;; --gpu) GPU="$2"; shift 2;; --source-receipt) SOURCE="$2"; shift 2;;
    --teacher-checkpoint) TEACHER="$2"; shift 2;; --teacher-receipt) RECEIPT="$2"; shift 2;;
    --launch) MODE=launch; shift;; --dry-run) shift;; *) usage; exit 2;;
  esac
done
[[ "$SEED" == 43 || "$SEED" == 44 ]] && [[ "$GPU" =~ ^[0-9]+$ && -f "$SOURCE" && -f "$TEACHER" && -f "$RECEIPT" ]] || { usage; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
REPL="$ROOT/sua_exploration/results/m2_t4_clean_spint_replication_v1"; PROTOCOL="$REPL/protocol_receipt.json"; ADDENDUM="$REPL/pre_heldout_addendum_v1.json"
PROTO_SHA=c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb; ADD_SHA=f87d3d0fbd2231062c7c2b430bd8da7128d583dbf5a1010940b37bb9adb2866f
[[ "$(sha256sum "$PROTOCOL" | awk '{print $1}')" == "$PROTO_SHA" && "$(sha256sum "$ADDENDUM" | awk '{print $1}')" == "$ADD_SHA" ]] || { echo 'receipt/addendum SHA drift' >&2; exit 1; }
read -r CKPT CKPT_SHA < <("$PYTHON" - "$SOURCE" "$SEED" "$PROTO_SHA" "$ADD_SHA" "$TEACHER" "$RECEIPT" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1])); h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest(); teacher,receipt=sys.argv[5],sys.argv[6]
if x.get('seed')!=int(sys.argv[2]) or x.get('protocol',{}).get('sha256')!=sys.argv[3] or x.get('pre_heldout_addendum',{}).get('sha256')!=sys.argv[4]: raise SystemExit('source receipt binding drift')
if x.get('heldin_performance_not_used_as_gate') is not True or x.get('contract_runtime_noncatastrophic') is not True or x.get('next_action')!='one_frozen_local_heldout_replay_required': raise SystemExit('source receipt is not contract-only authorization')
st=x.get('teacher',{}); sr=x.get('teacher_receipt',{})
if st.get('checkpoint_sha256')!=h(teacher) or sr.get('sha256')!=h(receipt) or st.get('receipt_sha256')!=h(receipt): raise SystemExit('provided teacher/receipt bytes differ from source receipt')
if sr.get('path')!=__import__('os').path.realpath(receipt): raise SystemExit('provided teacher receipt path differs from source receipt')
c=x.get('checkpoint',{}); p=c.get('path')
if not p or not c.get('sha256') or h(p)!=c['sha256']: raise SystemExit('source checkpoint drift')
print(p,c['sha256'])
PY
)
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$RECEIPT" >/dev/null
RUN_ID=m2_t4_clean_spint_replication_t4_heldout; OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
compgen -G "$OUT/${RUN_ID}_f1_s${SEED}_*" >/dev/null && { echo "existing heldout artifact for seed $SEED" >&2; exit 1; }
CMD=("$PYTHON" src/train.py experiment=b3s_t4_m2_m24_clean_teacher_loso_internal "seed=$SEED" "run_id=$RUN_ID" "paths.root_dir=$ROOT/streaming_calibration_exp" "model.teacher_ckpt_path=$(realpath "$TEACHER")" "model.teacher_receipt_path=$(realpath "$RECEIPT")" "train=false" "test=true" "ckpt_path=$(realpath "$CKPT")" "optimized_metric=null" "require_baseline_validation=false" "trainer.accelerator=gpu" "trainer.devices=1" "data.include_heldout_in_fit=false" "data.include_heldout_in_test=true" "data.query_start_trial=24" "data.calibration_n_trials=24" "data.random_calibration=false")
if [[ "$MODE" == dry-run ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
cd "$ROOT/streaming_calibration_exp"; CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/m2_t4_clean_spint_heldout_s${SEED}_gpu${GPU}" "${CMD[@]}"
