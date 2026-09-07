#!/usr/bin/env bash
# One fail-closed continuation for one already-running clean-teacher seed.
set -euo pipefail
usage(){ echo "Usage: $0 --seed 43|44 --gpu INDEX --teacher-pid PID --teacher-run DIR --teacher-log LOG" >&2; }
SEED="";GPU="";TPID="";TRUN="";TLOG=""
while [[ $# -gt 0 ]]; do case "$1" in --seed)SEED="$2";shift 2;;--gpu)GPU="$2";shift 2;;--teacher-pid)TPID="$2";shift 2;;--teacher-run)TRUN="$2";shift 2;;--teacher-log)TLOG="$2";shift 2;;*)usage;exit 2;;esac;done
[[ "$SEED" == 43 || "$SEED" == 44 ]] && [[ "$GPU" =~ ^[0-9]+$ && "$TPID" =~ ^[0-9]+$ && -d "$TRUN" && -f "$TLOG" ]] || { usage;exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."&&pwd)";PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python;REPL="$ROOT/sua_exploration/results/m2_t4_clean_spint_replication_v1";PROTO="$REPL/protocol_receipt.json";ADD="$REPL/pre_heldout_addendum_v1.json";TRACE=/tmp/spint_clean_teacher_preflight_root.strace;TRACE_SHA=b1f93fa5de2661531468a0a329d973cbf5a6932d3497bd10d91370fea89f3de9
TRECEIPT="$REPL/receipts/clean_teacher_s${SEED}.json";SRECEIPT="$REPL/receipts/t4_source_s${SEED}.json";CREF="$REPL/receipts/clean_spint_primary_s${SEED}.json";OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
MANIFEST="$REPL/manifests/clean_teacher_input_manifest_s${SEED}.json"
"$PYTHON" - "$TRUN/.hydra/config.yaml" "$MANIFEST" "$SEED" <<'PY'
import hashlib,json,sys,yaml
cfg=yaml.safe_load(open(sys.argv[1])); manifest=sys.argv[2]; seed=int(sys.argv[3])
if cfg.get('seed') != seed: raise SystemExit('teacher Hydra seed does not match continuation seed')
if cfg.get('data',{}).get('clean_teacher_manifest_path') != manifest: raise SystemExit('teacher data manifest path does not match seed')
if cfg.get('callbacks',{}).get('clean_teacher_provenance',{}).get('input_manifest_path') != manifest: raise SystemExit('teacher callback manifest path does not match seed')
payload=json.load(open(manifest))
if payload.get('calibration_n_trials') != 24 or len(payload.get('files',[])) != 14: raise SystemExit('seed manifest contents invalid')
print(hashlib.sha256(open(manifest,'rb').read()).hexdigest())
PY
while kill -0 "$TPID" 2>/dev/null;do sleep 30;done
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_completion.py" --log "$TLOG"
mapfile -t CKPTS < <(find "$TRUN/checkpoints/clean_teacher_best" -maxdepth 1 -type f -name '*.ckpt' -print|sort);[[ ${#CKPTS[@]} == 1 ]]||{ echo 'expected one clean teacher checkpoint' >&2;exit 1;};TEACHER="${CKPTS[0]}"
[[ ! -e "$TRECEIPT" ]]||{ echo 'teacher receipt already exists' >&2;exit 1;}
"$PYTHON" "$ROOT/sua_exploration/scripts/finalize_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --resolved-config "$TRUN/.hydra/config.yaml" --output "$TRECEIPT" --strace-path "$TRACE" --strace-sha256 "$TRACE_SHA"
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$TRECEIPT"
"$ROOT/sua_exploration/scripts/run_m2_t4_clean_spint_replication_source.sh" --seed "$SEED" --gpu "$GPU" --teacher-checkpoint "$TEACHER" --teacher-receipt "$TRECEIPT" --launch
mapfile -t SOURCE_ARTIFACT < <(find "$OUT" -maxdepth 1 -type d -name "m2_t4_clean_spint_replication_t4_source_f1_s${SEED}_*" -print|sort);[[ ${#SOURCE_ARTIFACT[@]} == 1 ]]||{ echo 'expected one source artifact' >&2;exit 1;}
[[ ! -e "$SRECEIPT" ]]||{ echo 'source receipt already exists' >&2;exit 1;}
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_t4_clean_spint_source.py" --seed "$SEED" --artifact "${SOURCE_ARTIFACT[0]}" --teacher-receipt "$TRECEIPT" --protocol "$PROTO" --out "$SRECEIPT"
"$ROOT/sua_exploration/scripts/run_m2_t4_clean_spint_replication_heldout.sh" --seed "$SEED" --gpu "$GPU" --source-receipt "$SRECEIPT" --teacher-checkpoint "$TEACHER" --teacher-receipt "$TRECEIPT" --launch
mapfile -t HELDOUT_ARTIFACT < <(find "$OUT" -maxdepth 1 -type d -name "m2_t4_clean_spint_replication_t4_heldout_f1_s${SEED}_*" -print|sort);[[ ${#HELDOUT_ARTIFACT[@]} == 1 ]]||{ echo 'expected one heldout artifact' >&2;exit 1;}
"$PYTHON" "$ROOT/sua_exploration/scripts/write_m2_t4_clean_spint_replication_provenance.py" --seed "$SEED" --artifact "${HELDOUT_ARTIFACT[0]}" --source-receipt "$SRECEIPT" --teacher-receipt "$TRECEIPT" --protocol "$PROTO" --addendum "$ADD"
[[ ! -e "$CREF" ]]||{ echo 'clean primary reference already exists' >&2;exit 1;}
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/streaming_calibration_exp/scripts/eval_clean_teacher_m2_m24_heldout.py" --teacher-checkpoint "$TEACHER" --teacher-receipt "$TRECEIPT" --source-gate "$SRECEIPT" --source-arms ordinary_t4 --protocol-receipt "$PROTO" --expected-protocol-sha c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb --sampler-seed "$SEED" --role clean_full_spint_matched_primary_baseline --out "$CREF" --device cuda
echo "seed $SEED replication chain complete; awaiting joint aggregate"
