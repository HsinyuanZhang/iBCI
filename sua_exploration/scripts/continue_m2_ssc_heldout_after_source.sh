#!/usr/bin/env bash
# One fail-closed post-source transition for the frozen SSC-T4 M24 replay.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: continue_m2_ssc_heldout_after_source.sh --teacher-checkpoint PATH --teacher-receipt PATH --source-gate PATH

Only a source gate whose sole next action is the frozen local-heldout replay is
accepted.  It launches ordinary/SSC test-only arms in parallel, writes each
provenance receipt, evaluates the clean SPINT third reference once, and runs
the strict aggregate.  It never retries, overwrites, or opens held-out inputs
after a catastrophic source stop.
EOF
}

TEACHER=""; RECEIPT=""; SOURCE_GATE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-checkpoint) TEACHER="$2"; shift 2;;
    --teacher-receipt) RECEIPT="$2"; shift 2;;
    --source-gate) SOURCE_GATE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$TEACHER" && -n "$RECEIPT" && -n "$SOURCE_GATE" ]] || { usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
RESULT="$ROOT/sua_exploration/results/m2_ssc_t4_v1"
PROTOCOL="$RESULT/heldout_protocol_receipt.json"

# This read-only gate happens before either held-out runner is invoked.
"$PYTHON" - "$SOURCE_GATE" "$PROTOCOL" <<'PY'
import hashlib,json,sys
gate=json.load(open(sys.argv[1])); protocol=open(sys.argv[2],'rb').read()
if hashlib.sha256(protocol).hexdigest() != 'c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc': raise SystemExit('frozen protocol receipt SHA drift')
if gate.get('catastrophic_stop') is not False or gate.get('next_action') != 'one_frozen_local_heldout_replay_required': raise SystemExit('source gate forbids opening held-out')
for arm in ('ordinary_t4','ssc_t4'):
  value=gate.get('arms',{}).get(arm,{}).get('checkpoint_selection')
  if value != {'selected_by_metric':'val_heldin/r2_mean','no_early_stopping':True,'max_epochs':12}: raise SystemExit(f'{arm}: invalid source selection contract')
PY

mkdir -p "$RESULT/logs"
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_heldout_one_arm.sh" \
  --arm ordinary --gpu 0 --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --source-gate "$SOURCE_GATE" --launch \
  >"$RESULT/logs/ordinary_heldout.log" 2>&1 &
ORDINARY_PID=$!
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_heldout_one_arm.sh" \
  --arm ssc --gpu 1 --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --source-gate "$SOURCE_GATE" --launch \
  >"$RESULT/logs/ssc_heldout.log" 2>&1 &
SSC_PID=$!
STATUS=0
wait "$ORDINARY_PID" || STATUS=1
wait "$SSC_PID" || STATUS=1
[[ "$STATUS" == 0 ]] || { echo "held-out arm failed; refusing provenance/reference/aggregate" >&2; exit 1; }

OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
mapfile -t ORDINARY_ARTIFACTS < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_heldout_ordinary_f1_s42_*' -print | sort)
mapfile -t SSC_ARTIFACTS < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_heldout_ssc_f1_s42_*' -print | sort)
[[ "${#ORDINARY_ARTIFACTS[@]}" == 1 && "${#SSC_ARTIFACTS[@]}" == 1 ]] || { echo "expected exactly one held-out artifact per arm" >&2; exit 1; }

for spec in "${ORDINARY_ARTIFACTS[0]}:ordinary_t4" "${SSC_ARTIFACTS[0]}:ssc_t4"; do
  ARTIFACT="${spec%%:*}"; ARM="${spec##*:}"
  "$PYTHON" "$ROOT/sua_exploration/scripts/write_m2_ssc_t4_heldout_provenance.py" \
    --artifact "$ARTIFACT" --arm "$ARM" --source-gate "$SOURCE_GATE" --teacher-receipt "$RECEIPT" --protocol-receipt "$PROTOCOL"
done

REFERENCE="$RESULT/clean_spint_m24_local_heldout_reference.json"
[[ ! -e "$REFERENCE" ]] || { echo "clean SPINT reference already exists; refusing overwrite" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/streaming_calibration_exp/scripts/eval_clean_teacher_m2_m24_heldout.py" \
  --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --source-gate "$SOURCE_GATE" \
  --protocol-receipt "$PROTOCOL" --out "$REFERENCE" --device cuda

AGGREGATE="$RESULT/aggregate_heldout.json"
[[ ! -e "$AGGREGATE" ]] || { echo "held-out aggregate already exists; refusing overwrite" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/aggregate_m2_ssc_t4_heldout.py" \
  --ordinary "${ORDINARY_ARTIFACTS[0]}" --ssc "${SSC_ARTIFACTS[0]}" --source-gate "$SOURCE_GATE" \
  --protocol-receipt "$PROTOCOL" --clean-spint-reference "$REFERENCE" --out "$AGGREGATE"
echo "frozen SSC-T4 local-heldout replay complete: $AGGREGATE"
