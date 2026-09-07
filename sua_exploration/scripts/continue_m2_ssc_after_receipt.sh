#!/usr/bin/env bash
# Resume only the post-receipt source -> gate -> held-out chain, fail closed.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: continue_m2_ssc_after_receipt.sh --teacher-checkpoint PATH --teacher-receipt PATH

This intentionally cannot finalize or overwrite a clean-teacher receipt.  It
requires an already-valid receipt and pristine source/held-out targets, then
runs ordinary and SSC source arms.  Only two successful source exits may reach
the frozen source gate and the existing post-source driver.
EOF
}

TEACHER=""; RECEIPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-checkpoint) TEACHER="$2"; shift 2;;
    --teacher-receipt) RECEIPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$TEACHER" && -n "$RECEIPT" ]] || { usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
RESULT="$ROOT/sua_exploration/results/m2_ssc_t4_v1"
OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
SOURCE_GATE="$RESULT/source_gate.json"

# Validation happens before any GPU launch.  This script never invokes the
# receipt finalizer, so a recovered chain cannot rewrite its teacher evidence.
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" \
  --checkpoint "$TEACHER" --receipt "$RECEIPT"
for path in "$SOURCE_GATE" "$RESULT/aggregate_heldout.json" "$RESULT/clean_spint_m24_local_heldout_reference.json"; do
  [[ ! -e "$path" ]] || { echo "recovery target already exists: $path" >&2; exit 1; }
done
# run_id is overridden by the source runner, so preflight must use the actual
# artifact names rather than the experiment-file defaults.
mapfile -t EXISTING_SOURCE < <(find "$OUT" -maxdepth 1 -type d \( -name 'm2_ssc_t4_v1_ordinary_f1_s42_*' -o -name 'm2_ssc_t4_v1_ssc_f1_s42_*' \) -print)
[[ "${#EXISTING_SOURCE[@]}" == 0 ]] || { echo "existing source artifact prevents recovery" >&2; exit 1; }
for legacy_log in "$RESULT/logs/ordinary_source.log" "$RESULT/logs/ssc_source.log"; do
  [[ ! -e "$legacy_log" || ! -s "$legacy_log" ]] || { echo "nonempty legacy source log prevents recovery: $legacy_log" >&2; exit 1; }
done

mkdir -p "$RESULT/logs"
ORDINARY_LOG="$RESULT/logs/ordinary_source_recovery4.log"
SSC_LOG="$RESULT/logs/ssc_source_recovery4.log"
[[ ! -e "$ORDINARY_LOG" && ! -e "$SSC_LOG" ]] || { echo "non-overwrite recovery source log already exists" >&2; exit 1; }
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_clean_teacher_one_arm.sh" \
  --arm ordinary --gpu 0 --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --launch \
  >"$ORDINARY_LOG" 2>&1 &
ORDINARY_PID=$!
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_clean_teacher_one_arm.sh" \
  --arm ssc --gpu 1 --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --launch \
  >"$SSC_LOG" 2>&1 &
SSC_PID=$!
STATUS=0
wait "$ORDINARY_PID" || STATUS=1
wait "$SSC_PID" || STATUS=1
[[ "$STATUS" == 0 ]] || { echo "source arm failure: refusing source gate and held-out" >&2; exit 1; }

mapfile -t ORDINARY_ARTIFACTS < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_ordinary_f1_s42_*' -print | sort)
mapfile -t SSC_ARTIFACTS < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_ssc_f1_s42_*' -print | sort)
[[ "${#ORDINARY_ARTIFACTS[@]}" == 1 && "${#SSC_ARTIFACTS[@]}" == 1 ]] || { echo "expected exactly one successful source artifact per arm" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/aggregate_m2_ssc_t4_source.py" \
  --ordinary "${ORDINARY_ARTIFACTS[0]}" --ssc "${SSC_ARTIFACTS[0]}" \
  --protocol-receipt "$RESULT/heldout_protocol_receipt.json" --out "$SOURCE_GATE"
exec "$ROOT/sua_exploration/scripts/continue_m2_ssc_heldout_after_source.sh" \
  --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --source-gate "$SOURCE_GATE"
