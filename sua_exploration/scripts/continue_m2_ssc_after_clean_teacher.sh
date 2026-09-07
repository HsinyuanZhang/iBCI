#!/usr/bin/env bash
# Wait for the one clean teacher, receipt it, then launch the two source-only SSC arms.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: continue_m2_ssc_after_clean_teacher.sh --teacher-pid PID --teacher-run-dir DIR --teacher-log LOG

This fail-closed continuation never opens held-out data.  It requires the
teacher to exit with the fixed 35-epoch completion record, finalizes and
validates its receipt, then starts ordinary B3S+T4 on GPU0 and SSC-T4 on GPU1.
EOF
}

PID=""; TEACHER_RUN=""; TEACHER_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-pid) PID="$2"; shift 2 ;;
    --teacher-run-dir) TEACHER_RUN="$2"; shift 2 ;;
    --teacher-log) TEACHER_LOG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$PID" =~ ^[0-9]+$ && -d "$TEACHER_RUN" && -f "$TEACHER_LOG" ]] || { usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
RESULT="$ROOT/sua_exploration/results/m2_ssc_t4_v1"
RECEIPT="$RESULT/clean_teacher_receipt.json"
TRACE=/tmp/spint_clean_teacher_preflight_root.strace
TRACE_SHA=b1f93fa5de2661531468a0a329d973cbf5a6932d3497bd10d91370fea89f3de9

while kill -0 "$PID" 2>/dev/null; do sleep 30; done
# Lightning appends the marker directly to the CR progress record.  This
# validator reads raw bytes and requires one exact terminal marker attached to
# the final Epoch 34 record; it accepts neither a partial string nor a stale
# duplicated marker.
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_completion.py" --log "$TEACHER_LOG" || {
  echo "teacher did not complete its fixed 35 epochs; refusing receipt/student launch" >&2; exit 1;
}
mapfile -t CKPTS < <(find "$TEACHER_RUN/checkpoints/clean_teacher_best" -maxdepth 1 -type f -name '*.ckpt' -print | sort)
[[ "${#CKPTS[@]}" == 1 ]] || { echo "expected exactly one selected clean-teacher checkpoint" >&2; exit 1; }
[[ ! -e "$RECEIPT" ]] || { echo "clean-teacher receipt already exists; refusing overwrite" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/finalize_m2_clean_teacher_receipt.py" \
  --checkpoint "${CKPTS[0]}" --resolved-config "$TEACHER_RUN/.hydra/config.yaml" \
  --output "$RECEIPT" --strace-path "$TRACE" --strace-sha256 "$TRACE_SHA"
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" \
  --checkpoint "${CKPTS[0]}" --receipt "$RECEIPT"

mkdir -p "$RESULT/logs"
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_clean_teacher_one_arm.sh" \
  --arm ordinary --gpu 0 --teacher-checkpoint "${CKPTS[0]}" --teacher-receipt "$RECEIPT" --launch \
  >"$RESULT/logs/ordinary_source.log" 2>&1 &
ORDINARY_PID=$!
"$ROOT/sua_exploration/scripts/run_m2_ssc_t4_clean_teacher_one_arm.sh" \
  --arm ssc --gpu 1 --teacher-checkpoint "${CKPTS[0]}" --teacher-receipt "$RECEIPT" --launch \
  >"$RESULT/logs/ssc_source.log" 2>&1 &
SSC_PID=$!
wait "$ORDINARY_PID"
wait "$SSC_PID"
mapfile -t ORDINARY_ARTIFACTS < <(find "$ROOT/streaming_calibration_exp/outputs/streaming_calibration" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_ordinary_f1_s42_*' -print | sort)
mapfile -t SSC_ARTIFACTS < <(find "$ROOT/streaming_calibration_exp/outputs/streaming_calibration" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_ssc_f1_s42_*' -print | sort)
[[ "${#ORDINARY_ARTIFACTS[@]}" == 1 && "${#SSC_ARTIFACTS[@]}" == 1 ]] || {
  echo "expected exactly one completed source artifact per arm" >&2; exit 1;
}
SOURCE_GATE="$RESULT/source_gate.json"
[[ ! -e "$SOURCE_GATE" ]] || { echo "source gate already exists; refusing overwrite" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/aggregate_m2_ssc_t4_source.py" \
  --ordinary "${ORDINARY_ARTIFACTS[0]}" --ssc "${SSC_ARTIFACTS[0]}" \
  --protocol-receipt "$RESULT/heldout_protocol_receipt.json" --out "$SOURCE_GATE"
# The post-source driver independently rechecks the gate before it can touch a
# held-out path; a catastrophic source result therefore ends here without a
# local replay.  It also refuses all pre-existing artifacts/receipts.
exec "$ROOT/sua_exploration/scripts/continue_m2_ssc_heldout_after_source.sh" \
  --teacher-checkpoint "${CKPTS[0]}" --teacher-receipt "$RECEIPT" --source-gate "$SOURCE_GATE"
