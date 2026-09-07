#!/usr/bin/env bash
# Held-in-only ordinary T4 student source for one frozen replication seed.
set -euo pipefail

usage() { cat <<'EOF'
Usage: run_m2_t4_clean_spint_replication_source.sh --seed 43|44 --gpu INDEX \
  --teacher-checkpoint PATH --teacher-receipt PATH [--launch|--dry-run]

This command cannot test held-out data.  It accepts a teacher only through its
per-seed clean-teacher receipt and trains the fixed ordinary B3S+T4 source.
EOF
}

SEED=""; GPU=""; TEACHER=""; RECEIPT=""; MODE=dry-run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;; --gpu) GPU="$2"; shift 2;;
    --teacher-checkpoint) TEACHER="$2"; shift 2;; --teacher-receipt) RECEIPT="$2"; shift 2;;
    --launch) MODE=launch; shift;; --dry-run) MODE=dry-run; shift;;
    -h|--help) usage; exit 0;; *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$SEED" == 43 || "$SEED" == 44 ]] || { echo "seed must be 43 or 44" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ && -n "$TEACHER" && -n "$RECEIPT" ]] || { usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
REPL="$ROOT/sua_exploration/results/m2_t4_clean_spint_replication_v1"
PROTOCOL="$REPL/protocol_receipt.json"
PROTOCOL_SHA=c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb
[[ "$(sha256sum "$PROTOCOL" | awk '{print $1}')" == "$PROTOCOL_SHA" ]] || { echo "replication protocol SHA drift" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$RECEIPT" >/dev/null

RUN_ID="m2_t4_clean_spint_replication_t4_source"
OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
compgen -G "$OUT/${RUN_ID}_f1_s${SEED}_*" >/dev/null && { echo "existing source artifact for seed $SEED" >&2; exit 1; }
CMD=("$PYTHON" src/train.py experiment=b3s_t4_m2_m24_clean_teacher_loso_internal "seed=$SEED" "run_id=$RUN_ID"
  "paths.root_dir=$ROOT/streaming_calibration_exp"
  "model.teacher_ckpt_path=$(realpath "$TEACHER")" "model.teacher_receipt_path=$(realpath "$RECEIPT")"
  "require_baseline_validation=false" "test=true" "data.include_heldout_in_fit=false" "data.include_heldout_in_test=false")
if [[ "$MODE" == dry-run ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
cd "$ROOT/streaming_calibration_exp"
CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/m2_t4_clean_spint_source_s${SEED}_gpu${GPU}" "${CMD[@]}"
