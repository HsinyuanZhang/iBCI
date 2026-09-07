#!/usr/bin/env bash
# Execute one and only one inference-only replay from the frozen M1 held-in correction receipt.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m1_heldin_disjoint_replay_correction_one_arm.sh \
  --group f0|t4|ts4 --cell fold1_seed42|fold1_seed43|fold2_seed42 [--dry-run|--launch]

The historical contaminated M1 internal-LOSO CSV is never read as a score input.
--launch runs frozen test-only inference with chronological support trials [0:10]
and full-history-disjoint held-in-calib query windows starting at trial 10.
EOF
}

MODE=dry-run; GROUP=""; CELL=""; ATTEMPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --launch) MODE=launch; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --attempt) ATTEMPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == f0 || "$GROUP" == t4 || "$GROUP" == ts4 ]] || { echo "invalid --group" >&2; exit 2; }
[[ "$CELL" == fold1_seed42 || "$CELL" == fold1_seed43 || "$CELL" == fold2_seed42 ]] || { echo "invalid --cell" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCE="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="m1_heldin_disjoint_replay_v1"
PROTO="$ROOT/sua_exploration/results/$SCREEN/protocol_receipt.json"
NORM_ADDENDUM="$ROOT/sua_exploration/results/$SCREEN/normalization_parity_addendum_v1.json"
case "$CELL" in
  fold1_seed42) FOLD=1; SEED=42 ;;
  fold1_seed43) FOLD=1; SEED=43 ;;
  fold2_seed42) FOLD=2; SEED=42 ;;
esac
case "$GROUP" in
  f0) EXPERIMENT=m1_heldin_disjoint_replay_correction_f0 ;;
  t4) EXPERIMENT=m1_heldin_disjoint_replay_correction_t4 ;;
  ts4) EXPERIMENT=m1_heldin_disjoint_replay_correction_ts4 ;;
esac
RUN_ID="${SCREEN}_${GROUP}_m1"
if [[ -n "$ATTEMPT" ]]; then
  [[ "$ATTEMPT" =~ ^[a-z0-9_]+$ ]] || { echo "--attempt must be lowercase alnum/underscore" >&2; exit 2; }
  RUN_ID="${RUN_ID}_${ATTEMPT}"
fi

[[ -x "$PYTHON_BIN" && -f "$PROTO" && -f "$NORM_ADDENDUM" ]] || { echo "missing Python or immutable protocol/addendum receipts" >&2; exit 1; }
PROTO_SHA="$(sha256sum "$PROTO" | awk '{print $1}')"
NORM_SHA="$(sha256sum "$NORM_ADDENDUM" | awk '{print $1}')"
read -r CKPT CKPT_SHA < <("$PYTHON_BIN" - "$PROTO" "$GROUP" "$CELL" <<'PY'
import hashlib
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())
item = record['frozen_source_arms'][sys.argv[2]][sys.argv[3]]['checkpoint']
checkpoint = pathlib.Path(item['path'])
if not checkpoint.is_file():
    raise SystemExit(f'missing source checkpoint: {checkpoint}')
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != item['sha256']:
    raise SystemExit('source checkpoint SHA drift')
print(checkpoint, digest)
PY
)

if compgen -G "$SCE/outputs/streaming_calibration/${RUN_ID}_f${FOLD}_s${SEED}_*" >/dev/null; then
  echo "refusing to overwrite existing correction artifact for $GROUP/$CELL" >&2; exit 1
fi

ACCEL=gpu
DEVICES=1
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1 < 20 {found=1} END {exit !found}'; then
    ACCEL=cpu
    DEVICES=1
  fi
else
  ACCEL=cpu
fi

CMD=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED" "run_id=$RUN_ID"
  "train=false" "test=true" "optimized_metric=null" "ckpt_path=$CKPT" "require_baseline_validation=false"
  "data.include_heldout_in_fit=false" "data.include_heldout_in_test=false" "data.calibration_n_trials=10"
  "data.heldin_query_start_trial=10" "data.random_calibration=false"
  "trainer.accelerator=$ACCEL" "trainer.devices=$DEVICES"
)
if [[ "$MODE" == dry-run ]]; then
  echo "scope=test-only M1 held-in-calib post-support correction; historical contaminated scores are forbidden inputs"
  echo "source_checkpoint=$CKPT sha256=$CKPT_SHA"
  echo "accelerator=$ACCEL devices=$DEVICES"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"
LOG_SUFFIX="${ATTEMPT:-default}"
LOG="$ROOT/sua_exploration/results/$SCREEN/logs/${GROUP}_${CELL}_${LOG_SUFFIX}.log"
{
  echo "[$(date -Is)] start M1 held-in correction group=$GROUP cell=$CELL accelerator=$ACCEL"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  echo "normalization_addendum=$NORM_ADDENDUM sha256=$NORM_SHA"
  echo "source_checkpoint=$CKPT sha256=$CKPT_SHA"
  echo "support=[0:10] chronological; query from held-in-calib trial 10 onward"
  (cd "$SCE" && MPLCONFIGDIR="/tmp/${SCREEN}_${GROUP}_${CELL}" "${CMD[@]}")
  mapfile -t ARTIFACTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
  [[ ${#ARTIFACTS[@]} == 1 ]] || { echo "expected one artifact, found ${#ARTIFACTS[@]}" >&2; exit 1; }
  "$PYTHON_BIN" "$ROOT/sua_exploration/scripts/write_m1_heldin_disjoint_replay_correction_provenance.py" \
    --artifact "${ARTIFACTS[0]}" --protocol "$PROTO" --expected-protocol-sha "$PROTO_SHA" \
    --normalization-addendum "$NORM_ADDENDUM" --group "$GROUP" --cell "$CELL"
  echo "[$(date -Is)] complete M1 held-in correction group=$GROUP cell=$CELL"
} >"$LOG" 2>&1
