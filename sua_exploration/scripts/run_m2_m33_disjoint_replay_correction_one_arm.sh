#!/usr/bin/env bash
# Execute one and only one CPU-only test replay from the frozen M33 correction receipt.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_m33_disjoint_replay_correction_one_arm.sh \
  --group f0|t4|ts4 --cell fold1_seed42|fold1_seed43|fold2_seed42 [--dry-run|--launch]

The historical M33 CSV is never read as a score input. --launch runs frozen
test-only inference on CPU, with chronological support trials [0:33] and
full-history-disjoint query windows. It is intentionally non-self-authorising.
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
SCREEN="m2_m33_disjoint_replay_correction_v1"
PROTO="$ROOT/sua_exploration/results/$SCREEN/protocol_receipt.json"
PROTO_SHA="da4eb05db43401f2e314351e8882c6606cc4d020aed3ed45e860d52db9e59df1"
NORM_ADDENDUM="$ROOT/sua_exploration/results/$SCREEN/normalization_parity_addendum_v1.json"
NORM_ADDENDUM_SHA="8a103352eb6341152ec18096411563bc968162b7154074f80a958793b0a2d3e8"
case "$CELL" in
  fold1_seed42) FOLD=1; SEED=42 ;;
  fold1_seed43) FOLD=1; SEED=43 ;;
  fold2_seed42) FOLD=2; SEED=42 ;;
esac
case "$GROUP" in
  f0) EXPERIMENT=m2_m33_disjoint_replay_correction_f0 ;;
  t4) EXPERIMENT=m2_m33_disjoint_replay_correction_t4 ;;
  ts4) EXPERIMENT=m2_m33_disjoint_replay_correction_ts4 ;;
esac
RUN_ID="${SCREEN}_${GROUP}_m2"
if [[ -n "$ATTEMPT" ]]; then
  [[ "$ATTEMPT" =~ ^[a-z0-9_]+$ ]] || { echo "--attempt must be lowercase alnum/underscore" >&2; exit 2; }
  RUN_ID="${RUN_ID}_${ATTEMPT}"
fi

[[ -x "$PYTHON_BIN" && -f "$PROTO" ]] || { echo "missing Python or immutable protocol receipt" >&2; exit 1; }
[[ "$(sha256sum "$PROTO" | awk '{print $1}')" == "$PROTO_SHA" ]] || { echo "protocol receipt SHA drift" >&2; exit 1; }
[[ -f "$NORM_ADDENDUM" && "$(sha256sum "$NORM_ADDENDUM" | awk '{print $1}')" == "$NORM_ADDENDUM_SHA" ]] || { echo "normalization addendum missing or SHA drift" >&2; exit 1; }
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
CMD=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED" "run_id=$RUN_ID"
  "train=false" "test=true" "optimized_metric=null" "ckpt_path=$CKPT" "require_baseline_validation=false"
  "data.include_heldout_in_fit=false" "data.include_heldout_in_test=true" "data.calibration_n_trials=33"
  "data.query_start_trial=33" "data.random_calibration=false" "data.allow_empty_heldout_query=true"
  "trainer.accelerator=cpu" "trainer.devices=1"
)
if [[ "$MODE" == dry-run ]]; then
  echo "scope=CPU-only test-only M33 correction; historical scores are forbidden inputs"
  echo "source_checkpoint=$CKPT sha256=$CKPT_SHA"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"
LOG_SUFFIX="${ATTEMPT:-default}"
LOG="$ROOT/sua_exploration/results/$SCREEN/logs/${GROUP}_${CELL}_${LOG_SUFFIX}_cpu.log"
{
  echo "[$(date -Is)] start CPU-only M33 correction group=$GROUP cell=$CELL"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  echo "source_checkpoint=$CKPT sha256=$CKPT_SHA"
  echo "support=[0:33] chronological; zero-query sessions are ineligible"
  (cd "$SCE" && MPLCONFIGDIR="/tmp/${SCREEN}_${GROUP}_${CELL}" "${CMD[@]}")
  mapfile -t ARTIFACTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
  [[ ${#ARTIFACTS[@]} == 1 ]] || { echo "expected one artifact, found ${#ARTIFACTS[@]}" >&2; exit 1; }
  "$PYTHON_BIN" "$ROOT/sua_exploration/scripts/write_m2_m33_disjoint_replay_correction_provenance.py" --artifact "${ARTIFACTS[0]}" --protocol "$PROTO" --expected-protocol-sha "$PROTO_SHA" --normalization-addendum "$NORM_ADDENDUM" --group "$GROUP" --cell "$CELL"
  echo "[$(date -Is)] complete CPU-only M33 correction group=$GROUP cell=$CELL"
} >"$LOG" 2>&1
