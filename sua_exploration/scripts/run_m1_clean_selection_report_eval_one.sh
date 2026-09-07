#!/usr/bin/env bash
# Evaluate one checkpoint source on the sealed M1 report window [210, end).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m1_clean_selection_report_eval_one.sh \
  --group f0|t4|ts4 --cell fold1_seed42|fold1_seed43|fold2_seed42 \
  --source clean_best|fixed_last|frozen_best [--gpu 0|1] [--dry-run|--launch]
EOF
}

MODE=dry-run; GROUP=""; CELL=""; SOURCE=""; GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --launch) MODE=launch; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == f0 || "$GROUP" == t4 || "$GROUP" == ts4 ]] || { echo "invalid --group" >&2; exit 2; }
[[ "$CELL" == fold1_seed42 || "$CELL" == fold1_seed43 || "$CELL" == fold2_seed42 ]] || { echo "invalid --cell" >&2; exit 2; }
[[ "$SOURCE" == clean_best || "$SOURCE" == fixed_last || "$SOURCE" == frozen_best ]] || { echo "invalid --source" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCE="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="m1_clean_selection_v1"
PROTO="$ROOT/sua_exploration/results/$SCREEN/protocol_receipt.json"
case "$CELL" in
  fold1_seed42) FOLD=1; SEED=42 ;;
  fold1_seed43) FOLD=1; SEED=43 ;;
  fold2_seed42) FOLD=2; SEED=42 ;;
esac
case "$GROUP" in
  f0) EXPERIMENT=m1_clean_selection_report_f0 ;;
  t4) EXPERIMENT=m1_clean_selection_report_t4 ;;
  ts4) EXPERIMENT=m1_clean_selection_report_ts4 ;;
esac
TRAIN_RUN_ID="m1_clean_selection_v1_${GROUP}_m1"
EVAL_RUN_ID="${SCREEN}_report_${SOURCE}_${GROUP}_m1"

[[ -f "$PROTO" ]] || { echo "missing protocol receipt" >&2; exit 1; }
PROTO_SHA="$(sha256sum "$PROTO" | awk '{print $1}')"

case "$SOURCE" in
  clean_best)
    mapfile -t TRAIN_ARTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${TRAIN_RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
    [[ ${#TRAIN_ARTS[@]} == 1 ]] || { echo "missing training artifact for $GROUP/$CELL" >&2; exit 1; }
    CKPT="${TRAIN_ARTS[0]}/checkpoints/best.ckpt"
    ;;
  fixed_last)
    mapfile -t TRAIN_ARTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${TRAIN_RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
    [[ ${#TRAIN_ARTS[@]} == 1 ]] || { echo "missing training artifact for $GROUP/$CELL" >&2; exit 1; }
    CKPT="${TRAIN_ARTS[0]}/checkpoints/last.ckpt"
    ;;
  frozen_best)
    CKPT="$("$PYTHON_BIN" - "$PROTO" "$GROUP" "$CELL" <<'PY'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(record["frozen_source_arms"][sys.argv[2]][sys.argv[3]]["frozen_checkpoint"]["path"])
PY
)"
    ;;
esac
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"

if compgen -G "$SCE/outputs/streaming_calibration/${EVAL_RUN_ID}_f${FOLD}_s${SEED}_*" >/dev/null; then
  echo "refusing to overwrite existing report eval artifact for $SOURCE $GROUP/$CELL" >&2; exit 1
fi

if [[ -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi

CMD=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED" "run_id=$EVAL_RUN_ID"
  "train=false" "test=true" "optimized_metric=null" "ckpt_path=$CKPT" "require_baseline_validation=false"
  "data.calibration_n_trials=10" "data.random_calibration=false"
  "data.heldin_query_start_trial=210" "data.heldin_query_end_trial=null"
  "trainer.accelerator=gpu" "trainer.devices=1"
)
if [[ "$MODE" == dry-run ]]; then
  echo "scope=report-window eval source=$SOURCE group=$GROUP cell=$CELL"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  echo "checkpoint=$CKPT sha256=$CKPT_SHA"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"
LOG="$ROOT/sua_exploration/results/$SCREEN/logs/report_${SOURCE}_${GROUP}_${CELL}.log"
{
  echo "[$(date -Is)] start report eval source=$SOURCE group=$GROUP cell=$CELL"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  echo "checkpoint=$CKPT sha256=$CKPT_SHA"
  (cd "$SCE" && MPLCONFIGDIR="/tmp/${SCREEN}_report_${SOURCE}_${GROUP}_${CELL}" "${CMD[@]}")
  mapfile -t ARTIFACTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${EVAL_RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
  [[ ${#ARTIFACTS[@]} == 1 ]] || { echo "expected one artifact, found ${#ARTIFACTS[@]}" >&2; exit 1; }
  echo "[$(date -Is)] complete report eval artifact=${ARTIFACTS[0]}"
} >"$LOG" 2>&1
