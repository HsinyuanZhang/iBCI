#!/usr/bin/env bash
# Train one M1 clean-selection cell on the bounded selection window [10, 210).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m1_clean_selection_train_one_cell.sh \
  --group f0|t4|ts4 --cell fold1_seed42|fold1_seed43|fold2_seed42 [--gpu 0|1] [--dry-run|--launch]

Trains with val_heldin on selection window only. Report window [210, end) is never loaded.
EOF
}

MODE=dry-run; GROUP=""; CELL=""; GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --launch) MODE=launch; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == f0 || "$GROUP" == t4 || "$GROUP" == ts4 ]] || { echo "invalid --group" >&2; exit 2; }
[[ "$CELL" == fold1_seed42 || "$CELL" == fold1_seed43 || "$CELL" == fold2_seed42 ]] || { echo "invalid --cell" >&2; exit 2; }

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
  f0) EXPERIMENT=m1_clean_selection_f0 ;;
  t4) EXPERIMENT=m1_clean_selection_t4 ;;
  ts4) EXPERIMENT=m1_clean_selection_ts4 ;;
esac
RUN_ID="${SCREEN}_${GROUP}_m1"

[[ -f "$PROTO" ]] || { echo "missing protocol receipt: $PROTO" >&2; exit 1; }
PROTO_SHA="$(sha256sum "$PROTO" | awk '{print $1}')"

if compgen -G "$SCE/outputs/streaming_calibration/${RUN_ID}_f${FOLD}_s${SEED}_*" >/dev/null; then
  echo "refusing to overwrite existing training artifact for $GROUP/$CELL" >&2; exit 1
fi

ACCEL=gpu
DEVICES=1
if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

CMD=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED" "run_id=$RUN_ID"
  "train=true" "test=false" "require_baseline_validation=false"
  "data.calibration_n_trials=10" "data.random_calibration=false"
  "data.heldin_query_start_trial=10" "data.heldin_query_end_trial=210"
  "trainer.accelerator=$ACCEL" "trainer.devices=$DEVICES"
)
if [[ "$MODE" == dry-run ]]; then
  echo "scope=train M1 clean-selection selection window [10,210); report window sealed"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  echo "gpu=${GPU:-auto}"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"
LOG="$ROOT/sua_exploration/results/$SCREEN/logs/train_${GROUP}_${CELL}.log"
{
  echo "[$(date -Is)] start clean-selection training group=$GROUP cell=$CELL gpu=${GPU:-auto}"
  echo "protocol=$PROTO sha256=$PROTO_SHA"
  (cd "$SCE" && MPLCONFIGDIR="/tmp/${SCREEN}_train_${GROUP}_${CELL}" "${CMD[@]}")
  mapfile -t ARTIFACTS < <(find "$SCE/outputs/streaming_calibration" -maxdepth 1 -type d -name "${RUN_ID}_f${FOLD}_s${SEED}_*" -print | sort)
  [[ ${#ARTIFACTS[@]} == 1 ]] || { echo "expected one artifact, found ${#ARTIFACTS[@]}" >&2; exit 1; }
  ART="${ARTIFACTS[0]}"
  mapfile -t RUN_DIRS < <(find "$SCE/logs/train/runs" -maxdepth 1 -type d -name "*_rid-${RUN_ID}_f${FOLD}_s${SEED}" -print | sort)
  [[ ${#RUN_DIRS[@]} -ge 1 ]] || { echo "missing lightning run dir" >&2; exit 1; }
  LAST_SRC="${RUN_DIRS[-1]}/checkpoints/best_ckpt/last.ckpt"
  [[ -f "$LAST_SRC" ]] || { echo "missing last.ckpt at $LAST_SRC" >&2; exit 1; }
  mkdir -p "$ART/checkpoints"
  cp -f "$LAST_SRC" "$ART/checkpoints/last.ckpt"
  BEST_SRC="$("$PYTHON_BIN" - "$RUN_DIRS[-1]" <<'PY'
import sys
from pathlib import Path
import torch
run = Path(sys.argv[1])
best_path = None
best_score = None
for ckpt in sorted((run / 'checkpoints/best_ckpt').glob('epoch_*.ckpt')):
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    for value in state.get('callbacks', {}).values():
        if isinstance(value, dict) and value.get('monitor') == 'val_heldin/r2_mean':
            path = value.get('best_model_path')
            score = value.get('best_model_score')
            if path and score is not None:
                score_f = float(score.item() if hasattr(score, 'item') else score)
                if best_score is None or score_f >= best_score:
                    best_score = score_f
                    best_path = path
if best_path is None:
    raise SystemExit('no best checkpoint found')
print(best_path)
PY
)"
  cp -f "$BEST_SRC" "$ART/checkpoints/best.ckpt"
  sha256sum "$ART/checkpoints/best.ckpt" "$ART/checkpoints/last.ckpt"
  echo "[$(date -Is)] complete clean-selection training group=$GROUP cell=$CELL artifact=$ART"
} >"$LOG" 2>&1
