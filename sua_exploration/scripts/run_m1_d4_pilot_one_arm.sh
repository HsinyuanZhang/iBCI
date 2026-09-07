#!/usr/bin/env bash
# Train exactly one pre-authorized D4 or DS4 Phase-B pilot arm.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m1_d4_pilot_one_arm.sh --group d4|ds4 --gpu 0|1 [--dry-run|--launch]

Trains fold1_seed42 only on selection window [10,210).  It never evaluates the
sealed report window and refuses to overwrite a pilot artifact.
EOF
}

MODE=dry-run; GROUP=""; GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --launch) MODE=launch; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == d4 || "$GROUP" == ds4 ]] || { echo "invalid --group" >&2; exit 2; }
[[ "$GPU" == 0 || "$GPU" == 1 ]] || { echo "--gpu must be 0 or 1" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCE="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="m1_d4_pilot_v1"
CELL="fold1_seed42"; FOLD=1; SEED=42
PROTO="$ROOT/sua_exploration/results/$SCREEN/phase_b_receipt.json"
EXPERIMENT="m1_d4_pilot_${GROUP}"
RUN_ID="${SCREEN}_${GROUP}_m1"

[[ -f "$PROTO" ]] || { echo "missing approved Phase-B receipt: $PROTO" >&2; exit 1; }
PROTO_SHA="$(sha256sum "$PROTO" | awk '{print $1}')"
if compgen -G "$SCE/outputs/streaming_calibration/${RUN_ID}_f${FOLD}_s${SEED}_*" >/dev/null; then
  echo "refusing to overwrite existing pilot artifact for $GROUP/$CELL" >&2; exit 1
fi

CMD=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED" "run_id=$RUN_ID"
  "train=true" "test=false" "require_baseline_validation=false"
  "data.calibration_n_trials=10" "data.random_calibration=false"
  "data.heldin_query_start_trial=10" "data.heldin_query_end_trial=210"
  "trainer.accelerator=gpu" "trainer.devices=1"
)
if [[ "$MODE" == dry-run ]]; then
  echo "scope=training only, M1 D4 Phase-B selection window [10,210); report [210,end) remains sealed"
  echo "phase_b_receipt=$PROTO sha256=$PROTO_SHA"
  echo "group=$GROUP cell=$CELL fold=$FOLD seed=$SEED gpu=$GPU"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"
LOG="$ROOT/sua_exploration/results/$SCREEN/logs/train_${GROUP}_${CELL}.log"
{
  echo "[$(date -Is)] start D4 pilot training group=$GROUP cell=$CELL gpu=$GPU"
  echo "phase_b_receipt=$PROTO sha256=$PROTO_SHA"
  export CUDA_VISIBLE_DEVICES="$GPU"
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
  BEST_SRC="$("$PYTHON_BIN" - "${RUN_DIRS[-1]}" <<'PY'
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
            path, score = value.get('best_model_path'), value.get('best_model_score')
            if path and score is not None:
                score = float(score.item() if hasattr(score, 'item') else score)
                if best_score is None or score >= best_score:
                    best_path, best_score = path, score
if best_path is None:
    raise SystemExit('no best checkpoint found')
print(best_path)
PY
)"
  cp -f "$BEST_SRC" "$ART/checkpoints/best.ckpt"
  sha256sum "$ART/checkpoints/best.ckpt" "$ART/checkpoints/last.ckpt"
  echo "[$(date -Is)] complete D4 pilot training group=$GROUP cell=$CELL artifact=$ART"
} >"$LOG" 2>&1
