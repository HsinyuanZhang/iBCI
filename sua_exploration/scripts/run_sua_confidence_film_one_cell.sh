#!/usr/bin/env bash
# One-cell production runner for the reviewed confidence-conditioned T4 screen.
#
# Required environment:
#   ARM={t4_continuation,film,confidence_shuffle,nofilm_match,film_ts4}
#   SEED=<integer>
#   GPU=<CUDA device index>
#   ANCHOR=<ordinary T4 epoch_011.ckpt for the same seed and M_T4>
#
# Activity calibration is fixed at first 30 rewarded trials.  T4 and its
# confidence use the same first M_T4 labelled/rate trials.  Every arm starts
# evaluation at trial 50, so no arm evaluates a trial used by the largest
# calibration budget.  This runner is validation-only and never opens the
# sealed formal-test split.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_confidence_film_v1}"
ARM="${ARM:?ARM is required}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
ANCHOR="${ANCHOR:?ANCHOR is required}"
M_T4="${M_T4:-50}"
MODE="${1:---dry-run}"

M_ACTIVITY=30
EVAL_START=50
TOTAL_EPOCHS=12
BURN_IN=4
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOGS="$RESULTS/logs"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"

case "$ARM" in
  t4_continuation)
    VARIANT="B3S"
    SIDE_FEATURES="t4"
    ;;
  film)
    VARIANT="B3SCF"
    SIDE_FEATURES="t4cf"
    ;;
  confidence_shuffle)
    VARIANT="B3SCFS"
    SIDE_FEATURES="t4cf_confidence_shuffled"
    ;;
  nofilm_match)
    VARIANT="B3SCFA"
    SIDE_FEATURES="t4cf"
    ;;
  film_ts4)
    VARIANT="B3SCF"
    SIDE_FEATURES="t4cf_ts4"
    ;;
  *)
    echo "Unsupported ARM=$ARM" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: ARM=<arm> SEED=<seed> GPU=<index> ANCHOR=<ckpt> M_T4=<budget> $0 [--dry-run|--launch]" >&2
  exit 2
fi

"$PY" "$ROOT/sua_exploration/scripts/assert_confidence_film_protocol.py" \
  --t4-budget "$M_T4" >/dev/null

NAME="${SCREEN_ID}_${ARM}_m${M_T4}_dandi688_co_s${SEED}"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${ARM}_m${M_T4}_s${SEED}.json"
LOG="$LOGS/${ARM}_m${M_T4}_s${SEED}.log"

TRAIN_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
  --teacher_ckpt "$TEACHER"
  --variant "$VARIANT"
  --side_features "$SIDE_FEATURES"
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --encoder_warmstart_path "$ANCHOR"
  --out_name "$NAME"
  --data_dir "$DATA"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
  --signal_view sua
  --task CO
  --split_counts 27,6,6
  --max_units_exclusive 100
  --max_epochs "$TOTAL_EPOCHS"
  --no_early_stopping
  --checkpoint_every_epoch
  --lr 1e-4
  --batch_size 32
  --num_workers 4
  --seed "$SEED"
  --loss_mode task_only
  --identity_mode calibrated
  --require_gpu
  --disable_progress_bar
)

EVAL_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py"
  --run_dir "$RUN_DIR"
  --teacher_ckpt "$TEACHER"
  --data_dir "$DATA"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
  --total_epochs "$TOTAL_EPOCHS"
  --burn_in "$BURN_IN"
  --calibration_n "$M_ACTIVITY"
  --pool_size "$EVAL_START"
  --out_path "$RESULT"
)

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  echo
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${EVAL_COMMAND[@]}"
  echo
  exit 0
fi

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "Missing teacher checkpoint: $TEACHER" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing strict manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$ANCHOR" ]] || { echo "Missing ordinary-T4 anchor checkpoint: $ANCHOR" >&2; exit 1; }
[[ "$(basename "$ANCHOR")" == "epoch_011.ckpt" ]] || {
  echo "Anchor rule is the predeclared final epoch_011.ckpt, got: $ANCHOR" >&2
  exit 1
}
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to reuse run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || { echo "Refusing to overwrite result: $RESULT" >&2; exit 1; }

ANCHOR_RUN_DIR="$(dirname "$(dirname "$(realpath "$ANCHOR")")")"
ANCHOR_METADATA="$ANCHOR_RUN_DIR/run_metadata.json"
[[ -f "$ANCHOR_METADATA" ]] || {
  echo "Anchor run metadata is missing: $ANCHOR_METADATA" >&2
  exit 1
}
"$PY" - "$ANCHOR_METADATA" "$SEED" "$M_T4" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed = int(sys.argv[2])
budget = int(sys.argv[3])
metadata = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "variant": metadata.get("variant") == "B3S",
    "seed": metadata.get("seed") == seed,
    "side_group": (metadata.get("side_features") or {}).get("group") == "t4",
    "t4_budget": (metadata.get("side_features") or {}).get("pool_size") == budget,
    "activity_budget": (metadata.get("training") or {}).get("calibration_n_trials") == 30,
    "fixed_epochs": (metadata.get("training") or {}).get("max_epochs") == 12,
    "no_early_stopping": (metadata.get("training") or {}).get("no_early_stopping") is True,
    "completed": metadata.get("status") == "completed",
    "formal_test_unopened": metadata.get("held_out_test_evaluated") is False,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"{path}: invalid ordinary-T4 anchor metadata fields: {failed}")
PY

mkdir -p "$LOGS"
{
  echo "[$(date -Is)] START arm=$ARM seed=$SEED gpu=$GPU"
  echo "protocol=M_activity=$M_ACTIVITY; M_T4=$M_T4; evaluation=trials[$EVAL_START:]; strict 27/6; no formal test"
  echo "anchor=$ANCHOR"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL arm=$ARM seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE arm=$ARM seed=$SEED"
} >"$LOG" 2>&1
