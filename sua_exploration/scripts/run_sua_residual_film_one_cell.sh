#!/usr/bin/env bash
# Frozen one-cell runner for the residual-only confidence-FiLM fallback.
#
# This intentionally does not launch ordinary-T4 continuation or full-FiLM
# references: those are pre-existing, same-anchor validation artifacts.  The
# only executable arms are the three new, 1,208-parameter frozen-head arms.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_confidence_film_v1}"
ARM="${ARM:?ARM is required}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
ANCHOR="${ANCHOR:?ANCHOR is required}"
MODE="${1:---dry-run}"

M_ACTIVITY=30
M_T4=50
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
  residual_film) VARIANT="B3SCFR"; SIDE_FEATURES="t4cf_residual" ;;
  residual_shuffle) VARIANT="B3SCFRS"; SIDE_FEATURES="t4cf_residual_shuffled" ;;
  residual_nofilm) VARIANT="B3SCFRA"; SIDE_FEATURES="t4cf_residual" ;;
  *) echo "ARM must be residual_film, residual_shuffle, or residual_nofilm" >&2; exit 2 ;;
esac
case "$SEED" in 42|43|44) ;; *) echo "SEED must be one of 42,43,44" >&2; exit 2 ;; esac
if [[ "$MODE" != "--dry-run" && "$MODE" != "--launch" ]]; then
  echo "Usage: ARM=<residual arm> SEED=<42|43|44> GPU=<index> ANCHOR=<epoch_011.ckpt> $0 [--dry-run|--launch]" >&2
  exit 2
fi

NAME="${SCREEN_ID}_${ARM}_m50_dandi688_co_s${SEED}"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${ARM}_m50_s${SEED}.json"
LOG="$LOGS/${ARM}_m50_s${SEED}.log"

# Do every provenance check before the dry-run return.  Thus dry-run is a
# useful launch preflight rather than merely a pretty-printer.
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "Missing teacher checkpoint: $TEACHER" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing strict manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$ANCHOR" ]] || { echo "Missing ordinary-T4 anchor: $ANCHOR" >&2; exit 1; }
[[ "$(basename "$ANCHOR")" == "epoch_011.ckpt" ]] || {
  echo "Anchor must be the predeclared final epoch_011.ckpt: $ANCHOR" >&2; exit 1;
}
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to reuse run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || { echo "Refusing to overwrite result: $RESULT" >&2; exit 1; }
[[ ! -e "$LOG" ]] || { echo "Refusing to overwrite log: $LOG" >&2; exit 1; }

"$PY" - "$ROOT" "$ANCHOR" "$SEED" "$TEACHER" "$MANIFEST" "$RESULTS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, anchor_text, seed_text, teacher_text, manifest_text, results_text = sys.argv[1:]
anchor = Path(anchor_text).resolve()
seed = int(seed_text)
teacher = Path(teacher_text).resolve()
manifest = Path(manifest_text).resolve()
results = Path(results_text).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

anchor_run = anchor.parent.parent
metadata_path = anchor_run / "run_metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"anchor metadata missing: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
side = metadata.get("side_features") or {}
training = metadata.get("training") or {}
checks = {
    "anchor variant B3S": metadata.get("variant") == "B3S",
    "anchor seed": metadata.get("seed") == seed,
    "anchor T4 group": side.get("group") == "t4",
    "anchor T4 version": side.get("feature_version") == 1,
    "anchor T4 pool": side.get("pool_size") == 50,
    "anchor activity pool": training.get("calibration_n_trials") == 30,
    "anchor epochs": training.get("max_epochs") == 12,
    "anchor no early stopping": training.get("no_early_stopping") is True,
    "anchor every epoch": training.get("checkpoint_every_epoch") is True,
    "anchor completed": metadata.get("status") == "completed",
    "anchor formal seal": metadata.get("held_out_test_evaluated") is False,
    "anchor teacher": metadata.get("teacher_sha256") == sha256(teacher),
    "anchor manifest": metadata.get("train_val_manifest_sha256") == sha256(manifest),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"invalid selected-T4 anchor metadata: {failed}")
anchor_sha = sha256(anchor)

sys.path.insert(0, str(Path(root) / "sua_exploration" / "scripts"))
from aggregate_sua_residual_film import EXPECTED_VAL_SESSIONS, validate_arm

# The fallback does not create references.  Requiring their existing artifacts
# now prevents a later residual score from being interpreted against a different
# selected anchor or a silently changed validation protocol.
reference_shared = {}
for arm, variant, group, version in (
    ("t4_continuation", "B3S", "t4", 1),
    ("film", "B3SCF", "t4cf", 2),
):
    result = results / f"{arm}_m50_s{seed}.json"
    if not result.is_file():
        raise SystemExit(f"required existing reference is missing: {result}")
    payload = json.loads(result.read_text(encoding="utf-8"))
    run_metadata_path = Path(payload.get("run_metadata_path", ""))
    if not run_metadata_path.is_file():
        raise SystemExit(f"{result}: reference run metadata is missing")
    reference = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    reference_side = reference.get("side_features") or {}
    reference_checks = {
        "variant": payload.get("variant") == variant and reference.get("variant") == variant,
        "seed": payload.get("seed") == seed and reference.get("seed") == seed,
        "feature group": reference_side.get("group") == group,
        "feature version": reference_side.get("feature_version") == version,
        "T4 pool": reference_side.get("pool_size") == 50,
        "same anchor": reference.get("encoder_warmstart_sha256") == anchor_sha,
        "same teacher": reference.get("teacher_sha256") == metadata.get("teacher_sha256"),
        "same manifest": (
            reference.get("train_val_manifest_sha256")
            == metadata.get("train_val_manifest_sha256")
        ),
        "formal seal": reference.get("held_out_test_evaluated") is False,
    }
    bad = [name for name, passed in reference_checks.items() if not passed]
    if bad:
        raise SystemExit(f"{result}: invalid same-anchor reference fields: {bad}")
    # Reuse the same strict validator that will later consume the completed
    # round. This binds the reference score to exact checkpoints, sessions,
    # teacher/manifest bytes, normalization, and formal isolation before a new
    # residual arm is allowed to consume GPU time.
    validate_arm(
        result,
        arm,
        seed,
        expected_sessions=EXPECTED_VAL_SESSIONS,
        shared=reference_shared,
    )
PY

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
  --freeze_decoder
  --freeze_encoder_base
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
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${TRAIN_COMMAND[@]}"; echo
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${EVAL_COMMAND[@]}"; echo
  exit 0
fi

mkdir -p "$LOGS"
{
  echo "[$(date -Is)] START arm=$ARM seed=$SEED gpu=$GPU"
  echo "protocol=M_activity=30;M_T4=50;eval_trials=[50:];epochs=12;window=5..12;strict=27/6/6;formal=unopened"
  echo "anchor=$ANCHOR"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL arm=$ARM seed=$SEED"
  [[ ! -e "$RESULT" ]] || { echo "Refusing to overwrite result before evaluation: $RESULT" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE arm=$ARM seed=$SEED"
} >"$LOG" 2>&1
