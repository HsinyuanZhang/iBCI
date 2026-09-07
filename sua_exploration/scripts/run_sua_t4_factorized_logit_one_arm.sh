#!/usr/bin/env bash
# One-arm runner for the frozen selected-T4 factorized logit-residual screen.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="sua_t4_factorized_logit_residual_v1"
ARM="${ARM:?ARM is required: aligned_logit|shuffled_logit|additive_control}"
GPU="${GPU:?GPU is required}"
MODE="${1:---dry-run}"
SEED=42
M_ACTIVITY=30
M_T4=50
EVAL_START=50
TOTAL_EPOCHS=12
BURN_IN=4
ANCHOR="$ROOT/sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
PROTOCOL="$ROOT/sua_exploration/docs/SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md"
PREFLIGHT="$ROOT/sua_exploration/results/$SCREEN/preflight.json"
RESULTS="$ROOT/sua_exploration/results/$SCREEN"
LOGS="$RESULTS/logs"

case "$ARM" in
  aligned_logit)
    RESIDUAL_MODE="aligned"
    INTERACTION="attention_logit"
    ;;
  shuffled_logit)
    RESIDUAL_MODE="shuffled"
    INTERACTION="attention_logit"
    ;;
  additive_control)
    RESIDUAL_MODE="aligned"
    INTERACTION="additive_control"
    ;;
  *)
    echo "unsupported ARM=$ARM" >&2
    exit 2
    ;;
esac
[[ "$MODE" == "--dry-run" || "$MODE" == "--launch" ]] || {
  echo "usage: ARM=<arm> GPU=<0|1> $0 [--dry-run|--launch]" >&2
  exit 2
}

NAME="${SCREEN}_${ARM}_dandi688_co_s42"
RUN_DIR="$ROOT/sua_exploration/checkpoints/$NAME"
RESULT="$RESULTS/${ARM}_s42.json"
LOG="$LOGS/${ARM}_s42.log"

TRAIN_COMMAND=(
  "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features t4
  --side_feature_pool_size "$M_T4"
  --calibration_n_trials "$M_ACTIVITY"
  --encoder_warmstart_path "$ANCHOR"
  --t4_logit_residual_mode "$RESIDUAL_MODE"
  --t4_logit_interaction_mode "$INTERACTION"
  --t4_logit_rank 8
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

[[ -x "$PY" ]] || { echo "missing Python: $PY" >&2; exit 1; }
for path in "$ANCHOR" "$TEACHER" "$MANIFEST" "$PROTOCOL" "$PREFLIGHT"; do
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 1; }
done
[[ -d "$DATA" ]] || { echo "missing data directory: $DATA" >&2; exit 1; }
[[ "$(sha256sum "$ANCHOR" | awk '{print $1}')" == "cf533e7cd97801d53985383b37f3fa1ff72fb6c385eaeb8c81273c3fa128273d" ]] || {
  echo "selected T4 anchor SHA drift" >&2; exit 1;
}
[[ "$(sha256sum "$TEACHER" | awk '{print $1}')" == "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d" ]] || {
  echo "teacher SHA drift" >&2; exit 1;
}
"$PY" - "$PREFLIGHT" "$PROTOCOL" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); protocol = Path(sys.argv[2])
a = json.loads(p.read_text())
if a.get("gate") != "pass" or a.get("no_dataset_opened") is not True or a.get("no_formal_test_opened") is not True:
    raise SystemExit("preflight gate is not a strict pass")
if a.get("protocol", {}).get("sha256") != hashlib.sha256(protocol.read_bytes()).hexdigest():
    raise SystemExit("preflight protocol SHA drift")
if any(not row.get("inherited_coupled_bit_equal") or not row.get("cached_on_the_fly_bit_equal")
       for row in a.get("zero_and_cache_checks", {}).values()):
    raise SystemExit("preflight exactness contract failed")
PY
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || { echo "refusing to overwrite result: $RESULT" >&2; exit 1; }

mkdir -p "$LOGS"
{
  echo "[$(date -Is)] START arm=$ARM gpu=$GPU seed=$SEED"
  echo "protocol_sha=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
  echo "preflight_sha=$(sha256sum "$PREFLIGHT" | awk '{print $1}')"
  echo "scope=strict 27/6 validation; formal test unopened; activity=30; T4=50; eval_start=50; epochs=5-12"
  CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_COMMAND[@]}"
  echo "[$(date -Is)] EVAL arm=$ARM"
  CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_COMMAND[@]}"
  echo "[$(date -Is)] DONE arm=$ARM"
} >"$LOG" 2>&1
