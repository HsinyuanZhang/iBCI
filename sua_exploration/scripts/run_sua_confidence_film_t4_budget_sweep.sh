#!/usr/bin/env bash
# Stage-0 dry-run contract for selected-T4 confidence FiLM continuations.
#
# This script intentionally cannot launch jobs.  It emits commands only after
# checking the strict 27/6 manifest and protocol; a maintainer must review and
# replace this script's explicit --dry-run-only guard before any GPU scheduling.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-sua_t4_confidence_film_v1}"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
SEED=42
T4_BUDGETS=(10 15 20 30 50)

if [[ "${1:---dry-run}" != "--dry-run" ]]; then
  echo "This Stage-0 runner is dry-run-only pending maintainer review; GPU launch is intentionally disabled." >&2
  exit 2
fi
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing strict manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "Missing teacher checkpoint: $TEACHER" >&2; exit 1; }

for budget in "${T4_BUDGETS[@]}"; do
  "$PY" "$ROOT/sua_exploration/scripts/assert_confidence_film_protocol.py" --t4-budget "$budget" >/dev/null
  # Set this after selecting the ordinary T4 run for this same M_T4.  The
  # existing external M=50 run is named t4m50 under this screen; this script
  # never creates or overwrites that artifact.
  anchor_var="T4_SELECTED_M${budget}"
  anchor="${!anchor_var:-<SELECTED_T4_CHECKPOINT_FOR_M${budget}>}"
  for arm in t4_continuation film confidence_shuffle nofilm_match film_ts4; do
    case "$arm" in
      t4_continuation) variant=B3S; side=t4 ;;
      film) variant=B3SCF; side=t4cf ;;
      confidence_shuffle) variant=B3SCFS; side=t4cf_confidence_shuffled ;;
      nofilm_match) variant=B3SCFA; side=t4cf ;;
      film_ts4) variant=B3SCF; side=t4cf_ts4 ;;
    esac
    name="${SCREEN_ID}_${arm}_m${budget}_dandi688_co_s${SEED}"
    printf '%q ' "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
      --teacher_ckpt "$TEACHER" --variant "$variant" --side_features "$side" \
      --side_feature_pool_size "$budget" --calibration_n_trials 30 \
      --encoder_warmstart_path "$anchor" --out_name "$name" --data_dir "$DATA" \
      --cache_dir "$CACHE" --train_val_manifest "$MANIFEST" --signal_view sua --task CO \
      --split_counts 27,6,6 --max_units_exclusive 100 --max_epochs 12 --no_early_stopping \
      --checkpoint_every_epoch --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$SEED" \
      --loss_mode task_only --identity_mode calibrated --require_gpu --disable_progress_bar
    echo
    printf '%q ' "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
      --run_dir "$ROOT/sua_exploration/checkpoints/$name" --teacher_ckpt "$TEACHER" \
      --data_dir "$DATA" --cache_dir "$CACHE" --train_val_manifest "$MANIFEST" \
      --total_epochs 12 --burn_in 4 --calibration_n 30 --pool_size 50 \
      --out_path "$RESULTS/${arm}_m${budget}_s${SEED}.json"
    echo
  done
done

printf '%q ' "$PY" "$ROOT/sua_exploration/scripts/aggregate_sua_confidence_film_t4_budget.py" \
  --result-dir "$RESULTS" --out "$RESULTS/seed42_aggregate.json"
echo
