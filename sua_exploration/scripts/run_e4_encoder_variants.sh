#!/usr/bin/env bash
# E4 encoder architecture variant screen under MEASUREMENT_PROTOCOL_V4
# (sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md sections 2, 2.0, 4).
#   groups = {B3, B3T, B3A} x seeds (from --seeds), all with --side_features none
#     B3  -> baseline EarlyPoolEncoder (control; no shuffled control needed for E4 --
#            architecture variants have no "content vs width" confound, charter section 4)
#     B3T -> TemporalBasisEarlyPoolEncoder (fixed raised-cosine temporal basis + small
#            learned projection; params -31%, MAC -65% vs B3, same support state)
#     B3A -> TrialAttentionEarlyPoolEncoder (learned attention over the M calibration
#            trials per unit; params/MAC ~= B3, but support state is 30x B3's -- it must
#            retain every trial's features, breaking B3's O(1) running-sum streaming
#            accumulation. See E3_E4_ENCODER_PROGRAM.md section 2.0.)
#   M2: --no_early_stopping (fixed --max_epochs budget, equal draws for every group)
#   M3: --checkpoint_every_epoch, scored by eval_epoch_window_generic_dandi688.py over the
#       window burn_in+1..max_epochs (NOT the frozen eval_epoch_window_dandi688.py, which
#       hardcodes max_epochs=12/window 5-12 -- E4's epoch budget comes from E2 and may not
#       be 12; that script is deliberately left unmodified, see section 0).
#
# Hyperparameters other than --variant/--max_epochs/--seed are copied verbatim from
# scripts/run_side_feature_ablation_v2.sh so that any E4-vs-v2 difference is attributable
# only to the encoder architecture change. --variant B3T/B3A requires
# train_variant_dandi688.py's --variant choices to include them (added alongside this
# script; the encoders themselves -- streaming_encoders.py -- were already implemented and
# unit-tested and are not modified here).
#
# Validation-only. Never loads test-session spikes/behavior/trials (train_variant_dandi688.py
# and eval_epoch_window_generic_dandi688.py only ever touch validation-session spike data;
# test sessions are only ever used for their names and NWB unit-table row counts).
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
DOC="sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md"

usage() {
  cat <<EOF
Usage: $(basename "$0") --max_epochs N --seeds S1,S2,... [--burn_in B]

Required (NO DEFAULTS -- see "Why this refuses to run" below):
  --max_epochs N       Training epoch budget for every group/seed run. Comes from E2's
                        convergence measurement.
  --seeds S1,S2,...    Comma-separated seed list, e.g. 42,43,44. Comes from E1's measured
                        sigma_seed (more seeds if SWA does not shrink sigma_seed enough).

Optional:
  --burn_in B          Epochs excluded from the trailing-average scoring window; the scored
                        window is epochs (B+1)..max_epochs. Default: 4 (the frozen M3
                        estimator's burn-in count, unaffected by the epoch-budget question).
  -h, --help            Show this message and exit.

Why this refuses to run without --max_epochs/--seeds:
  $DOC section 0 documents that this project has TWICE set a screen's gate below its own
  measured noise floor by guessing an epoch budget or seed count instead of measuring it
  first (attention_arch_screen_v3's +0.005 threshold against sigma_epoch=0.0388; then
  side_feature_ablation_v2's +0.03 threshold against a seed-count formula that omitted
  sigma_seed entirely). Section 0's rule is: do not launch the formal E3/E4 screen until E2
  has reported the convergence-derived epoch budget AND E1 has reported sigma_seed (which
  sets the seed count needed to resolve the gate). This script will not silently default
  either value -- pass them explicitly once both are known.
EOF
}

MAX_EPOCHS=""
SEEDS=""
BURN_IN=4

while [ $# -gt 0 ]; do
  case "$1" in
    --max_epochs) MAX_EPOCHS="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --burn_in) BURN_IN="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "$MAX_EPOCHS" ] || [ -z "$SEEDS" ]; then
  echo "ERROR: --max_epochs and --seeds are required and have no default." >&2
  echo "See $DOC section 0: neither the epoch budget nor the seed count may be guessed --" >&2
  echo "that is the exact failure mode that has twice forced a gate retraction in this" >&2
  echo "project. Supply --max_epochs from E2 and --seeds from E1 before launching this" >&2
  echo "screen." >&2
  echo >&2
  usage >&2
  exit 1
fi

if ! [[ "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCHS" -lt 1 ]; then
  echo "ERROR: --max_epochs must be a positive integer, got: $MAX_EPOCHS" >&2
  exit 1
fi
if ! [[ "$BURN_IN" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --burn_in must be a non-negative integer, got: $BURN_IN" >&2
  exit 1
fi
if [ "$BURN_IN" -ge "$MAX_EPOCHS" ]; then
  echo "ERROR: --burn_in ($BURN_IN) must be strictly less than --max_epochs ($MAX_EPOCHS)." >&2
  exit 1
fi
if ! [[ "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERROR: --seeds must be a comma-separated list of non-negative integers, got: $SEEDS" >&2
  exit 1
fi
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

RES="$ROOT/sua_exploration/results/e4_encoder_variants"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

# group -> "variant side_features" (side_features is always none for E4 -- architecture
# variants only, charter section 2)
declare -A GROUP=( [b3]="B3 none" [b3t]="B3T none" [b3a]="B3A none" )
GROUPS_ORDER=(b3 b3t b3a)

echo "[$(date +%H:%M:%S)] E4 encoder variants: max_epochs=$MAX_EPOCHS burn_in=$BURN_IN seeds=${SEED_ARR[*]} groups=${GROUPS_ORDER[*]}"
echo "[$(date +%H:%M:%S)] Total runs: $(( ${#GROUPS_ORDER[@]} * ${#SEED_ARR[@]} ))"

run_one() {
  local gpu="$1" group="$2" seed="$3"
  read -r variant side <<<"${GROUP[$group]}"
  local name="e4_encoder_variants_${group}_dandi688_co_s${seed}"
  local log="$RES/logs/${group}_s${seed}.log"

  echo "[$(date +%H:%M:%S)] GPU$gpu START $group seed=$seed ($variant/$side) max_epochs=$MAX_EPOCHS" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$TEACHER" --variant "$variant" --side_features "$side" \
    --out_name "$name" --data_dir "$DATA" --cache_dir "$CACHE" \
    --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
    --max_epochs "$MAX_EPOCHS" --no_early_stopping --checkpoint_every_epoch \
    --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$seed" \
    --loss_mode task_only --identity_mode calibrated \
    --require_gpu --disable_progress_bar >>"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] GPU$gpu TRAIN-FAIL $group s$seed rc=$rc" | tee -a "$log"
    return $rc
  fi

  echo "[$(date +%H:%M:%S)] GPU$gpu EVAL $group seed=$seed" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
    --run_dir "$ROOT/sua_exploration/checkpoints/$name" \
    --teacher_ckpt "$TEACHER" --data_dir "$DATA" --cache_dir "$CACHE" \
    --total_epochs "$MAX_EPOCHS" --burn_in "$BURN_IN" \
    --out_path "$RES/${group}_s${seed}.json" >>"$log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE $group s$seed eval_rc=$rc" | tee -a "$log"
  return $rc
}

# Build the (3 groups x len(seeds)) job queue, then drain it over 2 GPUs.
JOBS=()
for seed in "${SEED_ARR[@]}"; do for g in "${GROUPS_ORDER[@]}"; do JOBS+=("$g $seed"); done; done

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  read -r g0 s0 <<<"${JOBS[$i]}"; run_one 0 "$g0" "$s0" & p0=$!
  j=$((i+1)); p1=""
  if [ $j -lt ${#JOBS[@]} ]; then
    read -r g1 s1 <<<"${JOBS[$j]}"; run_one 1 "$g1" "$s1" & p1=$!
  fi
  wait $p0; [ -n "$p1" ] && wait $p1
  i=$((i+2))
done

echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS FINISHED"
ls -1 "$RES"/*.json 2>/dev/null | wc -l
