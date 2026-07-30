#!/usr/bin/env bash
# F3 electrode-embedding ablation (UNIT_SIDE_FEATURE_ABLATION.md section 6, stage 2).
#   groups = {F3, FS3} x seeds (from --seeds)  -- F1/F2 are NOT re-run (v2 already has them).
#     F3  -> variant B3S, --side_features f3   (F2 waveform scalars + learned electrode embed)
#     FS3 -> variant B3S, --side_features fs3  (F3 with electrode ids permuted along units)
#   F0 baseline for aggregation is reused from results/e3_tuning_ablation/f0_s{seed}.json.
#
# M2/M3 budget matches E3 (run_e3_tuning_ablation.sh): --no_early_stopping,
# --checkpoint_every_epoch, scored by eval_epoch_window_generic_dandi688.py.
#
# Validation-only. Never loads test-session spikes/behavior/trials.
set -uo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
DOC="sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md"

usage() {
  cat <<EOF
Usage: $(basename "$0") --max_epochs N --seeds S1,S2,... [--burn_in B]

Required:
  --max_epochs N       Training epoch budget (same as E3).
  --seeds S1,S2,...    Comma-separated seeds, e.g. 42,43,44 (same as E3).

Optional:
  --burn_in B          Scored window is epochs (B+1)..max_epochs. Default: 4.
  -h, --help           Show this message and exit.
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
  echo "ERROR: --max_epochs and --seeds are required." >&2
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
  echo "ERROR: --seeds must be a comma-separated list of integers, got: $SEEDS" >&2
  exit 1
fi
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

RES="$ROOT/sua_exploration/results/electrode_ablation_f3"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RES/logs"

declare -A GROUP=( [f3]="B3S f3" [fs3]="B3S fs3" )
GROUPS_ORDER=(f3 fs3)

echo "[$(date +%H:%M:%S)] F3 electrode ablation: max_epochs=$MAX_EPOCHS burn_in=$BURN_IN seeds=${SEED_ARR[*]} groups=${GROUPS_ORDER[*]} ($DOC)"
echo "[$(date +%H:%M:%S)] Total runs: $(( ${#GROUPS_ORDER[@]} * ${#SEED_ARR[@]} )) (F0 reused from e3_tuning_ablation)"

run_one() {
  local gpu="$1" group="$2" seed="$3"
  read -r variant side <<<"${GROUP[$group]}"
  local name="electrode_ablation_f3_${group}_dandi688_co_s${seed}"
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
