#!/usr/bin/env bash
# Validation-only, recoverable first pilot for Stage-0 same-electrode relation.
#
# Nothing runs unless --launch is supplied.  This runner intentionally schedules
# only the pre-registered core arms at one explicitly recorded seed:
#   t4                 B3S    / t4
#   relation           B3SER  / t4rel
#   membership_shuffle B3SER  / t4rel_membership_shuffled
#   no_group           B3SERN / t4rel_nogroup
#
# It is not a raw-waveform/SNR screen: all arms use the existing T4 pool=50
# values. Relative amplitude is reserved for Stage 2 after the relation gate.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="/home/xinyuan/miniconda3/envs/spint/bin/python"
DATA="$ROOT/sua_exploration/data/dandi_000688/sub-C"
CACHE="$ROOT/sua_exploration/cache/dandi688_subc_co_v1"
TEACHER="$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST="$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
SCREEN_ID="sua_electrode_relation_pilot_v1"
GPU=""
SEED=42
ARMS_CSV="t4,relation,membership_shuffle,no_group"
LAUNCH=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --launch --gpu GPU_ID [--seed SEED] [--arms CSV] [--screen-id TOKEN]

Runs exactly the validation-only core matrix T4 / relation / membership-shuffle /
parameter-matched-no-group at the requested seed. By default all four arms run;
--arms may select a comma-separated subset for a pre-registered multi-GPU
partition. Every job is a fresh run directory and is evaluated with the
deterministic epoch window 5..12.

Frozen contract:
  SUA; CO split=27,6,6; units<100; T4 pool=50; activity calibration_n=10;
  12 epochs; no early stopping; checkpoint every epoch; formal test=false.

No command is started without --launch.  Run one copy per requested GPU only after
the maintainer has scheduled the resource.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --launch) LAUNCH=1; shift ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --arms) ARMS_CSV="${2:-}"; shift 2 ;;
    --screen-id) SCREEN_ID="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
if [ "$LAUNCH" -ne 1 ] || [ -z "$GPU" ]; then
  echo "Refusing to launch: --launch and --gpu are both required." >&2
  usage >&2
  exit 2
fi
if ! [[ "$SCREEN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--screen-id must be a filesystem-safe token" >&2; exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "--seed must be a non-negative integer" >&2; exit 2
fi

RES="$ROOT/sua_exploration/results/$SCREEN_ID"
if [ -e "$RES" ]; then
  echo "Refusing to reuse existing result directory: $RES" >&2
  exit 2
fi
mkdir -p "$RES/logs"

declare -A VARIANT=(
  [t4]="B3S"
  [relation]="B3SER"
  [membership_shuffle]="B3SER"
  [no_group]="B3SERN"
)
declare -A SIDE=(
  [t4]="t4"
  [relation]="t4rel"
  [membership_shuffle]="t4rel_membership_shuffled"
  [no_group]="t4rel_nogroup"
)
IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
if [ "${#ARMS[@]}" -eq 0 ]; then
  echo "--arms must select at least one arm" >&2; exit 2
fi
for arm in "${ARMS[@]}"; do
  if [ -z "${VARIANT[$arm]+x}" ]; then
    echo "Unknown --arms entry: $arm" >&2; exit 2
  fi
done

for arm in "${ARMS[@]}"; do
  name="${SCREEN_ID}_${arm}_dandi688_co_s${SEED}"
  log="$RES/logs/${arm}_s${SEED}.log"
  echo "[$(date --iso-8601=seconds)] start $arm" | tee "$log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" \
    --teacher_ckpt "$TEACHER" --variant "${VARIANT[$arm]}" --side_features "${SIDE[$arm]}" \
    --side_feature_pool_size 50 --out_name "$name" --data_dir "$DATA" --cache_dir "$CACHE" \
    --train_val_manifest "$MANIFEST" \
    --signal_view sua --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
    --max_epochs 12 --no_early_stopping --checkpoint_every_epoch \
    --lr 1e-4 --batch_size 32 --num_workers 4 --seed "$SEED" \
    --loss_mode task_only --identity_mode calibrated --require_gpu --disable_progress_bar >>"$log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/sua_exploration/scripts/eval_epoch_window_generic_dandi688.py" \
    --run_dir "$ROOT/sua_exploration/checkpoints/$name" --teacher_ckpt "$TEACHER" \
    --data_dir "$DATA" --cache_dir "$CACHE" --total_epochs 12 --burn_in 4 \
    --train_val_manifest "$MANIFEST" \
    --out_path "$RES/${arm}_s${SEED}.json" >>"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] done $arm" | tee -a "$log"
done

if [ "${#ARMS[@]}" -eq 4 ] \
  && [[ " ${ARMS[*]} " == *" t4 "* ]] \
  && [[ " ${ARMS[*]} " == *" relation "* ]] \
  && [[ " ${ARMS[*]} " == *" membership_shuffle "* ]] \
  && [[ " ${ARMS[*]} " == *" no_group "* ]]; then
  "$PY" -u "$ROOT/sua_exploration/scripts/aggregate_sua_electrode_relation_pilot.py" \
    --seed "$SEED" \
    --t4 "$RES/t4_s${SEED}.json" \
    --relation "$RES/relation_s${SEED}.json" \
    --membership-shuffle "$RES/membership_shuffle_s${SEED}.json" \
    --no-group "$RES/no_group_s${SEED}.json" \
    --out "$RES/seed${SEED}_strict_aggregate.json"
fi

printf '%s\n' \
  "Validation-only completion for seed $SEED arms $ARMS_CSV. Do not run a formal test from these artifacts." \
  > "$RES/README.txt"
