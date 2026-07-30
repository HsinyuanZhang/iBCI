#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_attention_architecture_screen_v4.sh [--dry-run|--launch] [--wait]

Runs the MEASUREMENT_PROTOCOL_V4 SUA attention-architecture screen
(attention_arch_screen_v4): variants B3/B15P/B15D/B15 x seeds 42/43/44 = 12 runs
on DANDI 000688 sub-C CO, 27/6/6 chronological session split. Never reads a
formal held-out test session (train+validation only); see
sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md section 6.

Each run is: train_variant_dandi688.py --max_epochs 12 --no_early_stopping
--checkpoint_every_epoch (M2/M3 fixed-epoch-budget + deterministic-checkpoint
mode), immediately followed by eval_epoch_window_dandi688.py on the resulting
checkpoint directory to score epochs 5-12 (M3 estimator; script is reused
unmodified). Both steps for one (variant, seed) share one log file.

Every non-protocol hyperparameter (lr, batch_size, num_workers, cache_dir,
data_dir, behavior scaling, freeze_decoder, identity_mode) is matched exactly
to the attention_arch_screen_v3 SUA run_metadata.json files (see the
"v3-matched hyperparameters" block below) so that v4-vs-v3 differences are
attributable only to the M1/M2/M3 measurement fixes, not to a changed
training recipe. --disable_progress_bar and --require_gpu are also passed;
neither affects training numerics (progress_bar only gates a Trainer/tqdm
display flag, require_gpu only raises if CUDA is unavailable) -- see the
handoff report for the evidence trail on both this and on num_workers, which
is not recorded in run_metadata.json and had to be reconstructed from
indirect evidence.

Refuses to overwrite an existing v4 result: if a run's epoch-window JSON
already exists, that run is skipped with a hard error (see the "refusing to
overwrite" guard below), and -- because this script uses `set -euo
pipefail` -- a crash or a refusal in one run aborts the REST of that GPU's
queue (the other GPU's queue is unaffected). This matches
run_attention_architecture_screen.sh's (v3) behavior: on failure this script
does not retry with different settings; it stops and the failure is visible
in the run's log file and in the worker_gpu{0,1}.log wrapper log.
EOF
}

MODE="dry-run"
WAIT_FOR_WORKERS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --launch) MODE="launch"; shift ;;
    --wait) WAIT_FOR_WORKERS=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="attention_arch_screen_v4"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS_DIR/logs"
SUA_DATA="$ROOT_DIR/sua_exploration/data/dandi_000688/sub-C"
SUA_CACHE="$ROOT_DIR/sua_exploration/cache/dandi688_subc_co_v1"
SUA_TEACHER="$ROOT_DIR/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

# --- v3-matched hyperparameters -----------------------------------------------
# Read from sua_exploration/checkpoints/attention_arch_screen_v3_*_dandi688_co_s4*/
# run_metadata.json (all 8 SUA artifacts agree on every one of these fields):
#   learning_rate=0.0001  batch_size=32  freeze_decoder=false  identity_mode=calibrated
#   (default, not passed)  loss_mode=task_only  data_dir=.../dandi_000688/sub-C
#   cache_dir=.../cache/dandi688_subc_co_v1  task=CO  split_counts=[27,6,6]
#   max_units_exclusive=100
# num_workers is NOT recorded in run_metadata.json (DataLoader construction arg,
# not logged). Reconstructed as 4 from two independent pieces of evidence: (1)
# run_attention_architecture_screen.sh (the v3 SUA launcher) passes
# --num_workers 4; (2) all 8 v3 SUA logs are silent on Lightning's
# PossibleUserWarning "The 'train_dataloader' does not have many workers", which
# Lightning's _worker_check fires whenever num_workers<2 and this machine's
# suggested_max_num_workers (32 CPUs, 1 device) is >1 -- so every v3 SUA run
# must have used num_workers>=2, consistent with 4 and inconsistent with the
# script's own --num_workers default of 0.
SUA_LR=0.0001
SUA_BATCH_SIZE=32
SUA_NUM_WORKERS=4

require_file() {
  [[ -e "$1" ]] || { echo "Missing required asset: $1" >&2; exit 1; }
}
for asset in "$PYTHON_BIN" "$SUA_DATA" "$SUA_TEACHER"; do
  require_file "$asset"
done

run_one() {
  local gpu="$1" variant="$2" seed="$3"
  local lower_variant="${variant,,}"
  local out_name="attention_arch_screen_v4_${lower_variant}_dandi688_co_s${seed}"
  local checkpoint_dir="$ROOT_DIR/sua_exploration/checkpoints/$out_name"
  local eval_result="$RESULTS_DIR/epoch_window_${lower_variant}_s${seed}.json"
  local log_file="$LOG_DIR/${variant}_s${seed}.log"

  if [[ -e "$eval_result" ]]; then
    echo "Refusing to overwrite existing v4 epoch-window result for $out_name: $eval_result" >&2
    return 1
  fi

  local train_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$SUA_TEACHER"
    --variant "$variant"
    --out_name "$out_name"
    --data_dir "$SUA_DATA"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --max_epochs 12
    --no_early_stopping
    --checkpoint_every_epoch
    --seed "$seed"
    --lr "$SUA_LR"
    --batch_size "$SUA_BATCH_SIZE"
    --num_workers "$SUA_NUM_WORKERS"
    --cache_dir "$SUA_CACHE"
    --loss_mode task_only
    --disable_progress_bar
    --require_gpu
  )
  local eval_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/eval_epoch_window_dandi688.py"
    --run_dir "$checkpoint_dir"
    --out_path "$eval_result"
  )

  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu $variant seed $seed -> $out_name"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}"; echo
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${eval_cmd[@]}"; echo
    return 0
  fi

  {
    echo "[$(date -Is)] start TRAIN variant=$variant seed=$seed gpu=$gpu out_name=$out_name"
    echo "[$(date -Is)] resolved checkpoint_dir=$checkpoint_dir"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${train_cmd[@]}"
    echo "[$(date -Is)] complete TRAIN variant=$variant seed=$seed gpu=$gpu"
    echo "[$(date -Is)] start EVAL variant=$variant seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${eval_cmd[@]}"
    echo "[$(date -Is)] complete EVAL variant=$variant seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

# 4 variants x 3 seeds = 12 runs, split 6/6 across the two GPUs. The first job in
# each queue (B3 seed 42 on GPU0, B3 seed 43 on GPU1) is deliberately the SAME
# variant with DIFFERENT seeds launched at the same instant on separate GPUs --
# this is the exact (variant, differing-seed, concurrent-launch) shape of the v3
# bug H.4 collision (two MUA seeds resolved to one timestamp-named Hydra dir).
# train_variant_dandi688.py's --out_name bakes in variant+seed with no timestamp
# component at all, and assert_run_dir_is_fresh() is a hard runtime backstop, so
# this pair is the sharpest available real-world test of the M1 fix.
gpu0_queue() {
  run_one 0 B3 42
  run_one 0 B15P 42
  run_one 0 B15D 42
  run_one 0 B15 42
  run_one 0 B3 44
  run_one 0 B15P 44
}

gpu1_queue() {
  run_one 1 B3 43
  run_one 1 B15D 43
  run_one 1 B15 43
  run_one 1 B15P 43
  run_one 1 B15D 44
  run_one 1 B15 44
}

if [[ "$MODE" == "dry-run" ]]; then
  gpu0_queue
  gpu1_queue
  exit 0
fi

mkdir -p "$LOG_DIR"
nvidia-smi -L >"$RESULTS_DIR/gpu_inventory.txt"
{
  echo "screen_id=$SCREEN_ID"
  echo "started_at=$(date -Is)"
  echo "protocol=MEASUREMENT_PROTOCOL_V4 SUA sub-C 27/6/6 train+validation only; fixed epoch window 5-12 mean; first/n=30/pool=50"
  echo "protocol_doc=sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md"
} >"$RESULTS_DIR/manifest.env"

gpu0_queue >"$LOG_DIR/worker_gpu0.log" 2>&1 &
pid0=$!
gpu1_queue >"$LOG_DIR/worker_gpu1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" >"$RESULTS_DIR/worker_gpu0.pid"
printf '%s\n' "$pid1" >"$RESULTS_DIR/worker_gpu1.pid"
echo "Launched screen $SCREEN_ID with workers $pid0 (gpu0) and $pid1 (gpu1)"
echo "Monitor logs in $LOG_DIR"
echo "Aggregate with: $PYTHON_BIN $ROOT_DIR/sua_exploration/scripts/aggregate_attention_architecture_screen_v4.py"

if [[ "$WAIT_FOR_WORKERS" == true ]]; then
  set +e
  wait "$pid0"
  worker0_status=$?
  wait "$pid1"
  worker1_status=$?
  set -e
  if [[ "$worker0_status" -ne 0 || "$worker1_status" -ne 0 ]]; then
    echo "Screen workers failed: GPU0=$worker0_status GPU1=$worker1_status" >&2
    exit 1
  fi
  echo "Screen $SCREEN_ID completed successfully"
fi
