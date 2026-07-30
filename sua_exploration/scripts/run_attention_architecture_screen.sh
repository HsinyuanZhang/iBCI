#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_attention_architecture_screen.sh [--dry-run|--launch] [--wait] [--screen-id ID]

Runs the preregistered 24-hour B15 architecture screen on two GPUs. The screen
never reads a formal held-out session: SUA uses DANDI 000688 sub-C train/val
only and MUA uses FALCON M2 internal LOSO cells only.
EOF
}

MODE="dry-run"
WAIT_FOR_WORKERS=false
SCREEN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --launch) MODE="launch"; shift ;;
    --wait) WAIT_FOR_WORKERS=true; shift ;;
    --screen-id) SCREEN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="${SCREEN_ID:-attention_screen_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS_DIR/logs"
SUA_DATA="$ROOT_DIR/sua_exploration/data/dandi_000688/sub-C"
SUA_CACHE="$ROOT_DIR/sua_exploration/cache/dandi688_subc_co_v1"
SUA_TEACHER="$ROOT_DIR/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MUA_ROOT="$ROOT_DIR/streaming_calibration_exp"

require_file() {
  [[ -e "$1" ]] || { echo "Missing required asset: $1" >&2; exit 1; }
}

for asset in "$PYTHON_BIN" "$SUA_DATA" "$SUA_TEACHER"; do
  require_file "$asset"
done
run_sua() {
  local gpu="$1" variant="$2" seed="$3"
  local lower_variant="${variant,,}"
  local run_name="${SCREEN_ID}_${lower_variant}_dandi688_co_s${seed}"
  local checkpoint_dir="$ROOT_DIR/sua_exploration/checkpoints/$run_name"
  local train_result="$ROOT_DIR/sua_exploration/results/p3_${run_name}_seed${seed}.json"
  local fixed_result="$RESULTS_DIR/sua_${lower_variant}_s${seed}.json"
  local log_file="$LOG_DIR/sua_${lower_variant}_s${seed}.log"

  if [[ -e "$checkpoint_dir" || -e "$train_result" || -e "$fixed_result" ]]; then
    echo "Refusing to overwrite existing SUA screen artifact for $run_name" >&2
    return 1
  fi
  local train_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$SUA_TEACHER"
    --variant "$variant"
    --out_name "$run_name"
    --data_dir "$SUA_DATA"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --max_epochs 20
    --patience 5
    --seed "$seed"
    --batch_size 32
    --num_workers 4
    --cache_dir "$SUA_CACHE"
    --loss_mode task_only
    --disable_progress_bar
    --require_gpu
  )
  local eval_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/select_gradient_free_protocol_dandi688.py"
    --ckpt ""
    --teacher_ckpt "$SUA_TEACHER"
    --variant "$variant"
    --data_dir "$SUA_DATA"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --cache_dir "$SUA_CACHE"
    --pool_size 50
    --fixed_selection_mode first
    --fixed_calibration_n 30
    --seed "$seed"
    --no_formal_lock
    --out_path "$fixed_result"
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu SUA $variant seed $seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start SUA variant=$variant seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${train_cmd[@]}"
    local checkpoint
    checkpoint="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$checkpoint_dir/run_metadata.json")"
    eval_cmd[3]="$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${eval_cmd[@]}"
    echo "[$(date -Is)] complete SUA variant=$variant seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

run_mua() {
  local gpu="$1" variant="$2" fold="$3" seed="$4"
  local lower_variant="${variant,,}"
  local experiment="${lower_variant}_m2_loso_internal"
  # The MUA artifact writer appends its own fold/seed/timestamp suffix.
  local run_name="${SCREEN_ID}_${lower_variant}_m2"
  local log_file="$LOG_DIR/mua_${lower_variant}_f${fold}_s${seed}.log"
  local command=(
    "$PYTHON_BIN" src/train.py
    "experiment=$experiment"
    "data.loso_fold=$fold"
    "seed=$seed"
    "run_id=$run_name"
    "data.include_heldout_in_fit=false"
    "data.include_heldout_in_test=false"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu MUA $variant fold $fold seed $seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start MUA variant=$variant fold=$fold seed=$seed gpu=$gpu"
    (
      cd "$MUA_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${command[@]}"
    )
    echo "[$(date -Is)] complete MUA variant=$variant fold=$fold seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

gpu0_queue() {
  run_sua 0 B3 42
  run_sua 0 B15P 42
  run_sua 0 B15D 42
  run_sua 0 B15 42
  run_mua 0 B3 1 42
  run_mua 0 B15P 1 42
  run_mua 0 B15D 1 42
  run_mua 0 B15 1 42
  run_mua 0 B3 2 42
  run_mua 0 B15P 2 42
}

gpu1_queue() {
  run_sua 1 B3 43
  run_sua 1 B15P 43
  run_sua 1 B15D 43
  run_sua 1 B15 43
  run_mua 1 B3 1 43
  run_mua 1 B15P 1 43
  run_mua 1 B15D 1 43
  run_mua 1 B15 1 43
  run_mua 1 B15D 2 42
  run_mua 1 B15 2 42
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
  echo "protocol=SUA sub-C 27/6/6 train+validation only; fixed first-30 validation evaluation; MUA M2 current-source internal LOSO only"
} >"$RESULTS_DIR/manifest.env"

gpu0_queue >"$LOG_DIR/worker_gpu0.log" 2>&1 &
pid0=$!
gpu1_queue >"$LOG_DIR/worker_gpu1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" >"$RESULTS_DIR/worker_gpu0.pid"
printf '%s\n' "$pid1" >"$RESULTS_DIR/worker_gpu1.pid"
echo "Launched screen $SCREEN_ID with workers $pid0 and $pid1"
echo "Monitor logs in $LOG_DIR and aggregate with:"
echo "  $PYTHON_BIN $ROOT_DIR/sua_exploration/scripts/aggregate_attention_architecture_screen.py --screen-id $SCREEN_ID"

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
