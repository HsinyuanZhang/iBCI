#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_pseudo_mua_attention_pilot.sh [--dry-run|--launch] [--wait] --screen-id ID

Runs a development-only paired pseudo-MUA bridge on DANDI 000688 sub-C CO.
Pseudo-MUA sums sorted-unit spike counts only within each NWB electrode id.
No formal held-out test sessions are read or evaluated.
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

[[ -n "$SCREEN_ID" ]] || { echo "--screen-id is required" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS_DIR/logs"
SUA_DATA="$ROOT_DIR/sua_exploration/data/dandi_000688/sub-C"
SUA_CACHE="$ROOT_DIR/sua_exploration/cache/dandi688_subc_co_pseudomua_v1"
SUA_TEACHER="$ROOT_DIR/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

require_file() {
  [[ -e "$1" ]] || { echo "Missing required asset: $1" >&2; exit 1; }
}

for asset in "$PYTHON_BIN" "$SUA_DATA" "$SUA_TEACHER"; do
  require_file "$asset"
done

run_sua() {
  local gpu="$1" variant="$2" seed="$3"
  local lower_variant="${variant,,}"
  local run_name="${SCREEN_ID}_${lower_variant}_pseudomua_co_s${seed}"
  local checkpoint_dir="$ROOT_DIR/sua_exploration/checkpoints/$run_name"
  local train_result="$ROOT_DIR/sua_exploration/results/p3_${run_name}_seed${seed}.json"
  local fixed_result="$RESULTS_DIR/pseudo_${lower_variant}_s${seed}.json"
  local log_file="$LOG_DIR/pseudo_${lower_variant}_s${seed}.log"
  if [[ -e "$checkpoint_dir" || -e "$train_result" || -e "$fixed_result" ]]; then
    echo "Refusing to overwrite existing pseudo-MUA artifact for $run_name" >&2
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
    --signal_view pseudo_mua
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
    --signal_view pseudo_mua
    --pool_size 50
    --fixed_selection_mode first
    --fixed_calibration_n 30
    --seed "$seed"
    --no_formal_lock
    --out_path "$fixed_result"
  )
  if [[ "$MODE" == "dry-run" ]]; then
    echo "GPU $gpu pseudo-MUA $variant seed $seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start pseudo-MUA variant=$variant seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${train_cmd[@]}"
    local checkpoint
    checkpoint="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$checkpoint_dir/run_metadata.json")"
    eval_cmd[3]="$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${SCREEN_ID}_mpl_${gpu}" "${eval_cmd[@]}"
    echo "[$(date -Is)] complete pseudo-MUA variant=$variant seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

gpu0_queue() {
  run_sua 0 B3 42
  run_sua 0 B15P 42
  run_sua 0 B15D 42
  run_sua 0 B15 42
}

gpu1_queue() {
  run_sua 1 B3 43
  run_sua 1 B15P 43
  run_sua 1 B15D 43
  run_sua 1 B15 43
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
  echo "protocol=DANDI 000688 sub-C CO pseudo-MUA; electrode-id spike-count pooling; 27/6/6 train+validation only; fixed first-30 validation evaluation"
} >"$RESULTS_DIR/manifest.env"

gpu0_queue >"$LOG_DIR/worker_gpu0.log" 2>&1 &
pid0=$!
gpu1_queue >"$LOG_DIR/worker_gpu1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" >"$RESULTS_DIR/worker_gpu0.pid"
printf '%s\n' "$pid1" >"$RESULTS_DIR/worker_gpu1.pid"
echo "Launched pseudo-MUA pilot $SCREEN_ID with workers $pid0 and $pid1"

if [[ "$WAIT_FOR_WORKERS" == true ]]; then
  set +e
  wait "$pid0"
  worker0_status=$?
  wait "$pid1"
  worker1_status=$?
  set -e
  if [[ "$worker0_status" -ne 0 || "$worker1_status" -ne 0 ]]; then
    echo "Pseudo-MUA workers failed: GPU0=$worker0_status GPU1=$worker1_status" >&2
    exit 1
  fi
  echo "Pseudo-MUA pilot $SCREEN_ID completed successfully"
fi
