#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_fixed_slot_router_sharp_pilot.sh [--dry-run|--launch] [--wait] [--pilot-id ID]

Runs the validation-only K=32 low-temperature fixed-slot router pilot on two GPUs.
It keeps the DANDI 000688 formal held-out test sessions completely unused.
EOF
}

mode="dry-run"
wait_for_workers=false
pilot_id=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) mode="dry-run"; shift ;;
    --launch) mode="launch"; shift ;;
    --wait) wait_for_workers=true; shift ;;
    --pilot-id) pilot_id="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
pilot_id="${pilot_id:-fixed_slot_router_sharp_$(date +%Y%m%d_%H%M%S)}"
results_dir="$root_dir/sua_exploration/results/$pilot_id"
log_dir="$results_dir/logs"
data_dir="$root_dir/sua_exploration/data/dandi_000688/sub-C"
cache_dir="$root_dir/sua_exploration/cache/dandi688_subc_co_v1"
teacher_ckpt="$root_dir/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

require_asset() {
  [[ -e "$1" ]] || { echo "Missing required asset: $1" >&2; exit 1; }
}

for asset in "$python_bin" "$data_dir" "$teacher_ckpt"; do
  require_asset "$asset"
done

run_case() {
  local gpu="$1" seed="$2"
  local run_name="${pilot_id}_b3_fsrk32_soft_t010_s${seed}"
  local checkpoint_dir="$root_dir/sua_exploration/checkpoints/$run_name"
  local train_result="$root_dir/sua_exploration/results/p3_${run_name}_seed${seed}.json"
  local eval_result="$results_dir/fsr_k32_soft_t010_s${seed}.json"
  local diagnostic_result="$results_dir/router_diagnostic_k32_soft_t010_s${seed}.json"
  local log_file="$log_dir/fsr_k32_soft_t010_s${seed}.log"
  if [[ -e "$checkpoint_dir" || -e "$train_result" || -e "$eval_result" ]]; then
    echo "Refusing to overwrite existing sharp-router artifact for $run_name" >&2
    return 1
  fi

  local train_command=(
    "$python_bin" "$root_dir/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$teacher_ckpt"
    --variant B3
    --fixed_slot_count 32
    --fixed_slot_dim 32
    --fixed_slot_mode soft
    --fixed_slot_fusion film
    --fixed_slot_temperature 0.1
    --out_name "$run_name"
    --data_dir "$data_dir"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --max_epochs 20
    --patience 5
    --seed "$seed"
    --batch_size 32
    --num_workers 4
    --cache_dir "$cache_dir"
    --loss_mode task_only
    --disable_progress_bar
    --require_gpu
  )
  local eval_command=(
    "$python_bin" "$root_dir/sua_exploration/scripts/select_gradient_free_protocol_dandi688.py"
    --ckpt ""
    --teacher_ckpt "$teacher_ckpt"
    --variant B3
    --data_dir "$data_dir"
    --task CO
    --split_counts 27,6,6
    --max_units_exclusive 100
    --cache_dir "$cache_dir"
    --pool_size 50
    --fixed_selection_mode first
    --fixed_calibration_n 30
    --seed "$seed"
    --no_formal_lock
    --out_path "$eval_result"
  )
  if [[ "$mode" == "dry-run" ]]; then
    echo "GPU $gpu sharp fixed-slot K=32 temperature=0.1 seed=$seed"
    printf '  %q ' CUDA_VISIBLE_DEVICES="$gpu" "${train_command[@]}"; echo
    return 0
  fi
  {
    echo "[$(date -Is)] start sharp fixed-slot K=32 temperature=0.1 seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${pilot_id}_mpl_${gpu}" "${train_command[@]}"
    local checkpoint
    checkpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best_checkpoint"])' "$checkpoint_dir/run_metadata.json")"
    eval_command[3]="$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${pilot_id}_mpl_${gpu}" "${eval_command[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR="/tmp/${pilot_id}_mpl_${gpu}" "$python_bin" \
      "$root_dir/sua_exploration/scripts/diagnose_fixed_slot_router.py" \
      --ckpt "$checkpoint" \
      --teacher_ckpt "$teacher_ckpt" \
      --variant B3 \
      --data_dir "$data_dir" \
      --task CO \
      --split_counts 27,6,6 \
      --max_units_exclusive 100 \
      --calibration_n 30 \
      --pool_size 50 \
      --selection_mode first \
      --seed "$seed" \
      --out_path "$diagnostic_result"
    echo "[$(date -Is)] complete sharp fixed-slot K=32 temperature=0.1 seed=$seed gpu=$gpu"
  } >"$log_file" 2>&1
}

if [[ "$mode" == "dry-run" ]]; then
  run_case 0 42
  run_case 1 43
  exit 0
fi

mkdir -p "$log_dir"
nvidia-smi -L >"$results_dir/gpu_inventory.txt"
"$python_bin" "$root_dir/sua_exploration/scripts/profile_fixed_slot_router_hardware.py" \
  --pilot-id "$pilot_id" --slots 32
{
  echo "pilot_id=$pilot_id"
  echo "started_at=$(date -Is)"
  echo "scope=validation-only DANDI 000688 sub-C CO; no formal held-out data"
  echo "protocol=first calibration n=30 from pool=50; evaluate trials[50:]"
  echo "architecture=B3 NeuronID, K32 soft FiLM, routing temperature=0.1"
} >"$results_dir/manifest.env"

run_case 0 42 >"$log_dir/worker_gpu0.log" 2>&1 &
worker0=$!
run_case 1 43 >"$log_dir/worker_gpu1.log" 2>&1 &
worker1=$!
printf '%s\n' "$worker0" >"$results_dir/worker_gpu0.pid"
printf '%s\n' "$worker1" >"$results_dir/worker_gpu1.pid"
echo "Launched sharp fixed-slot pilot $pilot_id with workers $worker0 and $worker1"

if [[ "$wait_for_workers" == true ]]; then
  set +e
  wait "$worker0"
  worker0_status=$?
  wait "$worker1"
  worker1_status=$?
  set -e
  if [[ "$worker0_status" -ne 0 || "$worker1_status" -ne 0 ]]; then
    echo "Sharp pilot workers failed: GPU0=$worker0_status GPU1=$worker1_status" >&2
    exit 1
  fi
  "$python_bin" "$root_dir/sua_exploration/scripts/aggregate_fixed_slot_router_sharp_pilot.py" --pilot-id "$pilot_id"
fi
