#!/usr/bin/env bash
set -euo pipefail

# Preregistered 8-run side-feature ablation (F0/F1/F2/FS × seeds 42/43).
# Validation-only; never loads formal test sessions.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RESULTS_DIR="$ROOT_DIR/sua_exploration/results/side_feature_ablation_v1"
SUA_DATA="$ROOT_DIR/sua_exploration/data/dandi_000688/sub-C"
SUA_CACHE="$ROOT_DIR/sua_exploration/cache/dandi688_subc_co_v1"
SUA_TEACHER="$ROOT_DIR/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"

mkdir -p "$RESULTS_DIR/logs"

run_train_eval() {
  local group="$1" variant="$2" side="$3" seed="$4"
  local run_name="side_feature_ablation_v1_${group,,}_dandi688_co_s${seed}"
  local log_file="$RESULTS_DIR/logs/${group,,}_s${seed}.log"
  local train_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/train_variant_dandi688.py"
    --teacher_ckpt "$SUA_TEACHER"
    --variant "$variant"
    --side_features "$side"
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
  echo "[train] ${group} seed=${seed}" | tee "$log_file"
  "${train_cmd[@]}" 2>&1 | tee -a "$log_file"
  local best_ckpt
  best_ckpt="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
meta = json.loads(Path("$ROOT_DIR/sua_exploration/checkpoints/$run_name/run_metadata.json").read_text())
print(meta["best_checkpoint"])
PY
)"
  "$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/select_gradient_free_protocol_dandi688.py" \
    --ckpt "$best_ckpt" \
    --teacher_ckpt "$SUA_TEACHER" \
    --variant "$variant" \
    --data_dir "$SUA_DATA" \
    --task CO \
    --split_counts 27,6,6 \
    --max_units_exclusive 100 \
    --fixed_selection_mode first \
    --fixed_calibration_n 30 \
    --pool_size 50 \
    --seed "$seed" \
    --cache_dir "$SUA_CACHE" \
    --no_formal_lock \
    --out_path "$RESULTS_DIR/sua_${group,,}_s${seed}.json" \
    2>&1 | tee -a "$log_file"
}

for seed in 42 43; do
  run_train_eval F0 B3 none "$seed"
  run_train_eval F1 B3S f1 "$seed"
  run_train_eval F2 B3S f2 "$seed"
  run_train_eval FS B3S fs "$seed"
done

"$PYTHON_BIN" "$ROOT_DIR/sua_exploration/scripts/aggregate_side_feature_ablation.py" \
  --ablation_dir "$RESULTS_DIR"
