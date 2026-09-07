#!/usr/bin/env bash
# Fixed-e49 launcher/evaluator for the predeclared H1 activity-only H-S width curve.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
readonly ARTIFACT_ROOT="$ROOT/pilot_artifacts/h1_spint_width_fold0"
readonly PREFLIGHT="$ARTIFACT_ROOT/H1_SPINT_IDENTITY_WIDTH_FOLD0_CPU_PREFLIGHT_v2.json"
readonly RUN_ROOT="$ARTIFACT_ROOT/gpu_runs_v1"
readonly RESULT="$ARTIFACT_ROOT/H1_SPINT_IDENTITY_WIDTH_FOLD0_TERMINAL_EVALUATION_v1.json"
readonly EVALUATOR="$ROOT/scripts/h1_spint_width_fold0_evaluate.py"

fail() { echo "[$(date -Is)] ERROR: $*" >&2; exit 2; }

verify_preflight() {
  [[ -x "$PYTHON" ]] || fail "validated Python is absent: $PYTHON"
  [[ -f "$PREFLIGHT" ]] || fail "required CPU preflight is missing: $PREFLIGHT"
  [[ "$(stat -c '%a' "$PREFLIGHT")" == "444" ]] || fail "CPU preflight must be immutable mode 0444"
  PYTHONNOUSERSITE=1 "$PYTHON" - "$PREFLIGHT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
x = json.loads(p.read_text())
assert x["schema"] == "h1_spint_identity_width_fold0_cpu_preflight_v2"
assert x["status"] == "PASS_H1_SPINT_IDENTITY_WIDTH_FOLD0_SOURCE_ONLY_CPU_PREFLIGHT"
assert x["scope"]["target_recordings_opened"] == 0 and x["scope"]["cuda_initialized"] is False
assert x["predeclared"]["arms"] == ["H-S-1024", "H-S-W224", "H-S-W32"]
assert x["predeclared"]["widths"] == [1024, 224, 32]
assert x["predeclared"]["noninferiority_margin_r2"] == 0.03
assert x["architecture"]["h1024_state_and_forward_bit_identical_to_standard_spint"] is True
assert x["architecture"]["carrier"] == "absent from dataset batches, model signature, parameter names, and model call boundary"
PY
}

experiment_for_arm() {
  case "$1" in
    H-S-1024) echo h1_spint_width_fold0 ;;
    H-S-W224) echo h1_spint_width_fold0_w224 ;;
    H-S-W32) echo h1_spint_width_fold0_w32 ;;
    *) fail "unknown predeclared arm: $1" ;;
  esac
}

directory_for_arm() {
  case "$1" in
    H-S-1024) echo hs1024 ;;
    H-S-W224) echo w224 ;;
    H-S-W32) echo w32 ;;
    *) fail "unknown predeclared arm: $1" ;;
  esac
}

run_arm() {
  local physical_gpu="$1" arm="$2" experiment dir run checkpoint
  verify_preflight
  experiment="$(experiment_for_arm "$arm")"
  dir="$(directory_for_arm "$arm")"
  run="$RUN_ROOT/$dir"
  checkpoint="$run/checkpoints/fixed_epoch50/epoch_049.ckpt"
  [[ ! -e "$run" ]] || fail "$arm run path already exists; refusing resume/overwrite: $run"
  mkdir -p "$RUN_ROOT"
  echo "[$(date -Is)] starting fresh $arm on physical GPU $physical_gpu"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" "$ROOT/src/train.py" \
    "experiment=$experiment" \
    "hydra.run.dir=$run" \
    "paths.root_dir=$ROOT" "paths.work_dir=$ROOT" "paths.data_dir=$ROOT/data" \
    ckpt_path=null test=false seed=42 \
    trainer.accelerator=gpu trainer.devices=1 trainer.precision=32-true \
    trainer.max_epochs=50 trainer.min_epochs=50
  [[ -f "$checkpoint" ]] || fail "$arm ended without its required fixed e49 checkpoint"
  echo "[$(date -Is)] completed $arm fixed e49"
}

evaluate_once() {
  verify_preflight
  [[ ! -e "$RESULT" ]] || fail "terminal target receipt already exists; never re-evaluate: $RESULT"
  local hs="$RUN_ROOT/hs1024" w224="$RUN_ROOT/w224" w32="$RUN_ROOT/w32"
  for run in "$hs" "$w224" "$w32"; do
    [[ -f "$run/checkpoints/fixed_epoch50/epoch_049.ckpt" ]] || fail "missing fixed e49 checkpoint under $run"
    [[ -f "$run/.hydra/config.yaml" ]] || fail "missing resolved config under $run"
  done
  echo "[$(date -Is)] all three checkpoints bound; starting the one strict fold0 target evaluation"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$EVALUATOR" \
    --data-dir "$ROOT/data/000954" \
    --hs1024-checkpoint "$hs/checkpoints/fixed_epoch50/epoch_049.ckpt" \
    --w224-checkpoint "$w224/checkpoints/fixed_epoch50/epoch_049.ckpt" \
    --w32-checkpoint "$w32/checkpoints/fixed_epoch50/epoch_049.ckpt" \
    --hs1024-config "$hs/.hydra/config.yaml" \
    --w224-config "$w224/.hydra/config.yaml" \
    --w32-config "$w32/.hydra/config.yaml" \
    --output "$RESULT" --device cuda
  [[ -f "$RESULT" && "$(stat -c '%a' "$RESULT")" == "444" ]] || fail "terminal evaluator did not publish immutable receipt"
}

after_w224_then_finish_curve() {
  # This watcher never inspects target data.  It starts the already
  # predeclared W32 arm only after W224 has exited successfully, then invokes
  # the single evaluator after all three fixed e49 files exist.
  while tmux has-session -t h1_spint_width_w224 2>/dev/null; do
    sleep 30
  done
  [[ -f "$RUN_ROOT/w224/checkpoints/fixed_epoch50/epoch_049.ckpt" ]] || fail "W224 ended without a fixed e49 checkpoint; W32/evaluation will not start"
  run_arm 1 H-S-W32
  evaluate_once
}

case "${1:-}" in
  --worker)
    [[ $# -eq 3 ]] || fail "usage: $0 --worker PHYSICAL_GPU ARM"
    run_arm "$2" "$3"
    ;;
  --evaluate)
    evaluate_once
    ;;
  --after-w224)
    after_w224_then_finish_curve
    ;;
  --launch)
    verify_preflight
    for name in h1_spint_width_hs1024 h1_spint_width_w224 h1_spint_width_w32; do
      tmux has-session -t "$name" 2>/dev/null && fail "tmux session already exists: $name"
    done
    mkdir -p "$ARTIFACT_ROOT/logs"
    # 224 and 32 run sequentially on GPU 1 to respect the two-idle-GPU budget.
    tmux new-session -d -s h1_spint_width_hs1024 "$(printf '%q' "$0") --worker 0 H-S-1024 2>&1 | tee '$ARTIFACT_ROOT/logs/hs1024_gpu0.log'"
    tmux new-session -d -s h1_spint_width_w224 "$(printf '%q' "$0") --worker 1 H-S-W224 2>&1 | tee '$ARTIFACT_ROOT/logs/w224_gpu1.log'"
    tmux new-session -d -s h1_spint_width_w32 "$(printf '%q' "$0") --after-w224 2>&1 | tee '$ARTIFACT_ROOT/logs/w32_and_evaluate.log'"
    echo "launched h1_spint_width_hs1024, h1_spint_width_w224, and a source-only W32/evaluation watcher"
    ;;
  *)
    echo "usage: $0 --launch | --worker PHYSICAL_GPU ARM | --evaluate | --after-w224" >&2
    exit 2
    ;;
esac
