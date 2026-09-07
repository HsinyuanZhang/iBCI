#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
readonly PHASE1="$ROOT/pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json"
readonly RUN_ROOT="$ROOT/pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/gpu_runs"
readonly EVALUATOR="$ROOT/scripts/h1_carrierid_hc0_fivedate_evaluate.py"

run_date() {
  local gpu="$1" date="$2" pair run_dir
  pair="$ROOT/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_${date}_PAIR_CPU_PREFLIGHT_v1.json"
  run_dir="$RUN_ROOT/$date"
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; return 2; }
  [[ -r "$PHASE1" && -r "$pair" ]] || { echo "missing source receipt for $date" >&2; return 2; }
  if [[ -f "$run_dir/checkpoints/fixed_epoch50/epoch_049.ckpt" ]]; then
    echo "[$(date -Is)] H-C0 $date already complete"
    return 0
  fi
  [[ ! -e "$run_dir" ]] || { echo "incomplete run directory exists: $run_dir" >&2; return 3; }
  mkdir -p "$RUN_ROOT"
  echo "[$(date -Is)] starting H-C0 $date on physical GPU $gpu"
  # The workstation also has a user-site CUDA-13 PyTorch that is newer than
  # the installed driver.  Pin to the validated conda environment used by the
  # earlier H1 runs.
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/src/train.py" \
    experiment=h1_carrierid_date_lodo_hc0_phase2 \
    "phase2.outer_date=$date" \
    "phase2.phase1_preflight_path=$PHASE1" \
    "phase2.pair_preflight_path=$pair" \
    "hydra.run.dir=$run_dir" \
    "paths.root_dir=$ROOT" "paths.work_dir=$ROOT" "paths.data_dir=$ROOT/data" \
    ckpt_path=null test=false seed=42 \
    trainer.accelerator=gpu trainer.devices=1 trainer.precision=32-true \
    trainer.max_epochs=50 trainer.min_epochs=50
  [[ -f "$run_dir/checkpoints/fixed_epoch50/epoch_049.ckpt" ]] || {
    echo "H-C0 $date did not create the required e49 checkpoint" >&2
    return 4
  }
  echo "[$(date -Is)] completed H-C0 $date"
}

case "${1:-}" in
  --worker)
    shift
    gpu="$1"
    shift
    for date in "$@"; do
      run_date "$gpu" "$date"
    done
    ;;
  --watch-evaluate)
    for date in 19250108 19250113 19250115 19250119 19250120; do
      checkpoint="$RUN_ROOT/$date/checkpoints/fixed_epoch50/epoch_049.ckpt"
      while [[ ! -f "$checkpoint" ]]; do
        sleep 30
      done
    done
    while tmux has-session -t h1_hc0_gpu0 2>/dev/null || tmux has-session -t h1_hc0_gpu1 2>/dev/null; do
      sleep 30
    done
    for date in 19250108 19250113 19250115 19250119 19250120; do
      PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$EVALUATOR" --date "$date" --device cuda
    done
    PYTHONNOUSERSITE=1 "$PYTHON" "$EVALUATOR" --aggregate
    ;;
  --launch)
    tmux has-session -t h1_hc0_gpu0 2>/dev/null && { echo "tmux h1_hc0_gpu0 already exists" >&2; exit 5; }
    tmux has-session -t h1_hc0_gpu1 2>/dev/null && { echo "tmux h1_hc0_gpu1 already exists" >&2; exit 5; }
    mkdir -p "$ROOT/pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/logs"
    tmux new-session -d -s h1_hc0_gpu0 "$(printf '%q' "$0") --worker 0 19250108 19250115 19250120 2>&1 | tee '$ROOT/pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/logs/gpu0.log'"
    tmux new-session -d -s h1_hc0_gpu1 "$(printf '%q' "$0") --worker 1 19250113 19250119 2>&1 | tee '$ROOT/pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/logs/gpu1.log'"
    tmux new-session -d -s h1_hc0_eval "$(printf '%q' "$0") --watch-evaluate 2>&1 | tee '$ROOT/pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/logs/evaluate.log'"
    echo "launched h1_hc0_gpu0, h1_hc0_gpu1, and h1_hc0_eval"
    ;;
  *)
    echo "usage: $0 --launch | --worker GPU DATE... | --watch-evaluate" >&2
    exit 2
    ;;
esac
