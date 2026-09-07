#!/usr/bin/env bash
set -euo pipefail

# Guarded launcher for the isolated M1 Version-B pilot. It prints the exact
# command unless --run is supplied; this prevents an accidental GPU launch
# before the CPU receipt is reviewed. Run from any directory.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_m1_version_b_5070ti.sh {hs|c0|c} [--run] [extra Hydra overrides ...]

Without --run the resolved command is printed and no process is started.
The three aliases map to H-S/B0, B-C0/B3S+zero4, and B-C/B3S+full.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  hs) EXPERIMENT="m1_version_b_hs_continuation" ;;
  c0) EXPERIMENT="m1_version_b_c0" ;;
  c) EXPERIMENT="m1_version_b_c" ;;
  *) echo "unknown arm: $1" >&2; usage >&2; exit 2 ;;
esac
shift

RUN=0
if [[ "${1:-}" == "--run" ]]; then
  RUN=1
  shift
fi

cd "$REPO_ROOT"
CMD=(
  "$PYTHON_BIN" -u streaming_calibration_exp/src/train.py
  "experiment=${EXPERIMENT}"
  "paths.root_dir=${REPO_ROOT}"
  "data.loso_fold=0"
  "data.source_session_names=[ses-20120926,ses-20120927,ses-20120928]"
  "seed=42"
  "trainer.accelerator=gpu"
  "trainer.devices=1"
  "train=true"
  "test=true"
  "require_baseline_validation=false"
  "data.include_heldout_in_fit=false"
  "data.include_heldout_in_test=false"
  "data.query_start_trial=0"
  "data.random_calibration=false"
  "data.heldin_query_start_trial=10"
  "data.heldin_query_end_trial=210"
  "data.validation_protocol=loso"
  "data.calibration_n_trials=10"
  "trainer.min_epochs=12"
  "trainer.max_epochs=12"
)
CMD+=("$@")

printf 'REPO_ROOT=%q\nCUDA_VISIBLE_DEVICES=%q\n' "$REPO_ROOT" "$VISIBLE_GPU"
printf 'COMMAND='
printf '%q ' env "CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}" "${CMD[@]}"
printf '\n'

if [[ "$RUN" -ne 1 ]]; then
  echo 'dry-run: pass --run after reviewing the CPU preflight receipt' >&2
  exit 0
fi

exec env "CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}" "${CMD[@]}"
