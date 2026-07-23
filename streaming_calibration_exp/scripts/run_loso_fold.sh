#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <experiment> <loso_fold> [extra hydra overrides...]" >&2
  echo "Example: $0 b2_d128_anchor 0 seed=42" >&2
  exit 1
fi

EXPERIMENT="$1"
FOLD="$2"
shift 2

echo "[$(date -Is)] Starting ${EXPERIMENT} LOSO fold=${FOLD}"
python src/train.py "experiment=${EXPERIMENT}" "data.loso_fold=${FOLD}" "$@"
