#!/usr/bin/env bash
# Inert runner for the single missing A14 B3T+TS4 content-control cell (seed 42).
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
CONTRACT="$SUA/docs/A14_B3T_T4_EFFICIENCY_COMPLETION_CONTRACT_20260812.md"
AUTH_ENV="A14_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_A14_B3T_EFFICIENCY_COMPLETION_GPU"
GPU="${GPU:?GPU index required}"
MODE="${1:---dry-run}"

PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
CONTRACT_SHA="$("$PY" - "$CONTRACT" <<'PY'
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
print(h.hexdigest())
PY
)"

echo "CONTRACT=$CONTRACT"
echo "CONTRACT_SHA256=$CONTRACT_SHA"
echo "MISSING_CELL=b3t_ts4_s42"
echo "EXPECTED_NEW_GPU_CELLS=1"
echo "AUTHORIZATION_REQUIRED=${AUTH_ENV}=${AUTH_VALUE}"

if [[ "$MODE" == "--dry-run" ]]; then
  ARM=b3t_ts4 SEED=42 GPU="$GPU" "$SUA/scripts/run_sua_b3t_t4_one_cell.sh" --dry-run
  exit 0
fi

if [[ "${A14_GPU_AUTHORIZATION:-}" != "$AUTH_VALUE" ]]; then
  echo "Refusing GPU launch: set ${AUTH_ENV}=${AUTH_VALUE}" >&2
  echo "This runner does not have GPU authorization." >&2
  exit 3
fi

ARM=b3t_ts4 SEED=42 GPU="$GPU" "$SUA/scripts/run_sua_b3t_t4_one_cell.sh" --launch
