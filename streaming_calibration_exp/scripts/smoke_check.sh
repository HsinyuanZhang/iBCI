#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Gate 0 (verify canonical B0 baseline) =="
python scripts/verify_gate0.py

echo "== Gate 1 =="
python scripts/verify_gate1.py

echo "== Unit tests =="
if python -m pytest --version >/dev/null 2>&1; then
  python -m pytest tests/ -q
else
  echo 'pytest not installed; skipping unit tests'
fi

echo "All smoke checks passed."
