#!/usr/bin/env bash
set -euo pipefail

workspace="${T4_C1_WORKSPACE:-/home/xinyuan/Work_host/SPINT}"
python_bin="${T4_C1_PYTHON:-/home/xinyuan/miniconda3/envs/spint/bin/python}"

exec "${python_bin}" -u \
  "${workspace}/sua_exploration/scripts/run_t4_paired_view_c1_one_cell.py" \
  --workspace "${workspace}" \
  --python "${python_bin}" \
  "$@"
