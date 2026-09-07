#!/usr/bin/env bash
set -euo pipefail

workspace="${T4_C1_WORKSPACE:-/home/xinyuan/Work_host/SPINT}"
python_bin="${T4_C1_PYTHON:-/home/xinyuan/miniconda3/envs/spint/bin/python}"

exec "${python_bin}" -u \
  "${workspace}/sua_exploration/scripts/schedule_t4_paired_view_c1_2gpu.py" \
  --workspace "${workspace}" \
  --python "${python_bin}" \
  "$@"
