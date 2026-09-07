#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED GPU_INDEX" >&2
  exit 2
fi
seed="$1"
gpu_index="$2"
case "${seed}" in 42|43|44) ;; *) echo "seed must be 42,43,44" >&2; exit 2;; esac
case "${gpu_index}" in 0|1) ;; *) echo "GPU_INDEX must be 0 or 1" >&2; exit 2;; esac

workspace="${T4_C1_QAT_WORKSPACE:-/home/xinyuan/Work_host/SPINT}"
python_bin="${T4_C1_QAT_PYTHON:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
prelaunch="${T4_C1_QAT_PRELAUNCH:-${workspace}/sua_exploration/results/t4_paired_view_c1_encoder_qat_prelaunch_v1_20260805/receipt.json}"
result_root="${T4_C1_QAT_RESULT_ROOT:-${workspace}/sua_exploration/results/t4_paired_view_c1_encoder_qat_v1_20260805}"
out_dir="${result_root}/seed${seed}"
log_dir="${result_root}/logs"
log_path="${log_dir}/seed${seed}.log"

if [[ ! -f "${prelaunch}" ]]; then
  echo "QAT prelaunch is absent; frozen three-seed PTQ aggregate has not authorized launch" >&2
  exit 3
fi
if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
  echo "refusing to overwrite seed${seed} QAT output/log" >&2
  exit 4
fi
mkdir -p "${log_dir}"

CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -u \
  "${workspace}/sua_exploration/scripts/train_t4_paired_view_c1_encoder_qat.py" \
  --qat-prelaunch "${prelaunch}" \
  --seed "${seed}" \
  --out-dir "${out_dir}" \
  --device cuda:0 \
  --epochs 8 \
  >"${log_path}" 2>&1
