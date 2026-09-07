#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED GPU_INDEX" >&2
  exit 2
fi

seed="$1"
gpu_index="$2"
case "${seed}" in
  42|43|44) ;;
  *) echo "seed must be 42, 43, or 44" >&2; exit 2 ;;
esac
case "${gpu_index}" in
  0|1) ;;
  *) echo "GPU_INDEX must be 0 or 1 on the two-3090 host" >&2; exit 2 ;;
esac

workspace="${T4_C1_INT8_WORKSPACE:-/home/xinyuan/Work_host/SPINT}"
python_bin="${T4_C1_INT8_PYTHON:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
prelaunch="${T4_C1_INT8_PRELAUNCH:-${workspace}/sua_exploration/results/t4_paired_view_c1_encoder_int8_prelaunch_v2_20260805/receipt.json}"
program_root="${T4_C1_INT8_RESULT_ROOT:-${workspace}/sua_exploration/results/t4_paired_view_c1_encoder_int8_ptq_v1_20260805}"
metadata="${workspace}/sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s${seed}/run_metadata.json"
checkpoint="${workspace}/sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s${seed}/epoch_ckpts/epoch_011.ckpt"
out_dir="${program_root}/seed${seed}"
log_dir="${program_root}/logs"
log_path="${log_dir}/seed${seed}.log"

if [[ ! -f "${prelaunch}" || ! -f "${metadata}" || ! -f "${checkpoint}" ]]; then
  echo "missing prelaunch, metadata, or checkpoint" >&2
  exit 3
fi
if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
  echo "refusing to overwrite seed${seed} immutable output/log" >&2
  exit 4
fi
mkdir -p "${log_dir}"

CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -u \
  "${workspace}/sua_exploration/scripts/eval_t4_paired_view_c1_encoder_int8.py" \
  --prelaunch "${prelaunch}" \
  --run-metadata "${metadata}" \
  --checkpoint "${checkpoint}" \
  --out-dir "${out_dir}" \
  --device cuda:0 \
  >"${log_path}" 2>&1
