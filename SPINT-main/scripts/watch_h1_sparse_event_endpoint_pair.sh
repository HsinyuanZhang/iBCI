#!/usr/bin/env bash
set -euo pipefail

project_root="/home/xinyuan/Work_host/SPINT/SPINT-main"
python_bin="/home/xinyuan/miniconda3/envs/spint/bin/python"
full_root="${project_root}/logs/h1_sparse_event_endpoint_m4_full_s42_v1"
zero_root="${project_root}/logs/h1_sparse_event_endpoint_m4_zero_s42_v1"
full_ckpt="${full_root}/checkpoints/fixed_epoch50/epoch_049.ckpt"
zero_ckpt="${zero_root}/checkpoints/fixed_epoch50/epoch_049.ckpt"
full_config="${full_root}/.hydra/config.yaml"
zero_config="${zero_root}/.hydra/config.yaml"
output="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"

cd "${project_root}"

while true; do
  if [[ -s "${full_ckpt}" && -s "${zero_ckpt}" && -s "${full_config}" && -s "${zero_config}" ]]; then
    if tmux has-session -t hse5_full 2>/dev/null || tmux has-session -t hse5_zero 2>/dev/null; then
      sleep 20
      continue
    fi
    if [[ -e "${output}" ]]; then
      echo "H-SE5 terminal output already exists; refusing overwrite: ${output}"
      exit 3
    fi
    echo "H-SE5 pair complete; starting isolated strict target evaluator"
    set +e
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "${python_bin}" \
      scripts/h1_sparse_event_endpoint_evaluate.py \
      --data-dir data/000954 \
      --full-checkpoint "${full_ckpt}" \
      --zero-checkpoint "${zero_ckpt}" \
      --full-config "${full_config}" \
      --zero-config "${zero_config}" \
      --output "${output}" \
      --device cuda
    evaluator_status=$?
    set -e
    if [[ "${evaluator_status}" -eq 0 || "${evaluator_status}" -eq 2 ]]; then
      echo "H-SE5 evaluator closed scientifically with status ${evaluator_status}"
      exit 0
    fi
    echo "H-SE5 evaluator infrastructure failure: ${evaluator_status}"
    exit "${evaluator_status}"
  fi

  if ! tmux has-session -t hse5_full 2>/dev/null && [[ ! -s "${full_ckpt}" ]]; then
    echo "H-SE5 Full training exited without terminal checkpoint"
    exit 10
  fi
  if ! tmux has-session -t hse5_zero 2>/dev/null && [[ ! -s "${zero_ckpt}" ]]; then
    echo "H-SE5 Zero5 training exited without terminal checkpoint"
    exit 11
  fi
  sleep 30
done
