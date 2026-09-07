#!/usr/bin/env bash
set -euo pipefail

repository_root="/home/xinyuan/Work_host/SPINT"
project_root="${repository_root}/SPINT-main"
python_bin="/home/xinyuan/miniconda3/envs/spint/bin/python"
terminal="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"
verification="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1_verification.json"
recomputation="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1_recomputed_r2.json"

cd "${repository_root}"

while [[ ! -s "${terminal}" ]]; do
  if ! tmux has-session -t hse5_watch 2>/dev/null; then
    echo "H-SE5 terminal watcher exited without producing the terminal receipt"
    exit 20
  fi
  sleep 30
done

# The terminal receipt is written immediately before the evaluator exits.  Do
# not compete with its final CUDA work; wait for the scientific watcher to end.
while tmux has-session -t hse5_watch 2>/dev/null; do
  sleep 5
done

if [[ -e "${verification}" || -e "${recomputation}" ]]; then
  echo "H-SE5 audit output already exists; refusing overwrite"
  exit 21
fi

echo "H-SE5 terminal receipt ready; running independent structural verifier"
PYTHONNOUSERSITE=1 "${python_bin}" \
  sua_exploration/scripts/verify_h1_sparse_event_endpoint_gpu_receipt.py \
  "${terminal}" --output "${verification}"

echo "H-SE5 structural verifier passed; independently recomputing Full/Zero5 R2"
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH="${project_root}" "${python_bin}" \
  sua_exploration/scripts/recompute_h1_sparse_event_endpoint_gpu_r2.py \
  "${terminal}" --data-dir "${project_root}/data/000954" --device cuda \
  --batch-size 29 --tolerance 2e-6 --output "${recomputation}"

echo "H-SE5 terminal audit chain completed"
