#!/usr/bin/env bash
set -euo pipefail

repository_root="/home/xinyuan/Work_host/SPINT"
project_root="${repository_root}/SPINT-main"
python_bin="/home/xinyuan/miniconda3/envs/spint/bin/python"
terminal="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"
structural="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1_verification.json"
recomputation="${project_root}/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1_recomputed_r2.json"
output="${repository_root}/sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_COMPLETION_v1.json"
verifier="${repository_root}/sua_exploration/scripts/verify_h1_sparse_event_endpoint_program.py"

cd "${repository_root}"

while [[ ! -s "${terminal}" || ! -s "${structural}" || ! -s "${recomputation}" ]]; do
  if ! tmux has-session -t hse5_audit 2>/dev/null; then
    echo "H-SE5 terminal audit watcher exited before all program inputs existed"
    for artifact in "${terminal}" "${structural}" "${recomputation}"; do
      [[ -s "${artifact}" ]] || echo "missing: ${artifact}"
    done
    exit 30
  fi
  sleep 30
done

# The audit watcher writes the recomputation artifact immediately before its
# final shell message.  Avoid racing its final CUDA/file operations.
while tmux has-session -t hse5_audit 2>/dev/null; do
  sleep 5
done

if [[ -e "${output}" ]]; then
  echo "H-SE5 program completion artifact already exists; refusing overwrite: ${output}"
  exit 31
fi

echo "H-SE5 terminal audit chain ready; running final program verifier"
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${python_bin}" "${verifier}" \
  --terminal "${terminal}" \
  --structural "${structural}" \
  --recomputation "${recomputation}" \
  --output "${output}"

echo "H-SE5 program completion artifact written: ${output}"
