#!/usr/bin/env bash
# CTXV2 Stage A GPU launch orchestrator v2 — DO NOT RUN FROM AUTOMATED AGENTS.
#
# Requires receipts at their real paths (Phase 1 preflight, config parity, fidelity
# gate, and 19250108 source snapshot).  Refuses while Stage B tmux sessions are
# alive so it cannot collide with running fold-0 control arms.
set -euo pipefail

repo_root="/home/xinyuan/Work_host/SPINT"
project_root="${repo_root}/SPINT-main"
python_bin="/home/xinyuan/miniconda3/envs/spint/bin/python"
artifact_root="${project_root}/pilot_artifacts/h1_ctxv2_19250108"
preflight_receipt="${repo_root}/sua_exploration/results/ctxv2_stage_a_preflight/CTXV2_STAGE_A_PREFLIGHT_19250108.json"
parity_receipt="${artifact_root}/CTXV2_STAGE_A_CONFIG_PARITY_v1.json"
fidelity_receipt="${artifact_root}/CTXV2_STAGE_A_FIDELITY_GATE_v1.json"
snapshot_receipt="${artifact_root}/H1_CONTEXT_SER_Q4_19250108_SOURCE_v1.json"
log_root="${artifact_root}/logs"

ctx_experiment="h1_ctxv2_context_full_19250108"
hse5_experiment="h1_ctxv2_hse5_full_19250108"
zero_experiment="h1_ctxv2_zero5_19250108"

ctx_run_dir="${project_root}/logs/h1_ctxv2_context_full_19250108_m4_s42_v1"
hse5_run_dir="${project_root}/logs/h1_ctxv2_hse5_full_19250108_m4_s42_v1"
zero_run_dir="${project_root}/logs/h1_ctxv2_zero5_19250108_m4_s42_v1"

ctx_log="${log_root}/ctxv2_ctx_train.log"
hse5_log="${log_root}/ctxv2_hse5_train.log"
zero_log="${log_root}/ctxv2_zero_train.log"

pythonpath="${project_root}:${repo_root}"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
fi

export PYTHONNOUSERSITE=1

require_receipt() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "missing required ${label} receipt: ${path}" >&2
    exit 2
  fi
  local mode
  mode="$(stat -c '%a' "${path}")"
  if [[ "${mode}" != "444" ]]; then
    echo "required ${label} receipt must be mode 0444: ${path} (got ${mode})" >&2
    exit 2
  fi
}

refuse_stage_b_sessions() {
  local session
  for session in ctxv2_ls ctxv2_rs; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "refusing to launch: Stage B tmux session ${session} is still alive" >&2
      exit 5
    fi
  done
}

refuse_existing_run_dir() {
  local run_dir="$1"
  local label="$2"
  if [[ -e "${run_dir}" ]]; then
    echo "refusing to launch ${label}: run directory already exists: ${run_dir}" >&2
    exit 3
  fi
}

refuse_existing_log() {
  local log_path="$1"
  local label="$2"
  if [[ -e "${log_path}" ]]; then
    echo "refusing to launch ${label}: log already exists: ${log_path}" >&2
    exit 3
  fi
}

print_train_cmd() {
  local gpu="$1"
  local experiment="$2"
  local run_dir="$3"
  local log_path="$4"
  local session="$5"
  echo "tmux session: ${session}"
  echo "CUDA_VISIBLE_DEVICES=${gpu} PYTHONNOUSERSITE=1 PYTHONPATH=${pythonpath} ${python_bin} ${project_root}/src/train.py experiment=${experiment} hydra.run.dir=${run_dir} paths.output_dir=${run_dir} seed=42 2>&1 | tee ${log_path}"
}

if [[ "${dry_run}" == "false" ]]; then
  require_receipt "${preflight_receipt}" "Stage A preflight"
  require_receipt "${parity_receipt}" "config parity"
  require_receipt "${fidelity_receipt}" "fidelity gate"
  require_receipt "${snapshot_receipt}" "19250108 source snapshot"
  refuse_stage_b_sessions
  mkdir -p "${log_root}"
  refuse_existing_run_dir "${ctx_run_dir}" "Context Full"
  refuse_existing_run_dir "${hse5_run_dir}" "H-SE5 Full"
  refuse_existing_run_dir "${zero_run_dir}" "Zero5"
  refuse_existing_log "${ctx_log}" "Context Full"
  refuse_existing_log "${hse5_log}" "H-SE5 Full"
  refuse_existing_log "${zero_log}" "Zero5"
  if tmux has-session -t ctxv2_ctx 2>/dev/null; then
    echo "tmux session ctxv2_ctx already exists" >&2
    exit 4
  fi
  if tmux has-session -t ctxv2_hse5 2>/dev/null; then
    echo "tmux session ctxv2_hse5 already exists" >&2
    exit 4
  fi
  if tmux has-session -t ctxv2_zero 2>/dev/null; then
    echo "tmux session ctxv2_zero already exists" >&2
    exit 4
  fi
else
  refuse_stage_b_sessions
fi

echo "CTXV2 Stage A launch plan v2 (two-wave, two-GPU)"
echo "Wave 1: Context Full on GPU 0; H-SE5 Full on GPU 1"
echo "Wave 2: Zero5 on GPU 0 (starts after wave 1 tmux sessions exit)"
echo

echo "Wave 1 — Context Full"
print_train_cmd 0 "${ctx_experiment}" "${ctx_run_dir}" "${ctx_log}" "ctxv2_ctx"
echo

echo "Wave 1 — H-SE5 Full"
print_train_cmd 1 "${hse5_experiment}" "${hse5_run_dir}" "${hse5_log}" "ctxv2_hse5"
echo

echo "Wave 2 — Zero5"
print_train_cmd 0 "${zero_experiment}" "${zero_run_dir}" "${zero_log}" "ctxv2_zero"
echo

if [[ "${dry_run}" == "true" ]]; then
  echo "--dry-run: no tmux sessions started."
  exit 0
fi

tmux new-session -d -s ctxv2_ctx \
  "cd ${project_root} && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=${pythonpath} ${python_bin} src/train.py experiment=${ctx_experiment} hydra.run.dir=${ctx_run_dir} paths.output_dir=${ctx_run_dir} seed=42 2>&1 | tee ${ctx_log}"

tmux new-session -d -s ctxv2_hse5 \
  "cd ${project_root} && CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=${pythonpath} ${python_bin} src/train.py experiment=${hse5_experiment} hydra.run.dir=${hse5_run_dir} paths.output_dir=${hse5_run_dir} seed=42 2>&1 | tee ${hse5_log}"

tmux new-session -d -s ctxv2_zero \
  "cd ${project_root} && while tmux has-session -t ctxv2_ctx 2>/dev/null || tmux has-session -t ctxv2_hse5 2>/dev/null; do sleep 30; done; CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=${pythonpath} ${python_bin} src/train.py experiment=${zero_experiment} hydra.run.dir=${zero_run_dir} paths.output_dir=${zero_run_dir} seed=42 2>&1 | tee ${zero_log}"

echo "Launched tmux sessions: ctxv2_ctx (GPU 0), ctxv2_hse5 (GPU 1), ctxv2_zero (GPU 0 after wave 1)."
exit 0
