#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/xinyuan/Work_host/SPINT
task_root="$repo_root/tfpd_exploration/h1_series_20260830/automation"
prompt_file="$task_root/LP_R3_EVALAI_20260904_0800_PROMPT.md"
log_file="$task_root/LP_R3_EVALAI_20260904_0800.log"
last_message="$task_root/LP_R3_EVALAI_20260904_0800_LAST.md"
export PATH=/home/xinyuan/.nvm/versions/node/v26.1.0/bin:/usr/local/bin:/usr/bin:/bin
export HOME=/home/xinyuan

exec /home/xinyuan/.nvm/versions/node/v26.1.0/bin/codex exec \
  -C "$repo_root" \
  -s danger-full-access \
  -a never \
  --color never \
  -o "$last_message" \
  - <"$prompt_file" >>"$log_file" 2>&1
