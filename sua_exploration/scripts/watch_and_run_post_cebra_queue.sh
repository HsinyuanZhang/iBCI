#!/usr/bin/env bash
# Continue from the frozen pseudo-session terminal gate into SetKV-delta Stage 0.
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
PSEUDO="$SUA/results/cebra_pseudosession_sua_v1"
STAGE_P="$PSEUDO/stage_p_seed42_aggregate.json"
PSEUDO_TERMINAL="$PSEUDO/terminal_three_seed_aggregate.json"
RESULT="$SUA/results/setkv_delta_forward_v1"
PREFLIGHT="$RESULT/official_cpu_preflight.json"
SCORER="$SUA/scripts/score_setkv_delta_forward.py"
AGGREGATOR="$SUA/scripts/aggregate_setkv_delta_forward.py"
LOG="$RESULT/continuous_queue.log"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUA"

fail() {
  echo "SetKV post-CEBRA queue: $*" >&2
  exit 2
}

verify_pair() {
  local body="$1"
  local sidecar="${body}.sha256"
  local expected
  local actual
  [[ -f "$body" && -f "$sidecar" && ! -L "$body" && ! -L "$sidecar" ]] || fail "missing/symlinked receipt pair: $body"
  [[ "$(stat -c '%a' "$body")" == "444" && "$(stat -c '%a' "$sidecar")" == "444" ]] || fail "mutable receipt pair: $body"
  # A2 immutable receipts use the standard sha256sum sidecar spelling:
  # "<digest>  <basename>\n".  Extract only the digest rather than
  # concatenating the required filename into it.
  expected="$(awk 'NR == 1 { print $1; exit }' "$sidecar")"
  actual="$(sha256sum "$body" | awk '{print $1}')"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "malformed receipt SHA sidecar: $body"
  [[ "$actual" == "$expected" ]] || fail "receipt SHA mismatch: $body"
}

if [[ "${1:-}" == "--shell-smoke" ]]; then
  smoke_dir="$(mktemp -d /tmp/setkv_queue_smoke.XXXXXX)"
  smoke_body="$smoke_dir/receipt.json"
  cleanup_smoke() {
    [[ "$smoke_dir" == /tmp/setkv_queue_smoke.* ]] || exit 2
    rm -rf -- "$smoke_dir"
  }
  trap cleanup_smoke EXIT
  printf '{"shell_smoke":true}\n' > "$smoke_body"
  smoke_sha="$(sha256sum "$smoke_body" | awk '{print $1}')"
  printf '%s  %s\n' "$smoke_sha" "$(basename "$smoke_body")" > "${smoke_body}.sha256"
  chmod 444 "$smoke_body" "${smoke_body}.sha256"
  verify_pair "$smoke_body"
  chmod 644 "${smoke_body}.sha256"
  printf '%s\n' "$smoke_sha" > "${smoke_body}.sha256"
  chmod 444 "${smoke_body}.sha256"
  verify_pair "$smoke_body"
  chmod 644 "${smoke_body}.sha256"
  if (verify_pair "$smoke_body") >/dev/null 2>&1; then
    fail "shell smoke expected mutable-sidecar rejection"
  elif [[ "$?" != "2" ]]; then
    fail "shell smoke rejection did not use fail()"
  fi
  echo '{"status":"SHELL_SMOKE_PASS__NO_DATA_NO_GPU"}'
  exit 0
fi

if [[ "${1:-}" != "--execute" ]]; then
  "$PY" - <<PY
import json
print(json.dumps({
  "status": "DRY_RUN__NO_DATA_NO_GPU",
  "waits_for_stage_p": "$STAGE_P",
  "waits_for_three_seed_terminal_only_if_stage_p_passes": "$PSEUDO_TERMINAL",
  "setkv_preflight": "$PREFLIGHT",
  "setkv_score_cells": 8,
  "setkv_aggregate": "$RESULT/terminal_forward_aggregate.json",
}, indent=2, sort_keys=True))
PY
  exit 0
fi

mkdir -p "$RESULT/logs"
if [[ -e "$LOG" ]]; then
  echo "refusing to reuse post-CEBRA queue log: $LOG" >&2
  exit 2
fi
exec > >(tee "$LOG") 2>&1

echo "waiting for immutable pseudo-session Stage-P receipt"
while [[ ! -f "$STAGE_P" || ! -f "$STAGE_P.sha256" ]]; do sleep 60; done
verify_pair "$STAGE_P"
stage_p_sha="$(sha256sum "$STAGE_P" | awk '{print $1}')"
if jq -e '.receipt_kind == "cebra_pseudosession_sua_stage_p_aggregate" and .screen_id == "cebra_pseudosession_sua_v1" and .seed == 42 and .formal_subc_test_nwb_opened == false and .passes_stage_p == true and .verdict == "STAGE_P_PASS__EXPAND_SEEDS_43_44"' "$STAGE_P" >/dev/null; then
  echo "pseudo-session Stage P passed; waiting for its three-seed terminal receipt"
  while [[ ! -f "$PSEUDO_TERMINAL" || ! -f "$PSEUDO_TERMINAL.sha256" ]]; do sleep 60; done
  verify_pair "$PSEUDO_TERMINAL"
  jq -e --arg stage_p_sha "$stage_p_sha" '.receipt_kind == "cebra_pseudosession_sua_terminal_aggregate" and .screen_id == "cebra_pseudosession_sua_v1" and .seeds == [42, 43, 44] and .formal_subc_test_nwb_opened == false and .receipt_sha256.stage_p == $stage_p_sha' "$PSEUDO_TERMINAL" >/dev/null || fail "pseudo terminal receipt is not bound to this Stage-P receipt"
else
  jq -e '.receipt_kind == "cebra_pseudosession_sua_stage_p_aggregate" and .screen_id == "cebra_pseudosession_sua_v1" and .seed == 42 and .formal_subc_test_nwb_opened == false and .passes_stage_p == false and .verdict == "STAGE_P_STOP__NO_PARAMETER_OR_ARCHITECTURE_TUNING"' "$STAGE_P" >/dev/null || fail "pseudo Stage-P stop receipt drift"
  echo "pseudo-session Stage P stopped; advancing without tuning"
fi

verify_pair "$PREFLIGHT"
"$PY" - "$PREFLIGHT" <<'PY'
import sys
from pathlib import Path
from mc_maze.setkv_delta_forward_core import load_official_preflight

payload, _digest = load_official_preflight(Path(sys.argv[1]))
assert payload["formal_subc_test_nwb_opened"] is False
assert payload["source_nwb_opened"] is False
assert payload["target_nwb_opened"] is False
assert payload["cuda_used"] is False
PY

# Reject a partial/duplicate Stage-0 launch before any GPU process is started.
for name in setkv_t4 setkv_z4 setkv_rs4 duplicate_activity_t4; do
  for domain in within_subject external_subject_M; do
    body="$RESULT/${domain}_${name}_s42.json"
    [[ ! -e "$body" && ! -e "${body}.sha256" ]] || fail "SetKV score cell already exists: $body"
  done
done
[[ ! -e "$RESULT/terminal_forward_aggregate.json" && ! -e "$RESULT/terminal_forward_aggregate.json.sha256" ]] || fail "SetKV terminal aggregate already exists"

score_chain() {
  local name="$1"
  local gpu="$2"
  local log="$RESULT/logs/${name}.log"
  test ! -e "$log"
  for domain in within_subject external_subject_M; do
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "$PY" -u "$SCORER" \
      --intervention "$name" --domain "$domain" --device cuda:0 --execute \
      >>"$log" 2>&1
  done
}

# The first pair measures the aligned T4 and matched Z4 token-set effects.
score_chain setkv_t4 0 &
pid0=$!
score_chain setkv_z4 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

# The second pair prices wrong attachment and generic set duplication.
score_chain setkv_rs4 0 &
pid0=$!
score_chain duplicate_activity_t4 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PY" "$AGGREGATOR" --execute
echo "SetKV-delta Stage-0 terminal aggregate complete; next queue bridge may inspect its frozen verdict"
