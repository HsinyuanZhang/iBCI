#!/usr/bin/env bash
# Continue from SetKV-delta into the source-gated carrier value-mask diagnostic.
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
SETKV="$SUA/results/setkv_delta_forward_v1/terminal_forward_aggregate.json"
RESULT="$SUA/results/carrier_value_mask_v1"
SOURCE_GATE="$RESULT/source_value_weighted_gate.json"
PROBE="$SUA/scripts/probe_carrier_value_mask_source.py"
SCORER="$SUA/scripts/score_carrier_value_mask.py"
AGGREGATOR="$SUA/scripts/aggregate_carrier_value_mask.py"
LOG="$RESULT/continuous_queue.log"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUA:$ROOT/streaming_calibration_exp"

fail() {
  echo "carrier-value-mask queue: $*" >&2
  exit 2
}

verify_pair() {
  local body="$1"
  local sidecar="${body}.sha256"
  local expected
  local actual
  [[ -f "$body" && -f "$sidecar" && ! -L "$body" && ! -L "$sidecar" ]] || fail "missing/symlinked receipt pair: $body"
  [[ "$(stat -c '%a' "$body")" == "444" && "$(stat -c '%a' "$sidecar")" == "444" ]] || fail "mutable receipt pair: $body"
  # A2/SetKV immutable sidecars use sha256sum's '<digest>  <basename>' spelling.
  expected="$(awk 'NR == 1 { print $1; exit }' "$sidecar")"
  actual="$(sha256sum "$body" | awk '{print $1}')"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "malformed receipt SHA sidecar: $body"
  [[ "$actual" == "$expected" ]] || fail "receipt SHA mismatch: $body"
}

absent_pair() {
  local body="$1"
  [[ ! -e "$body" && ! -L "$body" && ! -e "${body}.sha256" && ! -L "${body}.sha256" ]] ||
    fail "output already exists: $body"
}

if [[ "${1:-}" == "--shell-smoke" ]]; then
  smoke_dir="$(mktemp -d /tmp/carrier_value_mask_queue_smoke.XXXXXX)"
  smoke_body="$smoke_dir/receipt.json"
  cleanup_smoke() {
    [[ "$smoke_dir" == /tmp/carrier_value_mask_queue_smoke.* ]] || exit 2
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
  "waits_for_setkv": "$SETKV",
  "source_gate": "$SOURCE_GATE",
  "target_score_cells_if_source_gate_passes": 10,
  "aggregate": "$RESULT/terminal_mask_aggregate.json",
}, indent=2, sort_keys=True))
PY
  exit 0
fi

mkdir -p "$RESULT/logs"
[[ ! -e "$LOG" && ! -L "$LOG" ]] || fail "refusing to reuse/symlink carrier-mask queue log: $LOG"
exec > >(tee "$LOG") 2>&1

echo "waiting for immutable SetKV-delta aggregate"
while [[ ! -f "$SETKV" || ! -f "$SETKV.sha256" ]]; do sleep 60; done
verify_pair "$SETKV"
"$PY" - "$SETKV" <<'PY'
import sys
from pathlib import Path

from mc_maze import a2_matched_subject_shift_v2_core as a2
from mc_maze import setkv_delta_forward_core as setkv

terminal, _ = a2.load_verified_immutable_json(Path(sys.argv[1]), label="SetKV terminal aggregate")
_preflight, preflight_sha = setkv.load_official_preflight()
assert terminal["receipt_kind"] == "setkv_delta_forward_aggregate"
assert terminal["screen_id"] == setkv.SCREEN_ID
assert terminal["official_preflight_sha256"] == preflight_sha
assert terminal["parameter_delta"] == 0
assert terminal["formal_subc_test_nwb_opened"] is False
assert terminal["verdict"] in {
    "SETKV_FORWARD_SIGNAL_POSITIVE__DESIGN_JOINT_TRAINING",
    "SETKV_FORWARD_SIGNAL_INCOMPLETE_OR_FLAT__NO_JOINT_TRAINING_FROM_THIS_DIAGNOSTIC",
}
PY

# Reject partial or duplicate target cells before the source probe or any GPU process.
for name in low_t4 random_t4 high_t4 low_z4 random_z4; do
  for domain in within_subject external_subject_M; do
    absent_pair "$RESULT/${domain}_${name}_s42.json"
  done
done
absent_pair "$RESULT/terminal_mask_aggregate.json"

if [[ ! -e "$SOURCE_GATE" && ! -L "$SOURCE_GATE" && ! -e "$SOURCE_GATE.sha256" && ! -L "$SOURCE_GATE.sha256" ]]; then
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "$PY" -u "$PROBE" --execute
fi
[[ -f "$SOURCE_GATE" && -f "$SOURCE_GATE.sha256" ]] || fail "source gate missing after source probe"
verify_pair "$SOURCE_GATE"

gate_status="$(env CUDA_VISIBLE_DEVICES='' "$PY" - <<'PY'
from mc_maze import carrier_value_mask_core as core
p, _ = core.load_source_gate()
print(p['status'])
PY
)"
if [[ "$gate_status" == "SOURCE_GATE_STOP__NO_TARGET_SCORE" ]]; then
  echo "carrier value-mask source screen stopped before target scoring; advancing queue"
  exit 0
fi
[[ "$gate_status" == "SOURCE_GATE_PASS__TARGET_FORWARD_ALLOWED" ]] || fail "unexpected source-gate status: $gate_status"

score_chain() {
  local name="$1"
  local gpu="$2"
  local log="$RESULT/logs/${name}.log"
  [[ ! -e "$log" && ! -L "$log" ]] || fail "refusing to reuse/symlink carrier-mask score log: $log"
  for domain in within_subject external_subject_M; do
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "$PY" -u "$SCORER" \
      --intervention "$name" --domain "$domain" --device cuda:0 --execute \
      >>"$log" 2>&1
  done
}

wait_pair() {
  local left="$1"
  local right="$2"
  if ! wait "$left"; then
    kill "$right" 2>/dev/null || true
    wait "$right" 2>/dev/null || true
    fail "parallel carrier-mask score chain failed"
  fi
  if ! wait "$right"; then
    fail "parallel carrier-mask score chain failed"
  fi
}

score_chain low_t4 0 & pid0=$!
score_chain low_z4 1 & pid1=$!
wait_pair "$pid0" "$pid1"

score_chain random_t4 0 & pid0=$!
score_chain random_z4 1 & pid1=$!
wait_pair "$pid0" "$pid1"

score_chain high_t4 0
"$PY" "$AGGREGATOR" --execute
echo "carrier value-mask terminal aggregate complete; misleading-identity swap-v2 is next"
