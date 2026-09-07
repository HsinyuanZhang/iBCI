#!/usr/bin/env bash
set -euo pipefail

# Recovery wrapper for the 2026-08-04 launch incident.  The sealed scheduler
# created the result root and consumed the single-use authorization before its
# run_cell helper hit an unbound local-variable expansion.  No cell started.
# This wrapper validates that exact claim and invokes the already-sealed runner;
# it does not alter the model, data, evaluator, or aggregate definition.

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RUN="$ROOT/sua_exploration/results/sua_t4_m30_ac4_rs4_v1"
STATUS="$RUN/cell_status"
RUNNER="$ROOT/sua_exploration/scripts/run_t4_m30_ac4_rs4_v1_one_cell.sh"
CLAIM="$RUN/authorization_claim.json"
EXPECTED_RECEIPT_SHA="425d2b9818db82346ce21d31d0a00766636a247fefa07c81e392155823b4b451"
EXPECTED_AUTHORIZATION_ID="t4-m30-ac4-rs4-v1-user-continued-20260804"

[[ -d "$RUN" && -d "$STATUS" && -f "$CLAIM" ]]
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_ac4_rs4_v1.py" --verify
"$PY" - "$CLAIM" "$EXPECTED_RECEIPT_SHA" "$EXPECTED_AUTHORIZATION_ID" <<'PY'
import json
import sys

claim_path, expected_sha, expected_authorization = sys.argv[1:]
with open(claim_path, "r", encoding="utf-8") as handle:
    claim = json.load(handle)
assert claim == {
    "authorization_id": expected_authorization,
    "receipt_sha256": expected_sha,
}, claim
PY

# Fail closed: this recovery path is valid only before any cell artifact exists.
[[ -z "$(find "$STATUS" -mindepth 1 -maxdepth 1 -print -quit)" ]]
for seed in 42 43 44; do
    [[ ! -e "$RUN/ac4_rs4_s${seed}.json" ]]
    [[ ! -e "$ROOT/sua_exploration/checkpoints/sua_t4_m30_ac4_rs4_v1_ac4_rs4_dandi688_co_s${seed}" ]]
done

run_cell() {
    local seed="$1"
    local gpu="$2"
    local log="$STATUS/ac4_rs4_s${seed}.log"
    local record="$STATUS/ac4_rs4_s${seed}.json"
    local started ended code result metadata cost

    started="$(date --iso-8601=seconds)"
    set +e
    SEED="$seed" GPU="$gpu" bash "$RUNNER" >"$log" 2>&1
    code=$?
    set -e
    ended="$(date --iso-8601=seconds)"
    result="$RUN/ac4_rs4_s${seed}.json"
    metadata="$ROOT/sua_exploration/checkpoints/sua_t4_m30_ac4_rs4_v1_ac4_rs4_dandi688_co_s${seed}/run_metadata.json"
    cost="${metadata%/*}/post_run_cost_receipt.json"
    "$PY" - "$record" "$seed" "$gpu" "$started" "$ended" "$code" "$result" "$metadata" "$cost" <<'PY'
import hashlib
import json
import os
import sys


def sha256(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


record, seed, gpu, started, ended, code, result, metadata, cost = sys.argv[1:]
payload = {
    "arm": "ac4_rs4",
    "seed": int(seed),
    "gpu": int(gpu),
    "started_at": started,
    "ended_at": ended,
    "exit_code": int(code),
    "status": "completed" if int(code) == 0 else "failed",
    "result_sha256": sha256(result),
    "metadata_sha256": sha256(metadata),
    "cost_sha256": sha256(cost),
}
with open(record, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
    return "$code"
}

run_cell 42 0 &
pid42=$!
run_cell 43 1 &
pid43=$!

pair_failed=0
wait "$pid42" || pair_failed=1
wait "$pid43" || pair_failed=1
[[ "$pair_failed" -eq 0 ]]

run_cell 44 0

for seed in 42 43 44; do
    "$PY" - "$STATUS/ac4_rs4_s${seed}.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    status = json.load(handle)
assert status["status"] == "completed" and status["exit_code"] == 0
assert status["result_sha256"]
assert status["metadata_sha256"]
assert status["cost_sha256"]
PY
done

"$PY" "$ROOT/sua_exploration/scripts/aggregate_t4_m30_ac4_rs4_v1.py" \
    --result-dir "$RUN" \
    --status-dir "$STATUS" \
    --baseline-result-dir "$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v10" \
    --baseline-aggregate "$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v10/aggregate_r11.json" \
    --out "$RUN/aggregate_v1.json"
