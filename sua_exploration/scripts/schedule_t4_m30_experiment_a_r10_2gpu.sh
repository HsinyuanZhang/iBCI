#!/usr/bin/env bash
# REQUIRED: run in a managed foreground exec session; never use nohup/disown.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RUN="$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v10"
STATUS="$RUN/cell_status"
RUNNER="$ROOT/sua_exploration/scripts/run_t4_m30_experiment_a_r10_one_cell.sh"

[[ -r "$RUNNER" ]] || { echo "r10 runner is not readable" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "r10 run directory already exists" >&2; exit 1; }
mkdir "$RUN"
mkdir "$STATUS"

"$PY" \
    "$ROOT/sua_exploration/scripts/verify_t4_m30_experiment_a_r10_authorization.py" \
    --claim

cells=()
for arm in z4 ph4 ac4 mb4 b4 ls4; do
    for seed in 42 43 44; do
        cells+=("$arm:$seed")
    done
done
[[ "${#cells[@]}" -eq 18 ]]

for index in "${!cells[@]}"; do
    IFS=: read -r ARM SEED <<<"${cells[$index]}"
    GPU=$((index % 2))
    LOG="$STATUS/${ARM}_s${SEED}.log"
    RECORD="$STATUS/${ARM}_s${SEED}.json"
    (
        started_at="$(date --iso-8601=seconds)"
        set +e
        ARM="$ARM" SEED="$SEED" GPU="$GPU" bash "$RUNNER" >"$LOG" 2>&1
        exit_code=$?
        set -e
        ended_at="$(date --iso-8601=seconds)"
        RESULT="$RUN/${ARM}_s${SEED}.json"
        META="$ROOT/sua_exploration/checkpoints/sua_t4_m30_component_attribution_v10_${ARM}_dandi688_co_s${SEED}/run_metadata.json"
        COST="${META%/*}/post_run_cost_receipt.json"
        "$PY" - "$RECORD" "$ARM" "$SEED" "$GPU" "$started_at" "$ended_at" \
            "$exit_code" "$RESULT" "$META" "$COST" <<'PY'
import hashlib
import json
import os
import sys


def sha256_if_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


record, arm, seed, gpu, started, ended, code, result, metadata, cost = sys.argv[1:]
code_int = int(code)
payload = {
    "arm": arm,
    "seed": int(seed),
    "gpu": int(gpu),
    "started_at": started,
    "ended_at": ended,
    "exit_code": code_int,
    "status": "completed" if code_int == 0 else "failed",
    "result_sha256": sha256_if_file(result),
    "metadata_sha256": sha256_if_file(metadata),
    "cost_sha256": sha256_if_file(cost),
}
with open(record, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
        exit "$exit_code"
    ) &
    if (((index + 1) % 2 == 0)); then
        wait
    fi
done
wait

# Fail closed before aggregation unless every frozen cell completed successfully.
for cell in "${cells[@]}"; do
    IFS=: read -r arm seed <<<"$cell"
    "$PY" - "$STATUS/${arm}_s${seed}.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
assert record["status"] == "completed"
assert record["exit_code"] == 0
assert record["result_sha256"]
assert record["metadata_sha256"]
assert record["cost_sha256"]
PY
done

"$PY" "$ROOT/sua_exploration/scripts/aggregate_t4_m30_experiment_a_r10.py" \
    --result-dir "$RUN" \
    --reference-dir "$ROOT/sua_exploration/results/sua_spint_t4_mainline_fp32_v1" \
    --status-dir "$STATUS" \
    --out "$RUN/aggregate_r10.json"

