#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
RUN="$ROOT/sua_exploration/results/sua_t4_m30_ac4_rs4_v1"
STATUS="$RUN/cell_status"
RUNNER="$ROOT/sua_exploration/scripts/run_t4_m30_ac4_rs4_v1_one_cell.sh"

[[ ! -e "$RUN" ]] || { echo "AC4-RS4 result root already exists" >&2; exit 1; }
mkdir "$RUN"
mkdir "$STATUS"
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_ac4_rs4_v1.py" --claim

run_cell() {
    local seed="$1" gpu="$2" log="$STATUS/ac4_rs4_s${seed}.log"
    local record="$STATUS/ac4_rs4_s${seed}.json"
    (
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
import hashlib,json,os,sys
def h(path):
    return hashlib.sha256(open(path,'rb').read()).hexdigest() if os.path.isfile(path) else None
record,seed,gpu,started,ended,code,result,metadata,cost=sys.argv[1:]
payload={'arm':'ac4_rs4','seed':int(seed),'gpu':int(gpu),'started_at':started,'ended_at':ended,'exit_code':int(code),'status':'completed' if int(code)==0 else 'failed','result_sha256':h(result),'metadata_sha256':h(metadata),'cost_sha256':h(cost)}
with open(record,'x') as f: json.dump(payload,f,sort_keys=True);f.write('\n')
PY
        exit "$code"
    ) &
}

run_cell 42 0
run_cell 43 1
wait
run_cell 44 0
wait

for seed in 42 43 44; do
    "$PY" - "$STATUS/ac4_rs4_s${seed}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['status']=='completed' and d['exit_code']==0
assert d['result_sha256'] and d['metadata_sha256'] and d['cost_sha256']
PY
done

"$PY" "$ROOT/sua_exploration/scripts/aggregate_t4_m30_ac4_rs4_v1.py" \
    --result-dir "$RUN" \
    --status-dir "$STATUS" \
    --baseline-result-dir "$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v10" \
    --baseline-aggregate "$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v10/aggregate_r11.json" \
    --out "$RUN/aggregate_v1.json"
