#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT"; PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"; RUN="$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v4"; STATUS="$RUN/cell_status"
mkdir "$RUN" || { echo 'r4 scheduler reservation collision' >&2; exit 1; }; mkdir "$STATUS"
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_experiment_a_r4_authorization.py" --claim
cells=(); for a in z4 ph4 ac4 mb4 b4 ls4; do for s in 42 43 44; do cells+=("$a:$s"); done; done
for i in "${!cells[@]}"; do IFS=: read -r ARM SEED <<<"${cells[$i]}"; GPU=$((i%2)); LOG="$STATUS/${ARM}_s${SEED}.log"; RECEIPT="$STATUS/${ARM}_s${SEED}.json"; (
 start="$(date --iso-8601=seconds)"; set +e; ARM="$ARM" SEED="$SEED" GPU="$GPU" "$ROOT/sua_exploration/scripts/run_t4_m30_experiment_a_r4_one_cell.sh" >"$LOG" 2>&1; code=$?; end="$(date --iso-8601=seconds)"
 "$PY" -c 'import json,sys;open(sys.argv[1],"x").write(json.dumps({"arm":sys.argv[2],"seed":int(sys.argv[3]),"gpu":int(sys.argv[4]),"started_at":sys.argv[5],"ended_at":sys.argv[6],"exit_code":int(sys.argv[7]),"status":"completed" if int(sys.argv[7])==0 else "failed"},sort_keys=True)+"\n")' "$RECEIPT" "$ARM" "$SEED" "$GPU" "$start" "$end" "$code"; exit "$code"
 ) & if (( (i+1)%2==0 )); then wait; fi; done; wait
for c in "${cells[@]}";do IFS=: read -r a s<<<"$c"; "$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="completed" and d["exit_code"]==0' "$STATUS/${a}_s${s}.json";done
