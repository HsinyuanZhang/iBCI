#!/usr/bin/env bash
# Two-GPU scheduler; inert unless a separately sealed root authorization is supplied.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT"; PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"; AUTH_RECEIPT="${AUTH_RECEIPT:?AUTH_RECEIPT required}"; AUTH_SHA256="${AUTH_SHA256:?AUTH_SHA256 required}"
PRELAUNCH_RECEIPT="${PRELAUNCH_RECEIPT:?PRELAUNCH_RECEIPT required}"; PRELAUNCH_SHA256="${PRELAUNCH_SHA256:?PRELAUNCH_SHA256 required}"
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_experiment_a_v3_authorization.py" --authorization "$AUTH_RECEIPT" --authorization-sha256 "$AUTH_SHA256" --prelaunch "$PRELAUNCH_RECEIPT" --prelaunch-sha256 "$PRELAUNCH_SHA256"
cells=(); for a in z4 ph4 ac4 mb4 b4 ls4; do for s in 42 43 44; do cells+=("$a:$s"); done; done
STATUS_DIR="$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v3/cell_status"; mkdir -p "$STATUS_DIR"
for i in "${!cells[@]}"; do
  # GPU=$((i % 2)); the root verifier requires authorization.gpu_launch_authorized.
  IFS=: read -r ARM SEED <<<"${cells[$i]}"; GPU=$((i % 2)); STATUS="$STATUS_DIR/${ARM}_s${SEED}.json"; LOG="$STATUS_DIR/${ARM}_s${SEED}.log"
  [[ ! -e "$STATUS" && ! -e "$LOG" ]] || { echo "write-once cell status collision: $STATUS" >&2; exit 1; }
  (
    START="$(date --iso-8601=seconds)"; set +e
    ARM="$ARM" SEED="$SEED" GPU="$GPU" AUTH_RECEIPT="$AUTH_RECEIPT" AUTH_SHA256="$AUTH_SHA256" PRELAUNCH_RECEIPT="$PRELAUNCH_RECEIPT" PRELAUNCH_SHA256="$PRELAUNCH_SHA256" "$ROOT/sua_exploration/scripts/run_t4_m30_experiment_a_one_cell.sh" >"$LOG" 2>&1
    CODE=$?; END="$(date --iso-8601=seconds)"
    "$PY" -c 'import json,sys; open(sys.argv[1],"x").write(json.dumps({"schema_version":1,"arm":sys.argv[2],"seed":int(sys.argv[3]),"gpu":int(sys.argv[4]),"started_at":sys.argv[5],"ended_at":sys.argv[6],"exit_code":int(sys.argv[7]),"status":"completed" if int(sys.argv[7])==0 else "failed"},indent=2)+"\n")' "$STATUS" "$ARM" "$SEED" "$GPU" "$START" "$END" "$CODE"
    exit "$CODE"
  ) &
  if (( (i + 1) % 2 == 0 )); then wait; fi
done
wait
for CELL in "${cells[@]}"; do IFS=: read -r ARM SEED <<<"$CELL"; "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="completed" and d["exit_code"]==0' "$STATUS_DIR/${ARM}_s${SEED}.json"; done
"$PY" "$ROOT/sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py" --result-dir "$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v3" --reference-dir "$ROOT/sua_exploration/results/sua_spint_t4_mainline_fp32_v1" --prelaunch "$PRELAUNCH_RECEIPT" --prelaunch-sha256 "$PRELAUNCH_SHA256" --authorization "$AUTH_RECEIPT" --authorization-sha256 "$AUTH_SHA256" --out "${AGGREGATE_OUT:-$ROOT/sua_exploration/results/sua_t4_m30_component_attribution_v3/aggregate_v3.json}"
