#!/usr/bin/env bash
# Launch one preregistered replication cell only after the primary Gate-B pass.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_k4_gate_b_extension_one_arm.sh --cell f1s43|f2s42 --group k4|ks4 --gpu INDEX [--launch|--dry-run]

This launcher has exactly two allowable replication cells: M2 LOSO f1/seed43
and f2/seed42.  It fail-closes unless the primary f1/seed42 strict aggregate
has all three comparisons pass.  F0 and T4 are frozen reference artifacts;
the launcher never evaluates held-out data and refuses to overwrite output.
EOF
}

MODE=dry-run
CELL=""
GROUP=""
GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch) MODE=launch; shift ;;
    --dry-run) MODE=dry-run; shift ;;
    --cell) CELL="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$CELL" == f1s43 || "$CELL" == f2s42 ]] || { echo "--cell must be f1s43 or f2s42" >&2; exit 2; }
[[ "$GROUP" == k4 || "$GROUP" == ks4 ]] || { echo "--group must be k4 or ks4" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a numeric GPU index" >&2; exit 2; }

case "$CELL" in
  f1s43) FOLD=1; SEED=43 ;;
  f2s42) FOLD=2; SEED=42 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MUA_ROOT="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
PRIMARY="$ROOT/sua_exploration/results/m2_k4_gate_b_v1/aggregate_seed42.json"
SCREEN_ID="m2_k4_gate_b_extension_v1"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
RUN_ID="${SCREEN_ID}_${CELL}_${GROUP}_m2"
EXPERIMENT="b3s_${GROUP}_m2_loso_internal"

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$PRIMARY" ]] || { echo "primary strict aggregate missing; extension is not authorized" >&2; exit 1; }
"$PYTHON_BIN" - "$PRIMARY" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("formal_heldout_evaluated") is not False:
    raise SystemExit("primary aggregate must be internal-only")
if payload.get("cell") != {"task": "m2", "fold": 1, "seed": 42, "M": 33, "network": "fresh_B3S_side_dim4"}:
    raise SystemExit("primary aggregate cell contract mismatch")
if payload.get("gate", {}).get("all_three_pass") is not True:
    raise SystemExit("primary all-three Gate-B pass required before extensions")
PY
for reference in f0 t4; do
  count=$(find "$MUA_ROOT/outputs/streaming_calibration" -maxdepth 1 -type d -name "native_mua_t4_v1_${reference}_m2_f${FOLD}_s${SEED}_*" | wc -l)
  [[ "$count" == 1 ]] || { echo "expected one frozen $reference reference for f$FOLD/s$SEED, found $count" >&2; exit 1; }
done
if compgen -G "$MUA_ROOT/outputs/streaming_calibration/${RUN_ID}_f${FOLD}_s${SEED}_*" >/dev/null; then
  echo "refusing to overwrite existing output for $RUN_ID" >&2; exit 1
fi

command=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" "data.loso_fold=$FOLD" "seed=$SEED"
  "run_id=$RUN_ID" data.include_heldout_in_fit=false data.include_heldout_in_test=false
  trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false
)
if [[ "$MODE" == dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi

busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d')
if [[ -n "$busy" ]]; then
  echo "GPU $GPU has active compute processes; refusing to compete: $busy" >&2
  exit 1
fi
mkdir -p "$RESULTS/logs"
log="$RESULTS/logs/${CELL}_${GROUP}_f${FOLD}_s${SEED}_gpu${GPU}.log"
{
  echo "[$(date -Is)] Gate-B extension start cell=$CELL group=$GROUP gpu=$GPU"
  echo "primary_aggregate=$PRIMARY"
  echo "exposure_note=K4 raw-full-trial blocks; legacy T4 valid_prefix=max_trial_length=100"
  (cd "$MUA_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${SCREEN_ID}_${CELL}_${GROUP}_gpu${GPU}" "${command[@]}")
  echo "[$(date -Is)] Gate-B extension complete cell=$CELL group=$GROUP gpu=$GPU"
} >"$log" 2>&1
