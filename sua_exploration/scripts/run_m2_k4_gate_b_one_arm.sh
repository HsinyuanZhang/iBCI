#!/usr/bin/env bash
# Launch exactly one frozen M2 Gate-B K4/KS4 arm.  It never opens held-out data.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_k4_gate_b_one_arm.sh --group k4|ks4 --gpu INDEX [--launch|--dry-run]

The only legal cell is M2 internal LOSO fold1/seed42 with first-33 raw K4
calibration.  Existing native_mua_t4_v1 F0/T4/TS4 artifacts are references,
not rerun.  This launcher refuses an occupied GPU, an overwritten output, a
changed Gate-A artifact, or a missing reference artifact.
EOF
}

MODE=dry-run
GROUP=""
GPU=""
SCREEN_ID="m2_k4_gate_b_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch) MODE=launch; shift ;;
    --dry-run) MODE=dry-run; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --screen-id) SCREEN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == k4 || "$GROUP" == ks4 ]] || { echo "--group must be k4 or ks4" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a numeric GPU index" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MUA_ROOT="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
GATE_A="$ROOT/sua_exploration/results/general_carrier_proxy_v1/audit_m2_heldin_v2.json"
GATE_A_SHA="8767d7ce0fc852401273806a61492a607a2064a4546bbd1c21e25ce344cf4ec9"
RESULTS="$ROOT/sua_exploration/results/$SCREEN_ID"
LOG_DIR="$RESULTS/logs"
EXPERIMENT="b3s_${GROUP}_m2_loso_internal"
RUN_ID="${SCREEN_ID}_${GROUP}_m2"

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$GATE_A" ]] || { echo "missing required Gate-A artifact: $GATE_A" >&2; exit 1; }
[[ "$(sha256sum "$GATE_A" | awk '{print $1}')" == "$GATE_A_SHA" ]] || {
  echo "Gate-A SHA drift; refusing GPU screen" >&2; exit 1;
}
"$PYTHON_BIN" - "$GATE_A" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("formal_heldout_evaluated") is not False:
    raise SystemExit("Gate-A must be development-only")
if payload.get("summary", {}).get("stage1_candidate") is not True:
    raise SystemExit("Gate-A stage1_candidate is not true; refusing GPU screen")
PY
for group in f0 t4 ts4; do
  count=$(find "$MUA_ROOT/outputs/streaming_calibration" -maxdepth 1 -type d -name "native_mua_t4_v1_${group}_m2_f1_s42_*" | wc -l)
  [[ "$count" == 1 ]] || { echo "expected one native_mua_t4_v1 $group M2 f1/s42 reference, found $count" >&2; exit 1; }
done
if compgen -G "$MUA_ROOT/outputs/streaming_calibration/${RUN_ID}_f1_s42_*" >/dev/null; then
  echo "refusing to overwrite existing output for $RUN_ID" >&2; exit 1
fi

command=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" data.loso_fold=1 seed=42
  "run_id=$RUN_ID" data.include_heldout_in_fit=false data.include_heldout_in_test=false
  trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false
)
if [[ "$MODE" == dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi

# The primary B3T queue owns GPU1.  Do not silently queue behind or compete with
# another process: the caller must explicitly retry after a safe release.
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d')
if [[ -n "$busy" ]]; then
  echo "GPU $GPU has active compute processes; refusing to compete: $busy" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
log="$LOG_DIR/${GROUP}_f1_s42_gpu${GPU}.log"
{
  echo "[$(date -Is)] Gate-B start group=$GROUP gpu=$GPU"
  echo "gate_a_sha256=$GATE_A_SHA"
  echo "exposure_note=K4 raw-full-trial blocks; legacy T4 valid_prefix=max_trial_length=100"
  (cd "$MUA_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${SCREEN_ID}_${GROUP}_gpu${GPU}" "${command[@]}")
  echo "[$(date -Is)] Gate-B complete group=$GROUP gpu=$GPU"
} >"$log" 2>&1
