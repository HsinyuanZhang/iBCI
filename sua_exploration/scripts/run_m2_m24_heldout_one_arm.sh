#!/usr/bin/env bash
# Run exactly one test-only M2 M24 chronological-disjoint held-out replay.
# This script is deliberately non-self-authorising: it can run only after the
# four source arms passed the internal M24 gate written by the strict aggregate.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_m24_heldout_one_arm.sh --group f0|t4|k4|ks4 --gpu INDEX [--launch|--dry-run]

Uses the frozen source checkpoint selected only on the held-in minival set.
The held-out session supplies the chronological first 24 trials only for local
calibration; scoring begins only at windows whose complete 50-bin history lies
after that support boundary.  It performs no fit, no backward pass, and no
held-out checkpoint selection.
EOF
}

MODE=dry-run
GROUP=""
GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch) MODE=launch; shift ;;
    --dry-run) MODE=dry-run; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GROUP" == f0 || "$GROUP" == t4 || "$GROUP" == k4 || "$GROUP" == ks4 ]] || { echo "invalid --group" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be numeric" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MUA_ROOT="$ROOT/streaming_calibration_exp"
PYTHON_BIN="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="m2_m24_disjoint_source_v1"
HELDOUT_SCREEN="m2_m24_disjoint_heldout_v1"
INTERNAL="$ROOT/sua_exploration/results/$SCREEN/aggregate_internal.json"
RUN_ID="${HELDOUT_SCREEN}_${GROUP}_m2"
case "$GROUP" in
  f0) EXPERIMENT=b3_native_mua_f0_m2_m24_loso_internal ;;
  t4) EXPERIMENT=b3s_t4_m2_m24_loso_internal ;;
  k4) EXPERIMENT=b3s_k4_m2_m24_loso_internal ;;
  ks4) EXPERIMENT=b3s_ks4_m2_m24_loso_internal ;;
esac

[[ -x "$PYTHON_BIN" && -f "$INTERNAL" ]] || { echo "missing Python or passed internal aggregate" >&2; exit 1; }
read -r SOURCE_CKPT SOURCE_CKPT_SHA < <("$PYTHON_BIN" - "$INTERNAL" "$GROUP" <<'PY'
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
cell = record.get("cell", {})
if record.get("formal_heldout_evaluated") is not False:
    raise SystemExit("internal record unexpectedly accessed held-out data")
if not record.get("gate", {}).get("all_three_pass"):
    raise SystemExit("internal M24 source gate did not pass; held-out replay is prohibited")
if (cell.get("task"), cell.get("fold"), cell.get("seed"), cell.get("M")) != ("m2", 1, 42, 24):
    raise SystemExit("internal aggregate is not the frozen M2/fold1/seed42/M24 source cell")
arm = record.get("arms", {}).get(sys.argv[2], {})
ckpt = arm.get("checkpoint", {})
if not ckpt.get("path") or not ckpt.get("sha256"):
    raise SystemExit("internal aggregate lacks a source checkpoint receipt")
print(ckpt["path"], ckpt["sha256"])
PY
)
[[ -f "$SOURCE_CKPT" ]] || { echo "source checkpoint from internal receipt is missing: $SOURCE_CKPT" >&2; exit 1; }
[[ "$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')" == "$SOURCE_CKPT_SHA" ]] || { echo "source checkpoint SHA drift before held-out launch" >&2; exit 1; }
if compgen -G "$MUA_ROOT/outputs/streaming_calibration/${RUN_ID}_f1_s42_*" >/dev/null; then
  echo "refusing to overwrite existing held-out replay artifact for $RUN_ID" >&2; exit 1
fi

command=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" data.loso_fold=1 seed=42 run_id="$RUN_ID"
  train=false test=true optimized_metric=null ckpt_path="$SOURCE_CKPT"
  data.include_heldout_in_fit=false data.include_heldout_in_test=true data.query_start_trial=24
  data.random_calibration=false trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false
)
if [[ "$MODE" == dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d')
if [[ -n "$busy" ]]; then
  echo "GPU $GPU has active compute processes; refusing to compete: $busy" >&2; exit 1
fi
mkdir -p "$ROOT/sua_exploration/results/$HELDOUT_SCREEN/logs"
log="$ROOT/sua_exploration/results/$HELDOUT_SCREEN/logs/${GROUP}_f1_s42_gpu${GPU}.log"
{
  echo "[$(date -Is)] M24 held-out test-only replay start group=$GROUP gpu=$GPU"
  echo "source_internal_aggregate=$INTERNAL"
  echo "source_checkpoint=$SOURCE_CKPT sha256=$SOURCE_CKPT_SHA"
  echo "scope=test-only; held-out support=trials[0:24]; query=full windows after trial 24"
  (cd "$MUA_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${HELDOUT_SCREEN}_${GROUP}_gpu${GPU}" "${command[@]}")
  artifact=$(find "$MUA_ROOT/outputs/streaming_calibration" -maxdepth 1 -type d -name "${RUN_ID}_f1_s42_*" -print -quit)
  [[ -n "$artifact" ]] || { echo "test-only replay produced no artifact" >&2; exit 1; }
  "$PYTHON_BIN" "$ROOT/sua_exploration/scripts/write_m2_m24_heldout_provenance.py" \
    --artifact "$artifact" --internal-aggregate "$INTERNAL" --group "$GROUP"
  echo "[$(date -Is)] M24 held-out test-only replay complete group=$GROUP"
} >"$log" 2>&1
