#!/usr/bin/env bash
# Launch exactly one frozen, held-in-only M2 M24 source-training arm.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_m2_m24_internal_one_arm.sh --group f0|t4|k4|ks4 --gpu INDEX [--launch|--dry-run]

The sole legal source cell is M2 LOSO fold1/seed42, chronological first-24
calibration trials, 12 epochs, held-in minival checkpoint selection, and no
held-out loading.  It is a prerequisite to, not an evaluation of, the later
local held-out chronological-disjoint replay.
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
RECEIPT="$ROOT/sua_exploration/results/general_carrier_proxy_v1/audit_m2_m24_heldin_v2.json"
RECEIPT_SHA="84af6055db3840b7f68bfeb74b61f9d028311343211eadba67b6d5a33345bdd3"
CORRECTION="$ROOT/sua_exploration/results/general_carrier_proxy_v1/audit_m2_m24_correction_receipt_v1.json"
CORRECTION_SHA="c7a038da879621d9352199b361dd1ebd980a9ca76e79855a61041a2e38ed15e2"
SCREEN_ID="m2_m24_disjoint_source_v1"
RUN_ID="${SCREEN_ID}_${GROUP}_m2"
case "$GROUP" in
  f0) EXPERIMENT=b3_native_mua_f0_m2_m24_loso_internal ;;
  t4) EXPERIMENT=b3s_t4_m2_m24_loso_internal ;;
  k4) EXPERIMENT=b3s_k4_m2_m24_loso_internal ;;
  ks4) EXPERIMENT=b3s_ks4_m2_m24_loso_internal ;;
esac

[[ -x "$PYTHON_BIN" && -f "$RECEIPT" && -f "$CORRECTION" ]] || { echo "missing Python, M24 v2 receipt, or correction receipt" >&2; exit 1; }
[[ "$(sha256sum "$RECEIPT" | awk '{print $1}')" == "$RECEIPT_SHA" ]] || { echo "M24 qualification receipt SHA drift" >&2; exit 1; }
[[ "$(sha256sum "$CORRECTION" | awk '{print $1}')" == "$CORRECTION_SHA" ]] || { echo "M24 correction receipt SHA drift" >&2; exit 1; }
"$PYTHON_BIN" - "$RECEIPT" "$CORRECTION" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8")); correction = json.load(open(sys.argv[2], encoding="utf-8"))
s = p.get("summary", {})
if p.get("formal_heldout_evaluated") is not False or p.get("protocol", {}).get("name") != "m24" or p.get("protocol", {}).get("fit_trials") != "0:12" or p.get("protocol", {}).get("prediction_trials") != "12:24":
    raise SystemExit("receipt scope/protocol mismatch")
if not (s.get("all_7_sessions_beat_all_100_w_only_nulls") and s.get("kreg_beats_baseline_sessions") == 7 and s.get("kreg_beats_w_only_median_sessions") == 7 and s.get("mean_kreg_over_baseline_ratio", 1.0) <= .95 and s.get("mean_kreg_over_w_only_null_median_ratio", 1.0) <= .95 and min(s.get("W_A_B_correlations", [0.0])) > .5 and s.get("wilcoxon_two_sided_kreg_vs_baseline_mse", 1.0) <= .05 and s.get("wilcoxon_two_sided_kreg_vs_w_only_median_mse", 1.0) <= .05):
    raise SystemExit("M24 qualification receipt does not pass frozen consistency conditions")
if not correction.get("numeric_rows_verified_identical"):
    raise SystemExit("M24 v1/v2 correction receipt is incomplete")
PY
if compgen -G "$MUA_ROOT/outputs/streaming_calibration/${RUN_ID}_f1_s42_*" >/dev/null; then
  echo "refusing to overwrite existing source artifact for $RUN_ID" >&2; exit 1
fi
command=(
  "$PYTHON_BIN" src/train.py "experiment=$EXPERIMENT" data.loso_fold=1 seed=42 run_id="$RUN_ID"
  data.include_heldout_in_fit=false data.include_heldout_in_test=false data.query_start_trial=0
  trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false
)
if [[ "$MODE" == dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d')
if [[ -n "$busy" ]]; then
  echo "GPU $GPU has active compute processes; refusing to compete: $busy" >&2; exit 1
fi
mkdir -p "$ROOT/sua_exploration/results/$SCREEN_ID/logs"
log="$ROOT/sua_exploration/results/$SCREEN_ID/logs/${GROUP}_f1_s42_gpu${GPU}.log"
{
  echo "[$(date -Is)] M24 source start group=$GROUP gpu=$GPU"
  echo "qualification_receipt_sha256=$RECEIPT_SHA"
  echo "scope=held-in-only; no held-out loading; minival selects best checkpoint"
  (cd "$MUA_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${SCREEN_ID}_${GROUP}_gpu${GPU}" "${command[@]}")
  echo "[$(date -Is)] M24 source complete group=$GROUP gpu=$GPU"
} >"$log" 2>&1
