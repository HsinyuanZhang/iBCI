#!/usr/bin/env bash
# Frozen held-in-only source for the M24 B3TS+T4 exploratory program.
set -euo pipefail
GROUP=""; GPU=""; MODE=dry-run
while [[ $# -gt 0 ]]; do case "$1" in --group) GROUP="$2"; shift 2;; --gpu) GPU="$2"; shift 2;; --launch) MODE=launch; shift;; --dry-run) MODE=dry-run; shift;; *) echo "usage: $0 --group t4|b3ts --gpu N [--launch]" >&2; exit 2;; esac; done
[[ "$GROUP" == t4 || "$GROUP" == b3ts ]] || { echo "invalid group" >&2; exit 2; }; [[ "$GPU" =~ ^[0-9]+$ ]] || exit 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; SCE="$ROOT/streaming_calibration_exp"; PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN=m2_m24_b3ts_t4_v1; FEAS="$ROOT/sua_exploration/results/$SCREEN/feasibility.json"; FEAS_SHA=6fbfea441acb0e9f98b75fae40b48bbf757f939eb4b4671cc10f69ddd7024171
[[ -f "$FEAS" && "$(sha256sum "$FEAS" | awk '{print $1}')" == "$FEAS_SHA" ]] || { echo "missing/drifted feasibility receipt" >&2; exit 1; }
if [[ "$GROUP" == t4 ]]; then EXP=b3s_t4_m2_m24_b3ts_matched_loso_internal; else EXP=b3ts_t4_m2_m24_loso_internal; fi
RID="${SCREEN}_source_${GROUP}_m2"; compgen -G "$SCE/outputs/streaming_calibration/${RID}_f1_s42_*" >/dev/null && { echo "refusing overwrite" >&2; exit 1; }
CMD=("$PY" src/train.py "experiment=$EXP" data.loso_fold=1 seed=42 run_id="$RID" data.include_heldout_in_fit=false data.include_heldout_in_test=false data.query_start_trial=0 data.random_calibration=false trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false)
if [[ "$MODE" == dry-run ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/No running/d;/^$/d'); [[ -z "$busy" ]] || { echo "GPU busy: $busy" >&2; exit 1; }
mkdir -p "$ROOT/sua_exploration/results/$SCREEN/logs"; log="$ROOT/sua_exploration/results/$SCREEN/logs/source_${GROUP}_gpu${GPU}.log"
{ echo "feasibility_sha256=$FEAS_SHA"; echo "scope=held-in-only M24 source; no heldout"; (cd "$SCE" && CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/${SCREEN}_${GROUP}" "${CMD[@]}"); echo "source complete"; } >"$log" 2>&1
