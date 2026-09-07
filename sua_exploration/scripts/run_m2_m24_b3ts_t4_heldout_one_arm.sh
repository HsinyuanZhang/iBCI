#!/usr/bin/env bash
set -euo pipefail
g=""; gpu=""; mode=dry-run
while [[ $# -gt 0 ]]; do case "$1" in --group) g="$2";shift 2;;--gpu)gpu="$2";shift 2;;--launch)mode=launch;shift;;--dry-run)shift;;*)exit 2;;esac;done
[[ "$g" == t4 || "$g" == b3ts ]] && [[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
r="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."&&pwd)"; s="$r/streaming_calibration_exp"; p="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"; x=m2_m24_b3ts_t4_v1; a="$r/sua_exploration/results/$x/aggregate_source.json"
receipt="$r/sua_exploration/results/$x/protocol_stage0_and_conditional_expansion.json"; receipt_sha=5779aaee90f1a4050b135803ef6a84bd2b556330ef00ed171ca8f071c6e2f286
[[ -f "$receipt" && "$(sha256sum "$receipt"|awk '{print $1}')" == "$receipt_sha" ]] || { echo "frozen protocol receipt missing or changed" >&2; exit 1; }
read -r c h < <("$p" - "$a" "$g" <<'PY'
import json,sys
z=json.load(open(sys.argv[1]));
if z.get('formal_heldout_evaluated') or not z.get('gate',{}).get('pass'): raise SystemExit('source gate')
k=z['arms'][sys.argv[2]]['checkpoint'];print(k['path'],k['sha256'])
PY
)
[[ -f "$c" && "$(sha256sum "$c"|awk '{print $1}')" == "$h" ]]||exit 1
[[ "$g" == t4 ]]&&e=b3s_t4_m2_m24_b3ts_matched_loso_internal||e=b3ts_t4_m2_m24_loso_internal;i="${x}_heldout_${g}_m2"
compgen -G "$s/outputs/streaming_calibration/${i}_f1_s42_*">/dev/null&&exit 1
cmd=("$p" src/train.py "experiment=$e" data.loso_fold=1 seed=42 run_id="$i" train=false test=true optimized_metric=null ckpt_path="$c" data.include_heldout_in_fit=false data.include_heldout_in_test=true data.query_start_trial=24 data.random_calibration=false trainer.accelerator=gpu trainer.devices=1 require_baseline_validation=false)
[[ "$mode" == dry-run ]]&&{ printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu";printf '%q ' "${cmd[@]}";printf '\n';exit 0;}
busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader|sed '/No running/d;/^$/d');[[ -z "$busy" ]]||exit 1
mkdir -p "$r/sua_exploration/results/$x/logs";l="$r/sua_exploration/results/$x/logs/heldout_${g}_gpu${gpu}.log"
{ (cd "$s"&&CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"); art=$(find "$s/outputs/streaming_calibration" -maxdepth 1 -type d -name "${i}_f1_s42_*" -print -quit); [[ -n "$art" ]]; "$p" "$r/sua_exploration/scripts/write_m2_m24_b3ts_t4_heldout_provenance.py" --artifact "$art" --group "$g"; } >"$l" 2>&1
