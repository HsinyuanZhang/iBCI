#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_fixed_slot_router_followup.sh --initial-pilot-id ID [--sharp-pilot-id ID] [--dry-run|--launch]

Runs spike-only diagnostics for a completed initial fixed-slot pilot and starts
the predeclared K=32 temperature-0.1 follow-up only when the mean route entropy
is at least 0.95. This script never accesses formal held-out test sessions.
EOF
}

mode="dry-run"
initial_pilot_id=""
sharp_pilot_id="fixed_slot_router_sharp_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) mode="dry-run"; shift ;;
    --launch) mode="launch"; shift ;;
    --initial-pilot-id) initial_pilot_id="$2"; shift 2 ;;
    --sharp-pilot-id) sharp_pilot_id="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$initial_pilot_id" ]] || { echo "--initial-pilot-id is required" >&2; exit 2; }

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
initial_results="$root_dir/sua_exploration/results/$initial_pilot_id"
sharp_results="$root_dir/sua_exploration/results/$sharp_pilot_id"
teacher_ckpt="$root_dir/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
data_dir="$root_dir/sua_exploration/data/dandi_000688/sub-C"

run_diagnostic() {
  local gpu="$1" slot_count="$2" seed="$3"
  local run_name="${initial_pilot_id}_b3_fsrk${slot_count}_soft_s${seed}"
  local metadata="$root_dir/sua_exploration/checkpoints/$run_name/run_metadata.json"
  local output="$initial_results/router_diagnostic_k${slot_count}_soft_s${seed}.json"
  local checkpoint
  checkpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best_checkpoint"])' "$metadata")"
  "$python_bin" "$root_dir/sua_exploration/scripts/diagnose_fixed_slot_router.py" \
    --ckpt "$checkpoint" \
    --teacher_ckpt "$teacher_ckpt" \
    --variant B3 \
    --data_dir "$data_dir" \
    --task CO \
    --split_counts 27,6,6 \
    --max_units_exclusive 100 \
    --calibration_n 30 \
    --pool_size 50 \
    --selection_mode first \
    --seed "$seed" \
    --out_path "$output"
}

run_cached_decode_verifier() {
  local slot_count="$1" seed="$2"
  local run_name="${initial_pilot_id}_b3_fsrk${slot_count}_soft_s${seed}"
  local metadata="$root_dir/sua_exploration/checkpoints/$run_name/run_metadata.json"
  local output="$initial_results/cached_decode_k${slot_count}_soft_s${seed}.json"
  local checkpoint
  checkpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best_checkpoint"])' "$metadata")"
  CUDA_VISIBLE_DEVICES="" "$python_bin" "$root_dir/sua_exploration/scripts/verify_fixed_slot_cached_decode.py" \
    --ckpt "$checkpoint" \
    --teacher_ckpt "$teacher_ckpt" \
    --variant B3 \
    --data_dir "$data_dir" \
    --task CO \
    --split_counts 27,6,6 \
    --max_units_exclusive 100 \
    --calibration_n 30 \
    --pool_size 50 \
    --selection_mode first \
    --windows_per_session 8 \
    --seed "$seed" \
    --out_path "$output"
}

if [[ "$mode" == "dry-run" ]]; then
  for slot_count in 32 16; do
  for seed in 42 43; do
    echo "diagnose completed K=${slot_count}, seed=${seed} with CUDA_VISIBLE_DEVICES=$((seed - 42))"
    echo "verify cached decode K=${slot_count}, seed=${seed} on CPU"
  done
  done
  echo "aggregate $initial_pilot_id; launch $sharp_pilot_id only if K32 mean entropy >= 0.95"
  exit 0
fi

[[ -f "$initial_results/aggregate.json" ]] || {
  echo "Initial pilot aggregate is missing: $initial_results/aggregate.json" >&2
  exit 1
}
[[ ! -e "$sharp_results" ]] || {
  echo "Refusing to overwrite sharp-pilot results: $sharp_results" >&2
  exit 1
}

for slot_count in 32 16; do
  for seed in 42 43; do
    gpu=$((seed - 42))
    CUDA_VISIBLE_DEVICES="$gpu" run_diagnostic "$gpu" "$slot_count" "$seed"
    run_cached_decode_verifier "$slot_count" "$seed"
  done
done
"$python_bin" "$root_dir/sua_exploration/scripts/aggregate_fixed_slot_router_pilot.py" \
  --pilot-id "$initial_pilot_id"

decision_path="$initial_results/low_temperature_followup_decision.json"
route_uniform="$($python_bin - "$initial_results" "$decision_path" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

result_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
entropies = []
for seed in (42, 43):
    payload = json.loads(
        (result_dir / f"router_diagnostic_k32_soft_s{seed}.json").read_text(encoding="utf-8")
    )
    entropies.append(payload["mean_across_sessions"]["mean_assignment_normalized_entropy"])
mean_entropy = sum(entropies) / len(entropies)
threshold = 0.95
payload = {
    "schema_version": 1,
    "created_at": datetime.now().astimezone().isoformat(),
    "purpose": "predeclared_low_temperature_followup_decision",
    "initial_pilot_id": result_dir.name,
    "criterion": "mean K32 per-unit normalized assignment entropy >= 0.95",
    "threshold": threshold,
    "per_seed_mean_assignment_normalized_entropy": {"42": entropies[0], "43": entropies[1]},
    "mean_assignment_normalized_entropy": mean_entropy,
    "launch_low_temperature_followup": mean_entropy >= threshold,
    "no_test_files_accessed": True,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("true" if payload["launch_low_temperature_followup"] else "false")
PY
)"

if [[ "$route_uniform" != "true" ]]; then
  echo "Low-temperature follow-up not launched; see $decision_path"
else
  bash "$root_dir/sua_exploration/scripts/run_fixed_slot_router_sharp_pilot.sh" \
    --launch --wait --pilot-id "$sharp_pilot_id"
fi

"$python_bin" "$root_dir/sua_exploration/scripts/write_fixed_slot_router_report.py" \
  --initial-pilot-id "$initial_pilot_id" --sharp-pilot-id "$sharp_pilot_id"
