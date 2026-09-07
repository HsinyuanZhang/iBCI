#!/usr/bin/env bash
# Fixed, receipt-bound CI32/CI64 source-only schedule.
#
# This runner deliberately contains no metric, checkpoint, or target-data
# branch.  Its only scheduling decision is the predeclared date partition:
# GPU 0 owns 19250108 -> 19250115 -> 19250120; GPU 1 owns 19250113 ->
# 19250119.  Each date runs the complete frozen five-arm grid serially; an
# error stops its partition before another arm or date can start.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "usage: $0 {0|1}" >&2
  exit 64
fi

partition="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
receipt="$root/pilot_artifacts/h1_carrierid_date_lodo_ci/H1_CARRIERID_DATE_LODO_CI64_SLODO_LAUNCH_RECEIPT_v1.json"
authority="$root/pilot_artifacts/h1_carrierid_date_lodo_ci/H1_CARRIERID_DATE_LODO_CI64_SLODO_EXECUTION_AUTHORITY_SUPPLEMENT_v1.json"
arms=(CI32-FULL CI64-FULL CI64-C0 CI64-LS CI64-RS)

if [[ "$partition" == "0" ]]; then
  dates=(19250108 19250115 19250120)
else
  dates=(19250113 19250119)
fi

CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 "$python_executable" \
  "$root/scripts/h1_carrierid_date_lodo_ci_launch_authority.py" --verify "$authority" >/dev/null

for date in "${dates[@]}"; do
  for arm in "${arms[@]}"; do
    run_dir="$root/pilot_artifacts/h1_carrierid_date_lodo_ci/gpu_runs/$date/$arm"
    CUDA_VISIBLE_DEVICES="$partition" PYTHONNOUSERSITE=1 "$python_executable" \
      "$root/scripts/h1_carrierid_date_lodo_ci_source_executor.py" \
      --launch-receipt "$receipt" \
      --outer-date "$date" \
      --arm "$arm" \
      --run-dir "$run_dir" \
      --python-executable "$python_executable" \
      --execute-source-training
  done
done
