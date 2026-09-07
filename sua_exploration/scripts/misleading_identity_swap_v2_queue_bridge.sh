#!/usr/bin/env bash
# Post-training bridge: validates the four immutable terminal receipts and prints
# the explicit target-authority/scoring matrix. It never launches by default.
set -euo pipefail
ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
SCORER="$SUA/scripts/misleading_identity_swap_v2_scorer.py"
SOURCE_LINEAGE="$SUA/results/misleading_identity_swap_v2_source_authority_dev/strict27_m30_source_lineage_v3.json"
OFFICIAL="$SUA/results/misleading_identity_swap_v2_source_authority_dev/official_cpu_preflight_v3.json"
SCORE_ROOT="$SUA/results/misleading_identity_swap_v2_stage_p_successor_v3"
CELL_ROOT="$SUA/checkpoints/misleading_identity_swap_v2_stage_p_successor_v3_seed42"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUA:$ROOT/streaming_calibration_exp${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:---dry-run}"
[[ "$MODE" == "--dry-run" ]] || { echo "queue bridge is plan-only until separate independent ROOT GO" >&2; exit 3; }
"$PY" "$SUA/scripts/misleading_identity_swap_v2_queue_verify.py" \
  --official-preflight "$OFFICIAL" --source-lineage "$SOURCE_LINEAGE"
for domain in within_subject external_subject_M; do
  authority="$SCORE_ROOT/${domain}_target_authority.json"
  lineage="$SCORE_ROOT/${domain}_target_lineage.json"
  printf '%q ' "$PY" "$SCORER" --domain "$domain" --source-lineage "$SOURCE_LINEAGE" \
    --official-preflight "$OFFICIAL" --target-authority "$authority" --target-lineage "$lineage" --prepare-target-authority
  echo
  for cell in clean_z4 clean_t4 swap_z4 swap_t4; do
    terminal="$CELL_ROOT/${cell}/terminal_receipt.json"
    for mode in clean swapped_diagnostic; do
      out="$SCORE_ROOT/${cell}__${domain}__${mode}.json"
      printf 'CUDA_VISIBLE_DEVICES=GPU_INDEX '
      printf '%q ' "$PY" "$SCORER" --cell "$cell" --domain "$domain" --evaluation-input-mode "$mode" \
        --source-lineage "$SOURCE_LINEAGE" --target-authority "$authority" --target-lineage "$lineage" \
        --official-preflight "$OFFICIAL" --cell-terminal-receipt "$terminal" --out-path "$out" --launch
      echo
    done
  done
done
