#!/usr/bin/env bash
# Dedicated additive Stage-P orchestrator. Default is a read-only command plan.
set -euo pipefail

ROOT=/home/xinyuan/Work_host/SPINT
SUA="$ROOT/sua_exploration"
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
BUILDER="$SUA/scripts/misleading_identity_swap_v2_build_source_authority.py"
TRAINER="$SUA/scripts/misleading_identity_swap_v2_train_cell.py"
AUTH_ROOT="$SUA/results/misleading_identity_swap_v2_source_authority_dev"
AUTHORITY="$AUTH_ROOT/strict27_m30_matching_authority_v3.json"
LINEAGE="$AUTH_ROOT/strict27_m30_source_lineage_v3.json"
INITIAL="$AUTH_ROOT/shared_initial_state_v2.pt"
OFFICIAL="$AUTH_ROOT/official_cpu_preflight_v3.json"
RESULT="$SUA/results/misleading_identity_swap_v2_stage_p_successor_v3"
CELL_ROOT="$SUA/checkpoints/misleading_identity_swap_v2_stage_p_successor_v3_seed42"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUA:$ROOT/streaming_calibration_exp${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:---dry-run}"

print_cell() {
  local cell="$1"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 %q -u %q --authority %q --lineage %q --initial-state %q --official-preflight %q --cell %q --output-dir %q --launch\n' \
    "${GPU:-GPU_INDEX}" "$PY" "$TRAINER" "$AUTHORITY" "$LINEAGE" "$INITIAL" "$OFFICIAL" "$cell" "$CELL_ROOT/$cell"
}

case "$MODE" in
  --dry-run)
    "$PY" "$BUILDER"
    echo "STATUS=DRY_RUN__LIVE_INTEGRATION_PLAN__NO_DATA_NO_GPU"
    echo "SOURCE_ONLY_AUTHORITY_COMMAND=$PY $BUILDER --execute-source-only"
    echo "DEVELOPMENT_INITIAL_STATE_COMMAND=$PY $TRAINER --authority $AUTHORITY --initial-state $INITIAL --mint-development-initial-state"
    for cell in clean_z4 clean_t4 swap_z4 swap_t4; do print_cell "$cell"; done
    echo "EPOCH_WINDOW_ONE_BASED=5,6,7,8,9,10,11,12"
    echo "TARGET_SCORING=separate queue bridge; never automatic"
    ;;
  --mint-development-initial-state)
    [[ -f "$AUTHORITY" && -f "${AUTHORITY}.sha256" ]] || { echo "missing immutable source authority" >&2; exit 2; }
    "$PY" "$TRAINER" --authority "$AUTHORITY" --initial-state "$INITIAL" --mint-development-initial-state
    ;;
  --launch-cell)
    : "${CELL:?CELL must be one of clean_z4 clean_t4 swap_z4 swap_t4}"
    : "${GPU:?GPU index is required}"
    case "$CELL" in clean_z4|clean_t4|swap_z4|swap_t4) ;; *) echo "invalid CELL" >&2; exit 2;; esac
    [[ "${SWAP_V2_STAGE_P_GPU_AUTHORIZATION:-}" == "I_AUTHORIZE_SWAP_V2_STAGE_P_GPU" ]] || { echo "explicit GPU authorization missing" >&2; exit 3; }
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TRAINER" --authority "$AUTHORITY" --lineage "$LINEAGE" \
      --initial-state "$INITIAL" --official-preflight "$OFFICIAL" --cell "$CELL" \
      --output-dir "$CELL_ROOT/$CELL" --launch
    ;;
  *) echo "usage: $0 [--dry-run|--mint-development-initial-state|--launch-cell]" >&2; exit 2;;
esac
