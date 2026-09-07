#!/usr/bin/env bash
# Complete receipts only for the one already-executed SSC-T4 held-out replay.
# It deliberately has no path for rerunning either primary arm.
set -euo pipefail

MODE=dry-run
case "${1:-}" in
  --dry-run) shift ;;
  --launch) MODE=launch; shift ;;
  *) echo "Usage: $0 [--dry-run|--launch]" >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { echo "unexpected argument: $1" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
RESULT="$ROOT/sua_exploration/results/m2_ssc_t4_v1"
OUT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"
PROTOCOL="$RESULT/heldout_protocol_receipt.json"
SOURCE_GATE="$RESULT/source_gate.json"
RECOVERY="$RESULT/heldout_testonly_postexport_recovery_receipt.json"
TEACHER="$ROOT/SPINT-main/logs/clean_teacher_runs/m2_ssc_t4_v1_clean_teacher_f1_s42_20260731_2352/checkpoints/clean_teacher_best/epoch_031.ckpt"
TEACHER_RECEIPT="$RESULT/clean_teacher_receipt.json"
REFERENCE="$RESULT/clean_spint_m24_local_heldout_reference.json"
AGGREGATE="$RESULT/aggregate_heldout.json"
PROTOCOL_SHA=c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc

[[ "$(sha256sum "$PROTOCOL" | awk '{print $1}')" == "$PROTOCOL_SHA" ]] || { echo "protocol receipt SHA drift" >&2; exit 1; }
"$PYTHON" "$ROOT/sua_exploration/scripts/validate_m2_clean_teacher_receipt.py" --checkpoint "$TEACHER" --receipt "$TEACHER_RECEIPT" >/dev/null
[[ ! -e "$AGGREGATE" ]] || { echo "aggregate already exists; salvage refuses overwrite" >&2; exit 1; }

mapfile -t ORDINARY < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_heldout_ordinary_f1_s42_*' -print | sort)
mapfile -t SSC < <(find "$OUT" -maxdepth 1 -type d -name 'm2_ssc_t4_v1_heldout_ssc_f1_s42_*' -print | sort)
[[ ${#ORDINARY[@]} == 1 && ${#SSC[@]} == 1 ]] || { echo "salvage requires exactly one existing primary artifact per arm" >&2; exit 1; }

# This validates the recovery receipt, source authorization, and all files that
# must already exist.  It intentionally never opens per-session metric values.
"$PYTHON" - "$RECOVERY" "$SOURCE_GATE" "${ORDINARY[0]}" "${SSC[0]}" <<'PY'
import json, sys
from pathlib import Path
recovery, gate_path, ordinary, ssc = map(Path, sys.argv[1:])
r = json.loads(recovery.read_text()); g = json.loads(gate_path.read_text())
if r.get('decision') != 'REUSE_EXISTING_HELDOUT_ARTIFACTS_NO_ORDINARY_OR_SSC_RERUN': raise SystemExit('invalid recovery decision')
if g.get('catastrophic_stop') is not False or g.get('next_action') != 'one_frozen_local_heldout_replay_required': raise SystemExit('source gate does not authorize one replay')
for arm, actual in [('ordinary_t4', ordinary), ('ssc_t4', ssc)]:
    declared = Path(r.get('arms', {}).get(arm, {}).get('artifact', '')).resolve()
    if declared != actual.resolve(): raise SystemExit(f'{arm}: recovery receipt artifact mismatch')
    if not r['arms'][arm].get('artifact_exported_before_nonzero_exit'): raise SystemExit(f'{arm}: no post-export recovery authorization')
    for name in ('resolved_config.yaml','split_manifest.json','checkpoint_manifest.json','metrics_per_session.csv','teacher_metadata.json','checkpoints/best.ckpt'):
        if not (actual / name).is_file(): raise SystemExit(f'{arm}: missing {name}')
PY

if [[ "$MODE" == dry-run ]]; then
  printf 'dry-run salvage: will write provenance only for existing artifacts:\n  ordinary=%s\n  ssc=%s\n' "${ORDINARY[0]}" "${SSC[0]}"
  if [[ -e "$REFERENCE" ]]; then printf 'dry-run salvage: existing clean third reference will be reused: %s\n' "$REFERENCE"; else printf 'dry-run salvage: will run exactly one predeclared clean third reference\n'; fi
  printf 'dry-run salvage: will write strict aggregate: %s\n' "$AGGREGATE"
  exit 0
fi

for spec in "${ORDINARY[0]}:ordinary_t4" "${SSC[0]}:ssc_t4"; do
  ARTIFACT="${spec%%:*}"; ARM="${spec##*:}"
  [[ ! -e "$ARTIFACT/heldout_ssc_t4_provenance.json" ]] || { echo "$ARM provenance already exists; refusing overwrite" >&2; exit 1; }
  "$PYTHON" "$ROOT/sua_exploration/scripts/write_m2_ssc_t4_heldout_provenance.py" \
    --artifact "$ARTIFACT" --arm "$ARM" --source-gate "$SOURCE_GATE" --teacher-receipt "$TEACHER_RECEIPT" --protocol-receipt "$PROTOCOL"
done

if [[ ! -e "$REFERENCE" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/streaming_calibration_exp/scripts/eval_clean_teacher_m2_m24_heldout.py" \
    --teacher-checkpoint "$TEACHER" --teacher-receipt "$TEACHER_RECEIPT" --source-gate "$SOURCE_GATE" \
    --protocol-receipt "$PROTOCOL" --out "$REFERENCE" --device cuda
fi

"$PYTHON" "$ROOT/sua_exploration/scripts/aggregate_m2_ssc_t4_heldout.py" \
  --ordinary "${ORDINARY[0]}" --ssc "${SSC[0]}" --source-gate "$SOURCE_GATE" \
  --protocol-receipt "$PROTOCOL" --clean-spint-reference "$REFERENCE" --out "$AGGREGATE"
printf 'salvage complete: %s\n' "$AGGREGATE"
