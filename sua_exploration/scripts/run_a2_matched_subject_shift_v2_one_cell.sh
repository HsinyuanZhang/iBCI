#!/usr/bin/env bash
# One-source-run runner for A2 matched subject-shift v2.
#
# This runner has two deliberately separate phases:
#   1. one official immutable, CPU-only preflight is minted before *all* six
#      source cells; and
#   2. each cell verifies that same receipt and only its own freshness before
#      it can launch.
#
# Thus a completed cell never makes the official preflight look stale and
# cannot block cells 2--6.  The default --dry-run is inert: it neither loads
# Torch/CUDA/NWB/checkpoints nor writes a receipt.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="a2_matched_subject_shift_v2"
CONTRACT="$SUA/docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"
CONFIG="$SUA/configs/a2_matched_subject_shift_v2.json"
PREFLIGHT="$SUA/scripts/a2_matched_subject_shift_v2_preflight.py"
SCORER="$SUA/scripts/a2_matched_subject_shift_v2_score.py"
TRAINER="$SUA/scripts/train_variant_dandi688.py"
MANIFEST="$SUA/configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER="$SUA/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
DATA_SUBC="$SUA/data/dandi_000688/sub-C"
CACHE="$SUA/cache/dandi688_subc_co_v1"
RESULTS="$SUA/results/$SCREEN_ID"
AUTH_ENV="A2_V2_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_A2_MATCHED_SUBJECT_SHIFT_V2_GPU"
ROOT_GO_ENV="A2_V2_ROOT_GO"
ROOT_GO_VALUE="I_SIGN_A2_MATCHED_SUBJECT_SHIFT_V2_ROOT_GO"

# Never inherit a user-site CUDA/Torch installation.  The explicit export is
# intentionally unconditional: all Python commands below (including CPU-only
# preflight verification) receipt-bind this isolation.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$SUA${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_ARM="${SOURCE_ARM:?SOURCE_ARM required: source_z4|source_t4}"
SEED="${SEED:?SEED required: 42|43|44}"
GPU="${GPU:?GPU index required}"
MODE="${1:---dry-run}"
OFFICIAL_PREFLIGHT="${OFFICIAL_PREFLIGHT:-$RESULTS/official_cpu_preflight.json}"

case "$SOURCE_ARM" in
  source_z4) SIDE="z4" ;;
  source_t4) SIDE="t4" ;;
  *) echo "Unsupported SOURCE_ARM=$SOURCE_ARM; expected source_z4 or source_t4" >&2; exit 2 ;;
esac
case "$SEED" in 42|43|44) ;; *) echo "Unsupported SEED=$SEED; expected 42, 43, or 44" >&2; exit 2 ;; esac
case "$MODE" in --dry-run|--launch) ;; *) echo "Mode must be --dry-run or --launch" >&2; exit 2 ;; esac

[[ "$PYTHONNOUSERSITE" == "1" ]] || { echo "PYTHONNOUSERSITE=1 enforcement failed" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }

CONTRACT_SHA="$($PY - "$CONTRACT" <<'PY'
import hashlib
import sys
from pathlib import Path
path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"

NAME="${SCREEN_ID}_${SOURCE_ARM}_dandi688_co_s${SEED}"
RUN_DIR="$SUA/checkpoints/$NAME"
WITHIN_RESULT="$RESULTS/within_subject_${SOURCE_ARM}_s${SEED}.json"
EXTERNAL_RESULT="$RESULTS/external_subject_M_${SOURCE_ARM}_s${SEED}.json"
CELL_LAUNCH_RECEIPT="$RESULTS/cell_launch_${SOURCE_ARM}_s${SEED}.json"
LOG="$RESULTS/logs/${SOURCE_ARM}_s${SEED}.log"

TRAIN_COMMAND=(
  "$PY" -u "$TRAINER"
  --teacher_ckpt "$TEACHER"
  --variant B3S
  --side_features "$SIDE"
  --side_feature_pool_size 30
  --calibration_n_trials 30
  --chronological_calibration
  --out_name "$NAME"
  --data_dir "$DATA_SUBC"
  --cache_dir "$CACHE"
  --train_val_manifest "$MANIFEST"
  --signal_view sua
  --task CO
  --split_counts 27,6,6
  --max_units_exclusive 100
  --max_epochs 12
  --no_early_stopping
  --checkpoint_every_epoch
  --lr 1e-4
  --batch_size 32
  --num_workers 4
  --seed "$SEED"
  --loss_mode task_only
  --identity_mode calibrated
  --require_gpu
  --disable_progress_bar
)
WITHIN_SCORE_COMMAND=(
  "$PY" -u "$SCORER"
  --source-arm "$SOURCE_ARM" --seed "$SEED" --domain within_subject
  --run-dir "$RUN_DIR" --out-path "$WITHIN_RESULT" --device cuda:0
  --official-preflight "$OFFICIAL_PREFLIGHT" --result-root "$RESULTS"
  --cell-launch-receipt "$CELL_LAUNCH_RECEIPT"
)
EXTERNAL_SCORE_COMMAND=(
  "$PY" -u "$SCORER"
  --source-arm "$SOURCE_ARM" --seed "$SEED" --domain external_subject_M
  --run-dir "$RUN_DIR" --out-path "$EXTERNAL_RESULT" --device cuda:0
  --official-preflight "$OFFICIAL_PREFLIGHT" --result-root "$RESULTS"
  --cell-launch-receipt "$CELL_LAUNCH_RECEIPT"
)

echo "SCREEN_ID=$SCREEN_ID"
echo "CONTRACT=$CONTRACT"
echo "CONTRACT_SHA256=$CONTRACT_SHA"
echo "CONFIG=$CONFIG"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "SOURCE_ARM=$SOURCE_ARM SIDE=$SIDE SEED=$SEED GPU=$GPU"
echo "OFFICIAL_PREFLIGHT=$OFFICIAL_PREFLIGHT"
echo "EXPECTED_FRESH_GPU_CELLS=6 (2 source arms x 3 seeds; domain retraining forbidden)"
echo "SAME_UNCHANGED_SOURCE_EPOCH_BUNDLE_SCORED_ON=within_subject,external_subject_M"
echo "OFFICIAL_PREFLIGHT_REQUIRED=one immutable CPU receipt shared by all six cells"
echo "AUTHORIZATION_REQUIRED=${AUTH_ENV}=${AUTH_VALUE} AND ${ROOT_GO_ENV}=${ROOT_GO_VALUE}"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "--- official CPU preflight command (not executed; writes one immutable 0444 body+SHA sidecar) ---"
  printf 'PYTHONNOUSERSITE=1 '
  printf '%q ' "$PY" "$PREFLIGHT" --result-root "$RESULTS" --receipt "$OFFICIAL_PREFLIGHT"
  echo
  echo "--- candidate-only official-preflight/freshness verification (not executed) ---"
  printf 'PYTHONNOUSERSITE=1 '
  printf '%q ' "$PY" - "$OFFICIAL_PREFLIGHT" "$RESULTS" "$SOURCE_ARM" "$SEED"
  echo "# core.load_verified_official_preflight(...); core.assert_cell_fresh(...)"
  echo "--- isolated Torch/CUDA identity check before training (not executed; only after ROOT GO) ---"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 ' "$GPU"
  printf '%q ' "$PY" - "$OFFICIAL_PREFLIGHT" "$RESULTS" "$SOURCE_ARM" "$SEED" "$CELL_LAUNCH_RECEIPT"
  echo "# validates official SHA+implementation hashes; writes immutable launch receipt; checks env Torch 2.5.1.post303/cu118/device"
  echo "--- source training command (not executed) ---"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 ' "$GPU"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  echo
  echo "--- within-domain scoring command (not executed; reuses source run) ---"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 ' "$GPU"
  printf '%q ' "${WITHIN_SCORE_COMMAND[@]}" --launch
  echo
  echo "--- external-domain scoring command (not executed; reuses SAME source run) ---"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 ' "$GPU"
  printf '%q ' "${EXTERNAL_SCORE_COMMAND[@]}" --launch
  echo
  echo "SCORER_DRY_RUN_RECEIPT=DEFERRED: source run does not exist until explicit ROOT GO plus GPU authorization."
  exit 0
fi

if [[ "${!ROOT_GO_ENV:-}" != "$ROOT_GO_VALUE" ]]; then
  echo "Refusing GPU launch: root must explicitly set ${ROOT_GO_ENV}=${ROOT_GO_VALUE}" >&2
  exit 3
fi
if [[ "${!AUTH_ENV:-}" != "$AUTH_VALUE" ]]; then
  echo "Refusing GPU launch: set ${AUTH_ENV}=${AUTH_VALUE}" >&2
  exit 3
fi
[[ -f "$CONTRACT" && -f "$CONFIG" && -f "$MANIFEST" && -f "$TEACHER" ]] || { echo "A2 v2 binding is missing" >&2; exit 1; }
[[ -f "$OFFICIAL_PREFLIGHT" ]] || { echo "Missing one official immutable preflight: $OFFICIAL_PREFLIGHT" >&2; exit 1; }
[[ ! -e "$CELL_LAUNCH_RECEIPT" && ! -e "${CELL_LAUNCH_RECEIPT}.sha256" ]] || { echo "Refusing existing cell launch receipt: $CELL_LAUNCH_RECEIPT" >&2; exit 1; }

# Do not run the global root audit here.  Verify its sealed record and only
# this candidate arm/seed's outputs, so a completed prior cell remains valid.
"$PY" - "$OFFICIAL_PREFLIGHT" "$RESULTS" "$SOURCE_ARM" "$SEED" <<'PY'
import sys
from pathlib import Path
from mc_maze import a2_matched_subject_shift_v2_core as core

preflight, digest = core.load_verified_official_preflight(Path(sys.argv[1]), result_root=Path(sys.argv[2]))
targets = core.assert_cell_fresh(source_arm=sys.argv[3], seed=int(sys.argv[4]), result_root=Path(sys.argv[2]))
print({"official_preflight_sha256": digest, "candidate_fresh_targets": targets, "status": preflight["status"]})
PY

# The pre-training launch receipt records the exact isolated Torch package and
# visible device.  It is written before trainer code runs; a failure leaves a
# non-reusable immutable trace rather than a silently retryable source cell.
CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$OFFICIAL_PREFLIGHT" "$RESULTS" "$SOURCE_ARM" "$SEED" "$CELL_LAUNCH_RECEIPT" <<'PY'
import sys
from pathlib import Path
from mc_maze import a2_matched_subject_shift_v2_core as core

official_path = Path(sys.argv[1])
result_root = Path(sys.argv[2])
source_arm = sys.argv[3]
seed = int(sys.argv[4])
out = Path(sys.argv[5])
official, official_sha = core.load_verified_official_preflight(official_path, result_root=result_root)
targets = core.assert_cell_fresh(source_arm=source_arm, seed=seed, result_root=result_root)
runtime = core.torch_runtime_binding(visible_device_index=0)
payload = {
    "schema_version": 3,
    "receipt_kind": core.CELL_LAUNCH_KIND,
    "screen_id": core.SCREEN_ID,
    "status": "ROOT_GO_AND_GPU_AUTHORIZED_PRETRAINING_ENVIRONMENT_VERIFIED",
    "source_arm": source_arm,
    "seed": seed,
    "official_preflight_path": str(official_path.expanduser().resolve()),
    "official_preflight_sha256": official_sha,
    "implementation_bindings": official["implementation_bindings"],
    "implementation_bindings_sha256": official["implementation_bindings_sha256"],
    "python_isolation": core.python_isolation_binding(),
    "torch_runtime": runtime,
    "candidate_fresh_targets": targets,
    "formal_subc_test_nwb_opened": False,
    "training_started": False,
}
_body, sidecar, digest = core.write_immutable_json(out, payload)
print({"cell_launch_receipt": str(out), "sidecar": str(sidecar), "sha256": digest, "torch": runtime})
PY

mkdir -p "$RESULTS/logs"
{
  echo "[$(date -Is)] START source_arm=$SOURCE_ARM seed=$SEED gpu=$GPU contract_sha=$CONTRACT_SHA pythonnouser=$PYTHONNOUSERSITE"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 "${TRAIN_COMMAND[@]}"
  # Both score calls receive the precise source run, the single official
  # preflight, and the pre-training immutable Torch/runtime receipt.
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 "${WITHIN_SCORE_COMMAND[@]}" --launch
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 "${EXTERNAL_SCORE_COMMAND[@]}" --launch
  "$PY" - "$WITHIN_RESULT" "$EXTERNAL_RESULT" <<'PY'
import sys
from pathlib import Path
from mc_maze import a2_matched_subject_shift_v2_core as core

left, left_sha = core.load_verified_immutable_json(Path(sys.argv[1]), label="within A2 v2 domain receipt")
right, right_sha = core.load_verified_immutable_json(Path(sys.argv[2]), label="external A2 v2 domain receipt")
for key in (
    "source_run_metadata_sha256", "source_checkpoint_sha256_bundle",
    "source_checkpoint_sha256_bundle_sha256", "query_policy", "normalizer_authority",
    "official_preflight_sha256", "implementation_bindings", "implementation_bindings_sha256",
    "cell_launch_receipt_sha256",
):
    if left.get(key) != right.get(key):
        raise SystemExit(f"cross-domain receipt mismatch: {key}")
if left.get("domain") != "within_subject" or right.get("domain") != "external_subject_M":
    raise SystemExit("domain receipt identity drift")
print({"A2_V2_CROSS_DOMAIN_SOURCE_CHECKPOINT_REUSE_VERIFIED": True, "within_sha256": left_sha, "external_sha256": right_sha})
PY
  echo "[$(date -Is)] DONE"
} >"$LOG" 2>&1
