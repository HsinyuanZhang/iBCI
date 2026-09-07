#!/usr/bin/env bash
# Inert one-cell runner for C1 teacher-domain ablation.
#
# Default --dry-run prints the frozen 2x2 cell and never loads Torch, NWB,
# checkpoints, or CUDA.  --launch always refuses: this scaffolding authorizes
# no GPU run.  Root's gate additionally requires a source-only teacher
# compatibility receipt before any future launch can even be reviewed.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
SUA="$ROOT/sua_exploration"
PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN_ID="c1_teacher_domain_ablation_v1"
CONTRACT="$SUA/docs/C1_TEACHER_DOMAIN_ABLATION_CONTRACT_20260813.md"
CONFIG="$SUA/configs/c1_teacher_domain_ablation.json"
PREFLIGHT="$SUA/scripts/c1_teacher_domain_ablation_preflight.py"
TRAINER="$SUA/scripts/train_variant_dandi688.py"
MANIFEST="$SUA/configs/subc_co_27_6_strict_train_val_manifest.json"
MC_TEACHER="$SUA/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
RESULTS="$SUA/results/$SCREEN_ID"
COMPATIBILITY="$RESULTS/source_only_teacher_compatibility_receipt.json"
AUTH_ENV="C1_TEACHER_DOMAIN_GPU_AUTHORIZATION"
AUTH_VALUE="I_AUTHORIZE_C1_TEACHER_DOMAIN_ABLATION_GPU"
ROOT_GO_ENV="C1_TEACHER_DOMAIN_ROOT_GO"
ROOT_GO_VALUE="I_SIGN_C1_TEACHER_DOMAIN_ABLATION_ROOT_GO"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT:$SUA${PYTHONPATH:+:$PYTHONPATH}"

TEACHER_DOMAIN="${TEACHER_DOMAIN:?TEACHER_DOMAIN required: mc_maze|co_native}"
CARRIER="${CARRIER:?CARRIER required: t4|z4}"
SEED="${SEED:?SEED required: 42|43|44}"
GPU="${GPU:?GPU index required (inert; never used)}"
MODE="${1:---dry-run}"

case "$TEACHER_DOMAIN" in mc_maze|co_native) ;; *) echo "Unsupported TEACHER_DOMAIN=$TEACHER_DOMAIN" >&2; exit 2 ;; esac
case "$CARRIER" in t4|z4) ;; *) echo "Unsupported CARRIER=$CARRIER" >&2; exit 2 ;; esac
case "$SEED" in 42|43|44) ;; *) echo "Unsupported SEED=$SEED" >&2; exit 2 ;; esac
case "$MODE" in --dry-run|--launch) ;; *) echo "Mode must be --dry-run or --launch" >&2; exit 2 ;; esac

[[ "$PYTHONNOUSERSITE" == "1" ]] || { echo "PYTHONNOUSERSITE=1 enforcement failed" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }

CONTRACT_SHA="$($PY - "$CONTRACT" <<'PY'
import hashlib, sys
from pathlib import Path
path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"

NAME="${SCREEN_ID}_${TEACHER_DOMAIN}_${CARRIER}_dandi688_co_s${SEED}"
RUN_DIR="$SUA/checkpoints/$NAME"
WITHIN_RESULT="$RESULTS/${TEACHER_DOMAIN}_${CARRIER}_s${SEED}_within_subject.json"
EXTERNAL_RESULT="$RESULTS/${TEACHER_DOMAIN}_${CARRIER}_s${SEED}_external_subject_M.json"
LOG="$RESULTS/logs/${TEACHER_DOMAIN}_${CARRIER}_s${SEED}.log"
TEACHER_CKPT="$MC_TEACHER"
if [[ "$TEACHER_DOMAIN" == "co_native" ]]; then
  TEACHER_CKPT="$RESULTS/TRAINED_CO_NATIVE_TEACHER_NOT_BOUND.ckpt"
fi

echo "SCREEN_ID=$SCREEN_ID"
echo "CONTRACT=$CONTRACT"
echo "CONTRACT_SHA256=$CONTRACT_SHA"
echo "CONFIG=$CONFIG"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "TEACHER_DOMAIN=$TEACHER_DOMAIN CARRIER=$CARRIER ADD_SITE=W-add SAMPLING=legacy SEED=$SEED GPU=$GPU"
echo "EXPECTED_FRESH_GPU_CELLS=12 (2 teachers x 2 carriers x 3 seeds; W-add fixed; H-add forbidden)"
echo "SAME_UNCHANGED_SOURCE_EPOCH_BUNDLE_SCORED_ON=within_subject,external_subject_M"
echo "COMPATIBILITY_RECEIPT_REQUIRED=$COMPATIBILITY"
echo "AUTHORIZATION_REQUIRED=${AUTH_ENV}=${AUTH_VALUE} AND ${ROOT_GO_ENV}=${ROOT_GO_VALUE}"
echo "HYPOTHESIS_STATUS=plausible_contributor_not_isolated_cause"
echo "SEALED_FORMAL_TEST_SESSIONS=refused"

echo "--- candidate source training command (not executed) ---"
printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=1 ' "$GPU"
printf '%q ' "$PY" -u "$TRAINER" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --variant B3S \
  --side_features "$CARRIER" \
  --loss_mode task_only \
  --identity_mode calibrated \
  --seed "$SEED" \
  --train_val_manifest "$MANIFEST"
echo
echo "--- within/external scoring (not executed; would reuse the same source run) ---"
echo "WITHIN_RESULT=$WITHIN_RESULT"
echo "EXTERNAL_RESULT=$EXTERNAL_RESULT"
echo "RUN_DIR=$RUN_DIR"
echo "LOG=$LOG"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY_RUN_INERT=true"
  echo "TORCH_IMPORTED=false"
  echo "NWB_OPENED=false"
  echo "GPU_LAUNCHED=false"
  echo "TRAINING_RUN=false"
  echo "SEALED_FORMAL_TEST_SESSIONS_OPENED=false"
  exit 0
fi

echo "Refusing GPU launch: C1 teacher-domain scaffolding authorizes no GPU run." >&2
echo "Root gate: source-only teacher compatibility receipt first; then a trained CO-native teacher binding; then explicit ${ROOT_GO_ENV}=${ROOT_GO_VALUE}." >&2
echo "This runner has no receipt-preparation bypass." >&2
exit 3
