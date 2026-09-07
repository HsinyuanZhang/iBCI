"""Frozen literals for the fold-local EMG-rSyn3 Stage-0 audit successor."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import plan as rsyn3_plan


class PlanError(RuntimeError):
    """Fail closed for static contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PHASE = "m1_emg_rsyn3_fold_local_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md"
DESIGN_SHA256 = "bd69577189f10363f1a741bdd24e00e7e5d944dc318d6a3ee0b7c4a7f2d600fa"
WORKORDER_SHA256 = "bdf4999baaab280cb83335ac9f7dc3590d8fcfaf35bdf8a323a7a3b41c5668af"

PARENT_FAIL_ROOT_RELATIVE = rsyn3_plan.PARENT_STAGE0_ROOT_RELATIVE
PARENT_FAIL_DECISION_SHA256 = rsyn3_plan.PARENT_DECISION_SHA256
PARENT_FAIL_TERMINAL_SHA256 = rsyn3_plan.PARENT_TERMINAL_SHA256
RSYN3_PASS_ROOT_RELATIVE = rsyn3_plan.STAGE0_ROOT_RELATIVE
RSYN3_PASS_DECISION_SHA256 = "bbc6815a551406e94e330965fcd77cc1221a82be06a3c37428ed5066a1b53dad"
RSYN3_PASS_TERMINAL_SHA256 = "aa98889c41e4c1a839dcf5e51d36008a4cdedc2d442770f31ffde1a7b747788a"
RSYN3_PASS_FOLD_RECEIPTS_SHA256 = "ffe98a39df8dcfef5b971ce2230006e587cb17d318893aad75320fd7c0e3e537"
RSYN3_PASS_RELIABILITY_SHA256 = "e9776b82e7f48eca7f8ceec599e9c9b93deef58ada8876d88895176859fe67c4"

SESSIONS = rsyn3_plan.SESSIONS
FOLD0_TARGET_SESSION = rsyn3_plan.FOLD0_TARGET_SESSION
FOLD0_SOURCE_SESSIONS = rsyn3_plan.FOLD0_SOURCE_SESSIONS
FOLD_TARGETS = rsyn3_plan.FOLD_TARGETS
SOURCE_RELATIVE = rsyn3_plan.SOURCE_RELATIVE
SOURCE_FILE_SHA256 = rsyn3_plan.SOURCE_FILE_SHA256
SUPPORT_TRIALS = rsyn3_plan.SUPPORT_TRIALS
QUERY_START = rsyn3_plan.QUERY_START
QUERY_STOP_EXCLUSIVE = rsyn3_plan.QUERY_STOP_EXCLUSIVE
CARRIER_BUDGETS = rsyn3_plan.CARRIER_BUDGETS
ACTIVITY_CYCLE = rsyn3_plan.ACTIVITY_CYCLE
SEED = rsyn3_plan.SEED
PACK_LIMIT = 2
OWN_TOKEN = "run_m1_emg_rsyn3_fold_local_stage1.py"
FULL_QUERY_OWN_TOKEN = "run_m1_emg_rsyn3_fold_local_full_query.py"
TOKEN_PROBE_OWN_TOKEN = "run_m1_emg_rsyn3_fold_local_token_probe.py"
BUDGET_PROBE_OWN_TOKEN = "run_m1_emg_rsyn3_fold_local_budget_probe.py"
OWN_TOKENS = (OWN_TOKEN, FULL_QUERY_OWN_TOKEN, TOKEN_PROBE_OWN_TOKEN, BUDGET_PROBE_OWN_TOKEN)
# Inference-only rescoring is light; three arms share one card.
FULL_QUERY_PACK_LIMIT = 3
RANK = rsyn3_plan.RANK
RECTIFIER_LAW = dict(rsyn3_plan.RECTIFIER_LAW)
STAGE1_ARMS = rsyn3_plan.STAGE1_ARMS
STATIC_CONTENT_GATE = rsyn3_plan.STATIC_CONTENT_GATE
TEACHER_CKPT_RELATIVE = rsyn3_plan.TEACHER_CKPT_RELATIVE
TEACHER_CKPT_SHA256 = rsyn3_plan.TEACHER_CKPT_SHA256

LS4_LAW = {
    "name": "ls4_unequal_length_disclosure_v1",
    "unequal_length": "cyclic_repeat_or_truncate",
    "true_resampling": False,
    "operator": "take = np.arange(dest_n) % source_n",
    "enabled_in_stage1_pilot": False,
    "must_repair_before": "stage2_mechanism_controls",
    "stage1_pilot_arms": list(STAGE1_ARMS),
}

RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1"
STAGE0_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/stage0"
# pilot_r2 is retained as failure evidence: all three arms trained 12 epochs but
# the publisher looked for `epoch_011.ckpt` while Lightning 2.4 writes
# `epoch_epoch=011.ckpt`. Fixed by `stage1.resolve_fixed_last_checkpoint`.
STAGE1_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3"
STAGE1_ARM_ROOT_RELATIVE = {
    "Z-Fix": "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/z_fix",
    "S-Fix": "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix",
    "S-Acyc": "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_acyc",
}
ARM_SLUG = {"Z-Fix": "z_fix", "S-Fix": "s_fix", "S-Acyc": "s_acyc"}
FULL_QUERY_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/full_query_v1"
FULL_QUERY_ARM_ROOT_RELATIVE = {arm: f"{FULL_QUERY_ROOT_RELATIVE}/{slug}" for arm, slug in ARM_SLUG.items()}
# Only the three pre-declared M1 internal-LOSO windows are legal (mirrors
# streaming_calibration_exp/src/data/falcon_datamodule.py allowed_windows).
FULL_QUERY_ALLOWED_WINDOWS = frozenset({(10, None), (10, 210), (210, None)})
# Ordered: anchor first, then late half, then full official extent.
FULL_QUERY_WINDOWS = (
    ("q10_210", (10, 210)),
    ("q210_end", (210, None)),
    ("q10_end", (10, None)),
)
FULL_QUERY_ANCHOR_WINDOW = "q10_210"
FULL_QUERY_OFFICIAL_EXTENT_WINDOW = "q10_end"
FULL_QUERY_LATE_HALF_WINDOW = "q210_end"
FULL_QUERY_ANCHOR_TOLERANCE = 1.0e-5
FULL_QUERY_PILOT_BODIES = ("epoch_011.pt", "score_static.json", "score_cdm_a.json", "arm_table.json")
TOKEN_PROBE_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/token_probe_v1"
TOKEN_PROBE_RIDGE_LAMBDA = 1.0
TOKEN_PROBE_NULL_PERMUTATIONS = 200
TOKEN_PROBE_NULL_SEED = 42
TOKEN_PROBE_DIRECTION_EPS = 1.0e-6
TOKEN_PROBE_ARMS = ("Z-Fix", "S-Fix", "S-Acyc")
TOKEN_PROBE_BASELINE = "RAW_STATS"
TOKEN_PROBE_READ_RULE = {
    "primary_arm": "Z-Fix",
    "primary_target": "weights_r2",
    "redundant": {"min_r2": 0.50, "min_advantage": 0.10, "min_positive_sessions": 3},
    "partially_recoverable": {"min_advantage": 0.10},
    "not_recoverable": {"max_advantage": 0.05},
}
BUDGET_PROBE_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/budget_probe_v1"
BUDGET_PROBE_BUDGETS = CARRIER_BUDGETS
BUDGET_PROBE_ARM = "Z-Fix"
BUDGET_PROBE_RELIABILITY_FLOOR = 0.30
BUDGET_PROBE_READ_RULE = {
    "primary_arm": "Z-Fix",
    "primary_metric": "weights_r2",
    "REDUNDANCY_PERSISTS": {
        "min_advantage": 0.10,
        "min_redundancy_fraction": 0.60,
    },
    "REDUNDANCY_BREAKS_AT_LOW_BUDGET": {
        "m10_min_weights_r2": 0.50,
        "m10_min_advantage": 0.10,
        "reliability_floor": 0.30,
        "max_redundancy_fraction_at_smallest_reliable": 0.40,
    },
    "INCONCLUSIVE_UNRELIABLE_TARGET": {
        "reliability_floor": 0.30,
        "budget_below": 10,
    },
    "PARTIAL_DECAY": {},
}
STAGE1_EPOCHS = rsyn3_plan.STAGE1_EPOCHS
FIXED_LAST_EPOCH_INDEX = rsyn3_plan.FIXED_LAST_EPOCH_INDEX
STAGE1_LR = rsyn3_plan.STAGE1_LR
STAGE1_BATCH_SIZE = rsyn3_plan.STAGE1_BATCH_SIZE
STAGE1_WEIGHT_DECAY = rsyn3_plan.STAGE1_WEIGHT_DECAY
WINDOW_SIZE = 100
TRIAL_LENGTH = 1024
DATA_DIR_RELATIVE = "SPINT-main/data/000941"
FOLD0_M10_RAW_CARRIER_DIGEST = "2d1a638e4816aa3254f5b1817601ddc4515cd6a24f4d5aa03e61ee7a8513ffe7"

GPU0_INDEX = 0
GPU1_INDEX = 1
GPU0_UUID = rsyn3_plan.REFUSED_GPU0_UUID
GPU1_UUID = rsyn3_plan.SELECTED_GPU_UUID
GPU0_ALLOWED = True

OWNED_PATHS = (
    "tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage0.py",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage1.py",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_full_query.py",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_token_probe.py",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_budget_probe.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_v1.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_stage1.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_full_query.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_token_probe.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_budget_probe.py",
    RESULT_ROOT_RELATIVE,
    DESIGN_RELATIVE,
    WORKORDER_RELATIVE,
)


def source_sessions_for_fold(fold: int) -> tuple[str, ...]:
    return rsyn3_plan.source_sessions_for_fold(fold)


def verify_bound_documents(root: Path) -> None:
    design = Path(root) / DESIGN_RELATIVE
    workorder = Path(root) / WORKORDER_RELATIVE
    _require(design.is_file() and workorder.is_file(), "fold-local docs missing")
    _require(sha256_bytes(design.read_bytes()) == DESIGN_SHA256, "design sha drift")
    _require(sha256_bytes(workorder.read_bytes()) == WORKORDER_SHA256, "workorder sha drift")


def dry_cli_payload() -> dict[str, object]:
    return {
        "phase": PHASE,
        "design_sha256": DESIGN_SHA256,
        "workorder_sha256": WORKORDER_SHA256,
        "parent_fail_root": PARENT_FAIL_ROOT_RELATIVE,
        "rsyn3_pass_root": RSYN3_PASS_ROOT_RELATIVE,
        "rectifier": dict(RECTIFIER_LAW),
        "ls4": dict(LS4_LAW),
        "fold0_source_sessions": list(FOLD0_SOURCE_SESSIONS),
        "fold0_target_session": FOLD0_TARGET_SESSION,
        "carrier_budgets": list(CARRIER_BUDGETS),
        "stage1_arms": list(STAGE1_ARMS),
        "static_content_gate": STATIC_CONTENT_GATE,
        "prospective_roots": {
            "stage0": STAGE0_ROOT_RELATIVE,
            "pilot": STAGE1_ROOT_RELATIVE,
            **STAGE1_ARM_ROOT_RELATIVE,
            "full_query": FULL_QUERY_ROOT_RELATIVE,
            **{f"full_query/{ARM_SLUG[arm]}": relative for arm, relative in FULL_QUERY_ARM_ROOT_RELATIVE.items()},
            "token_probe": TOKEN_PROBE_ROOT_RELATIVE,
            "budget_probe": BUDGET_PROBE_ROOT_RELATIVE,
        },
        "full_query_windows": [[name, [start, end]] for name, (start, end) in FULL_QUERY_WINDOWS],
        "gpu": {
            "gpu0_allowed": GPU0_ALLOWED,
            "gpu0_uuid": GPU0_UUID,
            "gpu1_uuid": GPU1_UUID,
            "eligible_indices": [GPU0_INDEX, GPU1_INDEX],
            "pack_limit": PACK_LIMIT,
            "own_token": OWN_TOKEN,
            "full_query_pack_limit": FULL_QUERY_PACK_LIMIT,
            "own_tokens": list(OWN_TOKENS),
        },
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_gpu_capability": False,
        "public_execution_authorized": False,
    }


@dataclass(frozen=True)
class StageRootSpec:
    root_relative: str


def require_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value
