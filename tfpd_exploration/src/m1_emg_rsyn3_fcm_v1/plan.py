"""Frozen literals for M1 EMG-rSyn3 successor one-fold V1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as parent_plan


class PlanError(RuntimeError):
    """Fail closed for static contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PHASE = "m1_emg_rsyn3_fcm_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_SUCCESSOR_20260902.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FCM_ONEFOLD_V1_20260902.md"
RESULT_NOTE_RELATIVE = "tfpd_exploration/docs/RESULT_M1_EMG_SYN3_STAGE0_SIGNED_REJECT_20260902.md"
DESIGN_SHA256 = "b5d12b51e74a70d22f50d84ab7078fd0592ba19aae637ed0c193f57497f97cd1"
WORKORDER_SHA256 = "ee9db113f69c80b706436757b0e1221e238acf4cd659d6b9da90bc895c116074"
RESULT_NOTE_SHA256 = "b81c4b2f43ffa2e6a8289b576954bb798e4d91831e9b36553ae3234df9c29b1f"

PARENT_DESIGN_RELATIVE = parent_plan.DESIGN_RELATIVE
PARENT_WORKORDER_RELATIVE = parent_plan.WORKORDER_RELATIVE
PARENT_DESIGN_SHA256 = parent_plan.DESIGN_SHA256
PARENT_WORKORDER_SHA256 = parent_plan.WORKORDER_SHA256
PARENT_STAGE0_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0"
PARENT_DECISION_SHA256 = "6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39"
PARENT_SIGNAL_VIEW_SHA256 = "27e681f86aca6bf7a2d7163cb422ba2c6b1f04c6bb8c5414687335c3453858cf"
PARENT_TERMINAL_SHA256 = "f8436c9e2e1f1e62a3a3fb4f78415cba7bef395c3821437ddb2ee67b99a3f7ba"

SESSIONS = parent_plan.SESSIONS
FOLD0_TARGET_SESSION = parent_plan.FOLD0_TARGET_SESSION
FOLD0_SOURCE_SESSIONS = parent_plan.FOLD0_SOURCE_SESSIONS
FOLD_TARGETS = parent_plan.FOLD_TARGETS
SOURCE_RELATIVE = parent_plan.SOURCE_RELATIVE
SOURCE_FILE_SHA256 = parent_plan.SOURCE_FILE_SHA256
FORBIDDEN_PATH_TOKENS = parent_plan.FORBIDDEN_PATH_TOKENS
SUPPORT_TRIALS = parent_plan.SUPPORT_TRIALS
QUERY_START = parent_plan.QUERY_START
QUERY_STOP_EXCLUSIVE = parent_plan.QUERY_STOP_EXCLUSIVE
CARRIER_BUDGETS = parent_plan.CARRIER_BUDGETS
ACTIVITY_CYCLE = parent_plan.ACTIVITY_CYCLE
CQC_CARRIER_CYCLE = parent_plan.CQC_CARRIER_CYCLE
BIN_SECONDS = parent_plan.BIN_SECONDS
RANK = parent_plan.RANK
CARRIER_DIM = parent_plan.CARRIER_DIM
RIDGE_LAMBDA = parent_plan.RIDGE_LAMBDA
SCALE_FLOOR = parent_plan.SCALE_FLOOR
SEED = parent_plan.SEED
LAG_BINS = parent_plan.LAG_BINS
NNMF_LAW = parent_plan.NNMF_LAW
STAGE1_ARMS = parent_plan.STAGE1_ARMS
STAGE1_EPOCHS = parent_plan.STAGE1_EPOCHS
FIXED_LAST_EPOCH_INDEX = parent_plan.FIXED_LAST_EPOCH_INDEX
STAGE1_LR = parent_plan.STAGE1_LR
STAGE1_BATCH_SIZE = parent_plan.STAGE1_BATCH_SIZE
STAGE1_WEIGHT_DECAY = parent_plan.STAGE1_WEIGHT_DECAY
STATIC_CONTENT_GATE = parent_plan.STATIC_CONTENT_GATE
TEACHER_CKPT_RELATIVE = parent_plan.TEACHER_CKPT_RELATIVE
TEACHER_CKPT_SHA256 = parent_plan.TEACHER_CKPT_SHA256

RECTIFIER_LAW = {
    "name": "relu_nonnegative_projection",
    "operator": "np.maximum(x_raw, 0.0)",
    "elementwise": True,
    "dtype": "float64",
    "threshold": 0.0,
    "learnable_parameters": 0,
    "per_session_threshold": "forbidden",
    "per_channel_threshold": "forbidden",
    "sweep": "forbidden",
    "forbidden_alternatives": ("abs", "offset", "envelope_filter", "smoother"),
    "query_values_read": False,
    "before_rms": True,
    "before_nnmf": True,
}

RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1"
STAGE0_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/stage0"
PILOT_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/pilot"
ARM_ROOT_RELATIVE = {
    "Z-Fix": "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/pilot/z_fix",
    "S-Fix": "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/pilot/s_fix",
    "S-Acyc": "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/pilot/s_acyc",
}

SELECTED_GPU_INDEX = parent_plan.SELECTED_GPU_INDEX
SELECTED_GPU_UUID = parent_plan.SELECTED_GPU_UUID
REFUSED_GPU0_UUID = parent_plan.REFUSED_GPU0_UUID

OWNED_PATHS = (
    "tfpd_exploration/src/m1_emg_rsyn3_fcm_v1/",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_stage0.py",
    "tfpd_exploration/scripts/run_m1_emg_rsyn3_pilot.py",
    "tfpd_exploration/tests/test_m1_emg_rsyn3_fcm_v1.py",
    RESULT_ROOT_RELATIVE,
    DESIGN_RELATIVE,
    WORKORDER_RELATIVE,
    RESULT_NOTE_RELATIVE,
)

SEALED_FOREIGN_ROOTS = (
    "tfpd_exploration/results/m1_emg_syn3_fcm_v1",
    "tfpd_exploration/results/m1_t0c1_prefix_v1",
    "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep",
)


def source_sessions_for_fold(fold: int) -> tuple[str, ...]:
    return parent_plan.source_sessions_for_fold(fold)


def verify_bound_documents(root: Path) -> None:
    design = Path(root) / DESIGN_RELATIVE
    workorder = Path(root) / WORKORDER_RELATIVE
    note = Path(root) / RESULT_NOTE_RELATIVE
    parent_design = Path(root) / PARENT_DESIGN_RELATIVE
    parent_workorder = Path(root) / PARENT_WORKORDER_RELATIVE
    _require(design.is_file() and workorder.is_file() and note.is_file(), "successor docs missing")
    _require(sha256_bytes(design.read_bytes()) == DESIGN_SHA256, "design sha drift")
    _require(sha256_bytes(workorder.read_bytes()) == WORKORDER_SHA256, "workorder sha drift")
    _require(sha256_bytes(note.read_bytes()) == RESULT_NOTE_SHA256, "result-note sha drift")
    _require(parent_design.is_file() and parent_workorder.is_file(), "parent docs missing")
    _require(sha256_bytes(parent_design.read_bytes()) == PARENT_DESIGN_SHA256, "parent design sha drift")
    _require(
        sha256_bytes(parent_workorder.read_bytes()) == PARENT_WORKORDER_SHA256,
        "parent workorder sha drift",
    )


@dataclass(frozen=True)
class StageRootSpec:
    root_relative: str

    def payload(self) -> dict[str, object]:
        return {"schema": "m1_emg_rsyn3_stage_root_v1", "root_relative": self.root_relative}


def dry_cli_payload() -> dict[str, object]:
    return {
        "phase": PHASE,
        "design_sha256": DESIGN_SHA256,
        "workorder_sha256": WORKORDER_SHA256,
        "parent_stage0": PARENT_STAGE0_ROOT_RELATIVE,
        "parent_decision_sha256": PARENT_DECISION_SHA256,
        "rectifier": dict(RECTIFIER_LAW),
        "fold0_source_sessions": list(FOLD0_SOURCE_SESSIONS),
        "fold0_target_session": FOLD0_TARGET_SESSION,
        "carrier_budgets": list(CARRIER_BUDGETS),
        "activity_cycle": list(ACTIVITY_CYCLE),
        "stage1_arms": list(STAGE1_ARMS),
        "static_content_gate": STATIC_CONTENT_GATE,
        "prospective_roots": {
            "stage0": STAGE0_ROOT_RELATIVE,
            "pilot": PILOT_ROOT_RELATIVE,
            **ARM_ROOT_RELATIVE,
        },
        "gpu": {
            "index": SELECTED_GPU_INDEX,
            "required_uuid": SELECTED_GPU_UUID,
            "refused_uuid_gpu0": REFUSED_GPU0_UUID,
        },
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_gpu_capability": False,
        "public_execution_authorized": False,
    }


def require_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value
