"""Frozen literals for M1 EMG-Syn3 FCM one-fold V1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


class PlanError(RuntimeError):
    """Fail closed for static contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PHASE = "m1_emg_syn3_fcm_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M1_EMG_SYN3_FCM_ONEFOLD_V1_20260902.md"
DESIGN_SHA256 = "766cb7952913fc0ee57f231bc398b1d50b4d768f4d361e77ca185c61ee6a8259"
WORKORDER_SHA256 = "7b4a002b39e4d86808931e2682b8cde2a9b55dc02a557d490da9e944da2c4378"

SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
FOLD0_TARGET_SESSION = "ses-20120924"
FOLD0_SOURCE_SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
FOLD_TARGETS = {
    0: "ses-20120924",
    1: "ses-20120926",
    2: "ses-20120927",
    3: "ses-20120928",
}

SOURCE_RELATIVE = {
    "ses-20120924": (
        "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
        "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb"
    ),
    "ses-20120926": (
        "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
        "sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb"
    ),
    "ses-20120927": (
        "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
        "sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb"
    ),
    "ses-20120928": (
        "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
        "sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb"
    ),
}
SOURCE_FILE_SHA256 = {
    "ses-20120924": "63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb",
    "ses-20120926": "9c72512308194b93cc19b51733514eb9721152ffe4dd357ee93aabd4be5caa91",
    "ses-20120927": "2d2fdc9be5ccb7a47969894ff1da43a994da11bfff298353f94d489c4af37b3a",
    "ses-20120928": "96e7f078c6acb89b802b6241a36b9c9e2d04df82c180a63354fd3f7e74357254",
}

FORBIDDEN_PATH_TOKENS = ("held-out", "minival", "evalai", "formal", "test")
SUPPORT_TRIALS = 10
QUERY_START = 10
QUERY_STOP_EXCLUSIVE = 210
CARRIER_BUDGETS = (10, 6, 4, 2)
ACTIVITY_CYCLE = (10, 5, 2)
CQC_CARRIER_CYCLE = (10, 6, 4)
BIN_SECONDS = 0.02
RANK = 3
CARRIER_DIM = 4
RIDGE_LAMBDA = 1.0
SCALE_FLOOR = 1.0e-8
SEED = 42
LAG_BINS = 0

NNMF_LAW = {
    "rank": RANK,
    "loss": "frobenius",
    "solver": "cd",
    "init": "nndsvda",
    "random_state": SEED,
    "tolerance": 1.0e-5,
    "maximum_iterations": 1000,
    "l1_regularization": 0.0,
    "l2_regularization": 0.0,
    "sklearn_kwargs": {
        "n_components": RANK,
        "init": "nndsvda",
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 1.0e-5,
        "max_iter": 1000,
        "random_state": SEED,
        "alpha_W": 0.0,
        "alpha_H": 0.0,
        "l1_ratio": 0.0,
    },
}

STAGE1_ARMS = ("Z-Fix", "S-Fix", "S-Acyc")
STAGE1_EPOCHS = 12
FIXED_LAST_EPOCH_INDEX = 11
STAGE1_LR = 1.0e-4
STAGE1_BATCH_SIZE = 32
STAGE1_WEIGHT_DECAY = 0.0
STATIC_CONTENT_GATE = 0.03

TEACHER_CKPT_RELATIVE = (
    "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/"
    "2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/"
    "checkpoints/best_ckpt/epoch_018.ckpt"
)
TEACHER_CKPT_SHA256 = "f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be"

RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_syn3_fcm_v1"
STAGE0_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0"
PILOT_ROOT_RELATIVE = "tfpd_exploration/results/m1_emg_syn3_fcm_v1/pilot"
ARM_ROOT_RELATIVE = {
    "Z-Fix": "tfpd_exploration/results/m1_emg_syn3_fcm_v1/pilot/z_fix",
    "S-Fix": "tfpd_exploration/results/m1_emg_syn3_fcm_v1/pilot/s_fix",
    "S-Acyc": "tfpd_exploration/results/m1_emg_syn3_fcm_v1/pilot/s_acyc",
}

SELECTED_GPU_INDEX = 1
SELECTED_GPU_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
REFUSED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"

OWNED_PATHS = (
    "tfpd_exploration/src/m1_emg_syn3_fcm_v1/",
    "tfpd_exploration/scripts/run_m1_emg_syn3_stage0.py",
    "tfpd_exploration/scripts/run_m1_emg_syn3_pilot.py",
    "tfpd_exploration/tests/test_m1_emg_syn3_fcm_v1.py",
    RESULT_ROOT_RELATIVE,
    DESIGN_RELATIVE,
    WORKORDER_RELATIVE,
)

SEALED_FOREIGN_ROOTS = (
    "tfpd_exploration/results/m1_t0c1_prefix_v1",
    "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep",
)


def source_sessions_for_fold(fold: int) -> tuple[str, ...]:
    _require(fold in FOLD_TARGETS, "fold drift")
    target = FOLD_TARGETS[fold]
    return tuple(session for session in SESSIONS if session != target)


def verify_bound_documents(root: Path) -> None:
    design = Path(root) / DESIGN_RELATIVE
    workorder = Path(root) / WORKORDER_RELATIVE
    _require(design.is_file() and workorder.is_file(), "design/workorder missing")
    _require(sha256_bytes(design.read_bytes()) == DESIGN_SHA256, "design sha drift")
    _require(sha256_bytes(workorder.read_bytes()) == WORKORDER_SHA256, "workorder sha drift")


@dataclass(frozen=True)
class StageRootSpec:
    root_relative: str

    def payload(self) -> dict[str, object]:
        return {"schema": "m1_emg_syn3_stage_root_v1", "root_relative": self.root_relative}


def dry_cli_payload() -> dict[str, object]:
    return {
        "phase": PHASE,
        "design_sha256": DESIGN_SHA256,
        "workorder_sha256": WORKORDER_SHA256,
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
