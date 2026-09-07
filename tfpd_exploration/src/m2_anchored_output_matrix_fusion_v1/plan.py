"""Torch-free immutable AOF-M V1 route contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "m2_anchored_output_matrix_fusion_v1"
ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_output_matrix_fusion_v1/source_oof"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_ANCHORED_OUTPUT_MATRIX_FUSION_V1_20260903.md"
DESIGN_SHA256 = "8a3e84af290c9253a518251192ab318c7a0d6b9c51f624e9d84973f559c3e55e"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_OUTPUT_MATRIX_FUSION_V1_20260903.md"
WORKORDER_SHA256 = "ed633a807449cb6be1fee5e295ea93c29ec34163cd705833a0c3c0270f8c47d1"

AOF_V1_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_output_fusion_v1/source_screen"
AOF_V1_CLOSURE_SHA256 = "28f54813f90563eb779cf76af931b3c9cef4dd1bc1eb34955c372874bfabdd5f"
AOF_V1_BODIES = {
    "attempt.json": "a624183bcd6d1e21dddbb03320aab19a0b156c22e7a350f49d7adb60e5c6518c",
    "launch.json": "d3b6240526b81946d0e43541bbdc960e3248d19ea28c59023371df1f757f877d",
    "source_authority.json": "ac3e7fb3158f2fecddeb27eeec39dabf6532be4699380a2c8cc3653d9b96ccf8",
    "paired_output_authority.json": "dc988ed139c89f07c2e43388402f4670eb7c74d62b308f9cc1f3bd8f567fc17a",
    "fit.json": "b0aba60f798e5364252b7da2b573b02064912f3e20baf0577fb85f5af4f6044a",
    "validation.json": "b0f196cfd554c7c0cb2ead44ca249fc13e4167ec99b5501812448dbb210aab8d",
    "terminal.json": "ab7f864e8922dc0afe7ce445f09e043f206a4eb55b70ce7620e9373ecc8c71cd",
}

GPU_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
ENV = {
    "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": "/home/xinyuan/Work_host/SPINT",
}

SESSIONS = (
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
OUTPUT_DIMENSION = 2
CONDITION_LIMIT = 1e8
LAMBDA_MAX_MINIMUM = 1e-12
EIGEN_RATIO_MINIMUM = 1e-8
RELATIVE_RESIDUAL_MAXIMUM = 1e-12

CLOSURE = (
    DESIGN_RELATIVE, WORKORDER_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/binding.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/core.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/runner.py",
    "tfpd_exploration/src/m2_anchored_output_matrix_fusion_v1/driver.py",
    "tfpd_exploration/scripts/run_m2_anchored_output_matrix_fusion_v1.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/core.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/runner.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/driver.py",
    "tfpd_exploration/docs/DESIGN_ANCHORED_OUTPUT_FUSION_V1_20260902.md",
    "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_OUTPUT_FUSION_V1_20260902.md",
    "tfpd_exploration/scripts/run_m2_anchored_output_fusion_v1.py",
    "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.receipt.json",
    "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/export_act30_dopt4_payload.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/runtime.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/selection.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/training.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/pit_m2_v1/__init__.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closure(root: Path) -> dict[str, str]:
    root = Path(root)
    if len(CLOSURE) != len(set(CLOSURE)):
        raise ValueError("AOF-M duplicate closure leaf")
    result: dict[str, str] = {}
    for relative in CLOSURE:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"AOF-M closure leaf missing/symlink: {relative}")
        result[relative] = file_sha256(path)
    if tuple(result) != CLOSURE:
        raise ValueError("AOF-M closure order/key-set drift")
    return result
