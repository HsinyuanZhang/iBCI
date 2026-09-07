"""Pure frozen contract for APFG V1; this module must not import Torch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class PlanError(ValueError):
    """Raised when a static APFG contract is not exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SCHEMA = "m2_anchored_postfusion_gate_v1"
CELL = "M2_ANCHORED_POSTFUSION_GATE_V1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_postfusion_gate_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_POSTFUSION_GATE_V1_20260902.md"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_ANCHORED_POSTFUSION_GATE_V1_20260902.md"
DESIGN_SHA256 = "2b78e5be4de814d5f0bf00048df781c80d5bed695cad8c550ddd3ea3021d4e4f"
WORKORDER_SHA256 = "825bffad524e2bfc65eecb65d82f1d369bc64fc9ad15233ca50a1754122957c4"
LIVE_WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_POSTFUSION_GATE_LIVE_V1_20260902.md"
LIVE_WORKORDER_SHA256 = "c173b69e47710122731b92648f1d2811bb24387642a1fa647c34c7c7e90d4454"

SELECTED_T4_POOLED_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SEED = 42
TRAIN_BATCH_SIZE = 32
TARGET_CPU_DECODE_BATCH_SIZE = 1024
EPOCHS = 12
MILESTONES = (1, 3, 6, 12)
ADAM_LR = 1.0e-4
ADAM_WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
ADAM_AMSGRAD = False
ADAM_CLIP_GRAD_NORM = None
ADAM_SCHEDULER = None
TRIAL_LENGTH = 100
WINDOW_BINS = 50
SIDE_DIM = 4
SOURCE_SESSION_COUNT = 7
FIT_SESSION_COUNT = 5
VALIDATION_SESSION_COUNT = 2
SUPPORT_COUNT = 4
POOL_CYCLE = (4, 10, 30)
NON_SUPPORT_COMPLETIONS = {4: 0, 10: 6, 30: 26}
SOURCE_BATCH_MAX_MEMBERS = 32
SOURCE_BATCH_CONTROLLER = "POOL_CYCLE[(epoch_one_indexed-1 + canonical_batch_ordinal)%3]"
SOURCE_SELECTION_OPERATOR = "source_only_b30_dopt_k4_selected_support4_fixed_ridge_carrier_decode_before_commit_uncapped_equal_session_r2"
TRAINING_LOSS = "source_behavior_task_only_scaled_last_bin_mse"
BEHAVIOR_SCALING_FACTOR = 5.0
PREDICT_SCALED_BEHAVIOR = True
TEACHER_FORWARD_CALLS = 0
ACTIVITY_AUTHORITY = "pooled_g00m_linear"
GPU_PHYSICAL_INDEX = 0
LIVE_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": "/home/xinyuan/Work_host/SPINT",
}

CORRECTED_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_operator_corrected_score_v1"
CORRECTED_CLOSURE_SHA256 = "95b8e9e07e39700089e8364f67b15a020d17eb723a56cb18c2222305a1b25225"
CORRECTED_BODIES = {
    "attempt.json": "0f4e441c77f6c48c4358f0face4048ecf668bf8a6936818df3c279f436e4f57d",
    "launch.json": "10f4ed88a31f2cf09016c854dbbf82d51e1e4f5e670a05fbbb1f18fb6ccfa9aa",
    "input_authority.json": "f0bd485ec578b208c1dddb2d2b8f9c5c90d9677262b7debcfef4df9697d3ec89",
    "score.json": "156d7fab27c70bb01a7804bfdbb3434164c71dcab81ce3b1ad8a371eb169175d",
    "terminal.json": "f0f137c14c99da6e843b4887b4f3102610844fc0a3ee73fc0dfae4c547c41b01",
}

# This immutable descriptor vocabulary is sufficient for a root-only opaque
# issuer.  It is not exposed through the public CLI and does not authorize a
# launch by itself; descriptor reads happen only after attempt publication.
LIVE_PRODUCER_LITERALS: Mapping[str, object] | None = {
    "corrected_root_relative": CORRECTED_ROOT_RELATIVE,
    "corrected_body_sha256": CORRECTED_BODIES,
    "corrected_closure_sha256": CORRECTED_CLOSURE_SHA256,
    "selected_t4_pooled_checkpoint_sha256": SELECTED_T4_POOLED_CHECKPOINT_SHA256,
    "pooled_comparator_root_relative": "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1",
    "pooled_comparator_score_sha256": "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6",
    "source_session_count": SOURCE_SESSION_COUNT,
    "surfaces": ("external_post30_local", "within_post30"),
}

CLOSURE_RELATIVES = (
    WORKORDER_RELATIVE,
    LIVE_WORKORDER_RELATIVE,
    DESIGN_RELATIVE,
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/laws.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/scoring.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/selection.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/driver.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/binding.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/runtime.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/training.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/lifecycle.py",
    "tfpd_exploration/scripts/run_m2_anchored_postfusion_gate_v1.py",
    # Frozen runtime authority to be composed only after post-attempt
    # selected-checkpoint admission.  These source leaves are explicit; tests
    # and result roots are intentionally absent.
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
)


def validate_static(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).absolute()
    observed = {
        "workorder_sha256": sha256_file(root / WORKORDER_RELATIVE),
        "design_sha256": sha256_file(root / DESIGN_RELATIVE),
    }
    require(observed["workorder_sha256"] == WORKORDER_SHA256, "APFG workorder drift")
    observed["live_workorder_sha256"] = sha256_file(root / LIVE_WORKORDER_RELATIVE)
    require(observed["live_workorder_sha256"] == LIVE_WORKORDER_SHA256, "APFG live workorder drift")
    require(observed["design_sha256"] == DESIGN_SHA256, "APFG design drift")
    require(len(POOL_CYCLE) == 3 and tuple(POOL_CYCLE) == (4, 10, 30), "APFG pool cycle drift")
    require(NON_SUPPORT_COMPLETIONS == {4: 0, 10: 6, 30: 26}, "APFG pool law drift")
    return observed


def closure(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).absolute()
    result: dict[str, str] = {}
    for relative in CLOSURE_RELATIVES:
        path = root / relative
        require(path.is_file(), f"APFG closure file missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def dry_plan(repo_root: Path) -> dict[str, Any]:
    """Public inert payload; it intentionally cannot create a capability."""
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "status": "INERT_PUBLIC_CLI__ROOT_ONLY_OPAQUE_CAPABILITY_REQUIRED",
        "selected_t4_pooled_checkpoint_sha256": SELECTED_T4_POOLED_CHECKPOINT_SHA256,
        "parameter_law": {"trainable": ["alpha"], "alpha_shape": [], "inherited_weights_frozen": True},
        "source_selection": {"lexical_fit_sessions": FIT_SESSION_COUNT,
                             "lexical_validation_sessions": VALIDATION_SESSION_COUNT,
                             "epochs": EPOCHS, "earliest_best": True, "refit_from_zero": True},
        "causal_pool": {"cycle": list(POOL_CYCLE), "support_count": SUPPORT_COUNT,
                        "recent_non_support_completed": dict(NON_SUPPORT_COMPLETIONS),
                        "clamp_pad_repeat_or_future": False},
        "authority": validate_static(Path(repo_root)),
        "live_literals_bound": LIVE_PRODUCER_LITERALS is not None,
    }


__all__ = (
    "PlanError", "require", "canonical_json", "sha256_bytes", "sha256_file", "SCHEMA", "CELL",
    "RESULT_ROOT_RELATIVE", "WORKORDER_RELATIVE", "DESIGN_RELATIVE", "DESIGN_SHA256",
    "SELECTED_T4_POOLED_CHECKPOINT_SHA256", "SEED", "TRAIN_BATCH_SIZE", "TARGET_CPU_DECODE_BATCH_SIZE", "EPOCHS", "MILESTONES",
    "ADAM_LR", "ADAM_WEIGHT_DECAY", "TRIAL_LENGTH", "WINDOW_BINS", "SIDE_DIM", "SOURCE_SESSION_COUNT",
    "FIT_SESSION_COUNT", "VALIDATION_SESSION_COUNT", "SUPPORT_COUNT", "POOL_CYCLE", "NON_SUPPORT_COMPLETIONS",
    "GPU_PHYSICAL_INDEX", "LIVE_ENV", "ADAM_BETAS", "ADAM_EPS", "ADAM_AMSGRAD", "ADAM_CLIP_GRAD_NORM", "ADAM_SCHEDULER",
    "LIVE_PRODUCER_LITERALS", "CLOSURE_RELATIVES", "validate_static", "closure", "dry_plan",
)
