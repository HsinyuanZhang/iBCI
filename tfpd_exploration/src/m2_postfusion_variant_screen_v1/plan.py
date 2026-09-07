"""Pure, frozen contract for the fast M2 post-fusion variant screen.

This module must stay free of Torch, data access, checkpoint access, and GPU
queries.  Runtime code may only be reached through the route-owned capability
in ``driver`` after an immutable attempt has been published.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class PlanError(ValueError):
    """Raised for a frozen-contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SCHEMA = "m2_postfusion_variant_screen_v1"
CELL = "M2_POSTFUSION_VARIANT_SCREEN_V1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_variant_screen_v1"
SCREEN_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/screen"
SCORE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/score"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTFUSION_VARIANT_SCREEN_V1_20260902.md"
WORKORDER_SHA256 = "6c7dfc9d8005181ef0273ac6f74fb05bbb23fbd10536ef9d3260d4291779280a"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md"
DESIGN_SHA256 = "29afc5991ca6f287c0a9de0f5a20f737bd2823a9131fd4501752e9775925d4e2"

PIT_PACKAGE = "tfpd_exploration/src/pit_m2_v1"
POSTFUSION_PROBE_PACKAGE = "tfpd_exploration/src/m2_postfusion_probe_v1"
KCURVE_EXTENSION_PACKAGE = "tfpd_exploration/src/m2_kcurve_ext_v1"

ARMS = ("PF-MEAN", "PF-R1", "PF-R50")
ARM_ADDED_PARAMETERS = {"PF-MEAN": 0, "PF-R1": 1, "PF-R50": 50}
ARM_TIEBREAK_ORDER = ARMS
POOL_CYCLE = (30, 10, 4)
SEED = 42
TRAIN_BATCH_SIZE = 32
TRAINING_CALIBRATION_MEMBERS = 33
TRIAL_LENGTH = 100
WINDOW_BINS = 50
SIDE_DIM = 4
SOURCE_EPOCHS_SCREEN = 12
SCREEN_MILESTONES = (3, 6, 12)
ADAM_LR = 1.0e-4
ADAM_WEIGHT_DECAY = 0.0
GPU_PHYSICAL_INDEX = 0
MIN_FREE_VRAM_GIB = 4
SCREEN_HARD_CAP_SECONDS = 180 * 60
SMOKE_SHARED_STEPS = 12
SOURCE_TIE_EPSILON = 0.002
SOURCE_MONITOR_SESSION_COUNT = 7
SOURCE_MONITOR_WINDOW_COUNT = 1011
SOURCE_TRAIN_WINDOW_COUNT = 115911

SEALED_AUTHORITY = {
    "pit_plan": "tfpd_exploration/src/pit_m2_v1/plan.py",
    "sealed_checkpoint_sha256": "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
    "teacher_checkpoint_sha256": "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec",
    "normalization_sha256": "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e",
    "decoder": "frozen",
    "source_only": True,
}

# Explicit, no-glob execution closure.  These are the route-owned controller
# plus every directly imported PIT, lifecycle, loader, and streaming-model
# source leaf that can affect a future admitted screen.  Checkpoint hashes do
# not attest these current source bytes.  It intentionally contains neither
# tests, result roots, nor a filesystem glob.
STATIC_CLOSURE_RELATIVES = (
    WORKORDER_RELATIVE,
    DESIGN_RELATIVE,
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/controller.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/variants.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/selection.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/lifecycle.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/runner.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/driver.py",
    "tfpd_exploration/src/pit_m2_v1/__init__.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    # Direct runtime imports reached through PIT's sealed construction stack.
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/src/metrics/__init__.py",
    "streaming_calibration_exp/src/metrics/baseline.py",
    "streaming_calibration_exp/src/metrics/run_artifacts.py",
    "streaming_calibration_exp/src/utils/__init__.py",
    "streaming_calibration_exp/src/utils/instantiators.py",
    "streaming_calibration_exp/src/utils/logging_utils.py",
    "streaming_calibration_exp/src/utils/pylogger.py",
    "streaming_calibration_exp/src/utils/rich_utils.py",
    "streaming_calibration_exp/src/utils/utils.py",
    "streaming_calibration_exp/src/data/__init__.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/src/data/validation_protocol.py",
    "streaming_calibration_exp/src/data/falcon_t4_features.py",
    "streaming_calibration_exp/src/data/falcon_d4_features.py",
    "streaming_calibration_exp/src/data/falcon_k4_features.py",
    "streaming_calibration_exp/src/data/afc4_xls_v2_adapter.py",
    "streaming_calibration_exp/src/data/falcon_n4_features.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/falcon_module.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/neuron_dropout.py",
    "streaming_calibration_exp/src/models/components/carrier_noise_augmentation.py",
    "streaming_calibration_exp/src/models/components/correspondence_breaking.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "streaming_calibration_exp/third_party/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
    "streaming_calibration_exp/third_party/falcon_challenge/__init__.py",
    "streaming_calibration_exp/third_party/falcon_challenge/filtering.py",
)


def validate_static_authority(repo_root: Path) -> Mapping[str, str]:
    """Check the two reviewed documents without opening data or Torch."""
    root = Path(repo_root).absolute()
    observed = {
        "workorder_sha256": sha256_file(root / WORKORDER_RELATIVE),
        "design_sha256": sha256_file(root / DESIGN_RELATIVE),
    }
    _require(observed["workorder_sha256"] == WORKORDER_SHA256, "postfusion workorder drift")
    _require(observed["design_sha256"] == DESIGN_SHA256, "postfusion design drift")
    return observed


def validate_pool_cycle(cycle: object = POOL_CYCLE) -> tuple[int, ...]:
    values = tuple(cycle) if isinstance(cycle, (tuple, list)) else ()
    _require(values == POOL_CYCLE, "postfusion pool cycle must be exactly (30,10,4)")
    _require(all(type(value) is int and 1 <= value <= TRAINING_CALIBRATION_MEMBERS for value in values),
             "postfusion pool cycle member range drift")
    return tuple(int(value) for value in values)


def dry_plan(repo_root: Path) -> dict[str, Any]:
    """Public inert CLI payload; intentionally contains no capability."""
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "status": "INERT_PLAN_ONLY__NO_CAPABILITY",
        "arms": list(ARMS),
        "added_parameter_counts": dict(ARM_ADDED_PARAMETERS),
        "pool_cycle": list(validate_pool_cycle()),
        "epochs": {"screen": SOURCE_EPOCHS_SCREEN, "milestones": list(SCREEN_MILESTONES)},
        "source_monitor": {
            "name": "source_heldin_in_sample_monitor",
            "checkpoint_selection_is_in_sample_descriptive_only": True,
            "all_three_source_best_checkpoints_retained": True,
        },
        "matched_prefusion_control_trained": False,
        "gpu_contract": {"physical_index": GPU_PHYSICAL_INDEX, "other_gpu_touched": False},
        "authority": dict(validate_static_authority(Path(repo_root))),
    }


__all__ = (
    "PlanError", "SCHEMA", "CELL", "RESULT_ROOT_RELATIVE", "SCREEN_ROOT_RELATIVE",
    "SCORE_ROOT_RELATIVE", "WORKORDER_RELATIVE",
    "WORKORDER_SHA256", "DESIGN_RELATIVE", "DESIGN_SHA256", "ARMS",
    "ARM_ADDED_PARAMETERS", "ARM_TIEBREAK_ORDER", "POOL_CYCLE", "SEED",
    "TRAIN_BATCH_SIZE", "TRAINING_CALIBRATION_MEMBERS", "TRIAL_LENGTH", "WINDOW_BINS",
    "SIDE_DIM", "SOURCE_EPOCHS_SCREEN", "SCREEN_MILESTONES", "ADAM_LR", "ADAM_WEIGHT_DECAY", "GPU_PHYSICAL_INDEX",
    "MIN_FREE_VRAM_GIB", "SCREEN_HARD_CAP_SECONDS", "SMOKE_SHARED_STEPS",
    "SOURCE_TIE_EPSILON", "SEALED_AUTHORITY", "STATIC_CLOSURE_RELATIVES",
    "canonical_json_bytes", "sha256_bytes", "sha256_file", "validate_static_authority",
    "validate_pool_cycle", "dry_plan",
)
