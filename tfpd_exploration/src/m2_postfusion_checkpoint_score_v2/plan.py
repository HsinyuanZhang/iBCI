"""Frozen V2 recovery constants; importing this module is stdlib-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


SCHEMA = "m2_postfusion_checkpoint_score_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_checkpoint_score_v2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_POSTFUSION_CHECKPOINT_SCORE_V2_IMPORT_RECOVERY_20260902.md"
WORKORDER_SHA256 = "44a081587804f11b157c5fa318c2e95ebf69bbdae7f34b35f8b0d15eaf0a77e4"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md"
DESIGN_SHA256 = "29afc5991ca6f287c0a9de0f5a20f737bd2823a9131fd4501752e9775925d4e2"

V1_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_checkpoint_score_v1"
V1_FAILURE_BODIES = {
    "attempt.json": "dc0ef024e657773cd58833288af524e5d2fb39af9ac7e496739133f6372f72d5",
    "launch.json": "aebce91e6a441db0c7a053e475f218b55019419abd9c4684130c760852a54fe3",
    "failure.json": "9b5dca51bea287d6620847a5e6877d0a7d59add8412027dae008b8225fdf6c04",
}
V1_FAILURE_ERROR_SHA256 = "f96a882aa286e4363a3142459c0ffe1c30793d3864dea9efff65f3bfdedc1d85"

# Every source leaf imported after V2 attempt is explicit.  This list is
# intentionally no-glob and excludes mutable result/test paths.
CLOSURE_RELATIVES = (
    WORKORDER_RELATIVE, DESIGN_RELATIVE,
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v2/__init__.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v2/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v2/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v2/driver.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/laws.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/lifecycle.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/driver.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/variants.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/directions.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/gates.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gates.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
)


def validate_static(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).absolute()
    observed = {"workorder_sha256": sha256_file(root / WORKORDER_RELATIVE),
                "design_sha256": sha256_file(root / DESIGN_RELATIVE)}
    require(observed["workorder_sha256"] == WORKORDER_SHA256, "V2 workorder drift")
    require(observed["design_sha256"] == DESIGN_SHA256, "V2 design drift")
    return observed
