"""Frozen operator-correction contract; no Torch or receipt reads at import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PlanError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PlanError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


SCHEMA = "m2_postfusion_operator_corrected_score_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_operator_corrected_score_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_POSTFUSION_OPERATOR_CORRECTED_SCORE_V1_20260902.md"
WORKORDER_SHA256 = "261925c75ba83c56e0f6a874e4d8a9db6144e269cf64e164a9aa42b3b8d3053a"
DESIGN_AUDIT_RELATIVE = "tfpd_exploration/docs/AUDIT_M2_POSTFUSION_TRAIN_SCORE_OPERATOR_MATCH_20260902.md"

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
ARMS = ("PF-MEAN", "PF-R1", "PF-R50")
LAWS = ("FIXED30", "UNCAPPED")
SURFACES = ("external_post30_local", "within_post30")
ROSTER_SIZES = {"external_post30_local": 6, "within_post30": 7}
BUDGET = 4
POOL_CAPACITY = 30
EXPECTED_ROWS = 78
CPU_DECODE_BATCH_SIZE = 1024

V2_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_checkpoint_score_v2"
V2_CLOSURE_SHA256 = "68d489aacd23611c625f98fee919430451fad251d3b681e0d01771bf546f982d"
V2_BODIES = {
    "attempt.json": "6bada93b3f6273a04866c0878f5367b9a9fcae4e5040e2a4ab2fcee7bb059bc5",
    "launch.json": "4e88d2b2df57430e36fcb15a5ab948ddb267534599f7507c2b53f65773a44e70",
    "input_authority.json": "92e85b8d0c1b27b4b40c857f4459969c76907e4a630057c5e697792f49cb657e",
    "score.json": "9dce9042360a4d835859532bd3ad97c0d586d74e25499a11f2898e2fce6adfef",
    "terminal.json": "75b5719e9cb46fa2c50e45131049ba1903a70a8923849f354a78b2f39eefd2f1",
}

# This is intentionally explicit rather than a glob.  It records the source
# graph that executes after the successor attempt, not test/result paths.
CLOSURE_RELATIVES = (
    WORKORDER_RELATIVE, DESIGN_AUDIT_RELATIVE,
    "tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/driver.py",
    "tfpd_exploration/scripts/run_m2_postfusion_operator_corrected_score_v1.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/laws.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/lifecycle.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
    "tfpd_exploration/src/m2_postfusion_variant_screen_v1/variants.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
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
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/src/data/__init__.py",
)


def validate_static(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).absolute()
    digest = sha256_file(root / WORKORDER_RELATIVE)
    require(digest == WORKORDER_SHA256, "operator-corrected workorder drift")
    return {"workorder_sha256": digest}
