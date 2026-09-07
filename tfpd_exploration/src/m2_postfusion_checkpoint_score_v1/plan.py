"""Pure frozen contract; importing it cannot import Torch or access a root."""
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


SCHEMA = "m2_postfusion_checkpoint_score_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_checkpoint_score_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_POSTFUSION_CHECKPOINT_SCORE_V1_20260902.md"
WORKORDER_SHA256 = "0e38a9a2afeeb474f240c07c9555561d9c7de2b3899e4df7191124f8879f95c0"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md"
DESIGN_SHA256 = "29afc5991ca6f287c0a9de0f5a20f737bd2823a9131fd4501752e9775925d4e2"
ARMS = ("PF-MEAN", "PF-R1", "PF-R50")
COMPLEXITY_ORDER = ARMS
LAWS = ("FIXED30", "UNCAPPED")
SURFACES = ("external_post30_local", "within_post30")
ROSTER_SIZES = {"external_post30_local": 6, "within_post30": 7}
BUDGET = 4
POOL_CAPACITY = 30
EXPECTED_ROWS = len(ARMS) * len(LAWS) * sum(ROSTER_SIZES.values())
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
# The reviewed CPU post-fusion probe used this batch to finish its complete
# frozen grid in minutes.  It is an inference chunking law, never a training
# batch/sampler setting.
CPU_DECODE_BATCH_SIZE = 1024

# Immutable, local-only historical POOLED k4 reference.  This is a practical
# comparator, not a matched Post-Fusion training control.
POOLED_COMPARATOR_ROOT_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1"
POOLED_COMPARATOR_BODY = "score.json"
POOLED_COMPARATOR_SHA256 = "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
POOLED_COMPARATOR_SCHEMA = "m2_precision_cdm_v2_screen_v1"
POOLED_COMPARATOR_CELL_ORDER = (
    "m30_static_t4", "m30_precision_cdmd_v2", "m10_static_t4", "m10_activity_only",
    "m10_ordinary_cdmd", "m10_precision_cdmd_v2", "m4_static_t4", "m4_activity_only",
    "m4_ordinary_cdmd", "m4_precision_cdmd_v2",
)
POOLED_COMPARATOR_PROTOCOL = {
    "m10_support": "first10_finite_direction_from_first30",
    "m4_support": "doptimal_four_from_first30",
    "official_submission_claimed": False,
    "parameter_updates": 0,
    "query": "post_first30_common_windows",
    "short_trial_policy": "activity_advances_carrier_rejects_velocity_shape_no_padding_no_cross_trial_history",
    "surfaces": ["within_post30", "external_post30_local"],
    "target_backward": 0,
    "target_gradients": 0,
    "target_state_uses": 0,
}
POOLED_SESSION_ORDER = {
    "within_post30": (
        "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
        "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
        "ses-2020-10-28-Run1",
    ),
    "external_post30_local": (
        "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
        "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
    ),
}

# Deliberately deferred: no canonical screen result is opened during code
# construction.  A post-terminal audit must replace None with exact literals.
LIVE_PRODUCER_LITERALS = {
    "screen_root_relative": "tfpd_exploration/results/m2_postfusion_variant_screen_v1/screen",
    "screen_closure_sha256": "d1f287671957e4244594530e9dbb2b9bec9555cd6c7fe280741e5b0087c4bf51",
    "sha256": {
        "attempt.json": "65b3959cb347f171b67e9c33225583bd72aa116f6a32c74ba945969ab6002a19",
        "launch.json": "1d48fcf92bab77713dba79badb174c0a7a8393f4b7a4e31867cd2840fd79e80c",
        "source_authority.json": "aaa7407fe974fe628f433236d0050e3a91bb7eb05ac0ce17606395b45baf6ff0",
        "screen.json": "3eb237ac5f87c6fb405f70ed7afaf9f8f930a6ab633fc32b92b9a9d969b18003",
        "terminal.json": "9919651c93b384c76e89fcc5248c9855f0b931604475b01b48611df104fb2625",
        "source_best_pf_mean.pt": "c1b4557e036796bd88f406cc4882821186a110001c1a88c47845bcad52c04a03",
        "source_best_pf_r1.pt": "48c5ad8f1cbd24f1ad2fa0d3dc4935045c187e8229e5864a868f42bd49b722a8",
        "source_best_pf_r50.pt": "2a0ccff4009444af3b72add02c6c11ca09bdd8a969abdb482714701ebfec718b",
    },
    "checkpoints": {"PF-MEAN": {"epoch": 12, "student_state_sha256": "80169f5e82d2d6e38f7fc5a6dfc65b7ed9739fd8e848862c93357d96370c5e2f"},
                    "PF-R1": {"epoch": 12, "student_state_sha256": "9ad23d0d857ee707e10a2b1b32444602a43daa629a4a73517d8f70ff57149a2c"},
                    "PF-R50": {"epoch": 12, "student_state_sha256": "7fc4d233650375f87cd402b84c69fe1396262b3ce6568badc66b44ffc8e06749"}},
}

CLOSURE_RELATIVES = (
    WORKORDER_RELATIVE, DESIGN_RELATIVE,
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
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    # The exact governed-array receipt framing used by sealed POOLED rows.
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    # Direct, post-attempt model/data/scoring runtime imports.  No result or
    # test path is admitted through this explicit closure.
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
)


def validate_static(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).absolute()
    observed = {"workorder_sha256": sha256_file(root / WORKORDER_RELATIVE),
                "design_sha256": sha256_file(root / DESIGN_RELATIVE)}
    require(observed["workorder_sha256"] == WORKORDER_SHA256, "checkpoint-score workorder drift")
    require(observed["design_sha256"] == DESIGN_SHA256, "checkpoint-score design drift")
    return observed
