"""Static AJPF V1 science and execution authorities."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M2_ANCHORED_JOINT_POSTFUSION_V1_20260903.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_JOINT_POSTFUSION_V1_20260903.md"
DESIGN_SHA256 = "e92eafea22f473ce91eee57223436c61df6d0f881690dd8a3eccc1f24e193523"
WORKORDER_SHA256 = "9b50f64b39fadf795c66d2aea45c1a0bd471976466ae16c96231b6bfb7999fc1"

ARMS = ("J-NATIVE", "J-R1", "J-MEAN")
ARM_ORDER = ARMS
POOL_CYCLE = (4, 10, 30)
EPOCHS = 12
SEED = 42
BATCH_SIZE = 32
SOURCE_GROUPS_PER_EPOCH = 3455
SOURCE_COORDINATES = 91717
SOURCE_COORDINATE_SHA256 = "06f19e267cb2029371bb4b3b3b6019314db593edfe44417d9305e7498deb7cba"
SOURCE_BATCH_SHA256 = "dffe4234befcc197aa47c2c6171b9c0fc9194085d96d2667d56b7308de533095"
SUPPORT_COUNT = 4
NON_SUPPORT_COMPLETIONS = {4: 0, 10: 6, 30: 26}
BEHAVIOR_SCALING_FACTOR = 5.0
MODEL_BIN_SECONDS = 0.020
ENCODER_ALPHA_LR = 1e-4
DECODER_LR = 1e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
STUDENT_STATE_SHA256 = "2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"
ACTIVITY_AUTHORITY = "pooled_g00m_linear"
TRAINING_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_joint_postfusion_v1/training"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_joint_postfusion_v1/score"
APFG_SOURCE_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_postfusion_gate_v1"
APFG_SOURCE_BODIES = {
    "attempt.json": "e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7",
    "launch.json": "9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef",
    "source_authority.json": "ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf",
    "alpha_selection.json": "c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908",
    "input_authority.json": "41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24",
    "failure.json": "6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d",
}
POOLED_ROOT_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1"
POOLED_SCORE_SHA256 = "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"

STATIC_CLOSURE_RELATIVES = (
    DESIGN_RELATIVE, WORKORDER_RELATIVE,
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/controller.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/adapter.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/dropout.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/runtime.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/runner.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/binding.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/lifecycle.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/scorer.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/driver.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/physical.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/production.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v1/joint_training.py",
    "tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v1.py",
    # Explicit production transitive closure.  These are source authorities,
    # not convenient imports: an admission made against a different one is a
    # different experiment.
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    # Clean qualified late-bound import trace (including package initializers).
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/__init__.py",
    "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/__init__.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/transition.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py",
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

# Every production late-bound repo-local module imported by the clean
# no-data trace.  This is deliberately an explicit tuple rather than a
# directory walk/glob: additions need review and enter the closure above.
LATE_BOUND_IMPORT_MODULES = (
    "tfpd_exploration.src.m2_anchored_joint_postfusion_v1.production",
    "tfpd_exploration.src.m2_anchored_joint_postfusion_v1.runner",
    "tfpd_exploration.src.m2_anchored_joint_postfusion_v1.scorer",
    "tfpd_exploration.src.m2_anchored_joint_postfusion_v1.joint_training",
    "tfpd_exploration.src.pit_m2_v1.trainer",
    "tfpd_exploration.src.m2_anchored_postfusion_gate_v1.source_replay",
    "tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical",
    "tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.binding",
    "tfpd_exploration.src.cdm_p1_m2_local_v1.replay",
    "tfpd_exploration.src.m2_precision_cdm_v2_screen_v1.physical",
    "tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1.core",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static(repo_root: Path = REPO_ROOT) -> None:
    for relative, expected in ((DESIGN_RELATIVE, DESIGN_SHA256), (WORKORDER_RELATIVE, WORKORDER_SHA256)):
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"AJPF static authority drift: {relative}")
    if ARMS != ("J-NATIVE", "J-R1", "J-MEAN") or POOL_CYCLE != (4, 10, 30):
        raise RuntimeError("AJPF arm/pool authority drift")
