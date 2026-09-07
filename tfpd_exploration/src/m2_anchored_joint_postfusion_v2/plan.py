"""Static authorities for the narrow AJPF V2 incident successor."""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M2_ANCHORED_JOINT_POSTFUSION_V2_20260903.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_ANCHORED_JOINT_POSTFUSION_V2_20260903.md"
INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_ANCHORED_JOINT_POSTFUSION_V1_20260903.md"
DESIGN_SHA256 = "7a80ac424268fef47153ec2c1da182b47f7415c13458e701de7079e90d2c75e0"
WORKORDER_SHA256 = "906a6792770a301ea57c5bc17f370dbe5021217ee98a2c66c72911e1321cde36"
INCIDENT_SHA256 = "97396859268e2753a228b5d8c0c330ddadde8b9fbfac279246779953f776fdd5"
V1_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_joint_postfusion_v1/training"
V2_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_joint_postfusion_v2"
V2_TRAINING_ROOT_RELATIVE = V2_ROOT_RELATIVE + "/training"
V2_SCORE_ROOT_RELATIVE = V2_ROOT_RELATIVE + "/score"
SELECTED_STUDENT_STATE_SHA256 = "2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"
V1_FAILURE_SHA256 = "dfebd25532f241c16131679491db19e2247fb9b310d64cac87d8be4f9fc6b269"
V1_ATTEMPT_SHA256 = "616e2e7b25c6d4abaef3dec487f955f1fdacd99a986121df831b4ed9b125f526"
V1_LAUNCH_SHA256 = "129f59866935cf9caf17664989ff1ee236f14af0ae6aa797aeb310aaa87a0c6f"
STATIC_CLOSURE_RELATIVES = (
    DESIGN_RELATIVE, WORKORDER_RELATIVE, INCIDENT_RELATIVE,
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v2/__init__.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v2/plan.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v2/lifecycle.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v2/runner.py",
    "tfpd_exploration/src/m2_anchored_joint_postfusion_v2/production.py",
    "tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v2.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
    "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
)
