"""Frozen constants for the M2 T4 activity-budget screen."""

from pathlib import Path


SCHEMA = "m2_t4_activity_budget_screen_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_M2_T4_ACTIVITY_BUDGET_SCREEN_V1_20260828.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1"
CHECKPOINT_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt"
)
CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
RIDGE_NORMALIZED_LAMBDA = 0.1
BUDGETS = (30, 10, 4)
ACTIVITY_HORIZON = 30
WINDOW_SIZE = 50
CHANNELS = 96
TRIAL_LENGTH = 100
BEHAVIOR_SCALE = 5.0
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE

