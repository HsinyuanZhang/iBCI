"""Frozen constants for the M2 same-query comparator matrix."""

from pathlib import Path


SCHEMA = "m2_same_query_comparator_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_M2_SAME_QUERY_COMPARATOR_V1_20260828.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_same_query_comparator_v1"
PARENT_SCORE_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
PARENT_SCORE_SHA256 = "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce"
T4_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SPINT_CHECKPOINT_SHA256 = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"

BUDGETS = (30, 10, 4)
ACTIVITY_HORIZON = 30
WINDOW_SIZE = 50
CHANNELS = 96
OUTPUT_DIM = 2
TRIAL_LENGTH = 100
BEHAVIOR_SCALE = 5.0
T4_RIDGE_NORMALIZED_LAMBDA = 0.1
DIRECT_RIDGE_NORMALIZED_LAMBDA = 1.0
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def parent_score(repo_root: Path) -> Path:
    return repo_root / PARENT_SCORE_RELATIVE
