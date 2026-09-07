"""Constants for the three-seed M2 activity-budget replication."""

from pathlib import Path


SCHEMA = "m2_t4_activity_budget_seed_replication_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_seed_replication_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_M2_T4_ACTIVITY_BUDGET_SEED_REPLICATION_V1_20260828.md"
)
PARENT_SCORE_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
PARENT_SCORE_SHA256 = "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
SEEDS = (42, 43, 44)
CHECKPOINTS = {
    42: (
        "streaming_calibration_exp/outputs/streaming_calibration/"
        "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt",
        "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
    ),
    43: (
        "streaming_calibration_exp/outputs/streaming_calibration/"
        "m2_spint_t4_mainline_fp32_v1_t4_m2_s43_20260730_133723/checkpoints/best.ckpt",
        "3723c173bf9135393eaa4fbf974c8135e591de75d00c6d79287d053d467aefa4",
    ),
    44: (
        "streaming_calibration_exp/outputs/streaming_calibration/"
        "m2_spint_t4_mainline_fp32_v1_t4_m2_s44_20260730_135649/checkpoints/best.ckpt",
        "e7789c332dcf5e4a09bdc8a3bf0f84a4e61c7775dabd791927819dd639582030",
    ),
}
DEFAULT_BATCH_SIZE = 1024
EXPECTED_SESSIONS = {"within_post30": 7, "external_official_query": 6}


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE
