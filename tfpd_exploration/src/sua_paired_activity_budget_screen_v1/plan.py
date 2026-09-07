"""Frozen constants for the paired DANDI 000688 activity-budget screen."""

from pathlib import Path


SCHEMA = "sua_paired_activity_budget_screen_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_SUA_PAIRED_ACTIVITY_BUDGET_SCREEN_V1_20260828.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/sua_paired_activity_budget_screen_v1"
V9_ROOT_RELATIVE = (
    "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
)
V9_MANIFEST_SHA256 = "f8f9c0371bdf98d72807bb0563041b384c4ed8e39b0b167e6ebde976b4f455b0"
V9_CONTRACT_SHA256 = "9f06357a46544312c3a8b405fa6fa8800c2bb5edb5c344764156294e773e1ca6"
LABEL_BUDGET_ROOT_RELATIVE = (
    "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full"
)
LABEL_BUDGET_AGGREGATE_RELATIVE = (
    f"{LABEL_BUDGET_ROOT_RELATIVE}/aggregate/endpoint_aggregate_torchmetrics151.json"
)
LABEL_BUDGET_AGGREGATE_SHA256 = (
    "12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2"
)
NWB_ROOT_RELATIVE = "sua_exploration/data/dandi_000688"
VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
EXPECTED_SESSIONS = 15
EXPECTED_QUERY_WINDOWS_PER_VIEW = 708_795
HISTORY_BINS = 50
TRIAL_LENGTH = 100
ACTIVITY_HORIZON = 30
M10 = 10
M4 = 4
RIDGE_NORMALIZED_LAMBDA = 0.1
BEHAVIOR_SCALE = 5.0
DEFAULT_BATCH_SIZE = 1024
EAGER_CACHED_PARITY_ATOL = 1.0e-5
BOOTSTRAP_SEED = 688_410_30
BOOTSTRAP_DRAWS = 10_000
REFERENCE_CELL = "ols_m10_activity30_reference"
NEW_CELLS = (
    "ols_m10_activity10",
    "ridge_m4_activity4",
    "ridge_m4_activity30",
)
CELL_ORDER = (REFERENCE_CELL, *NEW_CELLS)
EXPECTED_REFERENCE_ROWS = EXPECTED_SESSIONS * len(VIEWS) * len(SEEDS)
EXPECTED_NEW_ROWS = EXPECTED_REFERENCE_ROWS * len(NEW_CELLS)
EXPECTED_ALIGNED_ROWS = EXPECTED_REFERENCE_ROWS * len(CELL_ORDER)


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE
