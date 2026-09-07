"""Static contract for the pseudo-MUA Precision-CDM V2 engineering screen."""

from __future__ import annotations

from pathlib import Path

SCHEMA = "pseudo_mua_precision_cdm_v2_screen_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_PSEUDO_MUA_PRECISION_CDM_V2_SCREEN_V1_20260828.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/pseudo_mua_precision_cdm_v2_screen_v1"

VIEW = "pseudo_mua"
SEED = 42
WINDOW_BINS = 50
ACTIVITY_STACK_LIMIT = 30
BEHAVIOR_SCALE = 5.0
RIDGE_NORMALIZED_LAMBDA = 0.1
BUDGETS = (30, 10, 4)
TRANSITION_BUDGETS = (10, 4)

SYSTEM_STATIC = "static_t4"
SYSTEM_ACTIVITY = "activity_only"
SYSTEM_ORDINARY = "ordinary_cdmd"
SYSTEM_PRECISION = "precision_cdmd_v2"
SYSTEMS_BY_BUDGET = {
    30: (SYSTEM_STATIC, SYSTEM_PRECISION),
    10: (SYSTEM_STATIC, SYSTEM_ACTIVITY, SYSTEM_ORDINARY, SYSTEM_PRECISION),
    4: (SYSTEM_STATIC, SYSTEM_ACTIVITY, SYSTEM_ORDINARY, SYSTEM_PRECISION),
}
CELL_ORDER = tuple(
    f"m{budget}_{system}"
    for budget in BUDGETS
    for system in SYSTEMS_BY_BUDGET[budget]
)
EXPECTED_SESSIONS = 15
EXPECTED_ROWS = EXPECTED_SESSIONS * len(CELL_ORDER)
DEFAULT_BATCH_SIZE = 1024

V9_ROOT_RELATIVE = (
    "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
)
NWB_ROOT_RELATIVE = "sua_exploration/data/dandi_000688"


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
