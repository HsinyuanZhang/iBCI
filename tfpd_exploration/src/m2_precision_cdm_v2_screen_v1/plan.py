"""Static native-M2 Precision-CDM V2 screen contract."""

from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import plan as shared

SCHEMA = "m2_precision_cdm_v2_screen_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_PRECISION_CDM_V2_SCREEN_V1_20260828.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1"
SEED = 42
WINDOW_BINS = 50
ACTIVITY_STACK_LIMIT = 30
BEHAVIOR_SCALE = 5.0
MODEL_BIN_SECONDS = 0.020
RIDGE_NORMALIZED_LAMBDA = 0.1
BUDGETS = shared.BUDGETS
TRANSITION_BUDGETS = shared.TRANSITION_BUDGETS
SYSTEM_STATIC = shared.SYSTEM_STATIC
SYSTEM_ACTIVITY = shared.SYSTEM_ACTIVITY
SYSTEM_ORDINARY = shared.SYSTEM_ORDINARY
SYSTEM_PRECISION = shared.SYSTEM_PRECISION
SYSTEMS_BY_BUDGET = shared.SYSTEMS_BY_BUDGET
CELL_ORDER = shared.CELL_ORDER
SURFACES = ("within_post30", "external_post30_local")
EXPECTED_SESSIONS_BY_SURFACE = {"within_post30": 7, "external_post30_local": 6}
EXPECTED_SESSIONS = sum(EXPECTED_SESSIONS_BY_SURFACE.values())
EXPECTED_ROWS = EXPECTED_SESSIONS * len(CELL_ORDER)
DEFAULT_BATCH_SIZE = 1024


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
