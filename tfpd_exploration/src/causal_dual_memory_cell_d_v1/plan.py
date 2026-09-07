"""Static, standard-library-only plan for CDM-D Stage 0.

This module is deliberately safe for the zero-argument public CLI.  It does
not import NumPy, Torch, data adapters, a scorer, or any output primitive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"
PHASE = "stage0_synthetic_cpu_only"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md"
WORKORDER_SHA256 = "5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc"

B3S_ACTIVITY_STACK_LIMIT = 30
ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET = {4: 26, 10: 20, 30: 0}

PACKAGE_INIT_RELATIVE = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py"
PLAN_RELATIVE = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py"
CORE_RELATIVE = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py"
CLI_RELATIVE = "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py"

# These two production estimators are dependency-bound for the synthetic
# parity gates.  Stage-0 does not import either at runtime; tests independently
# compare its closed forms against them.
PRODUCTION_OLS_RELATIVE = "sua_exploration/mc_maze/d_optimal_calibration_design.py"
PRODUCTION_RIDGE_RELATIVE = "tfpd_exploration/src/calibration_budget_comparators_v1.py"

STAGE0_CLOSURE_PATHS = (
    WORKORDER_RELATIVE,
    PACKAGE_INIT_RELATIVE,
    PLAN_RELATIVE,
    CORE_RELATIVE,
    CLI_RELATIVE,
    TEST_RELATIVE,
    PRODUCTION_OLS_RELATIVE,
    PRODUCTION_RIDGE_RELATIVE,
)


class Stage0PlanError(ValueError):
    """Raised when the static Stage-0 closure is malformed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def closure_payload(root: Path) -> dict[str, Any]:
    """Return an explicit, non-globbed source closure and aggregate digest."""
    root = Path(root).resolve()
    rows: list[dict[str, str]] = []
    for relative in STAGE0_CLOSURE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise Stage0PlanError(f"Stage-0 closure path is not a regular file: {relative}")
        rows.append({"path": relative, "sha256": _sha256_bytes(path.read_bytes())})
    workorder = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
    if workorder != WORKORDER_SHA256:
        raise Stage0PlanError("CDM-D Stage-0 workorder SHA drift")
    return {
        "schema": "causal_dual_memory_cell_d_stage0_closure_v1",
        "paths": rows,
        "closure_sha256": _sha256_bytes(_canonical_json(rows)),
    }


def dry_plan(root: Path | None = None) -> dict[str, Any]:
    """Describe the inert Stage-0 route without constructing any runtime."""
    payload: dict[str, Any] = {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "groups": 4,
        "b3s_activity_stack_limit": B3S_ACTIVITY_STACK_LIMIT,
        "activity_fifo_capacity_by_support_budget": dict(ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET),
        "completed_trial_views": {
            "b3s_activity": "typed_interpolated_padded_spike_counts_[100,N]",
            "carrier_counts": "typed_native_rewarded_trial_binned_spike_counts_[T,N]",
            "carrier_rate_rule": "mean(native_counts_over_full_declared_rewarded_interval)/0.020",
            "velocity_validity": "typed_target_label_free_mask_with_own_prediction_interval",
        },
        "fit_modes": ["ordinary_ols_by_direction", "fixed_ridge_by_trial"],
        "execution_authorized": False,
        "opens_data": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_result_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        payload["closure"] = closure_payload(Path(root))
    return payload
