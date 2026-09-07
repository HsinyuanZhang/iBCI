from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1 import plan  # noqa: E402
from budget_matched_posterior_cal_aug_v1 import source_audit_v2 as v2  # noqa: E402


def _session(index: int, *, missing_positions=()):
    theta = np.resize(np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2]), 30).astype(float)
    truth = theta.copy()
    design = np.stack((np.ones(30), np.cos(truth), np.sin(truth)), axis=1)
    beta = np.asarray([[2.0, 1.0, -0.5], [1.0, -0.2, 0.8]])
    rates = beta @ design.T
    rates += (index + 1) * np.linspace(-0.01, 0.01, 30)[None, :]
    theta[list(missing_positions)] = np.nan
    ids = tuple(f"s{index}-trial-{position}" for position in range(30))
    return f"session-{index:02d}", rates, theta, ids, {"relative": f"source-{index}.nwb"}


def test_position_27_missing_is_explicit_and_does_not_change_m4_m10() -> None:
    rows = [_session(index, missing_positions=(27,) if index == 3 else ()) for index in range(27)]
    result = v2.build_products(rows)
    assert result["status"] == "SOURCE_POSTERIOR_AUTHORITY_V2_PASSED"
    row = result["rank_rows"][3]["budgets"]
    assert row["4"]["usable_direction_count"] == 4
    assert row["10"]["usable_direction_count"] == 10
    assert row["30"]["usable_direction_count"] == 29
    assert row["30"]["missing_direction_count"] == 1
    assert row["30"]["passed"] is True


def test_missing_label_is_never_imputed_or_replaced_by_future_trial() -> None:
    row = _session(0, missing_positions=(1,))
    audit = v2.audit_rank_rows(
        session=row[0], rates=row[1], theta_with_nan=row[2], support_ids=row[3],
        data_descriptor=row[4],
    )
    m4 = audit["budgets"]["4"]
    assert m4["physical_trial_count"] == 4
    assert m4["usable_direction_count"] == 3
    assert m4["residual_dof"] == 0
    assert m4["passed"] is False
    assert "no_imputation" in m4["law"]


def test_material_m4_missing_label_yields_typed_stop() -> None:
    rows = [_session(index, missing_positions=(1,) if index == 4 else ()) for index in range(27)]
    result = v2.build_products(rows)
    assert result["status"] == plan.M4_FAILURE
    assert result["source_prior"] is None
    assert any(row["session"] == "session-04" and row["budget"] == 4 for row in result["rank_failures"])


def test_v1_failure_graph_literals_and_dry_boundary_are_exact() -> None:
    assert v2.V1_ATTEMPT_SHA256.startswith("255d9cf4")
    assert v2.V1_FAILURE_SHA256.startswith("5894fd33")
    assert v2.V1_CLOSURE_SHA256.startswith("d535dda1")
    payload = v2.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["gpu_smoke_authorized"] is False

