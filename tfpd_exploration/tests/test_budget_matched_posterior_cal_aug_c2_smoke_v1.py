from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1 import c2_smoke  # noqa: E402


def test_runtime_verdict_requires_exact_cycle_finiteness_and_side_gradient() -> None:
    rows = [
        {"batch_ordinal": index, "budget": (30, 10, 4)[index % 3], "indices": [index]}
        for index in range(120)
    ]
    stats = {
        "optimizer_steps": 120,
        "visible_side_violation_count": 0,
        "nonfinite_loss_steps": 0,
        "nonfinite_grad_steps": 0,
        "w_side_grad_exact_zero_all_steps": False,
    }
    dropout = {"n_forwards_with_sampled_p": 120, "n_recorded_unit_mask_calls": 120}
    result = c2_smoke.validate_smoke_result(
        stats=stats, realized_rows=rows, dropout_summary=dropout,
        parameters_finite=True, optimizer_finite=True,
    )
    assert result["passed"] is True
    assert result["budget_counts"] == {30: 40, 10: 40, 4: 40}
    altered = list(rows)
    altered[1] = {**altered[1], "budget": 30}
    assert c2_smoke.validate_smoke_result(
        stats=stats, realized_rows=altered, dropout_summary=dropout,
        parameters_finite=True, optimizer_finite=True,
    )["passed"] is False


def test_dry_plan_binds_accepted_source_and_makes_no_performance_claim() -> None:
    payload = c2_smoke.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["source_authority_sha256"].startswith("92f898aa")
    assert payload["performance_claim"] is False
    assert payload["budget_cycle"] == [30, 10, 4]

