from __future__ import annotations

from pathlib import Path

from budget_matched_posterior_cal_aug_c3_v1 import smoke
from budget_matched_posterior_cal_aug_c3_v1.features import C3Arm


ROOT = Path(__file__).resolve().parents[2]


def _stats() -> dict[str, object]:
    return {
        "optimizer_steps": 120,
        "visible_side_violation_count": 0,
        "nonfinite_loss_steps": 0,
        "nonfinite_grad_steps": 0,
        "w_side_grad_exact_zero_all_steps": False,
    }


def _rows():
    return [
        {"batch_ordinal": index, "budget": (30, 10, 4)[index % 3], "indices": [index]}
        for index in range(120)
    ]


def _dropout() -> dict[str, object]:
    return {"n_forwards_with_sampled_p": 120, "n_recorded_unit_mask_calls": 120}


def test_arm_verdict_distinguishes_constant_and_real_q_semantics() -> None:
    constant = smoke._validate_arm(
        stats=_stats(), rows=_rows(), dropout=_dropout(), parameters_finite=True,
        optimizer_finite=True, arm=C3Arm.CONSTANT, q_weight_nonzero=0,
        q_moment_nonzero=0,
    )
    real = smoke._validate_arm(
        stats=_stats(), rows=_rows(), dropout=_dropout(), parameters_finite=True,
        optimizer_finite=True, arm=C3Arm.REAL, q_weight_nonzero=64,
        q_moment_nonzero=64,
    )
    assert constant["passed"] is True
    assert real["passed"] is True
    bad = smoke._validate_arm(
        stats=_stats(), rows=_rows(), dropout=_dropout(), parameters_finite=True,
        optimizer_finite=True, arm=C3Arm.CONSTANT, q_weight_nonzero=1,
        q_moment_nonzero=1,
    )
    assert bad["passed"] is False


def test_execution_and_review_closures_are_disjoint_and_current() -> None:
    assert set(smoke.EXECUTION_PATHS).isdisjoint(smoke.REVIEW_PATHS)
    assert not any("/tests/" in path or "/docs/" in path for path in smoke.EXECUTION_PATHS)
    execution = smoke.execution_closure(ROOT)
    review = smoke.review_closure(ROOT)
    assert len(execution["files"]) == len(smoke.EXECUTION_PATHS)
    assert len(review["files"]) == len(smoke.REVIEW_PATHS)


def test_review_drift_is_accepted_but_never_changes_numeric_acceptance() -> None:
    start = {"files": {"doc": {"sha256": "a"}}, "closure_sha256": "start"}
    end = {"files": {"doc": {"sha256": "b"}}, "closure_sha256": "end"}
    result = smoke._review_drift(start, end)
    assert result["status"] == "ACCEPTED_NON_NUMERIC_DRIFT"
    assert result["changed_paths"] == ["doc"]
    assert result["numerical_acceptance_affected"] is False


def test_dry_plan_is_inert_and_names_both_arms() -> None:
    payload = smoke.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["arms"] == ["constant", "real"]
    assert payload["smoke_steps_per_arm"] == 120
