from pathlib import Path

from src.budget_matched_posterior_cal_aug_v1 import c2_score


ROOT = Path(__file__).resolve().parents[2]


def test_dry_plan_is_inert_and_complete():
    plan = c2_score.dry_plan()
    assert plan["status"] == "DRY_NO_DATA_NO_MODEL_NO_WRITE"
    assert plan["surfaces"] == ["within", "external"]
    assert plan["budgets"] == [4, 10, 30]
    assert plan["review_drift_policy"] == "ACCEPTED_NON_NUMERIC_DRIFT"


def test_execution_and_review_closures_are_disjoint():
    execution = c2_score.execution_closure(ROOT)
    review = c2_score.review_closure(ROOT)
    assert execution["closure_sha256"] != review["closure_sha256"]
    execution_paths = set(execution["files"])
    review_paths = set(review["files"])
    assert execution_paths.isdisjoint(review_paths)
    assert all("/tests/" not in path and "/docs/" not in path for path in execution_paths)


def test_review_drift_never_blocks_numeric_result():
    launch = {"closure_sha256": "a" * 64}
    final = {"closure_sha256": "b" * 64}
    result = c2_score._review_drift(launch, final)
    assert result["disposition"] == "ACCEPTED_NON_NUMERIC_DRIFT"
    assert result["numerical_result_blocked"] is False
