from __future__ import annotations

from pathlib import Path

import pytest

from budget_matched_posterior_cal_aug_c3_v1 import full


ROOT = Path(__file__).resolve().parents[2]


def test_smoke_predecessor_exact_graph_is_current() -> None:
    row = full.validate_smoke_predecessor(ROOT)
    assert row["attempt_sha256"] == full.SMOKE_ATTEMPT_SHA256
    assert row["terminal_sha256"] == full.SMOKE_TERMINAL_SHA256
    assert row["terminal"]["arms"]["constant"]["verdict"]["q_weight_nonzero"] == 0
    assert row["terminal"]["arms"]["real"]["verdict"]["q_weight_nonzero"] > 0


def test_budget_counts_are_exact_over_epochs_and_global_cycle() -> None:
    total = {30: 0, 10: 0, 4: 0}
    for epoch in range(full.EPOCHS):
        rows = full.budget_counts_for_span(epoch * full.STEPS_PER_EPOCH, full.STEPS_PER_EPOCH)
        for budget in total:
            total[budget] += rows[budget]
    assert total == full.EXPECTED_TOTAL_COUNTS
    assert total == {30: 542800, 10: 542800, 4: 542800}


def test_execution_closure_excludes_review_only_files() -> None:
    assert set(full.EXECUTION_PATHS).isdisjoint(full.REVIEW_PATHS)
    assert not any("/tests/" in path or "/docs/" in path for path in full.EXECUTION_PATHS)
    strict = full.execution_closure(ROOT)
    review = full.review_closure(ROOT)
    assert len(strict["files"]) == len(full.EXECUTION_PATHS)
    assert len(review["files"]) == len(full.REVIEW_PATHS)


def test_dry_plan_has_separate_fresh_roots_and_unknown_arm_rejects() -> None:
    real = full.dry_plan("real")
    constant = full.dry_plan("constant")
    assert real["result_root_relative"] != constant["result_root_relative"]
    assert real["total_optimizer_steps"] == 1_628_400
    assert constant["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    with pytest.raises(ValueError):
        full.dry_plan("wrong")
