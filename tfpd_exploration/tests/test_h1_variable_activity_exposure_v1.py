from __future__ import annotations

from collections import Counter

import pytest

from tfpd_exploration.src.h1_variable_activity_exposure_v1.plan import TRAINING_PLAN
from tfpd_exploration.src.h1_variable_activity_exposure_v1.train import (
    BatchDirective,
    VariableActivityTrainingError,
    build_epoch_directives,
)


def _inputs():
    rows = {
        "ses-A": tuple(range(0, 1024)),
        "ses-B": tuple(range(1024, 1800)),
    }
    counts = {"ses-A": 15, "ses-B": 8}
    return rows, counts


def test_frozen_plan_is_the_small_warm_start_cell() -> None:
    assert TRAINING_PLAN.batch_size == 128
    assert TRAINING_PLAN.epochs == 5
    assert TRAINING_PLAN.learning_rate == 1e-5
    assert TRAINING_PLAN.support_trials == 4


def test_directives_are_deterministic_session_homogeneous_and_cover_both_regimes() -> None:
    rows, counts = _inputs()
    left = build_epoch_directives(epoch=0, rows_by_session=rows, trial_counts=counts)
    right = build_epoch_directives(epoch=0, rows_by_session=rows, trial_counts=counts)
    assert left == right
    assert {row.session for row in left} == set(rows)
    assert any(row.activity_trials == 4 for row in left)
    assert any(row.activity_trials > 4 for row in left)
    for row in left:
        assert 1 <= len(row.row_indices) <= 128
        assert len(set(row.row_indices)) == len(row.row_indices)
        assert row.activity_trials <= counts[row.session]
        assert row.replay_m4 is (row.activity_trials == 4)


def test_each_epoch_uses_every_source_row_once() -> None:
    rows, counts = _inputs()
    directives = build_epoch_directives(epoch=3, rows_by_session=rows, trial_counts=counts)
    observed = Counter((row.session, index) for row in directives for index in row.row_indices)
    expected = Counter((session, index) for session, indices in rows.items() for index in indices)
    assert observed == expected


def test_replay_is_exactly_alternating_within_each_session_before_global_shuffle() -> None:
    rows, counts = _inputs()
    directives = build_epoch_directives(epoch=1, rows_by_session=rows, trial_counts=counts)
    for session in rows:
        replay = [row.replay_m4 for row in directives if row.session == session]
        assert abs(sum(replay) - len(replay) / 2) <= 0.5


def test_epoch_changes_order_and_cardinality_schedule() -> None:
    rows, counts = _inputs()
    assert build_epoch_directives(epoch=0, rows_by_session=rows, trial_counts=counts) != build_epoch_directives(
        epoch=1, rows_by_session=rows, trial_counts=counts,
    )


def test_invalid_source_topology_fails_closed() -> None:
    with pytest.raises(VariableActivityTrainingError):
        build_epoch_directives(epoch=0, rows_by_session={"ses-A": (1, 2)}, trial_counts={"ses-A": 4})


def test_batch_directive_rejects_inconsistent_replay_label() -> None:
    with pytest.raises(VariableActivityTrainingError):
        BatchDirective(0, "ses-A", (1,), 5, True)


def test_successor_decision_requires_static_safety_and_growing_improvement() -> None:
    def verdict(static_delta: float, growing_delta: float) -> str:
        return (
            "PASS_VARIABLE_ACTIVITY_SUCCESSOR"
            if static_delta >= -0.01 and growing_delta > 0.0
            else "STOP_VARIABLE_ACTIVITY_SUCCESSOR"
        )

    assert verdict(-0.01, 1e-12) == "PASS_VARIABLE_ACTIVITY_SUCCESSOR"
    assert verdict(-0.0100001, 0.1) == "STOP_VARIABLE_ACTIVITY_SUCCESSOR"
    assert verdict(0.0, 0.0) == "STOP_VARIABLE_ACTIVITY_SUCCESSOR"
