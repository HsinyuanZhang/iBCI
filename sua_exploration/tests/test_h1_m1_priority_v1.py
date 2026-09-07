from __future__ import annotations

import numpy as np
import pytest

from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    PriorityAnalysisError,
    covariance_spectrum,
    direct_ridge_loso,
    regression_metrics,
    shared_axis_semantic_gate,
)


def _session(name: str, seed: int) -> DirectRidgeSession:
    rng = np.random.default_rng(seed)
    rows = 620
    starts = np.arange(10, 610, 50)
    change = np.zeros(rows, dtype=bool)
    change[starts] = True
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    weight = np.asarray([[1.2, -0.3], [0.1, 0.7], [-0.4, 0.2], [0.8, 0.1]], dtype=np.float32)
    y = np.zeros((rows, 2), dtype=np.float32)
    y[2:] = x[:-2] @ weight + 0.03 * rng.normal(size=(rows - 2, 2))
    mask = np.ones(rows, dtype=bool)
    mask[: starts[0]] = False
    return DirectRidgeSession(name, x, y, mask, change, ("a", "b"))


def test_same_trial_alignment_and_m10_boundary() -> None:
    session = _session("s0", 0)
    support_x, support_y, support_indices = session.aligned(2, split="support")
    query_x, query_y, query_indices = session.aligned(2, split="query")
    assert support_x.shape[0] == support_y.shape[0] > 0
    assert query_x.shape[0] == query_y.shape[0] > 0
    assert support_indices.max() < session.boundary
    assert query_indices.min() >= session.boundary + 2
    trial_id = session.trial_id
    assert np.all(trial_id[support_indices - 2] == trial_id[support_indices])
    assert np.all(trial_id[query_indices - 2] == trial_id[query_indices])


def test_directridge_nested_loso_finds_causal_signal_and_excludes_target() -> None:
    sessions = {f"s{index}": _session(f"s{index}", index) for index in range(4)}
    result = direct_ridge_loso(sessions, lag_grid=(0, 1, 2, 3), lambda_grid=(0.0, 1.0e-3, 1.0e-2))
    assert result["target_support_or_query_used_for_candidate_selection"] is False
    assert result["equal_session"]["mean_r2"] > 0.98
    for target, row in result["folds"].items():
        assert target not in row["source_sessions"]
        assert row["target_excluded_from_hyperparameter_selection"] is True
        assert row["selected"]["lag_bins"] == 2


def test_metric_is_per_output_variance_weighted() -> None:
    target = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float64)
    prediction = np.asarray([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]], dtype=np.float64)
    row = regression_metrics(target, prediction)
    assert row["r2_per_output"][0] == pytest.approx(1.0)
    assert row["r2_per_output"][1] == pytest.approx(0.0)
    assert row["pooled_variance_weighted_r2"] == pytest.approx(1.0 / 101.0)
    assert row["equal_output_mean_r2"] == pytest.approx(0.5)


def test_covariance_participation_ratio_and_rank90() -> None:
    values = np.stack([np.arange(100.0), 2.0 * np.arange(100.0), -np.arange(100.0)], axis=1)
    row = covariance_spectrum(values)
    assert row["participation_ratio"] == pytest.approx(1.0)
    assert row["components_for_90pct"] == 1


def test_shared_axis_gate_requires_metadata_identity_not_equal_dimension() -> None:
    stopped = shared_axis_semantic_gate(("tx", "ty"), ("arbitrary", "arbitrary"), ("index", "mrs"), ("AU", "AU"))
    assert stopped["passed"] is False
    passed = shared_axis_semantic_gate(("x", "y", "z"), ("m/s", "m/s", "m/s"), ("y", "x"), ("m/s", "m/s"))
    assert passed["passed"] is True
    assert passed["exact_name_and_unit_matches"] == ["x", "y"]


def test_invalid_session_is_fail_closed() -> None:
    session = _session("bad", 1)
    bad = DirectRidgeSession(session.name, session.neural, session.target, session.eval_mask, np.zeros_like(session.trial_change), session.output_names)
    with pytest.raises(PriorityAnalysisError, match="11 chronological"):
        bad.validate()
