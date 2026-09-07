from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.scripts import verify_h1_trial_aligned_decomposition as verify


def test_ridge_solve_matches_hand_computed_closed_form() -> None:
    rng = np.random.default_rng(0)
    n, latent, channels = 10, 4, 6
    z = rng.normal(size=(n, latent))
    y = rng.normal(size=(n, channels))
    design = np.column_stack((np.ones(n), z))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * verify.RIDGE_LAMBDA)
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    expected = np.column_stack((beta[1:].T, beta[0]))
    observed = verify.fit_carrier_ridge(z, y)
    assert observed is not None
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)


def test_ridge_intercept_is_unpenalised() -> None:
    rng = np.random.default_rng(1)
    n, channels = 12, 8
    z = rng.normal(size=(n, 4))
    y = rng.normal(size=(n, channels))
    design = np.column_stack((np.ones(n), z))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * verify.RIDGE_LAMBDA)
    assert penalty[0, 0] == 0.0
    assert np.all(penalty[0, 1:] == 0.0)
    assert np.all(penalty[1:, 0] == 0.0)
    expected_beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    expected_carrier = np.column_stack((expected_beta[1:].T, expected_beta[0]))
    carrier = verify.fit_carrier_ridge(z, y)
    assert carrier is not None
    np.testing.assert_allclose(carrier, expected_carrier, rtol=0.0, atol=1.0e-12)


def test_r2_matches_hand_computation_and_zero_variance_is_nan() -> None:
    observed = np.array([[1.0, 3.0], [3.0, 3.0], [5.0, 3.0]], dtype=np.float64)
    predicted = np.array([[1.5, 2.0], [2.5, 2.5], [4.5, 3.5]], dtype=np.float64)
    r2 = verify.r2_by_channel(observed, predicted)
    total = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    residual = np.sum((observed - predicted) ** 2, axis=0)
    expected = np.array([1.0 - residual[0] / total[0], np.nan], dtype=np.float64)
    np.testing.assert_allclose(r2[0], expected[0], rtol=0.0, atol=1.0e-12)
    assert np.isnan(r2[1])


def test_support_and_eval_selection_on_synthetic_session() -> None:
    events = tuple(
        v1.MovementEvent(
            row_id=index,
            tag="Reach",
            trial_value=float(trial),
            trial_index=trial,
            start_time=float(index),
            stop_time=float(index) + 0.1,
            duration_seconds=0.1,
            eval_bins=5,
            displacement=np.zeros(7, dtype=np.float64),
            log_rates=np.zeros(176, dtype=np.float64),
        )
        for index, trial in enumerate([0, 0, 1, 2, 3, 4, 4])
    )
    support = verify.select_support(events, 3)
    eval_events = verify.select_eval(events, 4)
    assert {event.trial_index for event in support} == {0, 1, 2}
    assert {event.trial_index for event in eval_events} == {4}
    assert len(support) == 4
    assert len(eval_events) == 2


def test_decomposition_identity_on_synthetic_values() -> None:
    cells = {
        verify.cell_key(3, 3): {"median_delta_intercept": 0.30},
        verify.cell_key(3, 4): {"median_delta_intercept": 0.20},
        verify.cell_key(4, 4): {"median_delta_intercept": 0.05},
    }
    result = verify.session_decomposition(cells)
    assert result["eval_effect"] == pytest.approx(0.10, abs=1.0e-12)
    assert result["support_effect"] == pytest.approx(0.15, abs=1.0e-12)
    assert result["total"] == pytest.approx(0.25, abs=1.0e-12)
    assert abs(result["total"] - (result["eval_effect"] + result["support_effect"])) <= 1.0e-12


def test_script_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text('{"placeholder": true}\n', encoding="utf-8")
    body = {
        "schema": verify.SCHEMA,
        "cells": list(verify.REQUIRED_CELLS),
        "per_session_cells": {},
        "per_session_decomposition": {},
        "decomposition_aggregates": {},
        "five_cell_table": [],
        "sealed_helper_comparison": {"comparisons": [], "max_abs_diff": 0.0},
        "source_binding": {"sessions": [], "dates": [], "files": []},
        "implementation_binding": {"script_path": str(verify.SCRIPT_PATH), "script_sha256": "0" * 64},
        "runtime_seconds": 0.0,
        "scope": {"cuda_used": False, "dense_velocity_series_opened": False},
    }
    with pytest.raises(verify.ReceiptExistsError):
        verify.write_receipt(output, body)
