"""Synthetic tests for B8 pseudo-label carrier gate and RLS scaffolding."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mc_maze.carrier_recursive_estimator import (
    CarrierRecursiveEstimator,
    batch_ols_carrier,
    design_row,
)
from mc_maze.decoder_attention_diagnostic import SEALED_FORMAL_TEST_SESSIONS, SealedSessionError
from mc_maze.pseudo_label_carrier_gate import (
    DEFAULT_BUDGET_M,
    FROZEN_GATE_PARAMETERS,
    evaluate_pseudo_label_gate,
    synthetic_tuning_rates,
    thetas_from_direction_indices,
)
from mc_maze.pseudo_label_carrier_gate_replay import (
    build_receipt,
    build_synthetic_session_payload,
    validate_receipt,
)
from mc_maze.d_optimal_calibration_design import CANONICAL_DIRECTIONS_RAD


def test_rls_converges_to_batch_ols_on_stationary_data() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    thetas = rng.uniform(-math.pi, math.pi, size=500)
    a, c, b = 2.5, -1.2, 4.0
    rates = b + a * np.cos(thetas) + c * np.sin(thetas)
    batch = batch_ols_carrier(thetas, rates)
    beta_init = np.array([b, a, c], dtype=np.float64)
    estimator = CarrierRecursiveEstimator(beta_init, forgetting_factor=1.0, departure_threshold=1e6)
    estimator.run_sequence(thetas, rates)
    rls_carrier = estimator.state.carrier_t4
    assert np.allclose(rls_carrier[:2], batch[:2], atol=1e-4)
    assert abs(rls_carrier[3] - batch[3]) < 1e-4


def test_rls_tracks_step_change_with_forgetting() -> None:
    thetas = np.linspace(-math.pi, math.pi, 400, endpoint=False)
    rates_pre = 1.0 + 2.0 * np.cos(thetas) + 0.5 * np.sin(thetas)
    rates_post = 1.0 - 2.0 * np.cos(thetas) - 0.5 * np.sin(thetas)
    init = batch_ols_carrier(thetas[:50], rates_pre[:50])
    beta_init = np.array([init[3], init[0], init[1]], dtype=np.float64)
    estimator = CarrierRecursiveEstimator(
        beta_init,
        forgetting_factor=0.95,
        departure_threshold=1e6,
    )
    estimator.run_sequence(thetas[:200], rates_pre[:200])
    estimator.run_sequence(thetas[200:], rates_post[200:])
    post_batch = batch_ols_carrier(thetas[200:], rates_post[200:])
    rls_ac = estimator.state.carrier_t4[:2]
    assert np.dot(rls_ac, post_batch[:2]) > np.dot(rls_ac, init[:2])


def test_departure_threshold_freezes_unit() -> None:
    theta = 0.0
    estimator = CarrierRecursiveEstimator(
        np.array([1.0, 1.0, 0.0], dtype=np.float64),
        forgetting_factor=1.0,
        departure_threshold=0.1,
    )
    for _ in range(20):
        estimator.update(theta, 100.0)
    assert estimator.state.frozen
    assert not estimator.update(theta, 100.0)


def test_pseudo_label_cosine_near_one_when_labels_match() -> None:
    directions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.int64)
    thetas = thetas_from_direction_indices(directions)
    rates = synthetic_tuning_rates(thetas, units=6, seed=11)
    result = evaluate_pseudo_label_gate(
        rates,
        directions,
        directions.copy(),
        budget_m=10,
        session_name="synthetic_match",
    )
    assert result.median_cosine_correct is not None
    assert result.median_cosine_correct > 0.99


def test_pseudo_label_cosine_near_chance_when_shuffled() -> None:
    rng = np.random.Generator(np.random.PCG64(99))
    directions = rng.integers(0, 8, size=40, dtype=np.int64)
    thetas = thetas_from_direction_indices(directions)
    rates = synthetic_tuning_rates(thetas, units=10, seed=12)
    shuffled = rng.permutation(directions)
    result = evaluate_pseudo_label_gate(
        rates,
        directions,
        shuffled,
        budget_m=30,
        session_name="synthetic_shuffle",
    )
    assert result.median_cosine_correct is not None
    assert result.median_cosine_shuffled is not None
    assert result.correct_minus_shuffle is not None
    assert result.median_cosine_correct < 0.6
    assert result.correct_minus_shuffle < 0.05


def test_sealed_session_guard_raises() -> None:
    with pytest.raises(SealedSessionError):
        build_synthetic_session_payload(SEALED_FORMAL_TEST_SESSIONS[0])


def test_receipt_determinism() -> None:
    payload = build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)
    first = build_receipt([payload], seed=42)
    second = build_receipt([payload], seed=42)
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_aggregator_rejects_sealed_opened() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["sealed_test_sessions_opened"] = True
    with pytest.raises(ValueError, match="sealed"):
        validate_receipt(receipt)


def test_aggregator_rejects_gate_parameter_drift() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["frozen_gates"]["shuffle_seed"] = 99
    with pytest.raises(ValueError, match="drift"):
        validate_receipt(receipt)


def test_aggregator_rejects_incomplete_sessions() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["session_results"] = []
    with pytest.raises(ValueError, match="incomplete"):
        validate_receipt(receipt)


def test_valid_receipt_passes() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    summary = validate_receipt(receipt)
    assert summary["status"] == "valid"


def test_design_row_matches_cosine_model() -> None:
    row = design_row(CANONICAL_DIRECTIONS_RAD[2])
    assert row[0] == 1.0
    assert abs(row[1] - math.cos(CANONICAL_DIRECTIONS_RAD[2])) < 1e-12


@pytest.mark.real_data
@pytest.mark.skip(reason="real checkpoint pseudo-label replay requires explicit authorization")
def test_real_checkpoint_pseudo_labels_not_run_by_default() -> None:
    raise AssertionError("real_data marker must stay skipped in default CI")


def test_frozen_gate_parameters_documented() -> None:
    assert FROZEN_GATE_PARAMETERS["budget_m_default"] == DEFAULT_BUDGET_M
