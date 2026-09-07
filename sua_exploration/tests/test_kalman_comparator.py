"""Unit tests for the velocity Kalman comparator (synthetic data only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import kalman_comparator as core


def _synthetic_linear_gaussian_system(
    rng: np.random.Generator,
    *,
    kinematic_dim: int,
    n_channels: int,
    n_steps: int,
) -> tuple[core.KalmanParameters, np.ndarray, np.ndarray]:
    state_dim = core.state_dim_for_kinematic_dim(kinematic_dim)
    dynamics = np.eye(state_dim, dtype=np.float64)
    dynamics[:kinematic_dim, core.velocity_slice(kinematic_dim)] = np.eye(kinematic_dim) * core.BIN_SIZE_S_DEFAULT
    A_true = core.fix_constant_dynamics(dynamics, kinematic_dim=kinematic_dim)
    H_true = rng.normal(size=(n_channels, state_dim)) * 0.1
    W_true = np.diag(np.full(state_dim, 0.05))
    Q_true = np.diag(np.full(n_channels, 0.05))

    states = np.zeros((n_steps, state_dim), dtype=np.float64)
    rates = np.zeros((n_steps, n_channels), dtype=np.float64)
    vel0 = rng.normal(size=kinematic_dim)
    states[0, :kinematic_dim] = 0.0
    states[0, core.velocity_slice(kinematic_dim)] = vel0
    states[0, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    for t in range(1, n_steps):
        process = rng.multivariate_normal(np.zeros(state_dim), W_true)
        states[t] = A_true @ states[t - 1] + process
        states[t, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    for t in range(n_steps):
        obs = rng.multivariate_normal(np.zeros(n_channels), Q_true)
        rates[t] = H_true @ states[t] + obs

    params = core.KalmanParameters(
        A=A_true,
        W=W_true,
        H=H_true,
        Q=Q_true,
        kinematic_dim=kinematic_dim,
        q_floor=core.DEFAULT_Q_FLOOR,
        w_floor=core.DEFAULT_W_FLOOR,
    )
    return params, states, rates


def test_closed_form_recovers_true_parameters() -> None:
    rng = np.random.default_rng(0)
    truth, states, rates = _synthetic_linear_gaussian_system(
        rng, kinematic_dim=2, n_channels=8, n_steps=400
    )
    fit = core.fit_kalman_parameters(states, rates, kinematic_dim=2)
    assert fit.success is True
    assert fit.parameters is not None
    assert np.allclose(fit.parameters.A, truth.A, atol=0.15)
    assert np.allclose(fit.parameters.H, truth.H, atol=0.15)
    assert np.allclose(fit.parameters.W, truth.W, atol=0.15)
    assert np.allclose(fit.parameters.Q, truth.Q, atol=0.15)


def test_constant_term_stays_exactly_one_through_recursion() -> None:
    rng = np.random.default_rng(1)
    truth, states, rates = _synthetic_linear_gaussian_system(
        rng, kinematic_dim=2, n_channels=5, n_steps=60
    )
    filtered = core.kalman_filter_query(rates[10:], truth, initial_state=states[9])
    for row in filtered:
        core.assert_constant_term(row, kinematic_dim=2)


def test_scalar_toy_posterior_matches_hand_computation() -> None:
    kinematic_dim = 1
    state_dim = core.state_dim_for_kinematic_dim(kinematic_dim)
    A = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    W = np.diag([0.01, 0.02, 0.0])
    H = np.array([[0.5, 0.0, 0.3]], dtype=np.float64)
    Q = np.array([[0.04]], dtype=np.float64)
    params = core.KalmanParameters(
        A=A, W=W, H=H, Q=Q, kinematic_dim=kinematic_dim, q_floor=0.0, w_floor=0.0
    )
    x0 = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    P0 = np.eye(state_dim, dtype=np.float64) * 0.1
    z = np.array([[0.7]], dtype=np.float64)
    x_pred = A @ x0
    x_pred[core.constant_index(state_dim)] = core.CONSTANT_VALUE
    P_pred = A @ P0 @ A.T + W
    S = H @ P_pred @ H.T + Q
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_manual = x_pred + (K @ (z[0] - H @ x_pred)).reshape(-1)
    x_manual[core.constant_index(state_dim)] = core.CONSTANT_VALUE
    filtered = core.kalman_filter_query(z, params, initial_state=x0, initial_covariance=P0)
    assert np.allclose(filtered[0], x_manual, atol=1.0e-12)


def test_singular_calibration_fails_closed_with_reason() -> None:
    rng = np.random.default_rng(2)
    kinematic_dim = 2
    state_dim = core.state_dim_for_kinematic_dim(kinematic_dim)
    states = np.tile(np.arange(state_dim, dtype=np.float64), (20, 1))
    states[:, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    rates = rng.normal(size=(20, 4))
    fit = core.fit_kalman_parameters(states, rates, kinematic_dim=kinematic_dim)
    assert fit.success is False
    assert fit.parameters is None
    assert fit.failure_reason is not None
    assert "singular" in fit.failure_reason.lower() or "ill-conditioned" in fit.failure_reason.lower()


def test_regularisation_floor_applied_and_recorded() -> None:
    rng = np.random.default_rng(3)
    state_dim = core.state_dim_for_kinematic_dim(2)
    kinematic_dim = 2
    dynamics = np.eye(state_dim, dtype=np.float64)
    dynamics[:kinematic_dim, core.velocity_slice(kinematic_dim)] = np.eye(kinematic_dim) * core.BIN_SIZE_S_DEFAULT
    A_true = core.fix_constant_dynamics(dynamics, kinematic_dim=kinematic_dim)
    H_true = rng.normal(size=(5, state_dim)) * 0.2
    W_true = np.diag(np.full(state_dim, 1.0e-8))
    Q_true = np.diag(np.full(5, 1.0e-8))
    truth = core.KalmanParameters(
        A=A_true,
        W=W_true,
        H=H_true,
        Q=Q_true,
        kinematic_dim=kinematic_dim,
        q_floor=core.DEFAULT_Q_FLOOR,
        w_floor=core.DEFAULT_W_FLOOR,
    )
    n_steps = 200
    states = np.zeros((n_steps, state_dim), dtype=np.float64)
    rates = np.zeros((n_steps, 5), dtype=np.float64)
    states[0, core.velocity_slice(kinematic_dim)] = rng.normal(size=kinematic_dim)
    states[0, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    for t in range(1, n_steps):
        states[t] = truth.A @ states[t - 1] + rng.multivariate_normal(np.zeros(state_dim), truth.W)
        states[t, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    for t in range(n_steps):
        rates[t] = truth.H @ states[t] + rng.multivariate_normal(np.zeros(5), truth.Q)
    q_floor = 1.0e-4
    w_floor = 1.0e-4
    fit = core.fit_kalman_parameters(
        states,
        rates,
        kinematic_dim=kinematic_dim,
        q_floor=q_floor,
        w_floor=w_floor,
    )
    assert fit.success is True
    assert fit.parameters is not None
    assert fit.q_floor_applied is True
    assert fit.w_floor_applied is True
    assert np.all(np.diag(fit.parameters.Q) >= q_floor - 1.0e-15)
    assert np.all(np.diag(fit.parameters.W) >= w_floor - 1.0e-15)
    assert fit.parameters.q_floor == q_floor
    assert fit.parameters.w_floor == w_floor


def test_r2_helpers_match_hand_computed_values() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 4.0]], dtype=np.float64)
    estimate = np.array([[1.1, 1.8], [2.2, 3.1], [2.7, 5.4], [3.8, 4.2]], dtype=np.float64)
    sse = float(np.square(truth - estimate).sum())
    centered = truth - truth.mean(axis=0, keepdims=True)
    pooled_manual = 1.0 - sse / float(np.square(centered).sum())
    residual = np.square(truth - estimate).sum(axis=0)
    total = np.square(truth - truth.mean(axis=0, keepdims=True)).sum(axis=0)
    vw_manual = 1.0 - float(residual.sum()) / float(total.sum())
    assert core.r2_pooled(truth, estimate) == pytest.approx(pooled_manual)
    assert core.r2_variance_weighted(estimate, truth) == pytest.approx(vw_manual)


def test_determinism_across_two_runs() -> None:
    rng = np.random.default_rng(4)
    truth, states, rates = _synthetic_linear_gaussian_system(
        rng, kinematic_dim=2, n_channels=6, n_steps=120
    )
    calib_states = states[:80]
    calib_rates = rates[:80]
    query_rates = rates[80:110]
    fit_a = core.fit_kalman_parameters(calib_states, calib_rates, kinematic_dim=2)
    fit_b = core.fit_kalman_parameters(calib_states, calib_rates, kinematic_dim=2)
    assert fit_a.success and fit_b.success
    assert fit_a.parameters is not None and fit_b.parameters is not None
    pred_a = core.kalman_filter_query(query_rates, fit_a.parameters, initial_state=calib_states[-1])
    pred_b = core.kalman_filter_query(query_rates, fit_b.parameters, initial_state=calib_states[-1])
    assert np.array_equal(pred_a, pred_b)


def test_evaluate_session_records_failure_instead_of_silent_pseudoinverse() -> None:
    rng = np.random.default_rng(5)
    kinematic_dim = 2
    state_dim = core.state_dim_for_kinematic_dim(kinematic_dim)
    states = np.zeros((12, state_dim), dtype=np.float64)
    states[:, core.constant_index(state_dim)] = core.CONSTANT_VALUE
    rates = rng.normal(size=(12, 3))
    blocks = core.SessionBlocks(
        session_name="synthetic",
        kinematic_dim=kinematic_dim,
        calibration_rates=rates,
        calibration_states=states,
        query_rates=rates[:5],
        query_states=states[:5],
        query_velocity_truth=rng.normal(size=(5, kinematic_dim)),
        metric_name="variance_weighted_r2",
        query_identity={"synthetic": True},
        calibration_rows=12,
        query_rows=5,
        n_channels=3,
    )
    result = core.evaluate_session(blocks)
    assert result.success is False
    assert result.r2 is None
    assert result.failure_reason is not None


def test_implementation_files_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "sua_exploration/mc_maze/kalman_comparator.py").is_file()
    assert (repo_root / "sua_exploration/scripts/run_kalman_comparator.py").is_file()
