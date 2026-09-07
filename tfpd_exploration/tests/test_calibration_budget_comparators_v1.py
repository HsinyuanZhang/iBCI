from __future__ import annotations

import numpy as np
import torch

from src import calibration_budget_comparators_v1 as compare


def test_ridge_t4_zero_lambda_matches_lstsq() -> None:
    theta = np.asarray([0.0, 0.5, 1.2, 2.0, 2.8, 4.0], dtype=np.float64)
    design = np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size)))
    coefficients = np.asarray([[2.0, -1.0], [-0.5, 3.0], [4.0, 7.0]])
    rates = design @ coefficients
    raw, evidence = compare.fit_ridge_t4(rates, theta, normalized_lambda=0.0)
    assert np.allclose(raw[:, :2], coefficients[:2].T, atol=1e-6)
    assert np.allclose(raw[:, 3], coefficients[2], atol=1e-6)
    assert np.allclose(raw[:, 2], np.linalg.norm(coefficients[:2].T, axis=1), atol=1e-6)
    assert evidence["design_rank"] == 3


def test_gcv_selection_is_grid_bound_and_deterministic() -> None:
    rng = np.random.default_rng(4)
    theta = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    rates = rng.normal(size=(10, 7))
    first, a = compare.fit_gcv_ridge_t4(rates, theta)
    second, b = compare.fit_gcv_ridge_t4(rates, theta)
    assert np.array_equal(first, second)
    assert a == b
    assert a["normalized_lambda"] in compare.RIDGE_T4_GCV_GRID


def test_closed_form_ridge_primal_and_dual_agree() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 5))
    y = rng.normal(size=(12, 2))
    primal = compare.fit_closed_form_ridge(x, y, normalized_lambda=0.7)
    # Force the mathematically equivalent dual computation directly.
    mean, scale = x.mean(0), x.std(0)
    z = (x - mean) / scale
    centered = y - y.mean(0)
    alpha = np.linalg.solve(z @ z.T + x.shape[0] * 0.7 * np.eye(x.shape[0]), centered)
    dual_w = z.T @ alpha
    assert np.allclose(primal.weights, dual_w, atol=2e-6, rtol=2e-6)


def test_session_r2_is_variance_weighted() -> None:
    target = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    prediction = target.copy()
    assert compare.session_r2_float32(prediction, target) == 1.0
    prediction[:, 0] = 0.0
    score = compare.session_r2_float32(prediction, target)
    assert score > 0.97  # high-variance second coordinate dominates


def test_session_r2_matches_frozen_house_scorer() -> None:
    from src.tfpd_lane.matched_scorer import session_r2

    generator = torch.Generator().manual_seed(42)
    target = torch.randn(257, 2, generator=generator, dtype=torch.float32)
    prediction = target + 0.4 * torch.randn(257, 2, generator=generator, dtype=torch.float32)
    observed = compare.session_r2_float32(prediction.numpy(), target.numpy())
    expected = session_r2(prediction, target)
    assert abs(observed - expected) <= 1.0e-6


def test_bootstrap_is_deterministic() -> None:
    assert compare.bootstrap_delta([1.0, -1.0, 2.0]) == compare.bootstrap_delta([1.0, -1.0, 2.0])


def test_float_encoded_integral_trial_bounds_are_normalized() -> None:
    trials = [{"start": 0.0, "stop": 52.0}, {"start": 60.0, "stop": 112.0}]
    bounds = compare._trial_bounds(trials, 2)
    assert bounds == [(0, 52), (60, 112)]
    assert np.array_equal(compare._support_starts(bounds), np.asarray([0, 1, 2, 60, 61, 62]))


def test_dry_plan_is_inert() -> None:
    plan = compare.dry_plan()
    assert plan["budgets"] == [4, 10, 30]
    assert plan["formal_opened"] is False
    assert plan["target_updates"] == 0


def test_real_cell_d_accepts_m4_m10_m30_calibration_without_graph_change() -> None:
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    torch.manual_seed(5)
    model = build_population_robustness_model(seed=42, cell="D").eval()
    state_keys = tuple(model.state_dict())
    neural = torch.randn(2, 50, 9)
    side = torch.randn(2, 9, 4)
    with torch.no_grad():
        for budget in compare.BUDGETS:
            calibration = torch.randn(2, budget, 100, 9)
            prediction, identity = model(neural, calib_trials=calibration, side_features=side)
            assert tuple(prediction.shape) == (2, 50, 2)
            assert tuple(identity.shape[:2]) == (2, 9)
            assert torch.isfinite(prediction).all()
    assert tuple(model.state_dict()) == state_keys
