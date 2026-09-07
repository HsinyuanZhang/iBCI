from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from budget_matched_posterior_cal_aug_v1 import contract, plan, posterior


def _theta(n: int) -> np.ndarray:
    base = np.asarray([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0], dtype=np.float64)
    return np.resize(base, n)


def _rates(theta: np.ndarray, coefficients=((2.0, 1.0, -0.5), (1.0, -0.2, 0.8))) -> np.ndarray:
    design = posterior.directional_design(theta)
    return np.stack([design @ np.asarray(beta, dtype=np.float64) for beta in coefficients], axis=0)


def test_plan_is_exact_cal_aug_schedule_and_inert() -> None:
    assert [plan.budget_at(i) for i in range(7)] == [30, 10, 4, 30, 10, 4, 30]
    payload = plan.dry_payload()
    assert payload["execution_authorized"] is False
    assert "torch_import" in payload["prohibitions"]
    with pytest.raises(ValueError):
        plan.budget_at(-1)


def test_workorder_binding_and_e02_e03_priority_are_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    workorder = root / plan.WORK_ORDER_RELATIVE
    assert hashlib.sha256(workorder.read_bytes()).hexdigest() == plan.WORK_ORDER_SHA256
    assert plan.NEXT_CELLS["c2"].startswith("primary_")
    assert plan.NEXT_CELLS["c3_real"].startswith("implement_now_")
    assert plan.NEXT_CELLS["c3_constant"].startswith("implement_now_")
    assert plan.NEXT_CELLS["e09"].startswith("park_")
    assert plan.NEXT_CELLS["e10"].startswith("park_")


def test_exact_posterior_matches_direct_solve_and_preserves_intercept() -> None:
    theta = _theta(8)
    rates = _rates(theta)
    rates[0, -1] += 0.4
    fit = posterior.fit_posterior_mean(rates, theta, prior_variance=0.7)
    design = posterior.directional_design(theta)
    gram = design.T @ design
    for unit in range(rates.shape[0]):
        ols = np.linalg.solve(gram, design.T @ rates[unit])
        residual = rates[unit] - design @ ols
        sigma2 = residual @ residual / (theta.size - 3)
        expected = np.linalg.solve(
            gram + np.diag([0.0, sigma2 / 0.7, sigma2 / 0.7]),
            design.T @ rates[unit],
        )
        np.testing.assert_allclose(fit.beta_bac[unit], expected, rtol=0.0, atol=2.0e-15)
        assert fit.raw_t4[unit, 3] == pytest.approx(expected[0], abs=2.0e-15)
        assert fit.raw_t4[unit, 2] == pytest.approx(np.hypot(expected[1], expected[2]), abs=2.0e-15)
    assert fit.design_rank == 3 and fit.residual_dof == 5


def test_noise_shrinks_ac_but_intercept_is_unpenalized() -> None:
    theta = _theta(8)
    clean = _rates(theta, coefficients=((4.0, 1.5, -0.7),))
    noisy = clean.copy()
    noisy[0] += np.asarray([0.0, 1.0, -1.0, 0.5, 0.0, -1.0, 1.0, -0.5])
    fit = posterior.fit_posterior_mean(noisy, theta, prior_variance=0.1)
    assert np.linalg.norm(fit.beta_bac[0, 1:]) < np.linalg.norm(fit.ols_beta_bac[0, 1:])
    # Symmetric directions make the intercept orthogonal to a/c.
    assert fit.beta_bac[0, 0] == pytest.approx(fit.ols_beta_bac[0, 0], abs=1e-14)


def test_trial_weighting_intentionally_differs_from_equal_direction_means() -> None:
    theta = np.asarray([0.0, 0.0, 0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    rates = np.asarray([[10.0, 10.0, 16.0, 4.0, 1.0, 3.0]], dtype=np.float64)
    trial_fit = posterior.fit_posterior_mean(rates, theta, prior_variance=1e12)
    unique = np.unique(theta)
    means = np.asarray([[rates[0, theta == value].mean() for value in unique]])
    equal_fit = posterior.fit_posterior_mean(means, unique, prior_variance=1e12)
    assert not np.allclose(trial_fit.raw_t4, equal_fit.raw_t4, atol=1e-10, rtol=0.0)


def test_rank_and_residual_dof_fail_closed_at_sparse_budget() -> None:
    with pytest.raises(posterior.PosteriorContractError, match=plan.M4_FAILURE):
        posterior.fit_posterior_mean(np.ones((2, 4)), np.zeros(4), prior_variance=1.0)
    with pytest.raises(posterior.PosteriorContractError, match=plan.M4_FAILURE):
        posterior.fit_posterior_mean(np.ones((2, 3)), np.asarray([0.0, 1.0, 2.0]), prior_variance=1.0)


def test_source_prior_uses_noise_corrected_second_moment_and_floor() -> None:
    theta = _theta(8)
    rates1 = _rates(theta)
    rates2 = _rates(theta, coefficients=((1.0, 0.4, 0.1),))
    prior = posterior.fit_source_prior([(rates1, theta), (rates2, theta)])
    assert prior.source_budget == 30
    assert prior.unit_fit_count == 3
    assert prior.variance == pytest.approx(
        max(prior.variance_floor, prior.raw_second_moment - prior.expected_ols_noise)
    )
    assert prior.variance != plan.E02_REFERENCE_PRIOR_VARIANCE_ONLY


def test_equal_budget_normalizer_is_budget_major_and_rejects_unequal_counts() -> None:
    rows = {
        4: np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.float64),
        10: np.asarray([[2, 3, 5, 7], [3, 5, 7, 11]], dtype=np.float64),
        30: np.asarray([[4, 6, 8, 10], [5, 7, 9, 12]], dtype=np.float64),
    }
    normalizer = posterior.fit_equal_budget_normalizer(rows)
    combined = np.concatenate([rows[4], rows[10], rows[30]], axis=0)
    np.testing.assert_array_equal(normalizer.mean, combined.mean(axis=0))
    np.testing.assert_array_equal(normalizer.std, combined.std(axis=0, ddof=0))
    np.testing.assert_allclose(normalizer.normalize(rows[4]), (rows[4] - normalizer.mean) / normalizer.std)
    bad = dict(rows)
    bad[4] = bad[4][:1]
    with pytest.raises(posterior.PosteriorContractError, match="equal source-row counts"):
        posterior.fit_equal_budget_normalizer(bad)


def test_budget_binding_requires_same_exact_chronological_trial_ids() -> None:
    ids = tuple(f"trial-{i}" for i in range(4))
    binding = contract.bind_budget_matched_step(
        step=2,
        activity_trial_ids=ids,
        carrier_trial_ids=ids,
        activity_rows=np.zeros((4, 100, 3), dtype=np.float32),
        carrier_rates=np.ones((3, 4), dtype=np.float64),
        direction_radians=_theta(4),
        source_prior_sha256="1" * 64,
        posterior_normalizer_sha256="2" * 64,
    )
    assert binding.budget == 4
    with pytest.raises(posterior.PosteriorContractError, match="trial IDs/order differ"):
        contract.bind_budget_matched_step(
            step=2,
            activity_trial_ids=ids,
            carrier_trial_ids=tuple(reversed(ids)),
            activity_rows=np.zeros((4, 100, 3), dtype=np.float32),
            carrier_rates=np.ones((3, 4), dtype=np.float64),
            direction_radians=_theta(4),
            source_prior_sha256="1" * 64,
            posterior_normalizer_sha256="2" * 64,
        )


def test_angular_reliability_is_finite_and_flat_units_fail_closed() -> None:
    theta = _theta(8)
    rates = _rates(theta, coefficients=((2.0, 0.0, 0.0), (1.0, 1.0, 0.4)))
    rates[1, -1] += 0.2
    fit = posterior.fit_posterior_mean(rates, theta, prior_variance=0.8)
    q = posterior.angular_reliability(fit)
    assert q.shape == (2,) and np.isfinite(q).all()
    assert q[0] == -20.0


def test_import_does_not_load_torch() -> None:
    before = "torch" in sys.modules
    importlib.reload(posterior)
    assert ("torch" in sys.modules) is before
