from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_estimator_v5 as v5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def _latent(rows: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    output = rng.normal(size=(rows, v5.RANK))
    assert np.linalg.matrix_rank(np.c_[np.ones(rows), output]) == v5.CARRIER_DIM
    return output


def test_v5_candidate_matrix_is_fixed_and_five_wide() -> None:
    assert [candidate.name for candidate in v5.CANDIDATES] == [
        "hse5_pca_delta_q4", "poisson_exposure_irls", "within_trial_contrast_ridge", "eb_channel_shrinkage",
    ]
    assert all(candidate.carrier_dim == 5 for candidate in v5.CANDIDATES)


def test_count_recovery_is_exact_for_bound_log_rate_contract() -> None:
    durations = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
    counts = np.zeros((3, v1.EXPECTED_NEURONS), dtype=np.float64)
    counts[:, :3] = np.asarray([[0, 1, 7], [2, 3, 0], [4, 8, 1]], dtype=np.float64)
    response = np.log1p(counts / durations[:, None])
    np.testing.assert_array_equal(v5.log_rates_to_counts(response, durations), counts)


def test_poisson_irls_recovers_one_exposure_invariant_log_rate() -> None:
    # Each duration group has zero-mean full-rank covariates.  The true count
    # rate is exactly two spikes/s for every event and channel, so a correct
    # exposure-offset GLM must recover beta_0=log(2), slopes=0 despite 4x
    # changing durations.  The V5r1 double-offset bug fails this assertion.
    rng = np.random.default_rng(93)
    durations = np.repeat(np.asarray([1.0, 2.0, 3.0, 4.0]), 5)
    z = rng.normal(size=(len(durations), v5.RANK))
    for duration in np.unique(durations):
        mask = durations == duration
        z[mask] -= z[mask].mean(axis=0, keepdims=True)
    assert np.linalg.matrix_rank(np.c_[np.ones(len(z)), z]) == v5.CARRIER_DIM
    counts = np.broadcast_to(2.0 * durations[:, None], (len(durations), v1.EXPECTED_NEURONS)).copy()
    response = np.log1p(counts / durations[:, None])
    observed, metadata = v5.fit_poisson_exposure_beta(z, response, durations)
    assert observed.shape == (v5.CARRIER_DIM, v1.EXPECTED_NEURONS)
    assert np.isfinite(observed).all() and 1 <= metadata["iterations"] <= v5.POISSON_MAX_ITERATIONS
    predicted = v5.poisson_predict(observed, z)
    assert predicted.shape == response.shape and np.isfinite(predicted).all()
    np.testing.assert_allclose(observed[0], np.log(2.0), atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(observed[1:], 0.0, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(predicted, response, atol=1.0e-10, rtol=0.0)


def test_contrast_fit_removes_trial_offsets_from_slopes() -> None:
    z = _latent(16)
    trials = np.repeat(np.arange(4), 4)
    true = np.zeros((v5.RANK, v1.EXPECTED_NEURONS))
    true[0, :2] = np.asarray([0.5, -0.25])
    trial_offset = np.repeat(np.asarray([-3.0, 2.0, 4.0, -2.5]), 4)[:, None]
    response = z @ true
    beta_without, _ = v5.fit_within_trial_contrast_beta(z, response, trials)
    beta, metadata = v5.fit_within_trial_contrast_beta(z, response + trial_offset, trials)
    # The fixed ridge penalty shrinks the absolute slope, but trial-level
    # nuisance offsets cannot change the contrast-estimated slopes.
    np.testing.assert_allclose(beta[1:, :2], beta_without[1:, :2], atol=1.0e-12, rtol=0.0)
    assert beta[1, 0] > 0 and beta[1, 1] < 0
    assert metadata["trial_count"] == 4


def test_eb_posterior_interpolates_between_target_and_source_prior() -> None:
    z = _latent(14)
    response = np.zeros((14, v1.EXPECTED_NEURONS), dtype=np.float64)
    response[:, 0] = 0.1 + 0.2 * z[:, 0]
    target, _sigma, target_cov = v5._linear_beta(z, response, ridge_lambda=v5.EB_TARGET_RIDGE_LAMBDA)
    prior = v5.EBPrior(
        outer_date="19250101", budget=3, source_sessions=("a",) * 10,
        mean=np.ones_like(target), tau2=np.maximum(target_cov * 0.5, v5.EB_VARIANCE_FLOOR),
        source_carrier_sha256=tuple("x" for _ in range(10)), prior_sha256="synthetic",
    )
    posterior, metadata = v5.fit_eb_beta(z, response, prior)
    assert np.all(posterior <= np.maximum(target, prior.mean) + 1.0e-12)
    assert np.all(posterior >= np.minimum(target, prior.mean) - 1.0e-12)
    assert 0.0 <= metadata["minimum_target_weight"] <= metadata["maximum_target_weight"] <= 1.0


def test_eb_ridge_variance_uses_sandwich_covariance_and_smoother_df() -> None:
    z = _latent(15)
    response = np.zeros((15, v1.EXPECTED_NEURONS), dtype=np.float64)
    response[:, 0] = 0.2 + 0.3 * z[:, 0] - 0.1 * z[:, 1]
    beta, sigma2, observed = v5._linear_beta(z, response, ridge_lambda=v5.EB_TARGET_RIDGE_LAMBDA)
    x = np.c_[np.ones(len(z)), z]
    penalty = np.diag([0.0] + [1.0] * v5.RANK) * (len(z) * v5.EB_TARGET_RIDGE_LAMBDA)
    inverse = np.linalg.inv(x.T @ x + penalty)
    residual_df = len(z) - np.trace((x.T @ x) @ inverse)
    expected_sigma = np.maximum(np.sum((response - x @ beta) ** 2, axis=0) / residual_df, v5.EB_VARIANCE_FLOOR)
    expected = np.diag(inverse @ (x.T @ x) @ inverse)[:, None] * expected_sigma[None, :]
    np.testing.assert_allclose(sigma2, expected_sigma, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(observed, expected, atol=1.0e-12, rtol=0.0)


def test_v5_gate_does_not_lower_material_threshold_or_skip_row_control() -> None:
    good = {"defined_sessions": 13, "mean": 0.0201, "median": 0.0101, "positive": 10,
            "leave_largest_absolute_out_mean": 0.0001}
    aggregate = {"correct_minus_hse5": good, "correct_minus_label_shuffle": good,
                 "correct_minus_intercept": good, "correct_minus_row_shuffle": good}
    assert v5.budget_gate(v5.CANDIDATES[1], aggregate)["passed"] is True
    aggregate["correct_minus_row_shuffle"] = {**good, "mean": -0.1}
    assert v5.budget_gate(v5.CANDIDATES[1], aggregate)["passed"] is False
