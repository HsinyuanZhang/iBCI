import numpy as np
import pytest

from fair_v2.numerics import (
    AlignedFAMap, CoralMap, FactorModel, _fa_em_step_sufficient,
    causal_lags, fit_ridge_grid, smooth_raw,
)


def test_smooth_raw_is_official_normalized_causal_12_tap_filter():
    x = np.zeros((16, 1)); x[0, 0] = 1
    got = smooth_raw(x)
    kernel = np.exp(-np.arange(12) * 20 / 240); kernel /= kernel.sum()
    np.testing.assert_allclose(got[:12, 0], kernel, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got[12:, 0], 0, rtol=0, atol=1e-15)
    x = np.zeros((20, 1)); x[9, 0] = 3
    np.testing.assert_allclose(smooth_raw(x)[10, 0], 3 * kernel[1])


def test_coral_maps_target_distribution_to_source_distribution():
    # Same full-rank base samples give exact finite-sample covariances, while
    # non-commuting source/target covariances exercise the affine orientation.
    rng = np.random.RandomState(10)
    base = rng.normal(size=(200, 3))
    source_mean = np.array([2.0, -1.0, .5])
    target_mean = np.array([-3.0, 4.0, 1.5])
    source_factor = np.array([[2., .3, .1], [.2, 1.2, .4], [.1, .2, .8]])
    target_factor = np.array([[.7, .4, .3], [-.2, 1.8, .1], [.5, .1, 1.1]])
    source = base @ source_factor.T + source_mean
    target = base @ target_factor.T + target_mean
    mapped = CoralMap.fit(source, target, ridge=0, shrinkage=0).transform(target)
    np.testing.assert_allclose(mapped.mean(0), source.mean(0), atol=2e-12)
    np.testing.assert_allclose(np.cov(mapped, rowvar=False), np.cov(source, rowvar=False),
                               rtol=2e-12, atol=2e-12)


def test_sufficient_stat_em_step_matches_full_covariance_reference():
    rng = np.random.RandomState(7)
    x = rng.normal(size=(300, 5))
    s = (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)
    load = rng.normal(size=(5, 2))
    psi = np.array([.3, .7, .5, .9, .4])
    got_load, got_psi = _fa_em_step_sufficient(s, load, psi, 1e-8)

    # Slow full-C Gaussian posterior reference, independently using Sigma^-1.
    beta = load.T @ np.linalg.inv(load @ load.T + np.diag(psi))
    ezz = np.eye(2) - beta @ load + beta @ s @ beta.T
    reference_load = (s @ beta.T) @ np.linalg.inv(ezz)
    reference_psi = np.diag(s) - np.diag(reference_load @ (beta @ s))
    np.testing.assert_allclose(got_load, reference_load, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(got_psi, reference_psi, rtol=1e-11, atol=1e-11)


def test_posterior_and_stable_rows_have_their_declared_behavior():
    rng = np.random.RandomState(13)
    x = rng.normal(size=(250, 5))
    fa = FactorModel.fit(x, latent_dim=2, seed=3, n_init=1, max_iter=500)
    beta = fa.posterior_matrix()
    full = fa.components_.T @ np.linalg.inv(
        fa.components_ @ fa.components_.T + np.diag(fa.noise_variance_)
    )
    np.testing.assert_allclose(beta, full, rtol=1e-10, atol=1e-10)
    aligned = AlignedFAMap.fit(fa, fa, stable_fraction=.6, posterior="stable", loading_threshold=0)
    altered = x.copy(); nonstable = np.setdiff1d(np.arange(x.shape[1]), aligned.stable_rows)
    altered[:, nonstable] += rng.normal(scale=100, size=(len(x), len(nonstable)))
    np.testing.assert_allclose(aligned.transform(x), aligned.transform(altered), rtol=0, atol=1e-12)


def test_alignment_rotation_maps_target_loadings_to_source_loadings():
    rng = np.random.RandomState(8); x = rng.normal(size=(180, 5))
    source = FactorModel.fit(x, 2, seed=2, n_init=1, max_iter=500)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    target = FactorModel(source.components_ @ q.T, source.mean_, source.noise_variance_, 2, source.diagnostics)
    aligned = AlignedFAMap.fit(source, target, stable_fraction=1, posterior="all", loading_threshold=0)
    np.testing.assert_allclose(target.components_ @ aligned.rotation, source.components_, rtol=1e-10, atol=1e-10)


def test_causal_lags_counts_total_bins_and_ridge_matches_sklearn():
    features = np.array([[1., 10.], [2., 20.], [3., 30.]])
    got = causal_lags(features, np.array([0, 2]), history_bins=2)
    np.testing.assert_allclose(got, [[1, 10, 0, 0], [3, 30, 2, 20]])
    with pytest.raises(ValueError): causal_lags(features, np.array([-1]), history_bins=1)

    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    rng = np.random.RandomState(11); x, y = rng.normal(size=(80, 7)), rng.normal(size=(80, 3))
    for alpha, ours in fit_ridge_grid(x, y, [.01, 1.]).items():
        reference = Ridge(alpha=alpha, fit_intercept=True).fit(x, y)
        np.testing.assert_allclose(ours.predict(x), reference.predict(x), rtol=1e-11, atol=1e-11)
