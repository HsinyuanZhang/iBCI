from __future__ import annotations

import numpy as np
import pytest

from src.data.afc4_features import (
    afc4_from_response_sums,
    circular_afc4_from_trial_sums,
    deterministic_afc4_row_permutation,
    deterministic_label_permutation,
    fit_train_circular_afc4_stats,
)
from src.data.falcon_t4_features import (
    deterministic_row_permutation,
    fit_train_t4_stats,
    t4_from_trial_sums,
)


def _fixture(seed: int = 7, trials: int = 33, channels: int = 11):
    rng = np.random.RandomState(seed)
    angles = np.linspace(-np.pi, np.pi, trials, endpoint=False).astype(np.float32)
    lengths = rng.randint(40, 101, size=trials).astype(np.int64)
    baseline = rng.uniform(0.2, 1.0, size=channels)
    weights = rng.normal(0.0, 0.1, size=(2, channels))
    rates = baseline + np.cos(angles)[:, None] * weights[0] + np.sin(angles)[:, None] * weights[1]
    sums = rates * lengths[:, None]
    return sums, lengths, angles, baseline, weights


def test_synthetic_coefficients_recovered():
    sums, lengths, angles, baseline, weights = _fixture()
    features = circular_afc4_from_trial_sums(sums, lengths, angles, source="synthetic")
    np.testing.assert_allclose(features[:, :2], weights.T, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(features[:, 2], np.linalg.norm(weights, axis=0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(features[:, 3], baseline, rtol=1e-6, atol=1e-7)


def test_circular_reduction_matches_t4_and_train_stats_exactly():
    sums, lengths, angles, _, _ = _fixture()
    legacy = t4_from_trial_sums(sums, lengths, angles, source="legacy")
    generalized = circular_afc4_from_trial_sums(sums, lengths, angles, source="generalized")
    np.testing.assert_array_equal(generalized, legacy)

    mappings = ({"s": sums}, {"s": lengths}, {"s": angles})
    legacy_mean, legacy_std = fit_train_t4_stats(*mappings, ["s"], 24)
    new_mean, new_std = fit_train_circular_afc4_stats(*mappings, ["s"], 24)
    np.testing.assert_array_equal(new_mean, legacy_mean)
    np.testing.assert_array_equal(new_std, legacy_std)


def test_general_two_basis_fit_and_intercept_unpenalized_ridge():
    sums, lengths, angles, _, _ = _fixture()
    basis = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ordinary = afc4_from_response_sums(sums, lengths, basis, source="ols")
    shrunk = afc4_from_response_sums(sums, lengths, basis, source="ridge", ridge=10.0)
    assert ordinary.shape == shrunk.shape == (sums.shape[1], 4)
    assert np.linalg.norm(shrunk[:, :2]) < np.linalg.norm(ordinary[:, :2])
    np.testing.assert_allclose(shrunk[:, 3], ordinary[:, 3], atol=1e-6)


def test_rank_and_shape_fail_closed():
    sums = np.ones((8, 4), dtype=np.float32)
    lengths = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="rank"):
        circular_afc4_from_trial_sums(sums, lengths, np.zeros(8), source="rank1")
    with pytest.raises(ValueError, match=r"\[M,2\]"):
        afc4_from_response_sums(sums, lengths, np.ones((8, 3)), source="bad-width")


def test_shuffle_contracts_are_deterministic_and_nonidentity():
    rows = deterministic_afc4_row_permutation(96, session_name="held_in_0", seed=42)
    np.testing.assert_array_equal(rows, deterministic_row_permutation(96, session_name="held_in_0", seed=42))
    assert not np.array_equal(rows, np.arange(96))
    labels = deterministic_label_permutation(33, context="held_in_0", seed=42)
    np.testing.assert_array_equal(labels, deterministic_label_permutation(33, context="held_in_0", seed=42))
    assert not np.array_equal(labels, np.arange(33))
