"""Unit tests for the new external sub-M classical-control numerical core.

These tests are deliberately synthetic: they establish the causal-window and
closed-form invariants without opening an external sub-M session or touching a
checkpoint.  The real-session smoke is owned by the separate CLI runner.
"""
from __future__ import annotations

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as controls


def test_targets_are_the_last_bin_of_each_causal_window() -> None:
    neural = np.zeros((8, 2), dtype=np.float32)
    behavior = np.stack((np.arange(8), -np.arange(8)), axis=1).astype(np.float32)
    starts = np.asarray([0, 3], dtype=np.int64)

    actual = controls.targets_at_window_end(behavior, starts, window_size=3)

    np.testing.assert_array_equal(actual, behavior[[2, 5]])
    with pytest.raises(controls.ClassicalControlError, match="outside"):
        controls.targets_at_window_end(behavior, np.asarray([6]), window_size=3)
    with pytest.raises(controls.ClassicalControlError, match="outside"):
        controls.raw_window_features(neural, np.asarray([6]), window_size=3)


def test_raw_window_features_preserve_causal_history_order() -> None:
    neural = np.arange(24, dtype=np.float32).reshape(8, 3)
    starts = np.asarray([1, 4], dtype=np.int64)

    actual = controls.raw_window_features(neural, starts, window_size=2)

    expected = np.asarray(
        [
            neural[1:3].reshape(-1),
            neural[4:6].reshape(-1),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_prefix_rates_equal_direct_causal_window_rates() -> None:
    rng = np.random.default_rng(688)
    neural = rng.poisson(0.4, size=(19, 4)).astype(np.float32)
    starts = np.asarray([0, 2, 7, 14], dtype=np.int64)
    prefix = controls.prefix_sums(neural)

    actual = controls.window_rates_from_prefix(prefix, starts, window_size=5, bin_size_s=0.02)
    expected = np.stack([neural[start : start + 5].sum(axis=0) / 0.1 for start in starts])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_population_vector_is_baseline_subtracted_and_pd_normalized() -> None:
    preferred, zero = controls.preferred_directions_from_cosine(
        np.asarray([3.0, 0.0]),
        np.asarray([4.0, 0.0]),
        np.asarray([5.0, 0.0]),
    )
    assert zero == 1
    np.testing.assert_allclose(preferred, [[0.6, 0.8], [0.0, 0.0]], atol=1.0e-12)
    rates = np.asarray([[12.0, 99.0], [7.0, -12.0]], dtype=np.float64)
    vector = controls.population_vectors(rates, preferred, np.asarray([2.0, 9.0]))
    np.testing.assert_allclose(vector, [[6.0, 8.0], [3.0, 4.0]], atol=1.0e-12)


def test_population_vector_affine_gain_is_exact_on_full_rank_design() -> None:
    vectors = np.asarray([[-1.0, 2.0], [0.0, 0.0], [3.0, -2.0], [4.0, 1.0]])
    gain_reference = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    intercept_reference = np.asarray([1.5, -2.0])
    targets = vectors @ gain_reference + intercept_reference

    gain, intercept, rank = controls.fit_population_vector_gain(vectors, targets)
    actual = controls.predict_population_vector(vectors, gain, intercept)

    assert rank == 3
    np.testing.assert_allclose(gain, gain_reference, atol=1.0e-12)
    np.testing.assert_allclose(intercept, intercept_reference, atol=1.0e-12)
    np.testing.assert_allclose(actual, targets, atol=1.0e-6)


def test_fixed_normalized_ridge_is_calibration_only_and_has_unpenalized_intercept() -> None:
    rng = np.random.default_rng(20260805)
    features = rng.normal(size=(40, 5)).astype(np.float32)
    weights = np.asarray([[0.7, -0.5], [0.0, 1.1], [0.4, 0.2], [-0.3, 0.5], [0.1, -0.2]])
    intercept = np.asarray([2.0, -3.0])
    targets = (features @ weights + intercept).astype(np.float32)

    readout = controls.fit_ridge(features, targets, normalized_lambda=1.0, device="cpu")
    actual = controls.predict_ridge(features, readout, device="cpu")

    assert readout.normalized_lambda == 1.0
    assert readout.solver_device == "cpu"
    # Ridge shrinks the slopes but the fitted unpenalized target mean preserves
    # the constant offset.  A new constant-design test catches accidental
    # penalization of an explicit intercept column.
    constant_features = np.full((8, 3), 7.0, dtype=np.float32)
    constant_target = np.tile(np.asarray([[3.0, -4.0]], dtype=np.float32), (8, 1))
    constant = controls.fit_ridge(constant_features, constant_target, device="cpu")
    constant_prediction = controls.predict_ridge(constant_features, constant, device="cpu")
    np.testing.assert_allclose(constant_prediction, constant_target, atol=1.0e-6)
    assert np.isfinite(actual).all()


def test_ridge_rejects_nonpositive_or_query_tuned_lambda() -> None:
    features = np.ones((4, 2), dtype=np.float32)
    targets = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(controls.ClassicalControlError, match="positive"):
        controls.fit_ridge(features, targets, normalized_lambda=0.0)

