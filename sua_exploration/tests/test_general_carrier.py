"""Fast contracts for the movement-aligned general-carrier Gate A primitives."""
from __future__ import annotations

import numpy as np
import pytest

from mc_maze.general_carrier import (
    CarrierFit,
    deterministic_row_shuffle,
    fit_encoding,
    pair_indices,
    predict_encoding,
    select_encoding_hyperparameters,
)


def test_pair_indices_never_crosses_mask_boundary() -> None:
    mask = np.zeros(30, dtype=bool)
    mask[10:24] = True
    segments = np.zeros(30, dtype=np.int64)
    response, labels = pair_indices(mask, lag_bins=3, segment_ids=segments)
    assert response.min() == 13
    assert labels.min() == 10
    assert np.all(mask[response])
    assert np.all(mask[labels])


def test_pair_indices_rejects_too_few_within_split_rows() -> None:
    with pytest.raises(ValueError, match="fewer than ten"):
        pair_indices(
            np.array([False, True, True]), lag_bins=0, segment_ids=np.zeros(3, dtype=np.int64)
        )


def test_pair_indices_rejects_lag_across_trial_boundary() -> None:
    mask = np.ones(30, dtype=bool)
    segments = np.repeat([0, 1, 2], 10)
    response, labels = pair_indices(mask, lag_bins=3, segment_ids=segments)
    assert np.all(segments[response] == segments[labels])
    assert response.min() == 3


def test_encoding_recovers_known_two_dimensional_carrier() -> None:
    rng = np.random.RandomState(4)
    behavior = rng.standard_normal((160, 2))
    true_w = np.array([[2.0, -0.5], [-1.0, 1.5], [0.3, 0.7]])
    true_b = np.array([4.0, 1.0, -2.0])
    rate = behavior @ true_w.T + true_b
    mask = np.ones(len(behavior), dtype=bool)
    segments = np.zeros(len(mask), dtype=np.int64)
    fit = fit_encoding(rate, behavior, mask, segments, lag_bins=0, alpha=0.0)
    assert np.allclose(fit.weights, true_w, atol=1e-10)
    assert np.allclose(fit.intercept, true_b, atol=1e-10)
    assert np.allclose(fit.features_2d()[:, :2], true_w)
    response, labels = pair_indices(mask, 0, segments)
    assert np.allclose(predict_encoding(fit, behavior, response, labels), rate)


def test_hyperparameter_selection_cannot_use_evaluation_mask() -> None:
    rng = np.random.RandomState(1)
    y = rng.standard_normal((100, 2))
    r = y @ np.array([[1.0, 2.0], [3.0, -1.0]]) + 0.5
    fit_mask = np.zeros(100, dtype=bool)
    select_mask = np.zeros(100, dtype=bool)
    evaluation_mask = np.zeros(100, dtype=bool)
    fit_mask[:40] = True
    select_mask[40:80] = True
    evaluation_mask[80:] = True
    lag, alpha, curve = select_encoding_hyperparameters(
        r, y, fit_mask, select_mask, np.zeros(100, dtype=np.int64), lags=(0,), alphas=(0.0, 1.0)
    )
    assert (lag, alpha) == (0, 0.0)
    assert set(curve) == {"lag=0,alpha=0", "lag=0,alpha=1"}
    assert not np.any(fit_mask & evaluation_mask)
    assert not np.any(select_mask & evaluation_mask)


def test_row_shuffle_moves_complete_descriptor_rows_deterministically() -> None:
    fit = CarrierFit(
        weights=np.arange(12, dtype=float).reshape(6, 2),
        intercept=np.arange(6, dtype=float),
        alpha=1.0,
        lag_bins=0,
    )
    one = deterministic_row_shuffle(fit, 9)
    two = deterministic_row_shuffle(fit, 9)
    assert np.array_equal(one.weights, two.weights)
    assert np.array_equal(one.intercept, two.intercept)
    # Each shuffled descriptor stays internally paired: b is its original row id.
    assert np.array_equal(one.weights[:, 0] // 2, one.intercept)
