"""E2 unit contracts. Synthetic only; no NWB, no decoder R2."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import m1_dynamic_audit as e2
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank


def test_lag_indices_stay_inside_one_trial() -> None:
    ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    valid = e2.lag_valid_indices(ids, lag_bins=2)
    for index in valid:
        assert ids[index - 2] == ids[index] == ids[index + 2]
    assert 5 not in set(valid.tolist())
    assert 6 not in set(valid.tolist())


def test_z_dynamic_concatenates_three_lags() -> None:
    z0 = np.arange(20 * 3, dtype=np.float64).reshape(20, 3)
    idx = np.array([5, 10], dtype=np.int64)
    out = e2.z_dynamic(z0, idx, lag_bins=5)
    assert out.shape == (2, 9)
    assert np.array_equal(out[0], np.concatenate((z0[0], z0[5], z0[10])))


def test_inherited_ridge_matches_bank_for_rank3() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=(40, 3))
    rates = z @ np.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]]).T + 0.5
    rates = np.column_stack((rates, rng.normal(size=(40, 2))))
    w_local, b_local = e2.inherited_unit_ridge(z, rates, 1.0)
    w_bank, b_bank = bank.fit_all_units(z, rates)
    assert w_local.shape == w_bank.shape
    assert np.allclose(w_local, w_bank)
    assert np.allclose(b_local, b_bank)


def test_lambda_does_not_change_with_dynamic_rank() -> None:
    rng = np.random.default_rng(3)
    z9 = rng.normal(size=(50, 9))
    rates = rng.normal(size=(50, 4))
    weights, intercepts = e2.inherited_unit_ridge(z9, rates, plan.M1_RIDGE_LAMBDA)
    assert weights.shape == (4, 9)
    assert intercepts.shape == (4,)
    assert plan.inherited_ridge_objective()["sample_count_changes_lambda"] is False
    assert plan.E2_DYNAMIC_RANK == 9


def test_variance_weighted_r2_perfect_is_one() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert e2.variance_weighted_r2(y, y) == pytest.approx(1.0)
    assert e2.mse(y, y) == pytest.approx(0.0)
