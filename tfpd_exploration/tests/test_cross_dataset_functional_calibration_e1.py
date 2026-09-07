"""E1 unit contracts. Synthetic only; no NWB, no decoder R2."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import h1_population_audit as e1
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan


def test_masks_keep_focal_and_are_deterministic() -> None:
    left = e1.other_channel_masks(176, 22)
    right = e1.other_channel_masks(176, 22)
    assert len(left) == 16
    assert left == right
    keep = left[0]["keep_indices"]
    drop = left[0]["drop_indices"]
    assert 22 in keep
    assert 22 not in drop
    n_other = 175
    assert len(keep) == 1 + int(0.75 * n_other)
    other = e1.other_channel_masks(176, 0)
    assert other[0]["keep_indices"] != left[0]["keep_indices"] or other[0]["drop_indices"] != left[0]["drop_indices"]


def test_intervention_preserves_focal_and_shape() -> None:
    rng = np.random.default_rng(0)
    rates = rng.normal(size=(40, 16))
    mean = e1.support_channel_mean(rates)
    drop = np.array([1, 3, 5], dtype=np.int64)
    out = e1.intervene_other_channels(rates, drop, mean)
    assert out.shape == rates.shape
    assert np.array_equal(out[:, 0], rates[:, 0])
    assert np.allclose(out[:, 1], mean[1])
    assert np.allclose(out[:, 2], rates[:, 2])


def test_forward_control_is_invariant_to_other_channels() -> None:
    rng = np.random.default_rng(1)
    rates = rng.normal(size=(30, 8))
    velocity = rng.normal(size=(30, 7))
    base = e1.forward_unit_ridge(rates, velocity, 100.0)
    drop = np.array([0, 2, 4, 6], dtype=np.int64)
    mean = rates.mean(axis=0)
    intervened = e1.intervene_other_channels(rates, drop, mean)
    other = e1.forward_unit_ridge(intervened, velocity, 100.0)
    assert np.allclose(base[:, 1], other[:, 1])
    assert np.allclose(base[:, 3], other[:, 3])
    assert not np.allclose(base[:, 0], other[:, 0])


def test_relative_change_undefined_for_zero_baseline() -> None:
    zeros = np.zeros(4)
    ones = np.ones(4)
    assert e1.relative_change(zeros, ones) is None
    assert e1.cosine(zeros, ones) is None
    assert e1.relative_change(ones, ones) == 0.0
    assert e1.cosine(ones, ones) == 1.0
    assert plan.E1_SEED == 20260905
