from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.m1_emg_afc4 import (  # noqa: E402
    SUPPORT_TRIALS,
    affine_design_report,
    afc4_from_encoding,
    deterministic_split_half_reliability,
    fit_source_frozen_emg_basis,
    fit_unit_encoding,
)


def test_source_pca_is_deterministic_and_sign_canonical() -> None:
    source = [
        np.array([[1.0, 0.0, 2.0], [2.0, 0.5, 3.0], [3.0, 1.0, 5.0]]),
        np.array([[4.0, 1.5, 6.0], [5.0, 2.0, 8.0], [6.0, 2.5, 9.0]]),
    ]
    first = fit_source_frozen_emg_basis(source, q=2)
    second = fit_source_frozen_emg_basis(source, q=2)
    assert np.allclose(first.mean, second.mean)
    assert np.allclose(first.components, second.components)
    assert np.all(first.components[np.arange(2), first.sign_anchor_indices] >= 0.0)
    assert first.source_energy_explained.sum() <= 1.0


def test_afc4_interfaces_have_four_coordinates() -> None:
    q2_weights = np.array([[3.0, 4.0], [0.0, 5.0]])
    intercept = np.array([7.0, 11.0])
    q2 = afc4_from_encoding(q2_weights, intercept)
    assert q2.shape == (2, 4)
    assert np.allclose(q2[:, 2], [5.0, 5.0])
    q3 = afc4_from_encoding(np.column_stack((q2_weights, [13.0, 17.0])), intercept)
    assert np.allclose(q3, [[3.0, 4.0, 13.0, 7.0], [0.0, 5.0, 17.0, 11.0]])


def test_design_and_split_half_are_support_only() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(SUPPORT_TRIALS, 2))
    neural = 2.0 + scores @ np.array([[1.5, -0.2, 0.4], [0.3, 1.1, -0.5]]) + rng.normal(scale=0.01, size=(SUPPORT_TRIALS, 3))
    report = affine_design_report(scores)
    assert report["full_rank"] is True
    weights, intercept = fit_unit_encoding(scores, neural)
    assert weights.shape == (3, 2)
    assert intercept.shape == (3,)
    reliability = deterministic_split_half_reliability(scores, neural)
    assert reliability["split"]["even_trial_indices"] == [0, 2, 4, 6, 8]
    assert reliability["split"]["odd_trial_indices"] == [1, 3, 5, 7, 9]
    assert reliability["weight_flattened_pearson"] is not None


def test_split_half_rejects_non_m10_input() -> None:
    with pytest.raises(ValueError, match="exactly 10"):
        deterministic_split_half_reliability(np.ones((9, 2)), np.ones((9, 3)))
