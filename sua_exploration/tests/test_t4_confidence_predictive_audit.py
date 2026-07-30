from __future__ import annotations

import numpy as np

from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD
from scripts.audit_t4_confidence_predictive_validity import (
    _direction_design_descriptors,
    _fit_t4_matrix,
    _standardized_ridge_loso,
)


def test_predictive_audit_recovers_exact_cosine_and_balanced_design():
    directions = np.tile(np.arange(8), 4)
    theta = np.asarray(CANONICAL_DIRECTIONS_RAD)
    base = 7.0 + 2.0 * np.cos(theta) - 1.5 * np.sin(theta)
    rates = np.stack([np.tile(base, 4), np.tile(2.0 * base, 4)])
    t4, confidence = _fit_t4_matrix(rates, directions)

    assert t4.shape == (2, 4)
    assert confidence.shape == (2, 2)
    assert np.allclose(t4[0, [0, 1, 3]], [2.0, -1.5, 7.0], atol=1.0e-6)
    entropy, log_se = _direction_design_descriptors(
        directions, np.ones(2)
    )
    assert abs(entropy - 1.0) < 1.0e-12
    assert log_se.shape == (2,)


def test_loso_ridge_detects_incremental_confidence_signal():
    rows = []
    for session_index in range(4):
        for unit in range(12):
            confidence = float((unit % 5) - 2 + 0.3 * session_index)
            rows.append(
                {
                    "session": f"s{session_index}",
                    "t4_a": 0.0,
                    "t4_c": 0.0,
                    "log1p_t4_m": 0.0,
                    "t4_b": float(unit),
                    "log_residual_variance": confidence,
                    "log_ac_covariance_shape": float(session_index),
                    "log_future_prediction_mse": (
                        2.0 * confidence + 0.1 * unit
                    ),
                }
            )
    baseline = _standardized_ridge_loso(
        rows, ("t4_a", "t4_c", "log1p_t4_m", "t4_b")
    )
    with_confidence = _standardized_ridge_loso(
        rows,
        (
            "t4_a",
            "t4_c",
            "log1p_t4_m",
            "t4_b",
            "log_residual_variance",
            "log_ac_covariance_shape",
        ),
    )
    assert with_confidence["global_r2"] > baseline["global_r2"] + 0.9
