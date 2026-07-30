from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_t4_confidence_shrinkage import (
    BASELINE,
    nested_loso_summary,
    shrink_factor,
)
from mc_maze.unit_side_features import (
    KNOWN_FEATURE_GROUPS,
    SIDE_FEATURE_DIMS,
    T4_WIENER_SHRINK_STRENGTH,
    base_feature_group,
    is_shuffled_control,
    uncertainty_wiener_shrink_t4,
)


def test_shrink_factor_is_bounded_and_zero_strength_is_identity():
    signal = np.asarray([0.0, 1.0, 4.0])
    uncertainty = np.asarray([1.0, 1.0, 2.0])
    for family in ("wiener", "positive_part"):
        identity = shrink_factor(
            signal,
            uncertainty,
            family=family,
            strength=0.0,
        )
        # The zero-signal direction is immaterial; all nonzero modulation
        # coefficients must remain exactly unshrunk at strength zero.
        assert np.allclose(identity[1:], 1.0)
        observed = shrink_factor(
            signal,
            uncertainty,
            family=family,
            strength=1.0,
        )
        assert np.all(observed >= 0.0)
        assert np.all(observed <= 1.0)
        assert observed[-1] > observed[1]


def test_shrink_factor_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shapes differ"):
        shrink_factor(
            np.ones(2),
            np.ones(3),
            family="wiener",
            strength=1.0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        shrink_factor(
            np.asarray([-1.0]),
            np.ones(1),
            family="wiener",
            strength=1.0,
        )
    with pytest.raises(ValueError, match="unsupported"):
        shrink_factor(
            np.ones(1),
            np.ones(1),
            family="unknown",
            strength=1.0,
        )


def test_nested_loso_selects_on_other_sessions_only():
    sessions = {}
    for index in range(4):
        # Candidate "good" is superior on three sessions and deliberately
        # catastrophic on one. Each fold must select using only the other rows.
        baseline = np.zeros(3)
        good = np.full(3, -0.2 if index < 3 else 10.0)
        bad = np.full(3, 0.1)
        sessions[f"s{index}"] = {
            "log_mse": {
                BASELINE: baseline,
                "good": good,
                "bad": bad,
            },
            "factors": {
                BASELINE: np.ones(3),
                "good": np.full(3, 0.5),
                "bad": np.full(3, 0.8),
            },
        }
    result = nested_loso_summary(
        sessions,
        (BASELINE, "good", "bad"),
    )
    # Holding out the catastrophic session exposes only the three good rows,
    # so that fold selects "good"; folds that train on the catastrophic row do
    # not leak the held-out good score into selection.
    assert result["per_session"]["s3"]["selected_candidate"] == "good"
    assert all(
        result["per_session"][session]["selected_candidate"] == BASELINE
        for session in ("s0", "s1", "s2")
    )


def test_deployment_t4w3_formula_matches_audit_wiener_family():
    directions = np.tile(np.arange(8, dtype=np.int64), 2)
    t4 = np.asarray(
        [[2.0, 1.0, np.sqrt(5.0), 4.0], [0.5, -0.5, np.sqrt(0.5), 2.0]]
    )
    log_variance = np.log(np.asarray([1.5, 0.25]))
    observed, factors = uncertainty_wiener_shrink_t4(
        t4,
        log_variance,
        directions,
    )
    theta = -3.0 * np.pi / 4.0 + directions * (np.pi / 4.0)
    design = np.stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta)],
        axis=1,
    )
    uncertainty = np.exp(log_variance) * np.trace(
        np.linalg.inv(design.T @ design)[1:3, 1:3]
    )
    expected = shrink_factor(
        np.square(t4[:, 0]) + np.square(t4[:, 1]),
        uncertainty,
        family="wiener",
        strength=T4_WIENER_SHRINK_STRENGTH,
    )
    assert np.allclose(factors, expected)
    assert np.allclose(observed[:, :2], t4[:, :2] * expected[:, None])
    assert np.allclose(observed[:, 2], np.hypot(observed[:, 0], observed[:, 1]))
    assert np.array_equal(observed[:, 3], t4[:, 3])


def test_t4w3_registry_and_shuffle_control_are_dimension_matched():
    assert SIDE_FEATURE_DIMS["t4w3"] == SIDE_FEATURE_DIMS["ts4w3"] == 4
    assert "t4w3" in KNOWN_FEATURE_GROUPS
    assert "ts4w3" not in KNOWN_FEATURE_GROUPS
    assert base_feature_group("ts4w3") == "t4w3"
    assert is_shuffled_control("ts4w3") is True
