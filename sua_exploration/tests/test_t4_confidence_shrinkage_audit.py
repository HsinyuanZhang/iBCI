from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_t4_confidence_shrinkage import (
    BASELINE,
    nested_loso_summary,
    shrink_factor,
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
