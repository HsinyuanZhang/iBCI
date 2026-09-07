"""Focused tests for the §8 affine diagnostics (no data, no CUDA)."""
from __future__ import annotations

import numpy as np
import pytest

from src.affine_diagnostics_v1 import affine


def test_fit_recovers_exact_affine_map():
    rng = np.random.default_rng(7)
    pred = rng.normal(size=(200, 2))
    a_true = np.array([[1.3, -0.2], [0.4, 0.9]])
    b_true = np.array([0.05, -0.1])
    target = pred @ a_true.T + b_true
    six = affine.fit_affine(pred, target)
    assert np.allclose(six[:4], a_true.reshape(-1), atol=1e-8)
    assert np.allclose(six[4:], b_true, atol=1e-8)
    assert np.allclose(affine.apply_affine(pred, six), target, atol=1e-8)


def test_six_vector_payload_order():
    payload = affine.six_vector_payload([1, 2, 3, 4, 5, 6])
    assert list(payload) == ["A00", "A01", "A10", "A11", "b0", "b1"]
    assert list(payload.values()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_dispersion_of_identical_vectors_is_zero():
    vectors = [np.ones(6)] * 4
    stats = affine.dispersion_stats(vectors)
    assert stats["covariance_trace"] == 0.0
    assert stats["leading_eigenvalue"] == 0.0
    assert stats["first_pc_explained_fraction"] == 0.0
    assert stats["median_distance_to_mean_six_vector"] == 0.0


def test_dispersion_first_pc_fraction_bounded():
    rng = np.random.default_rng(11)
    vectors = [rng.normal(size=6) for _ in range(12)]
    stats = affine.dispersion_stats(vectors)
    assert 0.0 <= stats["first_pc_explained_fraction"] <= 1.0
    assert stats["leading_eigenvalue"] <= stats["covariance_trace"] + 1e-12


def test_loo_r0_never_fits_the_held_session():
    # three sessions, the first two share one affine miscalibration
    rng = np.random.default_rng(3)
    rows = {}
    for name, a in (("s0", 1.2), ("s1", 1.25), ("s2", 0.7)):
        pred = rng.normal(size=(150, 2))
        rows[name] = (pred, pred @ np.diag([a, 1.0 / a]) @ np.eye(2))
    result = affine.loo_r0_rows(rows)
    assert result["n_sessions"] == 3
    held_s2 = [row for row in result["sessions"] if row["session"] == "s2"][0]
    # the mean of s0/s1 scalings must NOT recover s2's inverse scaling
    applied = held_s2["applied_six_vector"]
    assert applied["A00"] > 1.0
    for row in result["sessions"]:
        assert row["target_label_leakage"] is False
        assert row["deployment_eligible"] is False


def test_full_opportunity_rows_carry_leakage_labels():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=(100, 2))
    rows = {"s": (pred, pred + 0.3 * rng.normal(size=(100, 2)))}
    out = affine.full_opportunity_rows(rows)
    assert out[0]["target_label_leakage"] is True
    assert out[0]["checkpoint_selection_eligible"] is False
    assert out[0]["deployment_eligible"] is False
    # in-sample least squares can never lower the fit
    assert out[0]["gain"] >= -1e-9
    assert out[0]["gain"] == pytest.approx(0.0, abs=0.15)


def test_paired_session_stats_deterministic():
    first = affine.paired_session_stats([0.01, -0.02, 0.03, 0.04])
    second = affine.paired_session_stats([0.01, -0.02, 0.03, 0.04])
    assert first == second
    assert first["n_positive"] == 3
    assert first["n_total"] == 4


def test_r2_float64_matches_house_convention():
    rng = np.random.default_rng(13)
    pred = rng.normal(size=(500, 2))
    target = rng.normal(size=(500, 2)) * 2.0 + 0.5
    value = affine.r2_float64(pred, target)
    assert np.isfinite(value)
    assert affine.r2_float64(target, target) == pytest.approx(1.0)
