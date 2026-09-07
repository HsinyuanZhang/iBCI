"""Tests for the center-out T4 coordinate scale audit.

Most tests are synthetic and open no data.  Two are marked ``real_data`` and open two development
NWBs to prove the runner's single-pass fit is bit-identical to the sealed public estimator; those
are the licence for the runner assembling its pass from ``unit_side_features`` helpers instead of
calling ``compute_unit_side_features_uncached`` twice per session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "sua_exploration" / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import t4_coordinate_scale_audit as audit  # noqa: E402


# ---------------------------------------------------------------- scale statistics


def test_coordinate_scale_stats_matches_closed_form():
    values = np.array([-2.0, 0.0, 1.0, 5.0])
    stats = audit.coordinate_scale_stats(values)
    assert stats["n_units"] == 4
    assert stats["rms"] == pytest.approx(np.sqrt((4.0 + 0.0 + 1.0 + 25.0) / 4.0))
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["sd"] == pytest.approx(np.std(values))
    assert stats["median"] == pytest.approx(0.5)
    assert stats["iqr"] == pytest.approx(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    assert stats["mad"] == pytest.approx(np.median(np.abs(values - 0.5)))


def test_scale_stats_reject_non_finite_and_empty():
    with pytest.raises(audit.AuditError):
        audit.coordinate_scale_stats(np.array([1.0, np.nan]))
    with pytest.raises(audit.AuditError):
        audit.coordinate_scale_stats(np.array([]))


def test_rms_is_scale_equivariant():
    """The gauge claim rests on this: rescaling rate rescales every rate-carrying coordinate."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=64)
    for factor in (0.25, 3.0, 11.0):
        assert audit.rms(factor * values) == pytest.approx(factor * audit.rms(values))


# ---------------------------------------------------------------- within-session imbalance


def test_within_session_imbalance_recovers_a_known_ratio():
    n = 200
    rng = np.random.default_rng(1)
    a = rng.normal(scale=1.0, size=n)
    c = rng.normal(scale=1.0, size=n)
    m = np.hypot(a, c)
    b = 40.0 * rng.normal(scale=1.0, size=n)
    t4 = np.column_stack([a, c, m, b])
    imbalance = audit.within_session_imbalance(t4)
    pooled = np.sqrt(np.mean(np.concatenate([a, c]) ** 2))
    assert imbalance["rms_ac_pooled"] == pytest.approx(pooled)
    assert imbalance["b_over_ac_pooled"] == pytest.approx(np.sqrt(np.mean(b**2)) / pooled)
    assert imbalance["b_over_ac_pooled"] > 30.0


def test_imbalance_is_invariant_to_a_common_rate_rescaling():
    """A pure gauge change cannot move the within-session imbalance; only cross-session scale."""
    rng = np.random.default_rng(2)
    t4 = rng.normal(size=(50, 4))
    t4[:, 2] = np.hypot(t4[:, 0], t4[:, 1])
    base = audit.within_session_imbalance(t4)
    scaled = audit.within_session_imbalance(7.5 * t4)
    for key, value in base.items():
        if key == "rms_ac_pooled":
            assert scaled[key] == pytest.approx(7.5 * value)
        else:
            assert scaled[key] == pytest.approx(value)


def test_within_session_imbalance_rejects_wrong_width():
    with pytest.raises(audit.AuditError):
        audit.within_session_imbalance(np.zeros((5, 3)))


def test_zero_signed_tuning_yields_undefined_rather_than_inf():
    t4 = np.column_stack([np.zeros(4), np.zeros(4), np.zeros(4), np.arange(1.0, 5.0)])
    assert audit.within_session_imbalance(t4)["b_over_ac_pooled"] is None


# ---------------------------------------------------------------- cross-session ratio


def test_cross_session_scale_ratio_reports_extremes():
    result = audit.cross_session_scale_ratio({"s_a": 2.0, "s_b": 8.0, "s_c": 4.0})
    assert result["max_over_min"] == pytest.approx(4.0)
    assert result["max_session"] == "s_b"
    assert result["min_session"] == "s_a"
    assert result["median"] == pytest.approx(4.0)
    assert result["n_sessions"] == 3


def test_cross_session_ratio_rejects_empty():
    with pytest.raises(audit.AuditError):
        audit.cross_session_scale_ratio({})


# ---------------------------------------------------------------- rank/proportionality


def test_spearman_is_one_for_a_monotone_nonlinear_relation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert audit.spearman_rho(x, x**3) == pytest.approx(1.0)
    assert audit.spearman_rho(x, -(x**3)) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 20.0, 30.0])
    assert audit.spearman_rho(x, y) == pytest.approx(1.0)


def test_spearman_is_none_when_a_vector_is_constant():
    assert audit.spearman_rho(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) is None


def test_exactly_proportional_scale_passes_both_rate_tests():
    rates = np.array([3.0, 7.0, 11.0, 19.0, 25.0])
    tracking = audit.evaluate_rate_tracking(rates, {"b_hat": 2.5 * rates})
    assert tracking["b_hat"]["spearman_rho_with_mean_rate"] == pytest.approx(1.0)
    assert tracking["b_hat"]["scale_over_mean_rate_cv"] == pytest.approx(0.0)
    assert tracking["b_hat"]["rank_association_passes"]
    assert tracking["b_hat"]["proportionality_passes"]
    assert tracking["_declared_tracks_mean_rate"]


def test_rate_unrelated_scale_fails_the_declaration():
    rates = np.array([3.0, 7.0, 11.0, 19.0, 25.0])
    tracking = audit.evaluate_rate_tracking(rates, {"b_hat": np.array([9.0, 1.0, 30.0, 2.0, 14.0])})
    assert not tracking["_declared_tracks_mean_rate"]


def test_rate_tracking_rejects_length_mismatch():
    with pytest.raises(audit.AuditError):
        audit.evaluate_rate_tracking([1.0, 2.0, 3.0], {"b_hat": [1.0, 2.0]})


# ---------------------------------------------------------------- predeclared bands


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.0, "negligible"), (1.49, "negligible"), (1.5, "moderate"), (2.99, "moderate"),
     (3.0, "substantial"), (40.0, "substantial"), (None, "undefined")],
)
def test_cross_session_bands_are_the_predeclared_ones(ratio, expected):
    assert audit.classify_cross_session(ratio) == expected


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.0, "balanced"), (2.99, "balanced"), (3.0, "material_below_h1"), (31.9, "material_below_h1"),
     (32.0, "reproduces_h1"), (61.0, "reproduces_h1"), (None, "undefined")],
)
def test_imbalance_bands_are_the_predeclared_ones(ratio, expected):
    assert audit.classify_imbalance(ratio) == expected


def _cross_session(ratios):
    return {name: {"max_over_min": value} for name, value in zip(audit.COORDINATES, ratios)}


def test_gauge_reading_needs_both_substantial_spread_and_rate_tracking():
    fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([5.0, 6.0, 5.5, 4.0]),
        median_raw_imbalance=6.0,
        median_standardized_imbalance=2.0,
        tracks_mean_rate=True,
    )
    assert fired["gauge_reading"]
    assert "gauge_reading" in fired["supported_readings"]

    not_fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([5.0, 6.0, 5.5, 4.0]),
        median_raw_imbalance=6.0,
        median_standardized_imbalance=2.0,
        tracks_mean_rate=False,
    )
    assert not not_fired["gauge_reading"]


def test_two_substantial_coordinates_are_not_enough_for_substantial():
    fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([5.0, 6.0, 1.2, 1.1]),
        median_raw_imbalance=6.0,
        median_standardized_imbalance=2.0,
        tracks_mean_rate=True,
    )
    assert not fired["cross_session_substantial"]
    assert not fired["gauge_reading"]


def test_intercept_dominance_fires_from_either_raw_or_standardized_band():
    assert audit.select_readings(
        cross_session_by_coordinate=_cross_session([1.1, 1.1, 1.1, 1.1]),
        median_raw_imbalance=45.0,
        median_standardized_imbalance=1.5,
        tracks_mean_rate=False,
    )["intercept_dominance_reading"]
    assert audit.select_readings(
        cross_session_by_coordinate=_cross_session([1.1, 1.1, 1.1, 1.1]),
        median_raw_imbalance=1.5,
        median_standardized_imbalance=45.0,
        tracks_mean_rate=False,
    )["intercept_dominance_reading"]


def test_h1_is_special_needs_negligible_spread_and_balanced_imbalance():
    fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([1.1, 1.2, 1.3, 1.4]),
        median_raw_imbalance=1.8,
        median_standardized_imbalance=1.2,
        tracks_mean_rate=False,
    )
    assert fired["h1_is_special_reading"]
    assert fired["verdict"] == "h1_is_special_reading"


def test_no_reading_can_fire_and_is_reported_as_none():
    fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([2.0, 2.0, 2.0, 2.0]),
        median_raw_imbalance=5.0,
        median_standardized_imbalance=5.0,
        tracks_mean_rate=False,
    )
    assert fired["supported_readings"] == []
    assert fired["verdict"] == "none"


def test_multiple_readings_are_reported_together():
    fired = audit.select_readings(
        cross_session_by_coordinate=_cross_session([9.0, 9.0, 9.0, 9.0]),
        median_raw_imbalance=50.0,
        median_standardized_imbalance=50.0,
        tracks_mean_rate=True,
    )
    assert fired["supported_readings"] == ["gauge_reading", "intercept_dominance_reading"]


# ---------------------------------------------------------------- roster / seal guards


def test_roster_resolution_excludes_the_sealed_formal_test_sessions():
    rosters = audit.resolve_rosters()
    assert len(rosters["subC_train"]) == 27
    assert len(rosters["subC_val"]) == 6
    assert len(rosters["subM_external"]) == 15
    everything = set(rosters["subC_train"]) | set(rosters["subC_val"]) | set(rosters["subM_external"])
    assert everything.isdisjoint(set(audit.SEALED_TEST_SESSIONS))
    assert len(everything) == 48


def test_session_path_refuses_an_unknown_session():
    with pytest.raises(audit.AuditError):
        audit.session_path("sub-C_ses-CO-19000101")


def test_standardize_matches_the_sealed_z_score_expression():
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    std = np.array([2.0, 4.0, 0.5, 1.0], dtype=np.float32)
    assert np.array_equal(audit.standardize(raw, mean, std), ((raw - mean) / std).astype(np.float32))


# ---------------------------------------------------------------- real-data equivalence


REAL_SESSIONS = ("sub-C_ses-CO-20131003", "sub-M_ses-CO-20140307")


@pytest.mark.real_data
@pytest.mark.parametrize("session", REAL_SESSIONS)
def test_single_pass_t4_is_bit_identical_to_the_sealed_public_estimator(session):
    """The runner's shortcut is only legitimate if it reproduces the sealed entry point exactly."""
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    path = audit.session_path(session)
    sealed, sealed_metadata = compute_unit_side_features_uncached(
        path,
        feature_group=audit.FEATURE_GROUP,
        pool_size=audit.POOL_SIZE,
        bin_size_ms=audit.BIN_SIZE_MS,
        window_size=audit.WINDOW_SIZE,
        trial_result_filter=audit.TRIAL_RESULT_FILTER,
        signal_view=audit.SIGNAL_VIEW,
    )
    single_pass, rates, metadata = audit.session_t4_and_rates(path)

    assert single_pass.dtype == sealed.dtype == np.float32
    assert single_pass.shape == sealed.shape
    assert np.array_equal(single_pass, sealed), "single-pass T4 diverged from the sealed estimator"
    assert metadata["n_units"] == sealed.shape[0]
    assert metadata["zero_spike_unit_count"] == sealed_metadata.zero_spike_unit_count
    assert metadata["zero_modulation_unit_count"] == sealed_metadata.zero_modulation_unit_count
    assert rates.shape == (sealed.shape[0], audit.POOL_SIZE)
    assert np.isfinite(rates).all() and (rates >= 0.0).all()


@pytest.mark.real_data
def test_modulation_column_is_the_hypotenuse_of_the_signed_columns():
    t4, _rates, _metadata = audit.session_t4_and_rates(audit.session_path(REAL_SESSIONS[0]))
    assert np.allclose(t4[:, 2], np.hypot(t4[:, 0], t4[:, 1]), atol=1e-5)
