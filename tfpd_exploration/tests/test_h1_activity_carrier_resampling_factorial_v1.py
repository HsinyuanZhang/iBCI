from __future__ import annotations

from tfpd_exploration.h1_series_20260830.src.h1_activity_carrier_resampling_factorial_v1.evaluate import crossed_supports
from tfpd_exploration.h1_series_20260830.src.h1_activity_carrier_resampling_factorial_v1.plan import decide_factorial
from tfpd_exploration.h1_series_20260830.src.h1_support_resampled_postpool_v1.plan import DATE_ORDER


def _rows(ff, rr, rf, fr):
    return [
        {"outer_date": date, "factorial_scores": {"FF": a, "RR": b, "RF": c, "FR": d}}
        for date, a, b, c, d in zip(DATE_ORDER, ff, rr, rf, fr)
    ]


def test_crossed_supports_change_exactly_one_axis() -> None:
    fixed_activity, fixed_carrier = object(), object()
    random_activity, random_carrier = object(), object()
    result = crossed_supports((fixed_activity, fixed_carrier), (random_activity, random_carrier))
    assert result["LP-AR-CF"] == (random_activity, fixed_carrier)
    assert result["LP-AF-CR"] == (fixed_activity, random_carrier)


def test_factorial_recovers_activity_only_effect() -> None:
    rows = _rows([0.30] * 5, [0.32] * 5, [0.32] * 5, [0.30] * 5)
    result = decide_factorial(rows)
    assert result["classification"] == "ACTIVITY_DIVERSITY_DOMINANT"
    assert abs(result["summaries"]["activity_resampling_main_effect"]["mean"] - 0.02) < 1e-12
    assert abs(result["summaries"]["carrier_resampling_main_effect"]["mean"]) < 1e-12


def test_factorial_recovers_carrier_only_effect() -> None:
    rows = _rows([0.30] * 5, [0.32] * 5, [0.30] * 5, [0.32] * 5)
    result = decide_factorial(rows)
    assert result["classification"] == "CARRIER_DIVERSITY_DOMINANT"
    assert abs(result["summaries"]["carrier_resampling_main_effect"]["mean"] - 0.02) < 1e-12


def test_factorial_recovers_positive_interaction() -> None:
    rows = _rows([0.30] * 5, [0.32] * 5, [0.30] * 5, [0.30] * 5)
    result = decide_factorial(rows)
    assert result["classification"] == "MATCHED_CORESAMPLING_INTERACTION"
    assert abs(result["summaries"]["matched_interaction"]["mean"] - 0.02) < 1e-12
    assert result["authorizes_model_selection"] is False

