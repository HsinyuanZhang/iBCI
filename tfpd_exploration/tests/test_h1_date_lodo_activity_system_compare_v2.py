from __future__ import annotations

import pytest

from h1_date_lodo_activity_system_compare_v2.plan import DATE_ORDER, V1_SHA256, corrected_decision, dry_plan


def _rows(*, hs_gain: float, hc_gain: float, final_gap: float):
    rows = []
    for date in DATE_ORDER:
        hs_static = 0.4
        hs_growing = hs_static + hs_gain
        hc_growing = hs_growing + final_gap
        hc_static = hc_growing - hc_gain
        rows.append({"outer_date": date, "systems": {
            "H-S": {"STATIC_SUPPORT": {"equal_recording_mean_r2": hs_static}, "CAUSAL_GROWING_CAP30": {"equal_recording_mean_r2": hs_growing}},
            "H-C": {"STATIC_SUPPORT": {"equal_recording_mean_r2": hc_static}, "CAUSAL_GROWING_CAP30": {"equal_recording_mean_r2": hc_growing}},
        }})
    return rows


def test_difference_in_differences_not_final_gap_drives_interaction():
    result = corrected_decision(_rows(hs_gain=0.05, hc_gain=0.05, final_gap=0.04))
    assert result["interaction_verdict"] == "NO_MATERIAL_CARRIER_BY_ACTIVITY_INTERACTION"
    assert result["recommended_frozen_system_basis"] == "H-C_CAUSAL_GROWING_RETAINS_HIGHER_FINAL_LEVEL"
    assert result["equal_date_mean"]["interaction_hc_gain_minus_hs_gain"] == pytest.approx(0.0)


def test_material_interaction_signs_are_separate_from_final_level():
    positive = corrected_decision(_rows(hs_gain=0.03, hc_gain=0.05, final_gap=0.04))
    assert positive["interaction_verdict"] == "POSITIVE_CARRIER_BY_ACTIVITY_INTERACTION"
    negative = corrected_decision(_rows(hs_gain=0.06, hc_gain=0.04, final_gap=0.04))
    assert negative["interaction_verdict"] == "ACTIVITY_GAIN_MATERIALLY_LARGER_WITHOUT_CARRIER"


def test_exact_date_bootstrap_is_bound_and_finite():
    result = corrected_decision(_rows(hs_gain=0.05, hc_gain=0.04, final_gap=0.03))
    for interval in result["exact_five_date_bootstrap_95"].values():
        assert len(interval) == 2 and interval[0] <= interval[1]


def test_dry_plan_binds_exact_v1_and_is_inert():
    payload = dry_plan()
    assert payload["v1"]["sha256"] == V1_SHA256
    assert payload["status"] == "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE"
    assert payload["target_updates"] == 0
