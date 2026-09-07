from __future__ import annotations

import copy

import pytest

from h1_date_lodo_activity_system_compare_v1.plan import DATE_ORDER, HC_PREDECESSOR_SHA256, HS_AUTHORITIES, comparison_decision, dry_plan


def _rows(hs_gain: float, hc_gain: float, system_effect: float):
    rows = []
    for date in DATE_ORDER:
        hs_static = 0.4
        hc_static = 0.5
        hs_growing = hs_static + hs_gain
        hc_growing = hs_growing + system_effect
        rows.append({
            "outer_date": date,
            "systems": {
                "H-S": {
                    "STATIC_SUPPORT": {"equal_recording_mean_r2": hs_static},
                    "CAUSAL_GROWING_CAP30": {"equal_recording_mean_r2": hs_growing},
                },
                "H-C": {
                    "STATIC_SUPPORT": {"equal_recording_mean_r2": hc_static},
                    "CAUSAL_GROWING_CAP30": {"equal_recording_mean_r2": hc_growing},
                },
            },
        })
    return rows


def test_hs_authority_is_exact_five_date_seed42_terminal_set():
    assert tuple(HS_AUTHORITIES) == DATE_ORDER
    assert len({row.checkpoint_sha256 for row in HS_AUTHORITIES.values()}) == 5
    assert all(len(row.checkpoint_sha256) == len(row.config_sha256) == 64 for row in HS_AUTHORITIES.values())
    assert all(row.checkpoint_relative.endswith("hs_epoch_049.ckpt") for row in HS_AUTHORITIES.values())


@pytest.mark.parametrize(("effect", "verdict"), [
    (0.02, "HC_ACTIVITY_SYNERGY_RETAIN_CARRIER_SYSTEM"),
    (0.0, "ACTIVITY_MEMORY_DOMINANT_NO_MATERIAL_CARRIER_SYNERGY"),
    (-0.02, "HS_GROWING_OUTPERFORMS_HC"),
])
def test_predeclared_descriptive_system_verdict(effect, verdict):
    result = comparison_decision(_rows(0.03, 0.04, effect))
    assert result["verdict"] == verdict
    assert result["positive_hs_growing_dates"] == 5
    assert result["formal_selection_claim"] is False


def test_comparison_rejects_date_order_or_arm_drift():
    rows = _rows(0.03, 0.04, 0.0)
    with pytest.raises(ValueError, match="five-date order"):
        comparison_decision(list(reversed(rows)))
    broken = copy.deepcopy(rows)
    broken[0]["systems"]["H-S"].pop("STATIC_SUPPORT")
    with pytest.raises(ValueError, match="static/growing"):
        comparison_decision(broken)


def test_dry_plan_is_inert_and_binds_predecessor():
    payload = dry_plan()
    assert payload["status"] == "DRY_NO_TARGET_NO_CHECKPOINT_LOAD_NO_CUDA_NO_WRITE"
    assert payload["hc_predecessor"]["sha256"] == HC_PREDECESSOR_SHA256
    assert len(payload["matrix"]) == 4
    assert payload["target_updates"] == 0
