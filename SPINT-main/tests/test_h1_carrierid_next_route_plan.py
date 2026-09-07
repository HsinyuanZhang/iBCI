"""No-data contracts for the post-D-S4e/D-Q4e H1 route plan."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import h1_carrierid_next_route_plan as plan


def test_parameter_accounting_is_exact_for_ci32_ci64_and_h64():
    ci32 = plan.topology_parameters(hidden_dim=32, interface_dim=32)
    ci64 = plan.topology_parameters(hidden_dim=32, interface_dim=64)
    h64 = plan.topology_parameters(hidden_dim=64, interface_dim=64)

    assert (ci32["pre_pool"], ci32["post_pool"], ci32["identity_encoder"], ci32["whole_model"]) == (
        32_800, 25_340, 58_140, 10_947_836
    )
    assert (ci64["pre_pool"], ci64["post_pool"], ci64["identity_encoder"], ci64["whole_model"]) == (
        32_800, 27_548, 60_348, 10_950_044
    )
    assert (h64["pre_pool"], h64["post_pool"], h64["identity_encoder"], h64["whole_model"]) == (
        65_600, 54_076, 119_676, 11_009_372
    )
    assert ci64["identity_encoder"] - ci32["identity_encoder"] == 2_208
    assert h64["identity_encoder"] - ci32["identity_encoder"] == 61_536
    assert ci32["carrier_entry_weights"] == 128
    assert ci64["carrier_entry_weights"] == h64["carrier_entry_weights"] == 256


@pytest.mark.parametrize("hidden_dim,interface_dim", [(0, 32), (32, 0), (-1, 16)])
def test_invalid_topology_widths_fail_closed(hidden_dim, interface_dim):
    with pytest.raises(ValueError, match="positive"):
        plan.topology_parameters(hidden_dim=hidden_dim, interface_dim=interface_dim)


def test_routes_have_unique_canonical_names_and_consumer_does_not_jump_to_h64():
    payload = plan.build_static_plan()
    routes = payload["d_s4e_d_q4e_hypothesis_routes"]

    assert payload["schema"] == plan.SCHEMA
    assert payload["naming"]["retired_ambiguous_label"] == "C7"
    assert set(routes) == {
        "estimator-limited evidence",
        "consumer-limited evidence",
        "unresolved",
        "invalid_exposure_repair__q4e_minus_s4e_not_interpretable",
    }
    assert routes["estimator-limited evidence"]["next_preparation"] == "H1-EST4-SLODO"
    assert routes["consumer-limited evidence"]["next_preparation"] == "H1-CI64-SLODO"
    assert routes["unresolved"]["next_preparation"] == "H1-DIST-XDATE-REPL"
    assert routes["invalid_exposure_repair__q4e_minus_s4e_not_interpretable"]["next_preparation"] == "H1-DIST-REPAIR-ONLY"
    assert "H64 remains prohibited" in routes["consumer-limited evidence"]["selection_boundary"]


def test_source_only_gate_requires_new_lodo_implementation_and_all_content_controls():
    payload = plan.build_static_plan()
    prerequisite = payload["source_only_selection_prerequisite"]
    gate = payload["h64_escalation_gate"]

    assert prerequisite["status"] == "MISSING_IMPLEMENTATION_FAIL_CLOSED"
    assert prerequisite["source_dates"] == ["19250108", "19250113", "19250115", "19250119", "19250120"]
    assert "target score selection" in prerequisite["forbidden"]
    assert payload["controls_required_per_new_topology"] == ["full", "C0", "LS", "RS"]
    assert gate["run_first"] == ["H1-CI32", "H1-CI64"]
    assert len(gate["all_required"]) == 3
    assert gate["if_fail"] == "STOP_CONSUMER_WIDTH_ROUTE_NO_H64"


def test_plan_is_data_free_and_has_no_launch_or_model_imports():
    source = Path(plan.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "src.data" not in source
    assert "subprocess" not in source
    assert "DataModule" not in source
    scope = plan.build_static_plan()["scope"]
    assert scope == {
        "source_nwb_opened": 0,
        "target_nwb_opened": 0,
        "minival_or_formal_or_evalai_opened": False,
        "cuda_constructed_or_launched": False,
        "trainer_constructed": False,
        "checkpoint_created_or_selected": False,
    }
