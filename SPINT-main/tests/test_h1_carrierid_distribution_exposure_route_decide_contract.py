"""Receipt-only contracts for H1 D-S4e/D-Q4e route disposition."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from scripts import h1_carrierid_distribution_exposure_route_decide as route


def _terminal(*, decision: str, s4: float = 0.50, q4: float = 0.55) -> dict[str, object]:
    delta = q4 - s4
    return {
        "schema": route.TERMINAL_SCHEMA,
        "status": route.TERMINAL_STATUS,
        "claim_boundary": route.CLAIM_BOUNDARY,
        "metrics": {
            "d_s4e": {"pooled_r2": s4},
            "d_q4e": {"pooled_r2": q4},
            "d_q4e_minus_d_s4e": {"pooled_r2": delta},
        },
        "frozen_interpretation": {"decision": decision, "q4e_minus_s4e_pooled_r2": delta},
    }


@pytest.mark.parametrize(
    ("decision", "next_preparation"),
    [
        ("estimator-limited evidence", "H1-EST4-SLODO"),
        ("consumer-limited evidence", "H1-CI64-SLODO"),
        ("unresolved", "H1-DIST-XDATE-REPL"),
        ("invalid_exposure_repair__q4e_minus_s4e_not_interpretable", "H1-DIST-REPAIR-ONLY"),
    ],
)
def test_route_disposition_uses_exact_static_mapping_without_model_selection(decision, next_preparation):
    output = route.route_from_terminal_evaluation(_terminal(decision=decision))
    assert output["decision"] == decision
    assert output["next_preparation"] == next_preparation


def test_route_disposition_rejects_metric_arithmetic_or_claim_boundary_drift():
    bad = _terminal(decision="consumer-limited evidence")
    bad["metrics"]["d_q4e_minus_d_s4e"]["pooled_r2"] = 0.0  # type: ignore[index]
    with pytest.raises(NormalizedV2ContractError, match="arithmetic drift"):
        route.route_from_terminal_evaluation(bad)
    bad = _terminal(decision="consumer-limited evidence")
    bad["claim_boundary"] = "paper-main"
    with pytest.raises(NormalizedV2ContractError, match="leakage-only boundary"):
        route.route_from_terminal_evaluation(bad)


def test_route_disposition_tool_is_receipt_only_and_never_imports_target_or_torch():
    source = Path(route.__file__).read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "import torch" not in source
    assert "DataModule" not in source
    assert "subprocess" not in source
    assert "gpu_route_authorized\": False" in source
