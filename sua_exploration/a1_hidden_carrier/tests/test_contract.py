from __future__ import annotations

from copy import deepcopy

import pytest

from a1_hidden_carrier.a2_anchors import verify_sealed_a2_reuse
from a1_hidden_carrier.contract import (
    ATTACHMENT_CONTROL,
    FRESH_TRAINING_FAMILIES,
    LOGICAL_CELLS,
    PILOT_SEED,
    SESSIONS,
    aggregate_pilot,
    synthetic_fresh_score_receipt,
    validate_fresh_h_t4_score_receipt,
)


@pytest.fixture(scope="module")
def anchor():
    return verify_sealed_a2_reuse(seed=PILOT_SEED, verify_checkpoint_bytes=False)


def test_minimal_topology() -> None:
    assert LOGICAL_CELLS == ("W/Z4", "W/T4", "H/Z4", "H/T4")
    assert FRESH_TRAINING_FAMILIES == ("H/T4",)
    assert ATTACHMENT_CONTROL == "H/TS4"
    assert len(SESSIONS) == 6


def test_fresh_score_validator_rejects_ts4_retraining() -> None:
    receipt = synthetic_fresh_score_receipt(delta=0.04)
    validate_fresh_h_t4_score_receipt(receipt)
    invalid = deepcopy(receipt)
    invalid["attachment_control"]["training_run_created"] = True
    with pytest.raises(ValueError, match="TS4"):
        validate_fresh_h_t4_score_receipt(invalid)


def test_pass_and_stop_gate(anchor) -> None:
    w_t4 = anchor["sealed_w_cells"]["source_t4"]["per_session_mean_r2"]
    passed = synthetic_fresh_score_receipt(delta=0.0)
    passed["aligned"]["per_session_mean_r2"] = {
        session: float(w_t4[session]) + 0.04 for session in SESSIONS
    }
    result = aggregate_pilot(
        a2_anchor=anchor, fresh_score_receipt=passed,
        fresh_score_receipt_sha256="a" * 64, preflight_sha256="b" * 64,
    )
    assert result["passes"] is True
    assert result["overall"]["mean_primary_interaction"] == pytest.approx(0.04)
    assert all(row["H/Z4"] == row["W/Z4"] for row in result["per_session"])

    stopped = deepcopy(passed)
    stopped["aligned"]["per_session_mean_r2"] = dict(w_t4)
    result = aggregate_pilot(
        a2_anchor=anchor, fresh_score_receipt=stopped,
        fresh_score_receipt_sha256="a" * 64, preflight_sha256="b" * 64,
    )
    assert result["passes"] is False
    assert result["status"] == "PILOT_ROUTING_STOP"


def test_anchor_alias_binds_exact_w_z4_receipt(anchor) -> None:
    alias = anchor["h_z4_structural_alias"]
    z4 = anchor["sealed_w_cells"]["source_z4"]
    assert alias["separate_training_run"] is False
    assert alias["separate_scoring_run"] is False
    assert alias["carrier_port"] == "exact_zero"
    assert alias["w_z4_within_receipt_sha256"] == z4["within_receipt_sha256"]
