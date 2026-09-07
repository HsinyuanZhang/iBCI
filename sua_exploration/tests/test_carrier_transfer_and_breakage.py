from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_maze import carrier_transfer_and_breakage as ctb
from scripts.aggregate_carrier_transfer_and_breakage import (
    aggregate_breakage,
    aggregate_transfer,
)


def _load_screen_module():
    script = ROOT / "scripts/carrier_transfer_and_breakage_screen.py"
    spec = importlib.util.spec_from_file_location("carrier_transfer_and_breakage_screen", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def screen_module():
    return _load_screen_module()


def _synthetic_fixture(
    *,
    n_units: int = 8,
    carrier_sensitivity: float = 0.25,
    control_breakage_slope: float = 0.40,
    carrier_breakage_slope: float = 0.10,
) -> ctb.SyntheticSessionFixture:
    rng = np.random.RandomState(0)
    neural = rng.randn(12, n_units).astype(np.float32)
    calib = rng.randn(3, 10, n_units).astype(np.float32)
    carrier = rng.randn(n_units, 4).astype(np.float32)
    return ctb.SyntheticSessionFixture(
        session_name="synthetic_session",
        neural=neural,
        calib_trials=calib,
        carrier=carrier,
        carrier_sensitivity=carrier_sensitivity,
        control_breakage_slope=control_breakage_slope,
        carrier_breakage_slope=carrier_breakage_slope,
    )


def test_exact_null_breakage_zero_matches_unmodified_forward():
    fixture = _synthetic_fixture()
    baseline_carrier = ctb.synthetic_forward_score(fixture, arm="carrier", breakage_level=0.0)
    baseline_control = ctb.synthetic_forward_score(fixture, arm="control", breakage_level=0.0)
    keep = ctb.unit_dropout_indices(fixture.n_units, dropout_p=0.0, seed=123)
    dropped_carrier = ctb.synthetic_forward_score(
        fixture, arm="carrier", breakage_level=0.0, unit_indices=keep
    )
    dropped_control = ctb.synthetic_forward_score(
        fixture, arm="control", breakage_level=0.0, unit_indices=keep
    )
    assert dropped_carrier == baseline_carrier
    assert dropped_control == baseline_control


def test_transferred_carrier_substitutes_donor_rows_and_records_padding():
    donor = np.arange(16, dtype=np.float32).reshape(4, 4)
    transferred, receipt = ctb.apply_row_matching_rule(donor, recipient_n_units=6)
    assert receipt.matched_rows == 4
    assert receipt.donor_rows_dropped == 0
    assert receipt.recipient_rows_zero_filled == 2
    assert np.allclose(transferred[:4], donor)
    assert np.allclose(transferred[4:], 0.0)


def test_raw_shape_paste_without_rule_raises():
    donor = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot paste donor carrier"):
        ctb.require_matching_shapes_or_apply_rule(
            donor, recipient_n_units=6, rule="forbidden_rule"
        )


def test_inadmissible_transfer_pair_raises():
    with pytest.raises(ValueError, match="not on the pre-registered admissible list"):
        ctb.assert_transfer_pair_admissible(
            "sub-C_ses-CO-20151103",
            "sub-C_ses-CO-20151103",
        )


def test_unit_dropout_consistency_enforced():
    fixture = _synthetic_fixture(n_units=6)
    keep = np.array([0, 2, 4], dtype=np.int64)
    neural, side, calib = ctb.apply_unit_dropout_consistent(
        neural=fixture.neural,
        side_features=fixture.carrier,
        calib_trials=fixture.calib_trials,
        unit_indices=keep,
    )
    assert neural.shape[1] == calib.shape[-1] == side.shape[0] == 3
    bad_calib = fixture.calib_trials[..., :4]
    with pytest.raises(ValueError, match="disagree"):
        ctb.apply_unit_dropout_consistent(
            neural=fixture.neural,
            side_features=fixture.carrier,
            calib_trials=bad_calib,
            unit_indices=keep,
        )


def test_planted_positive_interaction_when_control_degrades_faster():
    fixture = _synthetic_fixture(
        control_breakage_slope=0.40,
        carrier_breakage_slope=0.10,
    )
    carrier_scores = [
        ctb.synthetic_forward_score(fixture, arm="carrier", breakage_level=level)
        for level in ctb.BREAKAGE_LEVELS
    ]
    control_scores = [
        ctb.synthetic_forward_score(fixture, arm="control", breakage_level=level)
        for level in ctb.BREAKAGE_LEVELS
    ]
    interaction = ctb.breakage_interaction_statistic(carrier_scores, control_scores)
    assert interaction > 0.0


def test_planted_zero_interaction_when_both_degrade_equally():
    fixture = _synthetic_fixture(
        control_breakage_slope=0.20,
        carrier_breakage_slope=0.20,
    )
    carrier_scores = [
        ctb.synthetic_forward_score(fixture, arm="carrier", breakage_level=level)
        for level in ctb.BREAKAGE_LEVELS
    ]
    control_scores = [
        ctb.synthetic_forward_score(fixture, arm="control", breakage_level=level)
        for level in ctb.BREAKAGE_LEVELS
    ]
    interaction = ctb.breakage_interaction_statistic(carrier_scores, control_scores)
    assert interaction == pytest.approx(0.0)


def test_receipt_determinism_excluding_timestamps(tmp_path: Path):
    payload_a = {
        "schema_version": ctb.SCHEMA_VERSION,
        "created_at": "2026-08-12T00:00:00+08:00",
        "scores": {"carrier": 0.5},
        "seed": ctb.derived_seed(family="x", base_seed=ctb.BASE_RANDOMIZATION_SEED, session="s"),
    }
    payload_b = {
        **payload_a,
        "created_at": "2026-08-12T12:34:56+08:00",
    }
    assert ctb.canonical_json_bytes(ctb.receipt_without_timestamps(payload_a)) == ctb.canonical_json_bytes(
        ctb.receipt_without_timestamps(payload_b)
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload_a, sort_keys=True), encoding="utf-8")
    reread = json.loads(path.read_text(encoding="utf-8"))
    assert ctb.canonical_json_bytes(ctb.receipt_without_timestamps(reread)) == ctb.canonical_json_bytes(
        ctb.receipt_without_timestamps(payload_a)
    )


def test_sealed_session_guard_raises():
    for session in ctb.SEALED_FORMAL_TEST_SESSIONS:
        with pytest.raises(ValueError, match="refusing sealed formal-test session"):
            ctb.assert_session_allowed(session)


def _transfer_receipt(
    *,
    donor: str,
    recipient: str,
    matched_rows: int | None = None,
    donor_rows_dropped: int = 0,
    recipient_rows_zero_filled: int | None = None,
    pool_size: int = 30,
) -> dict:
    n_donor = ctb.SESSION_UNIT_COUNTS[donor]
    n_recipient = ctb.SESSION_UNIT_COUNTS[recipient]
    if matched_rows is None:
        matched_rows = min(n_donor, n_recipient)
    if recipient_rows_zero_filled is None:
        recipient_rows_zero_filled = max(0, n_recipient - matched_rows)
    pair_subset = ctb.transfer_pair_subset(donor, recipient)
    scores = {
        "own_carrier": 0.40,
        "own_truncated": 0.35,
        "transferred_carrier": 0.20,
        "zero_carrier": 0.10,
    }
    if pair_subset == "primary":
        scores["own_truncated"] = scores["own_carrier"]
        recipient_rows_zero_filled = 0
    deltas = ctb.compute_transfer_deltas(
        {
            "own_carrier": scores["own_carrier"],
            "own_truncated": scores["own_truncated"],
            "transferred_carrier": scores["transferred_carrier"],
        }
    )
    return {
        "schema_version": ctb.SCHEMA_VERSION,
        "protocol_id": ctb.PROTOCOL_ID,
        "experiment": "A5_transferred_and_aged_carrier",
        "checkpoint_sha256": "abc",
        "donor_session": donor,
        "recipient_session": recipient,
        "pair_subset": pair_subset,
        "row_matching_rule": ctb.ROW_MATCHING_RULE,
        "matched_rows": matched_rows,
        "donor_rows_dropped": donor_rows_dropped,
        "recipient_rows_zero_filled": recipient_rows_zero_filled,
        "donor_unit_count": n_donor,
        "recipient_unit_count": n_recipient,
        "N_donor": n_donor,
        "N_recipient": n_recipient,
        "zero_fill_fraction": ctb.zero_fill_fraction(
            recipient_rows_zero_filled=recipient_rows_zero_filled,
            recipient_rows=n_recipient,
        ),
        "zero_fill_pattern_digest": ctb.ZeroFillPattern(
            recipient_rows=n_recipient,
            matched_rows=matched_rows,
        ).digest(),
        "per_session_scores": scores,
        "deltas": deltas.as_dict(),
        "primary_statistic": deltas.transferred_minus_own_truncated,
        "secondary_statistic": deltas.transferred_minus_own_full,
        "nuisance_statistic": deltas.own_truncated_minus_own_full,
        "query_window": {
            "pool_size": pool_size,
            "calibration_n": pool_size,
            "evaluation_start_trial": pool_size,
        },
        "sealed_test_sessions_opened": False,
    }


def _breakage_receipt(
    *,
    session: str,
    mode: str = "unit_dropout",
    levels: list[float] | None = None,
) -> dict:
    levels = list(levels or ctb.BREAKAGE_LEVELS)
    ladder = [{"breakage_level": level, "seed": int(level * 100), "kept_units": 10} for level in levels]
    carrier_scores = [0.5 - 0.05 * i for i in range(len(levels))]
    control_scores = [0.45 - 0.10 * i for i in range(len(levels))]
    return {
        "schema_version": ctb.SCHEMA_VERSION,
        "protocol_id": ctb.PROTOCOL_ID,
        "experiment": "A3_correspondence_breakage_dose_response",
        "checkpoint_sha256": "abc",
        "session": session,
        "breakage_mode": mode,
        "breakage_ladder": ladder,
        "carrier_scores": carrier_scores,
        "control_scores": control_scores,
        "interaction_statistic": ctb.breakage_interaction_statistic(carrier_scores, control_scores, levels),
        "query_window": {
            "pool_size": 30,
            "calibration_n": 30,
            "evaluation_start_trial": 30,
        },
        "sealed_test_sessions_opened": False,
    }


def test_aggregator_rejects_incomplete_ladder(tmp_path: Path):
    receipt = _breakage_receipt(session=ctb.VALIDATION_SESSIONS[0])
    receipt["carrier_scores"] = receipt["carrier_scores"][:-1]
    path = tmp_path / "breakage.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="carrier/control ladder length mismatch"):
        aggregate_breakage([path], session=ctb.VALIDATION_SESSIONS[0], breakage_mode="unit_dropout")


def test_aggregator_rejects_query_window_mismatch(tmp_path: Path):
    r1 = _transfer_receipt(
        donor=ctb.ADMISSIBLE_TRANSFER_PAIRS[0][0],
        recipient=ctb.ADMISSIBLE_TRANSFER_PAIRS[0][1],
        pool_size=30,
    )
    r2 = _transfer_receipt(
        donor=ctb.ADMISSIBLE_TRANSFER_PAIRS[1][0],
        recipient=ctb.ADMISSIBLE_TRANSFER_PAIRS[1][1],
        pool_size=20,
    )
    p1 = tmp_path / "t1.json"
    p2 = tmp_path / "t2.json"
    p1.write_text(json.dumps(r1), encoding="utf-8")
    p2.write_text(json.dumps(r2), encoding="utf-8")
    with pytest.raises(ValueError, match="query window"):
        aggregate_transfer([p1, p2], expected_rule=ctb.ROW_MATCHING_RULE)


def test_aggregator_rejects_sealed_session_contamination(tmp_path: Path):
    sealed = next(iter(ctb.SEALED_FORMAL_TEST_SESSIONS))
    receipt = _transfer_receipt(
        donor=ctb.PRIMARY_TRANSFER_PAIRS[0][0],
        recipient=ctb.PRIMARY_TRANSFER_PAIRS[0][1],
    )
    receipt["donor_session"] = sealed
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed session contamination"):
        aggregate_transfer([path], expected_rule=ctb.ROW_MATCHING_RULE)


def test_aggregator_rejects_row_matching_rule_drift(tmp_path: Path):
    receipt = _transfer_receipt(
        donor=ctb.ADMISSIBLE_TRANSFER_PAIRS[0][0],
        recipient=ctb.ADMISSIBLE_TRANSFER_PAIRS[0][1],
    )
    receipt["row_matching_rule"] = "other_rule"
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="row-matching rule"):
        aggregate_transfer([path], expected_rule=ctb.ROW_MATCHING_RULE)


def test_aggregator_rejects_non_admissible_pair(tmp_path: Path):
    receipt = _transfer_receipt(
        donor=ctb.VALIDATION_SESSIONS[0],
        recipient=ctb.VALIDATION_SESSIONS[0],
    )
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not admissible"):
        aggregate_transfer([path], expected_rule=ctb.ROW_MATCHING_RULE)


def test_aggregator_accepts_valid_transfer_and_breakage(tmp_path: Path):
    donor, recipient = ctb.ADMISSIBLE_TRANSFER_PAIRS[0]
    transfer_path = tmp_path / "transfer.json"
    transfer_path.write_text(
        json.dumps(_transfer_receipt(donor=donor, recipient=recipient)),
        encoding="utf-8",
    )
    transfer_agg = aggregate_transfer([transfer_path], expected_rule=ctb.ROW_MATCHING_RULE)
    assert transfer_agg["pairs"][0]["donor_session"] == donor

    breakage_path = tmp_path / "breakage.json"
    session = ctb.VALIDATION_SESSIONS[0]
    breakage_path.write_text(json.dumps(_breakage_receipt(session=session)), encoding="utf-8")
    breakage_agg = aggregate_breakage(
        [breakage_path],
        session=session,
        breakage_mode="unit_dropout",
    )
    assert breakage_agg["session"] == session


def test_transferred_path_uses_donor_rows_in_synthetic_score():
    fixture = _synthetic_fixture(n_units=6)
    donor = np.full((4, 4), 3.0, dtype=np.float32)
    transferred, receipt = ctb.apply_row_matching_rule(donor, recipient_n_units=fixture.n_units)
    own_score = ctb.synthetic_transfer_arm_score(
        fixture, fixture.carrier, zero_fill_fraction=0.0
    )
    transfer_score = ctb.synthetic_transfer_arm_score(
        fixture, transferred, zero_fill_fraction=receipt.zero_fill_fraction
    )
    assert receipt.matched_rows == 4
    assert receipt.recipient_rows_zero_filled == 2
    assert transfer_score != own_score


def test_primary_and_secondary_pair_counts_from_manifest():
    assert len(ctb.ADMISSIBLE_TRANSFER_PAIRS) == 30
    assert len(ctb.PRIMARY_TRANSFER_PAIRS) == 15
    assert len(ctb.SECONDARY_TRANSFER_PAIRS) == 15
    assert set(ctb.PRIMARY_TRANSFER_PAIRS) & set(ctb.SECONDARY_TRANSFER_PAIRS) == set()


def test_zero_fill_only_fixture_primary_delta_near_zero_secondary_strongly_negative():
    fixture = _synthetic_fixture(n_units=6)
    own = fixture.carrier.copy()
    donor = own[:4].copy()
    bundle = ctb.build_transfer_arm_carriers(own, donor, recipient_n_units=6)
    assert bundle.zero_fill_fraction > 0.0
    own_full = ctb.synthetic_transfer_arm_score(fixture, bundle.own_full, zero_fill_fraction=0.0)
    own_trunc = ctb.synthetic_transfer_arm_score(
        fixture, bundle.own_truncated, zero_fill_fraction=bundle.zero_fill_fraction
    )
    transferred = ctb.synthetic_transfer_arm_score(
        fixture, bundle.transferred, zero_fill_fraction=bundle.zero_fill_fraction
    )
    primary = transferred - own_trunc
    secondary = transferred - own_full
    assert primary == pytest.approx(0.0, abs=1e-9)
    assert secondary < -0.1


def test_primary_subset_wrong_donor_both_deltas_agree():
    fixture = _synthetic_fixture(n_units=4)
    own = np.arange(16, dtype=np.float32).reshape(4, 4)
    donor = own + 5.0
    bundle = ctb.build_transfer_arm_carriers(own, donor, recipient_n_units=4)
    assert bundle.zero_fill_fraction == 0.0
    own_full = ctb.synthetic_transfer_arm_score(fixture, bundle.own_full, zero_fill_fraction=0.0)
    own_trunc = ctb.synthetic_transfer_arm_score(
        fixture, bundle.own_truncated, zero_fill_fraction=0.0
    )
    transferred = ctb.synthetic_transfer_arm_score(
        fixture, bundle.transferred, zero_fill_fraction=0.0
    )
    assert own_trunc == pytest.approx(own_full)
    assert transferred - own_trunc == pytest.approx(transferred - own_full)


def test_shared_zero_fill_pattern_is_byte_identical_between_arms():
    own = np.arange(24, dtype=np.float32).reshape(6, 4)
    donor = np.arange(24, 32, dtype=np.float32).reshape(2, 4)
    bundle = ctb.build_transfer_arm_carriers(own, donor, recipient_n_units=6)
    tail_own = bundle.own_truncated[bundle.pattern.matched_rows :]
    tail_trans = bundle.transferred[bundle.pattern.matched_rows :]
    assert np.array_equal(tail_own, tail_trans)


def test_shared_zero_fill_pattern_divergence_raises():
    pattern = ctb.ZeroFillPattern(recipient_rows=6, matched_rows=4)
    own_truncated = ctb.apply_zero_fill_pattern(np.ones((6, 4), np.float32), pattern)
    transferred = ctb.apply_zero_fill_pattern(np.ones((6, 4), np.float32), pattern)
    transferred[5, 0] = 1.0
    with pytest.raises(ValueError, match="zero-fill tail bytes differ"):
        ctb.assert_shared_zero_fill_pattern(own_truncated, transferred, pattern)


def test_aggregator_refuses_to_pool_primary_and_secondary_subsets(tmp_path: Path):
    primary = ctb.PRIMARY_TRANSFER_PAIRS[0]
    secondary = ctb.SECONDARY_TRANSFER_PAIRS[0]
    p1 = tmp_path / "primary.json"
    p2 = tmp_path / "secondary.json"
    p1.write_text(json.dumps(_transfer_receipt(donor=primary[0], recipient=primary[1])), encoding="utf-8")
    p2.write_text(json.dumps(_transfer_receipt(donor=secondary[0], recipient=secondary[1])), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to pool primary and secondary"):
        aggregate_transfer([p1, p2], expected_rule=ctb.ROW_MATCHING_RULE)


def test_aggregator_subset_reports_zero_fill_with_deltas(tmp_path: Path):
    donor, recipient = ctb.SECONDARY_TRANSFER_PAIRS[0]
    path = tmp_path / "secondary.json"
    path.write_text(json.dumps(_transfer_receipt(donor=donor, recipient=recipient)), encoding="utf-8")
    agg = aggregate_transfer([path], expected_rule=ctb.ROW_MATCHING_RULE, subset="secondary")
    pair = agg["secondary_subset"]["per_pair"][0]
    assert "zero_fill_fraction" in pair
    assert pair["deltas_with_confound"]["transferred_minus_own_truncated"]["zero_fill_fraction"] > 0.0
