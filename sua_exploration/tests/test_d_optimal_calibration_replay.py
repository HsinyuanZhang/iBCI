"""Synthetic tests for B9 D-optimal calibration replay scaffolding."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mc_maze.decoder_attention_diagnostic import SEALED_FORMAL_TEST_SESSIONS, SealedSessionError
from mc_maze.d_optimal_calibration_design import (
    CANONICAL_DIRECTIONS_RAD,
    DEFAULT_BUDGET_LADDER,
    FROZEN_ALGORITHM_PARAMETERS,
    FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA,
    LeakageViolationError,
    assert_leakage_boundary,
    design_matrix_from_thetas,
    design_metrics,
    greedy_forward_d_optimal_indices,
    index_span,
    random_subset_indices,
    require_min_distinct_directions,
    select_calibration_trials,
    span_matched_random_indices,
    spans_match,
)
from mc_maze.d_optimal_calibration_replay import (
    build_phase_flip_session_payload,
    build_receipt,
    build_synthetic_session_payload,
    replay_session,
    validate_receipt,
)


def eight_direction_pool() -> np.ndarray:
    return np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)


def test_greedy_d_optimal_covers_eight_directions_at_m8_sanity() -> None:
    thetas = eight_direction_pool()
    selected = greedy_forward_d_optimal_indices(thetas, 8)
    metrics = design_metrics(design_matrix_from_thetas(thetas[selected]))
    assert metrics["design_rank"] == 3
    assert metrics["invertible"]
    assert len({int(i) for i in selected}) == 8


def test_greedy_d_optimal_m3_near_equiangular_sanity() -> None:
    thetas = eight_direction_pool()
    selected = greedy_forward_d_optimal_indices(thetas, 3)
    angles = sorted(float(thetas[i]) for i in selected)
    gaps = [angles[1] - angles[0], angles[2] - angles[1]]
    assert all(gap >= 0.6 * math.pi / 2.0 for gap in gaps)


def test_d_optimal_beats_random_det_sanity_check() -> None:
    """Implementation sanity: greedy D-optimal maximizes det(X'X) — not a primary gate."""
    thetas = eight_direction_pool()
    d_opt = greedy_forward_d_optimal_indices(thetas, 4)
    d_opt_det = design_metrics(design_matrix_from_thetas(thetas[d_opt]))["det_xtx"]
    random_dets = []
    for seed in range(200):
        random_idx = random_subset_indices(8, 4, seed)
        random_dets.append(
            design_metrics(design_matrix_from_thetas(thetas[random_idx]))["det_xtx"]
        )
    assert d_opt_det >= max(random_dets)


def test_nonstationary_d_opt_wins_det_loses_carrier_fidelity() -> None:
    """Primary endpoint is not tautological: better det need not imply better carrier."""
    payload = build_phase_flip_session_payload("sub-C_ses-CO-20151103", seed=1, units=4)
    result = replay_session(payload, budgets=[10])
    row = result["per_budget"]["M10"]
    det_gain = row["implementation_sanity_checks"]["det_xtx_gain_d_optimal_minus_chronological"]
    delta = row["primary_endpoint"]["paired_delta_d_optimal_minus_chronological"]
    assert det_gain > 0.0
    assert delta is not None and delta < 0.0


def test_stationary_d_opt_wins_det_and_does_not_lose_fidelity() -> None:
    payload = build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42, nonstationary=False)
    result = replay_session(payload, budgets=[10])
    row = result["per_budget"]["M10"]
    det_gain = row["implementation_sanity_checks"]["det_xtx_gain_d_optimal_minus_chronological"]
    delta = row["primary_endpoint"]["paired_delta_d_optimal_minus_chronological"]
    assert det_gain > 0.0
    assert delta is not None and delta >= -1e-9


def test_span_matched_null_matches_d_optimal_span() -> None:
    thetas = eight_direction_pool()
    d_opt = greedy_forward_d_optimal_indices(thetas, 4)
    target_span = index_span(d_opt)
    matched = span_matched_random_indices(8, 4, target_span, seed=42)
    assert spans_match(index_span(matched), target_span)


def test_span_mismatch_fails_spans_match_helper() -> None:
    assert not spans_match(5, 20)


def test_aggregator_rejects_span_mismatch_in_receipt() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    span_rows = receipt["session_results"][0]["per_budget"]["M10"]["random_m_span_matched_k"]["seeds"]
    span_rows[0]["selection"]["temporal_coverage"]["index_span"] = 0
    with pytest.raises(ValueError, match="span-matched null outside tolerance"):
        validate_receipt(receipt)


def test_degenerate_directions_refused() -> None:
    with pytest.raises(Exception):
        require_min_distinct_directions([0, 0], min_distinct=3, session_name="degenerate")


def test_leakage_assertion_fires_on_rate_data() -> None:
    trials = [{"target_dir": 0.1, "trial_rates": [1.0, 2.0]}]
    with pytest.raises(LeakageViolationError, match="forbidden keys"):
        assert_leakage_boundary(trials)
    thetas = eight_direction_pool()
    with pytest.raises(LeakageViolationError):
        select_calibration_trials(
            thetas,
            3,
            "d_optimal_prefix_k",
            candidate_trials_for_leakage_check=trials,
        )


def test_sealed_session_guard_raises() -> None:
    with pytest.raises(SealedSessionError):
        build_synthetic_session_payload(SEALED_FORMAL_TEST_SESSIONS[0])


def test_receipt_records_temporal_coverage() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    coverage = receipt["session_results"][0]["per_budget"]["M10"]["coverage_audit"]
    assert "chronological_temporal_coverage" in coverage
    assert "d_optimal_temporal_coverage" in coverage
    assert coverage["d_optimal_temporal_coverage"]["selected_indices"] is not None


def test_receipt_determinism_under_fixed_seed() -> None:
    first = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    second = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_aggregator_rejects_incomplete_ladder() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["budget_ladder"] = [10, 15]
    with pytest.raises(ValueError, match="budget ladder"):
        validate_receipt(receipt)


def test_aggregator_rejects_sealed_contamination() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["sealed_test_sessions_opened"] = True
    with pytest.raises(ValueError, match="sealed"):
        validate_receipt(receipt)


def test_aggregator_rejects_algorithm_drift() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["algorithm"]["algorithm_id"] = "other"
    with pytest.raises(ValueError, match="algorithm parameter drift"):
        validate_receipt(receipt)


def test_aggregator_rejects_missing_leakage_assertion() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    receipt["leakage_assertion"]["selector_uses_label_geometry_only"] = False
    with pytest.raises(ValueError, match="leakage"):
        validate_receipt(receipt)


def test_valid_receipt_passes_aggregator() -> None:
    receipt = build_receipt(
        [build_synthetic_session_payload("sub-C_ses-CO-20151103", seed=42)],
        seed=42,
    )
    summary = validate_receipt(receipt)
    assert summary["status"] == "valid"
    assert summary["budgets"] == list(DEFAULT_BUDGET_LADDER)
    assert summary["primary_gate_summary"]["minimum_median_session_delta"] == FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA


def test_frozen_algorithm_parameters_match_protocol() -> None:
    assert FROZEN_ALGORITHM_PARAMETERS["algorithm_id"] == "greedy_forward_d_optimal_v1"
    assert FROZEN_ALGORITHM_PARAMETERS["budget_ladder"] == list(DEFAULT_BUDGET_LADDER)
    assert "random_m_span_matched_k" in FROZEN_ALGORITHM_PARAMETERS["arms"]
    assert FROZEN_ALGORITHM_PARAMETERS["primary_gate"]["minimum_median_session_delta"] == 0.03


@pytest.mark.real_data
@pytest.mark.skip(reason="real NWB replay requires explicit authorization and sealed-checkpoint outputs")
def test_real_data_nwb_replay_not_run_by_default() -> None:
    raise AssertionError("real_data marker must stay skipped in default CI")
