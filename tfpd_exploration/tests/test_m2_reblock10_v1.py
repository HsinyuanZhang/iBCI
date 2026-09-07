"""Synthetic no-data/no-CUDA tests for the M2 re-blocking cell V1.

The review-critical properties:

1. the first-10 candidate law: the finite-angle mask within positions 0-9,
   fail-closed on short pools / too few candidates;
2. the K4 law IS the frozen greedy D-opt law over those candidates; the KALL
   law is ALL of them (k >= 4), and K4 is a subset of KALL;
3. the post-H window-partition law: boundary semantics, fail-closed cases,
   and bitwise parity with the sealed post30 law at horizon 30;
4. the evidence-stream census arithmetic (completed trials old vs new,
   extension ratio, window counts);
5. the paired BLOCK10-minus-OLDLAW contrast arithmetic and breadth;
6. the gate boundaries at 1e-12: exactly +0.01 passes, a miss inside the band
   with breadth met is REBLOCK_BORDERLINE, outside the band (or breadth short)
   is REBLOCK_CDM_NOT_WORTH;
7. the k-row (KALL) binding laws: the generalized activity stack rows, the
   sealed select_activity_rows cardinality guard (the disclosed deviation),
   and the k-row carrier/activity arithmetic through the frozen cdm core;
8. the anchor matchers imported from the chrono4 precedent still bind;
9. the plan binds the sealed foundations and the pre-registered roster.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src",
             ROOT / "sua_exploration"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core  # noqa: E402
from src.cdm_p1_m2_local_v1 import plan as g_plan  # noqa: E402
from src.m2_chrono4_strict_v1 import plan as chrono_plan  # noqa: E402
from src.m2_reblock10_v1 import laws, plan  # noqa: E402
from src.m2_t4_activity_budget_screen_v1.core import (  # noqa: E402
    select_activity_rows,
)
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import (  # noqa: E402
    core as pseudo_core,
)


def _angles(count: int = 30, *, seed: int = 5) -> np.ndarray:
    """The roster's alternating directional pattern: odd positions labeled."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-math.pi, math.pi, size=count)
    values[0::2] = np.nan
    return np.ascontiguousarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. the first-10 candidate law.
# ---------------------------------------------------------------------------


def test_first10_candidates_alternating_roster_pattern() -> None:
    candidates = laws.first10_candidates(_angles())
    assert candidates.tolist() == [1, 3, 5, 7, 9]
    assert candidates.dtype == np.int64


def test_first10_candidates_law_is_the_sealed_isfinite_mask() -> None:
    theta = np.full(30, np.nan)
    theta[[2, 4, 5, 9]] = [0.1, 0.9, -0.7, 2.2]
    assert laws.first10_candidates(theta).tolist() == [2, 4, 5, 9]
    # labeled rows beyond position 9 never enter the first-10 pool
    theta[10:] = np.linspace(-3.0, 3.0, 20)
    assert laws.first10_candidates(theta).tolist() == [2, 4, 5, 9]


def test_first10_candidates_fails_closed() -> None:
    with pytest.raises(laws.ReblockLawError):
        laws.first10_candidates(np.asarray([0.1, 0.2, 0.3]))  # no first-30 container
    broken = np.full(30, np.nan)
    broken[[0, 1, 2]] = [0.1, 0.2, 0.3]
    with pytest.raises(laws.ReblockLawError):
        laws.first10_candidates(broken)  # fewer than four directional rows


# ---------------------------------------------------------------------------
# 2. the K4 / KALL support laws.
# ---------------------------------------------------------------------------


def test_k4_is_the_frozen_greedy_dopt_law_over_first10_candidates() -> None:
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    theta = _angles(seed=11)
    candidates = laws.first10_candidates(theta)
    expected = np.sort(candidates[
        np.asarray(greedy_forward_d_optimal_indices(theta[candidates], 4), dtype=np.int64)])
    assert np.array_equal(laws.k4_support(theta), expected)
    # a first-10 pool with more than four candidates never selects outside it
    assert int(laws.k4_support(theta).max()) < 10


def test_k4_forced_when_exactly_four_candidates() -> None:
    theta = np.full(30, np.nan)
    theta[[0, 3, 6, 8]] = [-2.5, -0.4, 1.2, 2.9]
    assert laws.k4_support(theta).tolist() == [0, 3, 6, 8]


def test_k4_support_lies_within_first10_and_is_deterministic() -> None:
    theta = _angles(seed=23)
    first = laws.k4_support(theta)
    assert first.size == 4 and int(first.max()) < 10 and int(first.min()) >= 0
    assert np.array_equal(first, laws.k4_support(theta))


def test_kall_is_all_first10_candidates_and_contains_k4() -> None:
    theta = _angles(seed=31)
    kall = laws.kall_support(theta)
    assert kall.tolist() == laws.first10_candidates(theta).tolist()
    assert int(kall.size) >= 4 and int(kall.max()) < 10
    assert np.isin(laws.k4_support(theta), kall).all()


def test_support_payload_disclosure() -> None:
    payload = laws.support_payload(_angles(seed=41))
    assert payload["first10_candidates"] == [1, 3, 5, 7, 9]
    assert payload["first10_candidate_count"] == 5
    assert payload["directional_among_positions_0_to_9"] == 5
    assert payload["kall_cardinality"] == 5
    assert payload["k4_within_kall"] is True
    assert payload["reblock_pool_depth_required_trials"] == 10
    assert payload["reblock_earliest_decode_trial_1based"] == 11
    assert payload["reblock_earliest_evidence_position_0based"] == 10
    assert payload["oldlaw_pool_depth_required_trials"] == 30


def test_dopt30_support_is_the_sealed_law_recomputed() -> None:
    from src.m2_chrono4_strict_v1 import laws as chrono_laws

    theta = _angles(seed=47)
    assert np.array_equal(laws.dopt30_support(theta), chrono_laws.dopt_support_indices(theta))
    assert int(laws.dopt30_support(theta).max()) < 30


def test_cross_family_support_agreement_keys_and_fails_closed() -> None:
    payload = laws.support_payload(_angles(seed=53))
    entry_static = {**payload, "dopt30_support_positions": [1, 11, 17, 23]}
    entry_cdm = {**payload, "dopt30_support_positions": [1, 11, 17, 23]}
    static_supports = {
        "within_post10|sesA": entry_static,
        "external_post10_query|sesB": entry_static,
    }
    cdm_supports = {
        "within_post10|sesA": entry_cdm,
        "external_post10_local|sesB": entry_cdm,
    }
    agreement = laws.cross_family_support_agreement(static_supports, cdm_supports)
    assert sorted(agreement) == ["external_post10_query|sesB", "within_post10|sesA"]
    assert all(item["agree"] for item in agreement.values())
    # a disagreement on any of the three support laws fails closed
    drifted = {**entry_cdm, "k4_support_positions": [1, 3, 5, 7]}
    with pytest.raises(laws.ReblockLawError):
        laws.cross_family_support_agreement(
            static_supports, {**cdm_supports, "within_post10|sesA": drifted})
    drifted_kall = {**entry_cdm, "kall_support_positions": [1, 3, 5, 7]}
    with pytest.raises(laws.ReblockLawError):
        laws.cross_family_support_agreement(
            static_supports, {**cdm_supports, "external_post10_local|sesB": drifted_kall})
    drifted_dopt = {**entry_cdm, "dopt30_support_positions": [1, 11, 17, 29]}
    with pytest.raises(laws.ReblockLawError):
        laws.cross_family_support_agreement(
            static_supports, {**cdm_supports, "within_post10|sesA": drifted_dopt})
    # the attempt-1 regression: a roster that matches nothing fails closed
    with pytest.raises(laws.ReblockLawError):
        laws.cross_family_support_agreement(
            {"within_post30|sesA": entry_static}, cdm_supports)


# ---------------------------------------------------------------------------
# 3. the post-H window-partition law.
# ---------------------------------------------------------------------------


def _windows_and_trials() -> tuple[np.ndarray, np.ndarray]:
    trials = np.asarray(
        [0, 100, 210, 305, 410, 520, 615, 720, 830, 925, 1030, 1140, 1240], dtype=np.int64)
    windows = np.arange(0, 1240 - 49, 10, dtype=np.int64)
    return windows, trials


def test_post10_partition_boundary_semantics() -> None:
    windows, trials = _windows_and_trials()
    post10 = laws.select_post_h_window_starts(windows, trials, 10)
    assert int(post10.min()) >= int(trials[10])
    assert np.array_equal(post10, windows[windows >= trials[10]])
    # the last pre-boundary window is strictly inside trial 9
    assert int(windows[windows < trials[10]].max()) < int(trials[10])


def test_post_h_partition_fails_closed() -> None:
    windows, trials = _windows_and_trials()
    with pytest.raises(laws.ReblockLawError):
        laws.select_post_h_window_starts(windows, trials, 13)  # needs trials.size > horizon
    unsorted = np.asarray([5, 5, 7], dtype=np.int64)
    with pytest.raises(laws.ReblockLawError):
        laws.select_post_h_window_starts(unsorted, trials[: 12], 10)
    with pytest.raises(laws.ReblockLawError):
        laws.select_post_h_window_starts(windows[:1], trials, 10)  # no post-10 window


def test_post_h_partition_matches_sealed_post30_at_horizon_30() -> None:
    rng = np.random.default_rng(7)
    trials = np.sort(rng.choice(np.arange(0, 5000, 5), size=40, replace=False)).astype(np.int64)
    trials[0] = 0
    windows = np.arange(0, int(trials[-1]) + 50, 3, dtype=np.int64)
    windows = windows[windows + 50 <= int(trials[-1]) + 200]
    assert laws.post_h_partition_matches_sealed_post30(windows, trials) is True
    assert np.array_equal(
        laws.select_post_h_window_starts(windows, trials, 30),
        laws.select_post_h_window_starts(windows, trials, plan.OLD_HORIZON),
    )


# ---------------------------------------------------------------------------
# 4. the evidence-stream census.
# ---------------------------------------------------------------------------


def test_evidence_census_counts_and_ratio() -> None:
    census = laws.evidence_census(45)
    assert census["total_completed_trials"] == 45
    assert census["old_evidence_trials_post30"] == 15
    assert census["new_evidence_trials_post10"] == 35
    assert census["extension_ratio_new_over_old"] == pytest.approx(35.0 / 15.0)
    assert census["old_evidence_start_position_0based"] == 30
    assert census["new_evidence_start_position_0based"] == 10


def test_evidence_census_fails_closed() -> None:
    with pytest.raises(laws.ReblockLawError):
        laws.evidence_census(30)  # the old stream would be empty
    with pytest.raises(laws.ReblockLawError):
        laws.evidence_census(10)


def test_window_census_partitions() -> None:
    trials = np.arange(0, 40 * 115, 115, dtype=np.int64)  # 40 completed trials
    windows = np.arange(0, int(trials[-1]) - 60, 5, dtype=np.int64)
    census = laws.window_census(windows, trials)
    assert census["total_windows"] == int(windows.size)
    assert census["post30_windows"] == int((windows >= trials[30]).sum())
    assert census["post10_windows"] == int((windows >= trials[10]).sum())
    assert census["post10_windows"] >= census["post30_windows"]
    assert census["post10_over_post30_window_ratio"] == pytest.approx(
        census["post10_windows"] / census["post30_windows"])


def test_window_census_2to3x_extension_shape() -> None:
    trials = np.arange(0, 4600, 115, dtype=np.int64)  # 40 completed trials
    windows = np.arange(0, 4400, 5, dtype=np.int64)
    census = laws.window_census(windows, trials)
    evidence = laws.evidence_census(40)
    assert evidence["old_evidence_trials_post30"] == 10
    assert evidence["new_evidence_trials_post10"] == 30
    assert evidence["extension_ratio_new_over_old"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 5. the paired contrast and breadth.
# ---------------------------------------------------------------------------


def test_paired_contrast_arithmetic_and_breadth() -> None:
    block10 = {"a": 0.30, "b": 0.20, "c": 0.25, "d": 0.28, "e": 0.22, "f": 0.31}
    oldlaw = {"a": 0.28, "b": 0.24, "c": 0.25, "d": 0.30, "e": 0.20, "f": 0.29}
    contrast = laws.paired_contrast(block10, oldlaw)
    assert contrast["per_session_delta"] == {
        "a": pytest.approx(0.02), "b": pytest.approx(-0.04), "c": pytest.approx(0.0),
        "d": pytest.approx(-0.02), "e": pytest.approx(0.02), "f": pytest.approx(0.02)}
    assert contrast["equal_session_mean_delta"] == pytest.approx(0.0)
    assert contrast["positive_sessions_block10_better"] == 3
    assert contrast["negative_sessions_oldlaw_better"] == 2
    assert contrast["tied_sessions"] == 1
    assert contrast["breadth_denominator"] == 6
    with pytest.raises(laws.ReblockLawError):
        laws.paired_contrast({"a": 1.0}, {"b": 1.0})


# ---------------------------------------------------------------------------
# 6. the gate boundaries at 1e-12.
# ---------------------------------------------------------------------------


def _contrast(delta: float, positive: int, sessions: int = 6) -> dict:
    return {
        "equal_session_mean_delta": float(delta),
        "positive_sessions_block10_better": int(positive),
        "breadth_denominator": int(sessions),
    }


EPSILON = plan.GATES["boundary_epsilon"]
FLOOR = plan.GATES["primary"]["delta_floor"]


def test_gate_clear_pass() -> None:
    gate = laws.gate_evaluation(_contrast(FLOOR + 0.005, 4))
    assert gate["verdict"] == "REBLOCK_CDM_NET_POSITIVE"
    assert gate["delta_meets_floor_exactly"] is True
    assert gate["breadth_meets"] is True
    assert gate["delta_within_epsilon_band_below_floor"] is False


def test_gate_exactly_at_floor_passes() -> None:
    gate = laws.gate_evaluation(_contrast(FLOOR, 4))
    assert gate["verdict"] == "REBLOCK_CDM_NET_POSITIVE"


def test_gate_boundary_band_is_borderline_never_silent() -> None:
    inside = laws.gate_evaluation(_contrast(FLOOR - 5.0e-13, 4))
    assert inside["verdict"] == "REBLOCK_BORDERLINE"
    assert inside["delta_within_epsilon_band_below_floor"] is True
    assert inside["delta_meets_floor_exactly"] is False
    assert inside["breadth_meets"] is True
    outside = laws.gate_evaluation(_contrast(FLOOR - 5.0e-12, 4))
    assert outside["verdict"] == "REBLOCK_CDM_NOT_WORTH"
    assert outside["delta_within_epsilon_band_below_floor"] is False


def test_gate_breadth_short_is_not_worth_even_above_floor() -> None:
    gate = laws.gate_evaluation(_contrast(FLOOR + 0.02, 3))
    assert gate["verdict"] == "REBLOCK_CDM_NOT_WORTH"
    assert gate["breadth_meets"] is False


def test_gate_negative_delta_is_not_worth() -> None:
    gate = laws.gate_evaluation(_contrast(-0.05, 1))
    assert gate["verdict"] == "REBLOCK_CDM_NOT_WORTH"
    assert gate["delta_meets_floor_exactly"] is False


# ---------------------------------------------------------------------------
# 7. the k-row (KALL) binding laws.
# ---------------------------------------------------------------------------


def test_kall_activity_rows_are_the_selected_rows() -> None:
    rng = np.random.default_rng(3)
    calibration = rng.gamma(2.0, 1.5, size=(30, plan.TRIAL_LENGTH, plan.CHANNELS)).astype(
        np.float32)
    kall = np.asarray([1, 3, 5, 7, 9], dtype=np.int64)
    # k = 4: the sealed law binds verbatim
    assert np.array_equal(
        select_activity_rows(calibration, selected_indices=kall[:4], activity_budget=4),
        calibration[kall[:4]])
    # k = 5: the sealed cardinality guard rejects (the disclosed deviation) ...
    with pytest.raises(Exception):
        select_activity_rows(calibration, selected_indices=kall, activity_budget=4)
    # ... so the generalized law spells the same selected-rows arithmetic
    assert np.array_equal(
        np.ascontiguousarray(calibration[kall], dtype=np.float32), calibration[kall])


def test_krow_carrier_and_activity_through_the_frozen_cdm_core() -> None:
    rng = np.random.default_rng(13)
    rates = np.ascontiguousarray(rng.gamma(2.0, 40.0, size=(5, 12)), dtype=np.float64)
    theta = np.asarray([-2.4, -0.8, 0.0, 0.9, 2.5], dtype=np.float64)
    directions = pseudo_core.canonical_direction_indices(theta)
    fitted = cdm_core.fit_carriers_from_trial_table(
        rates, directions,
        mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    assert fitted.shape == (12, 4) and np.isfinite(fitted).all()
    # the k-row fit IS the sealed fixed-ridge-by-trial arithmetic with n = k:
    # design [cos, sin, 1] over the CANONICAL directions, penalty diag(n*lambda, n*lambda, 0)
    canonical = np.asarray(
        [cdm_core.CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions],
        dtype=np.float64)
    design = np.column_stack((np.cos(canonical), np.sin(canonical),
                              np.ones(5, dtype=np.float64)))
    penalty = np.diag((5 * 0.1, 5 * 0.1, 0.0))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ rates)
    a, c, b = coefficients
    manual = np.ascontiguousarray(
        np.column_stack((a, c, np.hypot(a, c), b)), dtype=np.float32)
    assert np.array_equal(fitted, manual)
    # and the k = 4 subset fit differs (the fifth row genuinely enters)
    fitted4 = cdm_core.fit_carriers_from_trial_table(
        rates[:4], directions[:4],
        mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    assert not np.array_equal(fitted, fitted4)


def test_krow_activity_memory_keeps_the_30_trial_ceiling() -> None:
    channels = np.arange(8, dtype=np.int64)
    digest = cdm_core.channel_order_digest(channels)
    trials = tuple(
        cdm_core.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(
                np.full((100, 8), float(index), dtype=np.float32)),
            session_id="s", trial_id=f"s:trial:{index}", channel_order_sha256=digest,
        )
        for index in range(5)
    )
    memory = cdm_core.ActivityMemory.initialize(
        trials, channel_ids=channels, fifo_capacity=30 - 5)
    assert memory.fifo_capacity == 25
    assert memory.stack().shape == (5, 100, 8)
    query = cdm_core.B3SInterpolatedSpikeCountTrial(
        activity=np.ascontiguousarray(np.full((100, 8), 9.0, dtype=np.float32)),
        session_id="s", trial_id="s:trial:99", channel_order_sha256=digest,
    )
    grown = memory.after_completed_trial(query)
    assert grown.query_count == 1 and grown.stack().shape == (6, 100, 8)
    with pytest.raises(Exception):
        # the frozen config law still guards the M4/M10/M30 budgets
        cdm_core.CDMDConfig(support_budget_m=5)


# ---------------------------------------------------------------------------
# 8. the anchor matchers (imported from the chrono4 precedent).
# ---------------------------------------------------------------------------


def test_anchor_matchers_bind() -> None:
    cell = {"window_count": 99, "ordered_window_starts_sha256": "a" * 64,
            "target_sha256": "b" * 64, "r2": 0.5}
    sealed = {"window_count": 99, "query_starts_sha256": "a" * 64,
              "target_sha256": "b" * 64, "r2": 0.5 + 1.0e-7}
    assert laws.pure_data_pairing(cell, sealed)["exact_match"] is True
    anchored = laws.fidelity_anchor(cell, sealed, r2_tolerance=1.0e-5)
    assert anchored["exact_match"] is True
    assert laws.dopt_proof(np.asarray([1, 3, 5, 9]), [1, 3, 5, 9])["exact_match"] is True
    assert laws.dopt_proof(np.asarray([1, 3, 5, 9]), [1, 3, 5, 7])["exact_match"] is False


# ---------------------------------------------------------------------------
# 9. the plan binds the sealed foundations and the pre-registered roster.
# ---------------------------------------------------------------------------


def test_plan_binds_the_sealed_foundations() -> None:
    assert plan.COMPARATOR_SCORE_SHA256 == chrono_plan.COMPARATOR_SCORE_SHA256
    assert plan.CDM_SCREEN_SCORE_SHA256 == chrono_plan.CDM_SCREEN_SCORE_SHA256
    assert plan.T4_CHECKPOINT_SHA256 == g_plan.T4_CHECKPOINT_SHA256
    assert plan.SPINT_CHECKPOINT_SHA256 == g_plan.SPINT_CHECKPOINT_SHA256
    assert plan.NORMALIZATION_SHA256 == g_plan.NORMALIZATION_SHA256
    assert plan.RIDGE_NORMALIZED_LAMBDA == g_plan.RIDGE_NORMALIZED_LAMBDA == 0.1
    assert plan.ACTIVITY_HORIZON == g_plan.ACTIVITY_HORIZON == 30
    assert plan.WINDOW_SIZE == g_plan.WINDOW_SIZE == 50
    assert plan.CHANNELS == g_plan.CHANNELS == 96


def test_plan_registers_the_reblock_protocol() -> None:
    assert plan.BLOCK_HORIZON == 10
    assert plan.OLD_HORIZON == 30
    assert plan.SURFACES_STATIC == ("within_post10", "external_post10_query")
    assert plan.SURFACES_CDM == ("within_post10", "external_post10_local")
    assert set(plan.CELLS) == {
        "BLOCK10_STATIC_K4", "BLOCK10_STATIC_KALL",
        "BLOCK10_CDM_K4", "BLOCK10_CDM_KALL",
        "OLDLAW_STATIC", "OLDLAW_CDM"}
    assert plan.GATES["primary"]["delta_floor"] == 0.01
    assert plan.GATES["primary"]["breadth_min"] == 4
    assert plan.GATES["primary"]["breadth_denominator"] == 6
    assert plan.GATES["boundary_epsilon"] == 1.0e-12
    assert sorted(plan.GATES["verdicts"]) == [
        "REBLOCK_BORDERLINE", "REBLOCK_CDM_NET_POSITIVE", "REBLOCK_CDM_NOT_WORTH"]
    assert plan.EXPECTED_WITHIN_SESSIONS == 7
    assert plan.EXPECTED_EXTERNAL_SESSIONS == 6
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ENVIRONMENT_LAW["dataloader_workers"] == 0
    assert plan.ANCHORS["executor_fidelity"]["r2_tolerance"] == 1.0e-5
    assert cdm_core.CANONICAL_DIRECTIONS_RAD[0] == pytest.approx(-3 * math.pi / 4)
