"""Synthetic no-data/no-CUDA tests for the M2 chrono4 strict-caliber cell.

The review-critical properties:

1. the chronological-first-4 law: strict branch when the first four trials are
   all directional, fallback branch (first four directional trials) otherwise,
   the earliest-decode arithmetic, and the disclosure fields;
2. the D-opt recompute law used for the receipts' selection proof;
3. the scatter statistics agree with the canonical-direction geometry;
4. the first-4 T4/activity fit consumes exactly the chronological support rows
   through the SEALED frozen laws (``_ridge_side`` / ``select_activity_rows``);
5. the paired selection-contribution arithmetic and breadth;
6. the verdict classes honor the pre-registered thresholds with the 1e-12
   boundary band and never silently flip;
7. the anchor matchers: pure-data pairing exact (both digest spellings),
   executor fidelity tolerance-bounded;
8. the plan constants bind the sealed foundations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src",
             ROOT / "sua_exploration"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core  # noqa: E402
from src.cdm_p1_m2_local_v1 import plan as g_plan  # noqa: E402
from src.m2_chrono4_strict_v1 import laws, plan  # noqa: E402
from src.m2_t4_activity_budget_screen_v1.core import select_activity_rows  # noqa: E402
from src.m2_t4_activity_budget_screen_v1.physical import _ridge_side  # noqa: E402
from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4  # noqa: E402


def _angles(count: int = 30, *, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-math.pi, math.pi, size=count)
    values[0::2] = np.nan  # the M2 roster's alternating non-directional pattern
    return np.ascontiguousarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. the chronological-first-4 law.
# ---------------------------------------------------------------------------


def test_chrono4_strict_branch() -> None:
    angles = np.linspace(-math.pi, math.pi - 0.1, 30).astype(np.float64)
    payload = laws.chrono4_support(angles)
    assert payload["support_positions"] == [0, 1, 2, 3]
    assert payload["strict_branch_taken"] is True
    assert payload["fallback_branch_taken"] is False
    assert payload["directional_among_positions_0_to_3"] == 4
    assert payload["earliest_decode_trial_1based"] == 5
    assert payload["earliest_decode_position_0based"] == 4
    assert payload["pool_depth_required_trials"] == 4


def test_chrono4_fallback_branch_first_four_directional() -> None:
    payload = laws.chrono4_support(_angles())
    assert payload["support_positions"] == [1, 3, 5, 7]
    assert payload["strict_branch_taken"] is False
    assert payload["fallback_branch_taken"] is True
    assert payload["directional_among_positions_0_to_3"] == 2
    assert payload["last_support_position_0based"] == 7
    assert payload["earliest_decode_trial_1based"] == 9
    assert payload["pool_depth_required_trials"] == 8


def test_chrono4_fallback_with_early_finite_gap_pattern() -> None:
    angles = np.full(30, np.nan)
    angles[[2, 4, 5, 9, 11]] = [0.1, 0.9, -0.7, 2.2, -2.0]
    payload = laws.chrono4_support(angles)
    assert payload["support_positions"] == [2, 4, 5, 9]
    assert payload["earliest_decode_trial_1based"] == 11


def test_chrono4_law_is_deterministic_and_fails_closed() -> None:
    angles = _angles(seed=17)
    assert laws.chrono4_support(angles) == laws.chrono4_support(angles)
    broken = np.full(30, np.nan)
    broken[:3] = [0.1, 0.2, 0.3]
    with pytest.raises(laws.Chrono4LawError):
        laws.chrono4_support(broken)
    with pytest.raises(laws.Chrono4LawError):
        laws.chrono4_support(np.asarray([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# 2. the D-opt recompute law (the receipts' selection proof).
# ---------------------------------------------------------------------------


def test_dopt_recompute_matches_the_sealed_greedy_law() -> None:
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    angles = _angles(seed=31)
    finite = np.flatnonzero(np.isfinite(angles))
    expected = np.sort(finite[greedy_forward_d_optimal_indices(angles[finite], 4)])
    assert np.array_equal(laws.dopt_support_indices(angles), expected)
    # only four candidates: the selection is forced and sorted
    tight = np.full(30, np.nan)
    tight[[3, 8, 13, 27]] = [0.2, 1.4, -1.1, 2.7]
    assert laws.dopt_support_indices(tight).tolist() == [3, 8, 13, 27]


# ---------------------------------------------------------------------------
# 3. the scatter statistics.
# ---------------------------------------------------------------------------


def test_scatter_statistics_canonical_geometry() -> None:
    spread = laws.scatter_statistics(
        np.asarray([-3 * math.pi / 4, -math.pi / 4, math.pi / 4, 3 * math.pi / 4]))
    assert spread["distinct_canonical_directions"] == 4
    assert spread["canonical_direction_indices"] == [0, 2, 4, 6]
    assert spread["min_pairwise_angular_separation_rad"] == pytest.approx(math.pi / 2)
    assert spread["max_pairwise_angular_separation_rad"] == pytest.approx(math.pi)
    assert spread["mean_pairwise_angular_separation_rad"] == pytest.approx(2 * math.pi / 3)
    clustered = laws.scatter_statistics(np.asarray([0.01, -0.01, 0.02, -0.02]))
    assert clustered["distinct_canonical_directions"] == 1
    assert clustered["min_pairwise_angular_separation_rad"] < 0.05
    assert clustered["min_pairwise_angular_separation_rad"] < (
        spread["min_pairwise_angular_separation_rad"])
    with pytest.raises(laws.Chrono4LawError):
        laws.scatter_statistics(np.asarray([np.nan, 0.1, 0.2, 0.3]))
    with pytest.raises(laws.Chrono4LawError):
        laws.scatter_statistics(np.asarray([0.1, 0.2, 0.3]))


def test_index_scatter() -> None:
    stats = laws.index_scatter(np.asarray([1, 3, 5, 7]))
    assert stats == {"support_positions": [1, 3, 5, 7], "index_span": 6,
                     "max_position": 7, "mean_position": 4.0}


# ---------------------------------------------------------------------------
# 4. the first-4 T4/activity fit through the SEALED frozen laws.
# ---------------------------------------------------------------------------


def _synthetic_dataset(trials: int = 30) -> SimpleNamespace:
    rng = np.random.default_rng(9)
    angles = _angles(seed=23)
    rates = rng.gamma(2.0, 3.0, size=(trials, plan.CHANNELS))
    lengths = rng.integers(80, 121, size=trials).astype(np.float64)
    return SimpleNamespace(
        calib_trial_spike_sums={"ses": rates * lengths[:, None]},
        calib_trial_lengths={"ses": lengths},
        calib_trial_target_angles={"ses": angles},
        side_feature_mean=np.full(4, 0.5, dtype=np.float32),
        side_feature_std=np.full(4, 2.0, dtype=np.float32),
    )


def test_first4_t4_fit_consumes_the_chronological_support_rows() -> None:
    dataset = _synthetic_dataset()
    angles = np.asarray(dataset.calib_trial_target_angles["ses"], dtype=np.float64)
    support = np.asarray(laws.chrono4_support(angles)["support_positions"], dtype=np.int64)
    assert support.tolist() == [1, 3, 5, 7]
    side, evidence = _ridge_side(dataset, "ses", support)
    # the sealed law fit over exactly the four directional rows
    rates = (np.asarray(dataset.calib_trial_spike_sums["ses"][support], dtype=np.float64)
             / np.asarray(dataset.calib_trial_lengths["ses"][support], dtype=np.float64)[:, None])
    manual, _ = fit_ridge_t4(
        rates, angles[support], normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA)
    assert np.array_equal(side, np.ascontiguousarray(
        (manual - dataset.side_feature_mean) / dataset.side_feature_std, dtype=np.float32))
    assert evidence["usable_directional_trials"] == 4
    # the sealed helper labels any four-row support by its size law; the
    # chrono4 labelling lives in this package's own side_evidence
    assert evidence["selected_indices"] == [1, 3, 5, 7]
    # the D-opt support produces a DIFFERENT carrier on the same session
    dopt = laws.dopt_support_indices(angles)
    side_dopt, _ = _ridge_side(dataset, "ses", dopt)
    assert not np.array_equal(side, side_dopt)


def test_first4_activity_pool_is_the_chronological_support_rows() -> None:
    rng = np.random.default_rng(3)
    calibration = rng.gamma(2.0, 1.5, size=(30, plan.TRIAL_LENGTH, plan.CHANNELS)).astype(
        np.float32)
    support = np.asarray([1, 3, 5, 7], dtype=np.int64)
    activity = select_activity_rows(
        calibration, selected_indices=support, activity_budget=plan.M4)
    assert np.array_equal(activity, calibration[[1, 3, 5, 7]])
    # the sealed law rejects a five-row "chrono4" support (cardinality drift)
    with pytest.raises(Exception):
        select_activity_rows(calibration, selected_indices=np.arange(5), activity_budget=4)


# ---------------------------------------------------------------------------
# 5. the paired selection contribution.
# ---------------------------------------------------------------------------


def test_paired_selection_contribution() -> None:
    dopt = {"a": 0.30, "b": 0.20, "c": 0.25}
    chrono = {"a": 0.28, "b": 0.24, "c": 0.25}
    contribution = laws.paired_selection_contribution(dopt, chrono)
    assert contribution["per_session_delta"] == {
        "a": pytest.approx(0.02), "b": pytest.approx(-0.04), "c": pytest.approx(0.0)}
    assert contribution["equal_session_mean_delta"] == pytest.approx(-0.02 / 3.0)
    assert contribution["positive_sessions_dopt_better"] == 1
    assert contribution["negative_sessions_chrono4_better"] == 1
    assert contribution["tied_sessions"] == 1
    assert contribution["dopt_equal_session_mean"] == pytest.approx(0.25)
    assert contribution["chrono4_equal_session_mean"] == pytest.approx(0.25 + 0.02 / 3.0)
    with pytest.raises(laws.Chrono4LawError):
        laws.paired_selection_contribution({"a": 1.0}, {"b": 1.0})


def test_equal_session_mean_and_sd() -> None:
    values = {"a": 0.2, "b": 0.4, "c": 0.6}
    assert laws.equal_session_mean(values) == pytest.approx(0.4)
    assert laws.session_sd(values) == pytest.approx(
        float(np.sqrt(((np.array([0.2, 0.4, 0.6]) - 0.4) ** 2).mean())), abs=1e-15)


# ---------------------------------------------------------------------------
# 6. the verdict classes and the 1e-12 boundary band.
# ---------------------------------------------------------------------------


EPSILON = plan.VERDICT_LAW["boundary_epsilon"]


def test_verdict_classes_clear_of_boundaries() -> None:
    assert laws.verdict_class(0.05)["verdict"] == "SELECTION_CONTRIBUTION_LARGE"
    assert laws.verdict_class(0.01)["verdict"] == "SELECTION_CONTRIBUTION_MODERATE"
    assert laws.verdict_class(0.001)["verdict"] == "SELECTION_CONTRIBUTION_SMALL"
    assert laws.verdict_class(-0.01)["verdict"] == "SELECTION_CONTRIBUTION_SMALL"
    zero = laws.verdict_class(0.0)
    assert zero["verdict"] == "SELECTION_CONTRIBUTION_SMALL"
    assert zero["boundary_bands"]["large"]["within_epsilon_band_of_boundary"] is False
    assert zero["boundary_bands"]["moderate"]["within_epsilon_band_of_boundary"] is False


def test_verdict_boundary_band_never_flips_silently() -> None:
    large_threshold = plan.VERDICT_LAW["thresholds"]["large_strictly_above"]
    moderate_floor = plan.VERDICT_LAW["thresholds"]["moderate_floor"]
    # exactly at 0.02: the pre-registered MODERATE interval includes its endpoint
    at_large = laws.verdict_class(large_threshold)
    assert at_large["verdict"] == "SELECTION_CONTRIBUTION_MODERATE"
    assert at_large["boundary_bands"]["large"]["within_epsilon_band_of_boundary"] is True
    # just above 0.02: LARGE
    assert laws.verdict_class(large_threshold + 1.0e-9)["verdict"] == \
        "SELECTION_CONTRIBUTION_LARGE"
    # inside the 1e-12 band below 0.02: pulled up to LARGE with disclosure
    pulled_up = laws.verdict_class(large_threshold - 5.0e-13)
    assert pulled_up["verdict"] == "SELECTION_CONTRIBUTION_LARGE"
    assert pulled_up["boundary_bands"]["large"]["within_epsilon_band_of_boundary"] is True
    # outside the band below 0.02: MODERATE, no band disclosure
    outside = laws.verdict_class(large_threshold - 5.0e-12)
    assert outside["verdict"] == "SELECTION_CONTRIBUTION_MODERATE"
    assert outside["boundary_bands"]["large"]["within_epsilon_band_of_boundary"] is False
    # exactly at 0.005: MODERATE (the floor is inclusive)
    at_floor = laws.verdict_class(moderate_floor)
    assert at_floor["verdict"] == "SELECTION_CONTRIBUTION_MODERATE"
    assert at_floor["boundary_bands"]["moderate"]["within_epsilon_band_of_boundary"] is True
    # inside the band below 0.005: pulled up to MODERATE with disclosure
    pulled = laws.verdict_class(moderate_floor - 5.0e-13)
    assert pulled["verdict"] == "SELECTION_CONTRIBUTION_MODERATE"
    assert pulled["boundary_bands"]["moderate"]["within_epsilon_band_of_boundary"] is True
    # outside the band below 0.005: SMALL
    assert laws.verdict_class(moderate_floor - 5.0e-12)["verdict"] == \
        "SELECTION_CONTRIBUTION_SMALL"


# ---------------------------------------------------------------------------
# 7. the anchor matchers.
# ---------------------------------------------------------------------------


def test_pure_data_pairing_both_digest_spellings() -> None:
    cell = {"window_count": 99, "ordered_window_starts_sha256": "a" * 64,
            "target_sha256": "b" * 64, "r2": 0.5}
    comparator_sealed = {"window_count": 99, "ordered_window_starts_sha256": "a" * 64,
                         "target_sha256": "b" * 64, "r2": 0.7}
    cdm_sealed = {"window_count": 99, "query_starts_sha256": "a" * 64,
                  "target_sha256": "b" * 64, "r2": 0.7}
    assert laws.pure_data_pairing(cell, comparator_sealed)["exact_match"] is True
    assert laws.pure_data_pairing(cell, cdm_sealed)["exact_match"] is True
    cell["window_count"] = 100
    assert laws.pure_data_pairing(cell, comparator_sealed)["exact_match"] is False
    assert laws.pure_data_pairing(cell, comparator_sealed)["field_matches"][
        "window_count"] is False


def test_fidelity_anchor_tolerance() -> None:
    sealed = {"window_count": 10, "query_starts_sha256": "a" * 64,
              "target_sha256": "b" * 64, "r2": 0.5}
    cell = {"window_count": 10, "ordered_window_starts_sha256": "a" * 64,
            "target_sha256": "b" * 64, "r2": 0.5 + 2.0 ** -10}
    result = laws.fidelity_anchor(cell, sealed, r2_tolerance=1.0e-3)
    assert result["exact_match"] is True
    assert result["r2_within_tolerance"] is True
    assert result["r2_delta"] == pytest.approx(2.0 ** -10, abs=1e-18)
    cell["r2"] = 0.5 + 2.0 ** -9
    assert laws.fidelity_anchor(cell, sealed, r2_tolerance=1.0e-3)["exact_match"] is False


def test_dopt_proof() -> None:
    proof = laws.dopt_proof(np.asarray([1, 3, 11, 13]), [1, 3, 11, 13])
    assert proof["exact_match"] is True
    assert laws.dopt_proof(np.asarray([1, 3, 11, 13]), [1, 3, 11, 14])["exact_match"] is False


# ---------------------------------------------------------------------------
# 8. the plan binds the sealed foundations.
# ---------------------------------------------------------------------------


def test_plan_binds_the_sealed_foundations() -> None:
    assert plan.COMPARATOR_SCORE_SHA256 == g_plan.SAME_QUERY_SCORE_SHA256
    assert plan.CDM_SCREEN_SCORE_SHA256 == g_plan.CDM_SCREEN_SCORE_SHA256
    assert plan.T4_CHECKPOINT_SHA256 == g_plan.T4_CHECKPOINT_SHA256
    assert plan.SPINT_CHECKPOINT_SHA256 == g_plan.SPINT_CHECKPOINT_SHA256
    assert plan.NORMALIZATION_SHA256 == g_plan.NORMALIZATION_SHA256
    assert plan.SURFACES_STATIC == ("within_post30", "external_official_query")
    assert plan.SURFACES_CDM == ("within_post30", "external_post30_local")
    assert plan.RIDGE_NORMALIZED_LAMBDA == g_plan.RIDGE_NORMALIZED_LAMBDA == 0.1
    assert plan.ACTIVITY_HORIZON == g_plan.ACTIVITY_HORIZON == 30
    assert plan.VERDICT_LAW["thresholds"] == {
        "large_strictly_above": 0.02, "moderate_floor": 0.005}
    assert plan.VERDICT_LAW["boundary_epsilon"] == 1.0e-12
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ENVIRONMENT_LAW["dataloader_workers"] == 0
    assert plan.ANCHORS["executor_fidelity"]["r2_tolerance"] == 1.0e-5
    assert cdm_core.CANONICAL_DIRECTIONS_RAD[0] == pytest.approx(-3 * math.pi / 4)
