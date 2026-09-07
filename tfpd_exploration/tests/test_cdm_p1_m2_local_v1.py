"""Synthetic no-data/no-CUDA tests for the LOCAL-M2 transfer laws.

The review-critical properties:

1. the M2 continuous-angle anchor reproduces the sealed ``fit_ridge_t4``
   carrier bit-for-bit through the selected-support design, the single
   full-unit ``b0`` product and the ``sqrt`` modulation mirror (the audit's
   prerequisite-(b) law);
2. the frozen Stage-P gate/bank/trust-region machinery consumes the M2 anchor
   unchanged, a committed pseudo row moves exactly the refit law, and the
   mass/movement accounting is internally consistent;
3. the ``alpha_M = 0`` no-op returns the sealed carrier itself;
4. the velocity unit law converts m/s views to cm/s exactly;
5. the surface/trial partition law assigns every surface window to exactly one
   completed trial and marks the evidence boundary;
6. the gate module's margins, epsilon band and stop conditions mirror the
   Part-A law on the M2 rosters.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.support_anchored_t4_stage_o_v1 import trust_region
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from src.cdm_p1_m2_local_v1 import anchor as m2_anchor
from src.cdm_p1_m2_local_v1 import gates, plan, replay


def _synthetic_carrier(seed: int = 3, units: int = 24):
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    rng = np.random.default_rng(seed)
    canonical = np.asarray(cdm_core.CANONICAL_DIRECTIONS_RAD)
    angles = np.asarray([canonical[index] + 0.03 * rng.standard_normal()
                         for index in (0, 2, 4, 6)])
    rates = rng.gamma(3.0, 0.4, size=(angles.size, units))
    raw, _evidence = fit_ridge_t4(
        rates, angles, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    return raw, rates, angles


def test_anchor_parity_bitwise_and_gate_compatibility() -> None:
    raw, rates, angles = _synthetic_carrier()
    channel_ids = np.arange(raw.shape[0], dtype=np.int64)
    groups = m2_anchor.build_groups(raw, channel_ids)
    assert groups.group_count == cdm_core.GROUP_COUNT
    sizes = [groups.unit_indices(g).size for g in range(cdm_core.GROUP_COUNT)]
    assert max(sizes) - min(sizes) <= 1

    support = m2_anchor.M2SupportAnchor.from_labeled_support(
        groups=groups, support_trial_rates=rates,
        support_angles_rad=angles.tolist(), support_t4=raw,
    )
    assert support.support_coefficient_parity["rebuilt_rows_bitwise_equal"] is True
    assert support.support_coefficient_parity["max_abs_coefficient_difference"] == 0.0
    for group_anchor in support.per_group:
        assert group_anchor.design_row_count == angles.size

    # The empty bank refit reproduces the sealed carrier through the anchor.
    empty = stage_p_gate.refit_from_anchor_p(support, stage_p_gate.EvidenceBankP.empty(), rho_M=1.0)
    assert np.array_equal(np.asarray(empty.candidate_t4), np.asarray(support.support_t4))


def _measurement(views, config, tau_d=math.pi / 4.0):
    return replay.direction_estimator.measure_trial(
        views, config=config, law="cross_group_circular", weight_law="resultant_length",
        binding="complementary_exclusion", tau_d=tau_d,
    )


def _canonical_reach_views(direction_index: int, windows: int = 8, speed_cms: float = 6.0):
    canonical = float(cdm_core.CANONICAL_DIRECTIONS_RAD[direction_index])
    unit = np.asarray([math.cos(canonical), math.sin(canonical)])
    rng = np.random.default_rng(11)
    views = []
    for _group in range(cdm_core.GROUP_COUNT):
        velocity = speed_cms * unit[None, :] * np.ones((windows, 1)) \
            + 0.01 * rng.standard_normal((windows, 2))
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(windows, dtype=np.bool_),
            session_id="synthetic", trial_id="synthetic:trial:0",
            prediction_interval_start_bin=49, prediction_interval_stop_bin=49 + windows,
        )
        views.append(cdm_core.CompletedVelocityPrediction(
            np.ascontiguousarray(velocity, dtype=np.float64), validity))
    return tuple(views)


def test_committed_block_moves_exactly_the_refit_law() -> None:
    raw, rates, angles = _synthetic_carrier()
    channel_ids = np.arange(raw.shape[0], dtype=np.int64)
    groups = m2_anchor.build_groups(raw, channel_ids)
    support = m2_anchor.M2SupportAnchor.from_labeled_support(
        groups=groups, support_trial_rates=rates,
        support_angles_rad=angles.tolist(), support_t4=raw,
    )
    config = cdm_core.CDMDConfig(support_budget_m=4)
    thresholds = stage_p_gate.GateThresholds(
        tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0,
    )
    measurement = _measurement(_canonical_reach_views(0), config)
    assert measurement.accepted is True
    facts = stage_p_gate.TrialFacts(
        session_id="synthetic", trial_id="synthetic:trial:30",
        chronology_position=30,
        scalar_rates=np.asarray(rates[0], dtype=np.float64),
        rate_sha256="a" * 64, native_counts_sha256="b" * 64,
        channel_order_sha256="c" * 64, valid_mask_sha256="d" * 64,
    )
    bank = stage_p_gate.EvidenceBankP.empty()
    decision = stage_p_gate.evaluate_block(
        support, bank, measurement.per_group, facts=facts,
        measurement_sha256=measurement.digest, rho_M=1.0, c_M=None,
        thresholds=thresholds, block_index=0,
    )
    assert decision.committed is True
    bank = bank.with_row(decision.row)
    refit = stage_p_gate.refit_from_anchor_p(support, bank, rho_M=1.0)
    outcome = trust_region.active_carrier(
        support, decision.coefficients, alpha_M=0.5, c_M=None,
    )
    # alpha_M = 0.5 moves exactly half the (unprojected) refit delta.
    expected = support.rebuild_t4([
        0.5 * (np.asarray(refit.coefficients[g], dtype=np.float64)
               - np.asarray(support.coefficients(support.support_t4)[g], dtype=np.float64))
        + np.asarray(support.coefficients(support.support_t4)[g], dtype=np.float64)
        for g in range(cdm_core.GROUP_COUNT)
    ])
    assert np.array_equal(np.asarray(outcome.active_t4), np.asarray(expected))
    assert outcome.movement_frobenius > 0.0


def test_alpha_zero_noop_returns_sealed_carrier() -> None:
    raw, rates, angles = _synthetic_carrier()
    channel_ids = np.arange(raw.shape[0], dtype=np.int64)
    groups = m2_anchor.build_groups(raw, channel_ids)
    support = m2_anchor.M2SupportAnchor.from_labeled_support(
        groups=groups, support_trial_rates=rates,
        support_angles_rad=angles.tolist(), support_t4=raw,
    )
    moved = support.rebuild_t4([
        np.asarray(support.coefficients(support.support_t4)[g], dtype=np.float64) + 0.05
        for g in range(cdm_core.GROUP_COUNT)
    ])
    outcome = trust_region.active_carrier(support, [
        np.asarray(block, dtype=np.float64)
        for block in support.coefficients(moved)
    ], alpha_M=0.0, c_M=None)
    assert np.shares_memory(np.asarray(outcome.active_t4), np.asarray(support.support_t4)) or \
        np.array_equal(np.asarray(outcome.active_t4), np.asarray(support.support_t4))
    assert outcome.movement_frobenius == 0.0


def test_velocity_unit_law_and_direction_gates() -> None:
    assert plan.VELOCITY_UNIT_LAW["view_scale"] == 100.0
    assert plan.VELOCITY_UNIT_LAW["dt_seconds"] == plan.BIN_SECONDS == 0.02
    # A 6 cm/s canonical reach in m/s (0.06) must pass the frozen gates only
    # after the exact conversion.
    views_ms = []
    direction_index = 2
    canonical = float(cdm_core.CANONICAL_DIRECTIONS_RAD[direction_index])
    unit = np.asarray([math.cos(canonical), math.sin(canonical)])
    for _group in range(cdm_core.GROUP_COUNT):
        velocity_ms = 0.06 * unit[None, :] * np.ones((8, 1))
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(8, dtype=np.bool_),
            session_id="synthetic", trial_id="synthetic:trial:1",
            prediction_interval_start_bin=49, prediction_interval_stop_bin=57,
        )
        views_ms.append(cdm_core.CompletedVelocityPrediction(
            np.ascontiguousarray(velocity_ms), validity))
    config = cdm_core.CDMDConfig(support_budget_m=4)
    rejected = replay.direction_estimator.measure_trial(
        tuple(views_ms), config=config, law="cross_group_circular",
        weight_law="resultant_length", binding="complementary_exclusion",
        tau_d=math.pi / 4.0,
    )
    assert rejected.accepted is False  # displacement ~0.0096 < 0.05 in metres
    converted = replay.direction_estimator.measure_trial(
        tuple(cdm_core.CompletedVelocityPrediction(
            replay._cm_s(view.velocity), view.validity) for view in views_ms),
        config=config, law="cross_group_circular", weight_law="resultant_length",
        binding="complementary_exclusion", tau_d=math.pi / 4.0,
    )
    assert converted.accepted is True
    assert converted.per_group[0].theta_index == direction_index


def test_surface_partition_and_evidence_boundary() -> None:
    starts_list = [49, 60, 70, 130, 145]
    trial_starts = [49]
    for index in range(1, 35):
        trial_starts.append(49 + index * 100)
    total = trial_starts[-1] + 60

    class _FakeDataset:
        window_indices = [("s", start) for start in starts_list]
        trial_start_indices = {"s": np.asarray(trial_starts, dtype=np.int64)}
        covariate_data = {"s": np.arange(total * 2, dtype=np.float32).reshape(total, 2)}
        neural_data = {"s": np.zeros((total, 8), dtype=np.float32)}

    material = replay.build_session_material(
        dataset=_FakeDataset(), session="s", surface="external_official_query",
    )
    assert material.starts.tolist() == starts_list
    assigned = [trial.position for trial in material.trials
                for _ in range(trial.metric_starts.size)]
    assert sorted(assigned) == [0, 0, 0, 1, 1]
    counts = {trial.position: trial.metric_starts.size
              for trial in material.trials if trial.metric_starts.size}
    assert counts == {0: 3, 1: 2}
    # The evidence boundary: only completed trials at/after position 30.
    assert all((trial.evidence_eligible == (trial.position >= 30))
               for trial in material.trials)
    assert sum(trial.evidence_eligible for trial in material.trials) == 5
    # The causal windows never cross a trial boundary.
    for trial in material.trials:
        assert trial.causal_starts.size == 0 or (
            trial.causal_starts[0] >= trial.start_bin
            and trial.causal_starts[-1] + plan.WINDOW_SIZE <= trial.stop_bin
        )


def test_gate_margins_epsilon_band_and_stop_conditions() -> None:
    law = {
        "promotion": {
            "expression": "expr", "delta_floor": 0.01,
            "breadth_min": 4, "breadth_denominator": 6,
        },
        "boundary_epsilon": 1.0e-12,
    }
    sessions = {f"s{index}": 0.20 + 0.01 * index for index in range(6)}
    candidate = dict(sessions)
    candidate["s0"] += 0.005
    promotion = gates.evaluate_promotion(
        candidate=candidate, baseline=sessions, promotion_law=law["promotion"],
        epsilon=law["boundary_epsilon"],
    )
    assert promotion["delta"]["equal_session_mean_delta"] == pytest.approx(0.005 / 6)
    assert promotion["promoted"] is False
    near = {name: value + 0.01 + 8.0e-13 for name, value in sessions.items()}
    near_promotion = gates.evaluate_promotion(
        candidate=near, baseline=sessions, promotion_law=law["promotion"],
        epsilon=1.0e-12,
    )
    margin = near_promotion["delta_margin"]
    assert margin["within_epsilon_band_of_boundary"] is True
    assert margin["meets_margin"] is True

    gate = gates.evaluate_m2_local_gate(
        external_m4={"F00m": dict(sessions), "F01m": dict(candidate)},
        within_by_budget={4: {"F00m": dict(sessions), "F01m": dict(sessions)},
                          30: {"F00m": dict(sessions), "F01m": dict(sessions)}},
        external_m30={"F00m": dict(sessions), "F01m": dict(sessions)},
        safety={
            "f00m_bit_anchor_vs_sealed_same_query_rows": True,
            "anchor_zero_evidence_parity_all_selected_supports": True,
            "m30_exact_noop": True,
            "causality_receipt_chains": True,
            "hyperparameters_reselected_on_m2_source_folds": True,
            "dandi_hyperparameters_ported_as_claims": False,
            "zero_target_updates": True,
        },
        gates_law=plan.GATES,
    )
    assert gate["safety_all_pass"] is True
    assert gate["promoted"] is False
    secondary = gates.evaluate_secondary_gate(
        external_m4={"G00m": dict(sessions), "G01m": dict(candidate)},
        gates_law=plan.GATES,
    )
    stops = gates.stop_conditions(gate, secondary)
    assert "no_candidate_beats_baseline" in stops["fired"]
    assert "secondary_full_stack_gate" not in stops["fired"]
    assert stops["conditions"]["secondary_full_stack_gate"]["fired"] is False


def test_short_trial_policy_law_is_frozen() -> None:
    assert "no_padding" in plan.SHORT_TRIAL_POLICY
    assert plan.EVIDENCE_START_POSITION == plan.ACTIVITY_HORIZON == 30
    assert plan.BUDGETS_F == (4, 10, 30)
    assert plan.GATES["promotion"]["breadth_denominator"] == 6
    assert plan.GATES["boundary_epsilon"] == 1.0e-12
    assert plan.M30_NOOP_LAW["alpha_M"] == 0.0
    assert plan.GATES["safety"]["within_floor"] == -0.02
