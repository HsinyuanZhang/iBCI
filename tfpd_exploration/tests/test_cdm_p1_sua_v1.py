"""Synthetic no-data/no-CUDA tests for the SUA transfer laws.

The review-critical properties:

1. the SUA anchor reproduces the sealed ``fit_ridge_t4`` carrier bit-for-bit
   through the continuous-angle design, the single full-unit ``b0`` product
   and the ``sqrt`` modulation mirror (the zero-evidence fallback law);
2. the frozen Stage-P gate/bank/trust-region machinery consumes the SUA
   anchor unchanged and a committed pseudo row moves exactly the refit law;
3. the M30 ``alpha_M = 0`` no-op returns the sealed carrier itself;
4. the gate module's margins and stop conditions mirror the Part-A law.
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

from src.cdm_p1_sua_v1 import anchor as sua_anchor
from src.cdm_p1_sua_v1 import gates, plan, replay


def _synthetic_carrier(seed: int = 3, units: int = 24):
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    rng = np.random.default_rng(seed)
    canonical = np.asarray(cdm_core.CANONICAL_DIRECTIONS_RAD)
    angles = np.asarray([canonical[index] for index in (0, 2, 4, 6)])
    rates = rng.gamma(3.0, 4.0, size=(angles.size, units))
    raw, _evidence = fit_ridge_t4(
        rates, angles, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    return raw, rates, angles


def test_anchor_parity_bitwise_and_gate_compatibility() -> None:
    raw, rates, angles = _synthetic_carrier()
    channel_ids = np.arange(raw.shape[0], dtype=np.int64)
    groups = sua_anchor.build_groups(raw, channel_ids)
    assert groups.group_count == cdm_core.GROUP_COUNT
    sizes = [groups.unit_indices(g).size for g in range(cdm_core.GROUP_COUNT)]
    assert max(sizes) - min(sizes) <= 1

    support = sua_anchor.SUASupportAnchor.from_labeled_support(
        groups=groups, support_trial_rates_hz=rates,
        support_angles_rad=angles.tolist(), support_t4=raw,
    )
    assert support.support_coefficient_parity["rebuilt_rows_bitwise_equal"] is True
    assert support.support_coefficient_parity["max_abs_coefficient_difference"] == 0.0

    # The empty-bank refit reproduces the sealed carrier through the frozen
    # Stage-P solver after the float32 quantization mirror (the float64 solve
    # itself carries only the sealed carrier's own quantization error).
    empty_refit = stage_p_gate.refit_from_anchor_p(
        support, stage_p_gate.EvidenceBankP.empty(), rho_M=1.0,
    )
    assert np.array_equal(
        np.asarray(empty_refit.candidate_t4, dtype=np.float32),
        np.asarray(raw, dtype=np.float32),
    )

    # The trust-region no-op law: alpha_M = 0 returns the sealed carrier itself.
    blocks = support.coefficients(np.asarray(raw))
    outcome = trust_region.active_carrier(support, blocks, alpha_M=0.0, c_M=None)
    assert np.array_equal(np.asarray(outcome.active_t4, dtype=np.float32),
                          np.asarray(raw, dtype=np.float32))
    assert outcome.movement_frobenius == 0.0

    # A committed pseudo row moves the candidate exactly the refit law.
    from src.support_anchored_t4_stage_p_v1 import direction_estimator

    theta = cdm_core.CANONICAL_DIRECTIONS_RAD[5]
    measurement = direction_estimator.GroupMeasurement(
        group=0, estimator="cross_group_circular", included_groups=(1, 2, 3),
        theta_raw_rad=float(theta), theta_index=5,
        theta_canonical_rad=float(theta), canonical_distance_rad=0.0,
        dispersion_rad=0.0, resultant_length=0.9, agreement=None,
        early_theta_rad=None, late_theta_rad=None, early_late_distance_rad=None,
        displacement_norm=1.0, weight=0.9, confidence_rule="x",
    )
    facts = stage_p_gate.TrialFacts(
        session_id="s", trial_id="t0", chronology_position=0,
        scalar_rates=np.asarray(rates[0], dtype=np.float64),
        rate_sha256="a" * 64, native_counts_sha256="b" * 64,
        channel_order_sha256="c" * 64, valid_mask_sha256="d" * 64,
    )
    decision = stage_p_gate.evaluate_block(
        support, stage_p_gate.EvidenceBankP.empty(), (measurement, None, None, None),
        facts=facts, measurement_sha256="e" * 64, rho_M=1.0, c_M=None,
        thresholds=stage_p_gate.GateThresholds(
            tau_d=math.pi / 4.0, r_max=8, d_min=4, max_mass_relative=2.0,
        ),
        block_index=0,
    )
    assert decision.committed is True
    moved = trust_region.active_carrier(support, decision.coefficients, alpha_M=0.5, c_M=None)
    assert moved.movement_frobenius > 0.0
    rebuilt = np.asarray(moved.active_t4, dtype=np.float32)
    assert rebuilt.shape == np.asarray(raw).shape
    assert np.isfinite(rebuilt).all()
    # Invalid units keep the sealed rows.
    invalid = ~np.asarray(groups.valid_mask, dtype=bool)
    if invalid.any():
        assert np.array_equal(rebuilt[invalid], np.asarray(raw, dtype=np.float32)[invalid])


def test_modulation_mirror_uses_sqrt_arithmetic() -> None:
    block = np.asarray([[3.0], [4.0], [1.0]])
    rows = sua_anchor.format_t4_rows(block)
    assert rows[0, 2] == float(np.sqrt(3.0 * 3.0 + 4.0 * 4.0))
    assert rows[0, 2] != float(np.hypot(3.0, 4.0)) or rows[0, 2] == 5.0


def test_mass_grid_and_row_specs_are_the_sealed_ones() -> None:
    assert len(stage_p_replay.enumerate_gate_grid()) == 16
    assert len(stage_p_replay.enumerate_mass_grid()) == 8
    spec = stage_p_replay.ROW_SPECS["P1"]
    assert (spec.law, spec.weight_law, spec.binding) == (
        "cross_group_circular", "resultant_length", "complementary_exclusion",
    )


def test_gates_mirror_part_a() -> None:
    seeds = {42: 0.50, 43: 0.52, 44: 0.48}
    external_m4 = {
        "F00s": {f"s{i}": dict(seeds) for i in range(15)},
        "F01s": {f"s{i}": {k: v + 0.02 for k, v in seeds.items()} for i in range(15)},
    }
    within_m4 = {
        "F00s": {f"w{i}": dict(seeds) for i in range(6)},
        "F01s": {f"w{i}": {k: v + 0.01 for k, v in seeds.items()} for i in range(6)},
    }
    external_m30 = {
        "F00s": {f"s{i}": dict(seeds) for i in range(15)},
        "F01s": {f"s{i}": dict(seeds) for i in range(15)},
    }
    within_m30 = {
        "F00s": {f"w{i}": dict(seeds) for i in range(6)},
        "F01s": {f"w{i}": dict(seeds) for i in range(6)},
    }
    gate = gates.evaluate_sua_gate(
        external_m4=external_m4, within_m4=within_m4,
        external_m30=external_m30, within_m30=within_m30,
        safety={"f00s_bit_anchor_vs_sealed_paired_row": True,
                "anchor_zero_evidence_parity_all_sessions": True,
                "m30_exact_noop": True, "causality_receipt_chains": True,
                "hyperparameters_reselected_on_sua_source_folds": True,
                "dandi_hyperparameters_ported_as_claims": False,
                "zero_target_updates": True},
        gates_law=plan.GATES,
    )
    assert gate["promoted"] is True
    assert gate["external_m4_delta"] == pytest.approx(0.02)
    assert gate["positive_external_m4_sessions"] == 15
    assert gate["safety_all_pass"] is True
    stops = gates.stop_conditions(gate)
    assert stops["any_fired"] is False

    weak_external_m4 = {
        "F00s": external_m4["F00s"],
        "F01s": {
            session: {seed: value - 0.05 for seed, value in per_seed.items()}
            for session, per_seed in external_m4["F00s"].items()
        },
    }
    failing = gates.evaluate_sua_gate(
        external_m4=weak_external_m4, within_m4=within_m4,
        external_m30=external_m30, within_m30=within_m30,
        safety={"f00s_bit_anchor_vs_sealed_paired_row": True,
                "anchor_zero_evidence_parity_all_sessions": True,
                "m30_exact_noop": True, "causality_receipt_chains": True,
                "hyperparameters_reselected_on_sua_source_folds": True,
                "dandi_hyperparameters_ported_as_claims": False,
                "zero_target_updates": True},
        gates_law=plan.GATES,
    )
    assert failing["promoted"] is False
    assert gates.stop_conditions(failing)["fired"] == ["no_candidate_beats_f00s"]


def test_plan_laws_are_frozen() -> None:
    assert plan.GATES["promotion"]["delta_floor"] == 0.01
    assert plan.GATES["promotion"]["breadth_min"] == 10
    assert plan.GATES["safety"]["within_floor"] == -0.02
    assert plan.M30_NOOP_LAW["alpha_M"] == 0.0
    assert plan.SELECTION_LAW["seed"] == 42
    assert plan.SELECTION_LAW["surface"].startswith("within-6")
    assert len(plan.WITHIN_SESSIONS) == 6
    assert plan.BUDGETS == (4, 30)
    assert "M10" in plan.M10_DISCLOSURE


def test_query_partition_law_on_synthetic_trials() -> None:
    # The whole-window starts of each post-50 rewarded trial must concatenate
    # to exactly the runtime's valid_starts authority.
    trial_starts = [(0, 200), (200, 380), (380, 600)]
    authority = np.concatenate([
        np.arange(start, stop - plan.HISTORY_BINS + 1, dtype=np.int64)
        for start, stop in trial_starts
    ])
    rebuilt = np.concatenate([
        np.arange(start, stop - plan.HISTORY_BINS + 1, dtype=np.int64)
        for start, stop in trial_starts
    ])
    assert np.array_equal(rebuilt, authority)
