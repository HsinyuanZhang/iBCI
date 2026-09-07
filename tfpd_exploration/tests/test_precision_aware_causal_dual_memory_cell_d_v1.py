"""Focused synthetic/no-data gates for Precision-Aware CDM-D V1.

These tests deliberately construct only typed in-memory CDM-D capabilities.
They never resolve source or target paths, load a checkpoint, import torch, or
initialize CUDA.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_v1 import physical, plan, transition  # noqa: E402


SESSION = "precision-aware-synthetic-session"


def _table(*, budget: int, noise_scale: float = 0.0) -> dict[str, np.ndarray]:
    directions = np.resize(np.asarray((0, 2, 4, 6, 1, 3, 5, 7), dtype=np.int64), budget)
    units = 9
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[int(item)] for item in directions], dtype=np.float64)
    a = np.linspace(0.7, 1.5, units, dtype=np.float64)
    c = np.linspace(-0.6, 0.6, units, dtype=np.float64)
    b = np.linspace(8.0, 11.0, units, dtype=np.float64)
    clean = b[None, :] + np.cos(theta)[:, None] * a[None, :] + np.sin(theta)[:, None] * c[None, :]
    pattern = np.sin(np.arange(budget * units, dtype=np.float64).reshape(budget, units) * 1.732)
    rates = np.ascontiguousarray(clean + noise_scale * pattern, dtype=np.float64)
    channels = np.arange(700, 700 + units, dtype=np.int64)
    valid = np.ones(units, dtype=np.bool_)
    valid[-1] = False
    return {"directions": directions, "rates": rates, "channels": channels, "valid": valid}


def _config(budget: int) -> core.CDMDConfig:
    return core.CDMDConfig(
        support_budget_m=budget,
        dt=0.02,
        minimum_movement_bins=2,
        minimum_displacement=0.001,
        minimum_mean_speed=0.001,
        max_canonical_distance_rad=math.pi / 8.0,
        max_group_direction_disagreement_rad=math.pi / 8.0,
        minimum_accepted_evidence=3,
        active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )


def _b3s(table: dict[str, np.ndarray], rates: np.ndarray, trial_id: str) -> core.B3SInterpolatedSpikeCountTrial:
    return core.B3SInterpolatedSpikeCountTrial(
        activity=np.repeat(np.asarray(rates, dtype=np.float32)[None, :], 100, axis=0),
        session_id=SESSION,
        trial_id=trial_id,
        channel_order_sha256=core.channel_order_digest(table["channels"]),
    )


def _native(table: dict[str, np.ndarray], rates: np.ndarray, trial_id: str) -> core.NativeRewardedTrialSpikeCounts:
    return core.NativeRewardedTrialSpikeCounts(
        counts=np.repeat((np.asarray(rates, dtype=np.float64) * 0.020)[None, :], 7, axis=0),
        session_id=SESSION,
        trial_id=trial_id,
        channel_order_sha256=core.channel_order_digest(table["channels"]),
        rewarded_interval_start_bin=10,
        rewarded_interval_stop_bin=17,
    )


def _predictions(trial_id: str, *, direction: int = 0, valid: bool = True) -> tuple[core.CompletedVelocityPrediction, ...]:
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(6, dtype=np.bool_) if valid else np.zeros(6, dtype=np.bool_),
        session_id=SESSION,
        trial_id=trial_id,
        prediction_interval_start_bin=20,
        prediction_interval_stop_bin=26,
    )
    theta = core.CANONICAL_DIRECTIONS_RAD[direction]
    velocity = np.repeat(np.asarray([[math.cos(theta), math.sin(theta)]], dtype=np.float64), 6, axis=0)
    return tuple(core.CompletedVelocityPrediction(velocity=velocity, validity=validity) for _ in range(core.GROUP_COUNT))


def _memory(
    *, budget: int, noise_scale: float = 0.0,
) -> tuple[core.IndependentActivityCausalDualMemory, transition.SupportPrecision | None, dict[str, np.ndarray]]:
    table = _table(budget=budget, noise_scale=noise_scale)
    initial = core.fit_carriers_from_trial_table(
        table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=initial,
        channel_ids=table["channels"],
        support_trial_rates=table["rates"],
        support_direction_indices=table["directions"],
        config=_config(budget),
        valid_mask=table["valid"],
    )
    activity = core.ActivityMemory.initialize(
        tuple(_b3s(table, row, f"support-{index}") for index, row in enumerate(table["rates"])),
        channel_ids=table["channels"],
        fifo_capacity=int(carrier.config.activity_fifo_capacity),
    )
    memory = core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)
    precision = (
        transition.SupportPrecision.from_support_only_fixed_ridge(
            support_rates=table["rates"],
            support_direction_indices=table["directions"],
            valid_mask=table["valid"],
            groups_sha256=carrier.groups.digest,
        ) if budget in plan.TRANSITION_BUDGETS else None
    )
    return memory, precision, table


def _expected_direction_rate(memory: core.IndependentActivityCausalDualMemory, *, direction: int = 0) -> np.ndarray:
    theta = core.CANONICAL_DIRECTIONS_RAD[direction]
    active = np.asarray(memory.state.carrier.active_t4, dtype=np.float64)
    result = active[:, 3] + math.cos(theta) * active[:, 0] + math.sin(theta) * active[:, 1]
    assert np.all(result > 0.0)
    return result


def _observe(wrapper: transition.PrecisionAwareIndependentActivity, table: dict[str, np.ndarray], *, budget: int,
             trial_id: str, valid_prediction: bool = True, rate_scale: float = 1.0) -> transition.PrecisionTransitionOutcome:
    rates = _expected_direction_rate(wrapper.memory) * rate_scale
    return wrapper.observe_and_commit(
        budget=budget,
        b3s_trial_activity=_b3s(table, rates, trial_id),
        carrier_trial_counts=_native(table, rates, trial_id),
        complementary_predictions=_predictions(trial_id, valid=valid_prediction),
    )


def test_static_plan_and_public_dry_cli_are_no_runtime() -> None:
    assert "torch" not in sys.modules
    dry = plan.dry_plan()
    assert dry["cell"] == plan.CELL
    assert dry["m30_deployment_noop"] is True
    assert dry["credible_region"]["typed_rejection_reason"] == "precision_credible_region"
    assert dry["decoder_contract"] == {
        "precision_token_to_decoder": False,
        "posterior_or_target_refit": False,
        "model_parameter_changes": False,
        "normalizer_changes": False,
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_v1.py"), "--dry"],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"public_cli_imports_runtime":false' in result.stdout
    assert "torch" not in sys.modules


def test_closure_is_explicit_bound_and_rejects_canonical_drift() -> None:
    payload = plan.implementation_closure(ROOT).payload()
    assert payload["paths"][0]["path"] == "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py"
    assert payload["paths"][-1]["path"] == "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_v1.py"
    assert plan.validate_implementation_closure(payload) == payload
    forged = copy.deepcopy(payload)
    forged["paths"][1]["sha256"] = "0" * 64
    with pytest.raises(plan.PrecisionAwarePlanError, match="canonical|workorder"):
        plan.validate_implementation_closure(forged)


@pytest.mark.parametrize("budget", (4, 10))
def test_float64_support_covariance_has_exact_fixed_ridge_parity_and_invalid_topology(budget: int) -> None:
    memory, precision, table = _memory(budget=budget, noise_scale=0.25)
    assert precision is not None
    expected = core.fit_carriers_from_trial_table(
        table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    assert precision.fixed_ridge_t4.dtype == np.float32
    assert np.array_equal(precision.fixed_ridge_t4, expected)
    core.assert_production_parity(
        precision.fixed_ridge_t4, table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    assert np.array_equal(memory.state.carrier.groups.valid_mask, table["valid"])
    assert np.isfinite(precision.covariance_ac[table["valid"]]).all()
    assert np.isnan(precision.covariance_ac[~table["valid"]]).all()
    assert np.isnan(precision.residual_variance[~table["valid"]]).all()
    assert precision.payload()["decoder_token_used"] is False
    assert precision.payload()["pseudo_labels_used"] is False
    with pytest.raises(transition.PrecisionTransitionError, match="closed form"):
        dataclasses.replace(precision, covariance_ac=np.asarray(precision.covariance_ac) * 2.0)


def test_high_precision_is_monotonically_stricter_than_identical_low_precision_delta() -> None:
    _, high, _ = _memory(budget=4, noise_scale=0.0)
    _, low, _ = _memory(budget=4, noise_scale=4.0)
    assert high is not None and low is not None
    current_high = np.asarray(high.fixed_ridge_t4, dtype=np.float64)
    proposed_high = current_high.copy()
    proposed_high[high.valid_mask, 0] += 0.40
    current_low = np.asarray(low.fixed_ridge_t4, dtype=np.float64)
    proposed_low = current_low.copy()
    proposed_low[low.valid_mask, 0] += 0.40
    high_decision = transition.decide_credible_region(
        high, current_active_t4=current_high, proposed_active_t4=proposed_high,
    )
    low_decision = transition.decide_credible_region(
        low, current_active_t4=current_low, proposed_active_t4=proposed_low,
    )
    assert high_decision.accepted is False
    assert low_decision.accepted is True
    assert high_decision.max_mahalanobis_squared > low_decision.max_mahalanobis_squared
    assert np.isnan(high_decision.mahalanobis_squared[~high.valid_mask]).all()


def test_precision_wrapper_preserves_independent_activity_on_precision_rejection() -> None:
    memory, precision, table = _memory(budget=4, noise_scale=0.0)
    assert precision is not None
    wrapper = transition.PrecisionAwareIndependentActivity(memory=memory, precision=precision)
    before_carrier = memory.state.carrier.digest
    before_activity = memory.state.activity.digest
    result = _observe(wrapper, table, budget=4, trial_id="precision-reject", rate_scale=1.15)
    assert result.core_outcome is not None and result.decision is not None
    assert result.decision.accepted is False
    assert result.core_outcome.activity_transition_committed is True
    assert result.core_outcome.carrier_transition_committed is False
    assert result.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION
    assert result.core_outcome.carrier_before_sha256 == result.core_outcome.carrier_after_sha256 == before_carrier
    assert result.core_outcome.activity_before_sha256 == before_activity != result.core_outcome.activity_after_sha256
    assert memory.state.committed_query_trials == 1


@pytest.mark.parametrize("budget", (4, 10))
def test_accepted_transition_changes_only_next_query_state_and_retains_core_gate_order(budget: int) -> None:
    memory, precision, table = _memory(budget=budget, noise_scale=4.0)
    assert precision is not None
    wrapper = transition.PrecisionAwareIndependentActivity(memory=memory, precision=precision)
    pre = memory.read_prediction_inputs()
    result = _observe(wrapper, table, budget=budget, trial_id=f"accepted-m{budget}")
    assert result.core_outcome is not None and result.decision is not None
    assert result.core_outcome.activity_transition_committed is True
    assert result.core_outcome.carrier_transition_committed is True
    assert result.decision.accepted is True
    assert result.state_before_sha256 == pre.state_digest
    assert result.state_after_sha256 == memory.state.digest != pre.state_digest
    assert memory.read_prediction_inputs().state_digest != pre.state_digest
    # An earlier base B8/velocity rejection is retained rather than relabelled
    # as a precision failure, while activity still advances under V5 semantics.
    rejected = _observe(wrapper, table, budget=budget, trial_id=f"base-reject-m{budget}", valid_prediction=False)
    assert rejected.core_outcome is not None and rejected.decision is not None
    assert rejected.core_outcome.activity_transition_committed is True
    assert rejected.core_outcome.carrier_transition_committed is False
    assert rejected.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.MOVEMENT_TOO_SHORT
    assert rejected.decision.accepted is True  # zero state delta, not a relabelled precision rejection


def test_m30_is_literal_state_prediction_and_proposal_noop() -> None:
    memory, _unused_precision, table = _memory(budget=30, noise_scale=0.2)
    assert _unused_precision is None
    wrapper = transition.PrecisionAwareIndependentActivity(memory=memory, precision=None)
    before = memory.state.digest
    before_prediction = transition.PrecisionAwareIndependentActivity._prediction_digest(memory.read_prediction_inputs())
    outcome = wrapper.observe_and_commit(
        budget=30,
        b3s_trial_activity=object(),
        carrier_trial_counts=object(),
        complementary_predictions=(),
    )
    assert outcome.m30_deployment_noop is True
    assert outcome.core_outcome is None and outcome.decision is None
    assert outcome.state_before_sha256 == outcome.state_after_sha256 == before == memory.state.digest
    assert outcome.prediction_before_sha256 == outcome.prediction_after_sha256 == before_prediction
    assert memory.state.activity.query_count == 0 and memory.state.committed_query_trials == 0
    assert table["rates"].shape[0] == 30
    m4_memory, m4_precision, _m4_table = _memory(budget=4, noise_scale=1.0)
    assert m4_precision is not None
    with pytest.raises(transition.PrecisionTransitionError, match="budget/memory"):
        transition.PrecisionAwareIndependentActivity(memory=m4_memory, precision=m4_precision).observe_and_commit(
            budget=30, b3s_trial_activity=object(), carrier_trial_counts=object(), complementary_predictions=(),
        )


def test_channel_group_permutation_and_invalid_mask_leave_statistic_equivariant() -> None:
    _memory_a, precision, table = _memory(budget=10, noise_scale=1.5)
    assert precision is not None
    current = np.asarray(precision.fixed_ridge_t4, dtype=np.float64)
    proposed = current.copy()
    proposed[precision.valid_mask, 1] += 0.015
    original = transition.decide_credible_region(precision, current_active_t4=current, proposed_active_t4=proposed)
    permutation = np.asarray((7, 3, 0, 5, 1, 8, 2, 6, 4), dtype=np.int64)
    permuted = transition.SupportPrecision.from_support_only_fixed_ridge(
        support_rates=table["rates"][:, permutation],
        support_direction_indices=table["directions"],
        valid_mask=table["valid"][permutation],
    )
    candidate = transition.decide_credible_region(
        permuted,
        current_active_t4=current[permutation],
        proposed_active_t4=proposed[permutation],
    )
    inverse = np.argsort(permutation)
    assert candidate.accepted == original.accepted
    assert np.allclose(candidate.mahalanobis_squared[inverse][table["valid"]],
                       original.mahalanobis_squared[table["valid"]], rtol=0.0, atol=1.0e-10)
    channels = table["channels"]
    groups_a = core.build_complementary_groups(precision.fixed_ridge_t4, channels, valid_mask=table["valid"])
    groups_b = core.build_complementary_groups(
        permuted.fixed_ridge_t4, channels[permutation], valid_mask=table["valid"][permutation],
    )
    assert groups_a.channel_to_group() == groups_b.channel_to_group()


def test_deferred_physical_adapter_enforces_post_first30_surface_and_sealed_invariants() -> None:
    ordered = tuple(f"trial-{index:02d}" for index in range(60))
    chronology = physical.SessionChronology(
        budget=4,
        ordered_trial_ids=ordered,
        support_trial_ids=tuple(ordered[index] for index in (1, 5, 11, 18)),
        query_trial_ids=tuple(ordered[index] for index in range(30, 36)),
        support_positions=(1, 5, 11, 18),
        query_positions=tuple(range(30, 36)),
    )
    binding = physical.SealedPhysicalBinding(
        sealed_authority=plan.SEALED_CELL_D_AUTHORITY.payload(),
        source_gate_terminal_sha256="a" * 64,
        independent_activity_contract_sha256="b" * 64,
    )
    memory, _precision, table = _memory(budget=4, noise_scale=1.0)
    assert _precision is not None
    adapter = physical.PrecisionAwarePhysicalAdapter(binding=binding)
    wrapper = adapter.bind_session(
        chronology=chronology, memory=memory,
        support_rates=table["rates"], support_direction_indices=table["directions"],
    )
    assert isinstance(wrapper, transition.PrecisionAwareIndependentActivity)
    unchanged = adapter.assert_no_live_model_mutation(
        before_model_state_sha256="c" * 64, after_model_state_sha256="c" * 64,
        before_normalizer_sha256="d" * 64, after_normalizer_sha256="d" * 64,
        target_gradient_or_update_count=0,
    )
    assert unchanged["decoder_precision_token_used"] is False
    with pytest.raises(physical.PrecisionAwarePhysicalError, match="post-first30"):
        physical.SessionChronology(
            budget=4, ordered_trial_ids=ordered,
            support_trial_ids=tuple(ordered[index] for index in (1, 5, 11, 18)),
            query_trial_ids=(ordered[29],), support_positions=(1, 5, 11, 18), query_positions=(29,),
        )


def test_authorized_new_reason_leaves_legacy_values_and_default_core_path_unchanged() -> None:
    legacy_values = tuple(item.value for item in core.UpdateRejectionReason if item is not core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION)
    assert legacy_values == (
        "activity_shape", "activity_nonfinite", "trial_capability", "trial_binding", "carrier_counts",
        "velocity_validity", "velocity_shape", "velocity_nonfinite", "movement_mask", "movement_too_short",
        "low_displacement", "low_mean_speed", "canonical_direction_too_far", "complementary_disagreement",
        "insufficient_evidence", "insufficient_design_rank", "ill_conditioned_design", "nonfinite_carrier",
        "departure_freeze", "stale_pending_update",
    )
    memory, precision, table = _memory(budget=4, noise_scale=0.0)
    assert precision is not None
    legacy_pending = memory.observe_completed_trial(
        b3s_trial_activity=_b3s(table, _expected_direction_rate(memory), "legacy"),
        carrier_trial_counts=_native(table, _expected_direction_rate(memory), "legacy"),
        complementary_predictions=_predictions("legacy", valid=False),
    )
    assert legacy_pending.carrier_rejection_reason is core.UpdateRejectionReason.MOVEMENT_TOO_SHORT
    wrapper = transition.PrecisionAwareIndependentActivity(memory=memory, precision=precision)
    outcome = _observe(wrapper, table, budget=4, trial_id="new-wrapper", rate_scale=1.15)
    assert outcome.core_outcome is not None
    assert outcome.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION
    # The successor's explicit closure includes the changed core byte; no
    # historical closure is claimed to certify that current shared byte.
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py" in plan.IMPLEMENTATION_PATHS
    assert "PRECISION_CREDIBLE_REGION" in inspect.getsource(core.UpdateRejectionReason)
