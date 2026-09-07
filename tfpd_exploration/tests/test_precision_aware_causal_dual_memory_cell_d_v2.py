"""Focused synthetic/no-data gates for Precision-Aware CDM-D V2.

The suite constructs only typed in-memory CDM-D capabilities. It does not
open source/target data, result roots, checkpoints, CUDA, or Torch.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_v1 import plan as v1_plan  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_v1 import transition as v1_transition  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_v2 import physical, plan, transition  # noqa: E402


SESSION = "precision-aware-v2-synthetic-session"


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
    channels = np.arange(900, 900 + units, dtype=np.int64)
    valid = np.ones((units,), dtype=np.bool_)
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
        valid_mask=np.ones((6,), dtype=np.bool_) if valid else np.zeros((6,), dtype=np.bool_),
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
) -> tuple[core.IndependentActivityCausalDualMemory, transition.SupportConditionalPosterior | None, dict[str, np.ndarray]]:
    table = _table(budget=budget, noise_scale=noise_scale)
    initial = core.fit_carriers_from_trial_table(
        table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
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
    posterior = (
        transition.SupportConditionalPosterior.from_support_only_fixed_ridge(
            support_rates=table["rates"],
            support_direction_indices=table["directions"],
            valid_mask=table["valid"],
            groups_sha256=carrier.groups.digest,
        ) if budget in plan.TRANSITION_BUDGETS else None
    )
    return memory, posterior, table


def _expected_direction_rate(memory: core.IndependentActivityCausalDualMemory, *, direction: int = 0) -> np.ndarray:
    theta = core.CANONICAL_DIRECTIONS_RAD[direction]
    active = np.asarray(memory.state.carrier.active_t4, dtype=np.float64)
    result = active[:, 3] + math.cos(theta) * active[:, 0] + math.sin(theta) * active[:, 1]
    assert np.all(result > 0.0)
    return result


def _observe(
    wrapper: transition.PrecisionAwareIndependentActivityV2,
    table: dict[str, np.ndarray],
    *,
    budget: int,
    trial_id: str,
    valid_prediction: bool = True,
    rate_scale: float = 1.0,
) -> transition.PrecisionTransitionOutcomeV2:
    rates = _expected_direction_rate(wrapper.memory) * rate_scale
    return wrapper.observe_and_commit(
        budget=budget,
        b3s_trial_activity=_b3s(table, rates, trial_id),
        carrier_trial_counts=_native(table, rates, trial_id),
        complementary_predictions=_predictions(trial_id, valid=valid_prediction),
    )


def _candidate_at_standardized_distance(
    posterior: transition.SupportConditionalPosterior, *, fraction_of_threshold: float,
) -> np.ndarray:
    reference = np.asarray(posterior.frozen_initial_active_t4, dtype=np.float64)
    candidate = reference.copy()
    threshold = plan.bonferroni_chi2_df2_threshold(int(posterior.valid_mask.sum()))
    for index in np.flatnonzero(posterior.valid_mask):
        chol = np.linalg.cholesky(posterior.posterior_covariance_ac[int(index)])
        candidate[int(index), :2] += chol @ np.asarray([math.sqrt(threshold * fraction_of_threshold), 0.0])
    return candidate


def test_static_plan_closure_and_dry_cli_are_torch_free() -> None:
    assert "torch" not in sys.modules
    dry = plan.dry_plan()
    assert dry["cell"] == plan.CELL
    assert dry["conditional_posterior"]["not_sampling_sandwich_covariance"] is True
    assert dry["conditional_posterior"]["reference"] == "frozen_support_only_initial_fixed_ridge_not_current_active"
    closure = plan.implementation_closure(ROOT).payload()
    assert closure["paths"][0]["path"] == "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py"
    assert closure["paths"][-1]["path"] == "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_v2.py"
    assert plan.validate_implementation_closure(closure) == closure
    forged = copy.deepcopy(closure)
    forged["paths"][1]["sha256"] = "0" * 64
    with pytest.raises(plan.PrecisionAwareV2PlanError, match="canonical|workorder"):
        plan.validate_implementation_closure(forged)
    script = ROOT / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_v2.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dry"], cwd=ROOT, check=True, text=True, capture_output=True,
        env={
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"],
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["public_cli_imports_runtime"] is False and payload["execution_capability_issued"] is False
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from src.precision_aware_causal_dual_memory_cell_d_v2 import plan; "
            "print('TORCH_PRESENT='+str('torch' in sys.modules)); print(plan.CELL)",
        ],
        cwd=ROOT, check=True, text=True, capture_output=True,
        env={
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{ROOT / 'tfpd_exploration'}:{ROOT / 'tfpd_exploration/src'}", "PATH": os.environ["PATH"],
        },
    )
    assert "TORCH_PRESENT=False" in probe.stdout


@pytest.mark.parametrize("budget", (4, 10))
def test_exact_conditional_gaussian_ridge_posterior_formula_and_not_sandwich(budget: int) -> None:
    _memory_value, posterior, table = _memory(budget=budget, noise_scale=0.35)
    assert posterior is not None
    x = np.asarray(posterior.design, dtype=np.float64)
    a = x.T @ x + np.diag((budget * 0.1, budget * 0.1, 0.0))
    inverse = np.linalg.inv(a)
    beta = np.linalg.solve(a, x.T @ table["rates"])
    residual = table["rates"] - x @ beta
    degrees = float(budget - np.trace(x @ inverse @ x.T))
    sigma2 = np.maximum(np.sum(residual * residual, axis=0) / degrees, plan.POSTERIOR_VARIANCE_FLOOR)
    expected = np.einsum("u,ij->uij", sigma2, inverse)[:, :2, :2]
    expected[~table["valid"]] = np.nan
    assert np.allclose(posterior.posterior_covariance_ac, expected, rtol=1.0e-12, atol=1.0e-14, equal_nan=True)
    sandwich = np.einsum("u,ij->uij", sigma2, inverse @ (x.T @ x) @ inverse)[:, :2, :2]
    sandwich[~table["valid"]] = np.nan
    assert not np.allclose(posterior.posterior_covariance_ac[table["valid"]], sandwich[table["valid"]])
    assert posterior.payload()["covariance_estimand"] == "conditional_gaussian_ridge_posterior_sigma2_A_inverse"
    assert posterior.payload()["sampling_sandwich_covariance_used"] is False
    with pytest.raises(transition.PrecisionAwareV2TransitionError, match="conditional Gaussian-ridge posterior"):
        dataclasses.replace(posterior, posterior_covariance_ac=sandwich)


def test_bonferroni_threshold_is_exact_n_dependent_familywise_law() -> None:
    assert plan.bonferroni_chi2_df2_threshold(4) == -2.0 * math.log(0.05 / 4.0)
    assert plan.bonferroni_chi2_df2_threshold(8) == -2.0 * math.log(0.05 / 8.0)
    assert plan.bonferroni_chi2_df2_threshold(8) > plan.bonferroni_chi2_df2_threshold(4)
    with pytest.raises(plan.PrecisionAwareV2PlanError, match="at least four"):
        plan.bonferroni_chi2_df2_threshold(3)
    _memory_value, posterior, _table_value = _memory(budget=4, noise_scale=0.5)
    assert posterior is not None
    candidate = _candidate_at_standardized_distance(posterior, fraction_of_threshold=0.5)
    decision = transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=posterior.frozen_initial_active_t4, proposed_active_t4=candidate,
    )
    assert decision.threshold == plan.bonferroni_chi2_df2_threshold(int(posterior.valid_mask.sum()))
    assert decision.accepted is True


def test_high_precision_is_monotonically_stricter_for_identical_physical_delta() -> None:
    _high_memory, high, _high_table = _memory(budget=4, noise_scale=0.0)
    _low_memory, low, _low_table = _memory(budget=4, noise_scale=5.0)
    assert high is not None and low is not None
    high_candidate = _candidate_at_standardized_distance(high, fraction_of_threshold=1.25)
    delta = high_candidate[:, :2] - high.frozen_initial_active_t4[:, :2]
    low_candidate = np.asarray(low.frozen_initial_active_t4, dtype=np.float64).copy()
    low_candidate[:, :2] += delta
    high_decision = transition.decide_against_frozen_support(
        high, frozen_initial_active_t4=high.frozen_initial_active_t4, proposed_active_t4=high_candidate,
    )
    low_decision = transition.decide_against_frozen_support(
        low, frozen_initial_active_t4=low.frozen_initial_active_t4, proposed_active_t4=low_candidate,
    )
    assert high_decision.accepted is False
    assert low_decision.accepted is True
    assert high_decision.max_mahalanobis_squared > low_decision.max_mahalanobis_squared


def test_v2_frozen_support_bound_rejects_cumulative_ratchet() -> None:
    _memory_value, posterior, _table_value = _memory(budget=4, noise_scale=0.8)
    assert posterior is not None
    first = _candidate_at_standardized_distance(posterior, fraction_of_threshold=0.20)
    reference = np.asarray(posterior.frozen_initial_active_t4, dtype=np.float64)
    increment = first[:, :2] - reference[:, :2]
    cumulative = reference.copy()
    cumulative[:, :2] += 3.0 * increment
    first_decision = transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=reference, proposed_active_t4=first,
    )
    cumulative_decision = transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=reference, proposed_active_t4=cumulative,
    )
    assert first_decision.accepted is True
    assert cumulative_decision.accepted is False
    with pytest.raises(transition.PrecisionAwareV2TransitionError, match="not exact frozen support"):
        transition.decide_against_frozen_support(
            posterior, frozen_initial_active_t4=first, proposed_active_t4=cumulative,
        )


def test_v1_current_relative_per_step_ratchet_is_reproduced_but_v2_rejects_cumulative_state() -> None:
    _memory_value, posterior, table = _memory(budget=4, noise_scale=0.7)
    assert posterior is not None
    legacy = v1_transition.SupportPrecision.from_support_only_fixed_ridge(
        support_rates=table["rates"], support_direction_indices=table["directions"], valid_mask=table["valid"],
    )
    reference = np.asarray(posterior.frozen_initial_active_t4, dtype=np.float64)
    assert np.array_equal(np.asarray(legacy.fixed_ridge_t4, dtype=np.float64), reference)
    # The constant belongs to V1's frozen plan, not a V2 tuning knob.
    increment = np.zeros((reference.shape[0], 2), dtype=np.float64)
    for index in np.flatnonzero(table["valid"]):
        chol = np.linalg.cholesky(legacy.covariance_ac[int(index)])
        increment[int(index)] = chol @ np.asarray([math.sqrt(0.25 * v1_plan.CREDIBLE_REGION_CHI2_DF2_95), 0.0])
    current = reference.copy()
    for _step in range(8):
        next_candidate = current.copy()
        next_candidate[:, :2] += increment
        legacy_decision = v1_transition.decide_credible_region(
            legacy, current_active_t4=current, proposed_active_t4=next_candidate,
        )
        assert legacy_decision.accepted is True
        current = next_candidate
    v2_decision = transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=reference, proposed_active_t4=current,
    )
    assert v2_decision.accepted is False
    assert v2_decision.max_mahalanobis_squared > v2_decision.threshold


def test_invalid_rows_and_channel_group_permutation_remain_equivariant() -> None:
    _memory_value, posterior, table = _memory(budget=10, noise_scale=1.2)
    assert posterior is not None
    candidate = _candidate_at_standardized_distance(posterior, fraction_of_threshold=0.4)
    original = transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=posterior.frozen_initial_active_t4, proposed_active_t4=candidate,
    )
    assert np.isnan(original.mahalanobis_squared[~table["valid"]]).all()
    permutation = np.asarray((7, 3, 0, 5, 1, 8, 2, 6, 4), dtype=np.int64)
    permuted = transition.SupportConditionalPosterior.from_support_only_fixed_ridge(
        support_rates=table["rates"][:, permutation], support_direction_indices=table["directions"],
        valid_mask=table["valid"][permutation],
    )
    candidate_permuted = candidate[permutation]
    decision_permuted = transition.decide_against_frozen_support(
        permuted, frozen_initial_active_t4=permuted.frozen_initial_active_t4, proposed_active_t4=candidate_permuted,
    )
    inverse = np.argsort(permutation)
    assert decision_permuted.accepted == original.accepted
    assert np.allclose(decision_permuted.mahalanobis_squared[inverse][table["valid"]],
                       original.mahalanobis_squared[table["valid"]], rtol=0.0, atol=1.0e-10)
    groups_a = core.build_complementary_groups(posterior.fixed_ridge_t4, table["channels"], valid_mask=table["valid"])
    groups_b = core.build_complementary_groups(
        permuted.fixed_ridge_t4, table["channels"][permutation], valid_mask=table["valid"][permutation],
    )
    assert groups_a.channel_to_group() == groups_b.channel_to_group()


def test_m30_is_literal_state_prediction_and_proposal_noop() -> None:
    memory, posterior, _table_value = _memory(budget=30, noise_scale=0.2)
    assert posterior is None
    wrapper = transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=None)
    before_state = memory.state.digest
    before_prediction = transition.PrecisionAwareIndependentActivityV2._prediction_digest(memory.read_prediction_inputs())
    outcome = wrapper.observe_and_commit(
        budget=30, b3s_trial_activity=object(), carrier_trial_counts=object(), complementary_predictions=(),
    )
    assert outcome.m30_deployment_noop is True and outcome.core_outcome is None and outcome.posterior_decision is None
    assert outcome.state_before_sha256 == outcome.state_after_sha256 == before_state == memory.state.digest
    assert outcome.prediction_before_sha256 == outcome.prediction_after_sha256 == before_prediction
    assert memory.state.activity.query_count == 0 and memory.state.committed_query_trials == 0


def test_v2_causality_preserves_activity_on_precision_rejection_and_core_reason_order() -> None:
    memory, posterior, table = _memory(budget=4, noise_scale=0.0)
    assert posterior is not None
    wrapper = transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
    pre = memory.read_prediction_inputs()
    before_carrier = memory.state.carrier.digest
    before_activity = memory.state.activity.digest
    rejected = _observe(wrapper, table, budget=4, trial_id="precision-reject", rate_scale=1.30)
    assert rejected.core_outcome is not None and rejected.posterior_decision is not None
    assert rejected.posterior_decision.accepted is False
    assert rejected.core_outcome.activity_transition_committed is True
    assert rejected.core_outcome.carrier_transition_committed is False
    assert rejected.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION
    assert rejected.state_before_sha256 == pre.state_digest
    assert rejected.core_outcome.carrier_before_sha256 == rejected.core_outcome.carrier_after_sha256 == before_carrier
    assert rejected.core_outcome.activity_before_sha256 == before_activity != rejected.core_outcome.activity_after_sha256
    assert memory.read_prediction_inputs().state_digest == rejected.state_after_sha256
    # A core velocity rejection retains its own reason and still independently appends activity; V2 does not relabel it.
    core_rejection = _observe(wrapper, table, budget=4, trial_id="base-reject", valid_prediction=False)
    assert core_rejection.core_outcome is not None and core_rejection.posterior_decision is None
    assert core_rejection.core_outcome.activity_transition_committed is True
    assert core_rejection.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.MOVEMENT_TOO_SHORT


def test_accepted_carrier_changes_only_the_next_query_state() -> None:
    memory, posterior, table = _memory(budget=4, noise_scale=4.0)
    assert posterior is not None
    wrapper = transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
    pre_query = memory.read_prediction_inputs()
    accepted = _observe(wrapper, table, budget=4, trial_id="accepted-next-state", rate_scale=1.0)
    assert accepted.core_outcome is not None and accepted.posterior_decision is not None
    assert accepted.posterior_decision.accepted is True
    assert accepted.core_outcome.carrier_transition_committed is True
    assert accepted.state_before_sha256 == pre_query.state_digest
    assert accepted.state_after_sha256 == memory.state.digest != pre_query.state_digest
    # The pre-observation snapshot is immutable and is the only state a
    # forward for this completed trial may have used; the changed carrier is
    # visible only to the following query.
    assert pre_query.state_digest != memory.read_prediction_inputs().state_digest
    with pytest.raises(transition.PrecisionAwareV2TransitionError, match="frozen support-only fixed-ridge initializer"):
        transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)


def test_v2_physical_seam_binds_only_sealed_authority_support_and_post_first30_chronology() -> None:
    ordered = tuple(f"trial-{index:02d}" for index in range(60))
    chronology = physical.SessionChronologyV2(
        budget=4,
        ordered_trial_ids=ordered,
        support_trial_ids=tuple(ordered[index] for index in (1, 5, 11, 18)),
        query_trial_ids=tuple(ordered[index] for index in range(30, 36)),
        support_positions=(1, 5, 11, 18),
        query_positions=tuple(range(30, 36)),
    )
    binding = physical.SealedPhysicalBindingV2(
        sealed_cell_d_authority=plan.SEALED_CELL_D_AUTHORITY.payload(),
        accepted_independent_activity_terminal_sha256="a" * 64,
        independent_activity_contract_sha256="b" * 64,
        source_input_authority_sha256="c" * 64,
    )
    memory, posterior, table = _memory(budget=4, noise_scale=1.0)
    assert posterior is not None
    adapter = physical.PrecisionAwarePhysicalAdapterV2(binding=binding)
    wrapper = adapter.bind_session(
        chronology=chronology, memory=memory, support_rates=table["rates"], support_direction_indices=table["directions"],
    )
    assert isinstance(wrapper, transition.PrecisionAwareIndependentActivityV2)
    assert adapter.assert_no_live_mutation(
        before_model_state_sha256="d" * 64, after_model_state_sha256="d" * 64,
        before_normalizer_sha256="e" * 64, after_normalizer_sha256="e" * 64,
        target_gradient_or_update_count=0,
    )["decoder_precision_token_used"] is False
    with pytest.raises(physical.PrecisionAwareV2PhysicalError, match="post-first30"):
        physical.SessionChronologyV2(
            budget=4, ordered_trial_ids=ordered, support_trial_ids=tuple(ordered[index] for index in (1, 5, 11, 18)),
            query_trial_ids=(ordered[29],), support_positions=(1, 5, 11, 18), query_positions=(29,),
        )


def test_v2_uses_frozen_reference_not_current_active_in_actual_wrapper_source() -> None:
    source = inspect.getsource(transition.PrecisionAwareIndependentActivityV2.observe_and_commit)
    assert "frozen_initial_active_t4=self._frozen_initial_active_t4" in source
    assert "current_active_t4" not in source
    assert "observe_completed_trial" in source and "commit_independent" in source
    assert source.index("observe_completed_trial") < source.index("decide_against_frozen_support") < source.index("commit_independent")
    assert "PRECISION_CREDIBLE_REGION" in inspect.getsource(transition.PrecisionAwareIndependentActivityV2._precision_rejected_pending)
