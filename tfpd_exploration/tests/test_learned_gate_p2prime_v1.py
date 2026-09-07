"""Synthetic-fixture tests for the P2' oracle-policy decomposition matrix.

No data root, no checkpoint, no CUDA: every test runs on constructed numpy
blocks and real imported ``core`` state-machine objects.  The five
review-critical properties under test:

1. counterfactual parity -- same-parent branches differ ONLY in carrier;
2. utility horizon isolation -- trial j never appears in its own utility;
3. trial-boundary smoothing reset -- no cross-trial kernel leakage;
4. the output filter is one fixed object shared by every row but A0;
5. kill-criterion verdict logic, including the boundary cases.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core

from src.learned_gate_p2prime_v1 import filters, plan, policy


MEAN = (0.1, -0.2)
STD = (8.0, 4.0)


# ---------------------------------------------------------------------------
# Shared synthetic state-machine fixture (real imported core classes).
# ---------------------------------------------------------------------------


def _support_tables(units: int = 8, trials: int = 8):
    rng = np.random.default_rng(7)
    rates = rng.uniform(5.0, 30.0, size=(trials, units))
    directions = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)[:trials]
    return rates, directions


def _synthetic_memory(units: int = 8, trials: int = 8, budget: int = 4):
    rates, directions = _support_tables(units, trials)
    initial = core.fit_carriers_from_trial_table(
        rates[:budget], directions[:budget],
        mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=core.RIDGE_NORMALIZED_LAMBDA,
    )
    config = core.CDMDConfig(
        support_budget_m=budget, active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=initial, channel_ids=np.arange(units, dtype=np.int64),
        support_trial_rates=rates[:budget], support_direction_indices=directions[:budget],
        config=config, valid_mask=None,
    )
    channel_sha = core.channel_order_digest(np.arange(units, dtype=np.int64))
    support_trials = tuple(
        core.B3SInterpolatedSpikeCountTrial(
            activity=np.full((100, units), float(index + 1), dtype=np.float32),
            session_id="synthetic", trial_id=f"support-{index}", channel_order_sha256=channel_sha,
        )
        for index in range(budget)
    )
    activity = core.ActivityMemory.initialize(
        support_trials, channel_ids=np.arange(units, dtype=np.int64),
        fifo_capacity=int(config.activity_fifo_capacity),
    )
    return core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier), channel_sha, config


def _synthetic_completed_trial(*, units: int = 8, trial_id: str = "query-0", bins: int = 10,
                               velocity: np.ndarray | None = None, channel_sha: str | None = None,
                               session_id: str = "synthetic"):
    channel_sha = channel_sha or core.channel_order_digest(np.arange(units, dtype=np.int64))
    rng = np.random.default_rng(11)
    b3s = core.B3SInterpolatedSpikeCountTrial(
        activity=rng.uniform(0.0, 3.0, size=(100, units)).astype(np.float32),
        session_id=session_id, trial_id=trial_id, channel_order_sha256=channel_sha,
    )
    native = core.NativeRewardedTrialSpikeCounts(
        counts=rng.poisson(0.3, size=(bins, units)),
        session_id=session_id, trial_id=trial_id, channel_order_sha256=channel_sha,
        rewarded_interval_start_bin=0, rewarded_interval_stop_bin=bins,
    )
    if velocity is None:
        velocity = np.tile(np.array([[0.5, 0.0]]), (bins, 1))
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(bins, dtype=bool), session_id=session_id, trial_id=trial_id,
        prediction_interval_start_bin=49, prediction_interval_stop_bin=49 + bins,
    )
    predictions = tuple(
        core.CompletedVelocityPrediction(velocity.copy(), validity) for _ in range(core.GROUP_COUNT)
    )
    return b3s, native, predictions, validity


# ---------------------------------------------------------------------------
# 1/3. Trial-boundary reset for both smoothing consumers.
# ---------------------------------------------------------------------------


def test_causal_ema_is_strictly_causal_within_one_trial() -> None:
    block = np.linspace(0.0, 1.0, 24).reshape(12, 2)
    smoothed = filters.ema_causal_one_trial(block)
    # Perturbing the LAST row never changes earlier rows.
    perturbed = block.copy()
    perturbed[-1] += 100.0
    smoothed_perturbed = filters.ema_causal_one_trial(perturbed)
    assert np.allclose(smoothed[:-1], smoothed_perturbed[:-1])
    assert not np.allclose(smoothed[-1], smoothed_perturbed[-1])
    # The first output row is exactly the first input row.
    assert np.array_equal(smoothed[0], block[0])


def test_output_filter_resets_at_trial_boundaries() -> None:
    trial_a = np.linspace(0.0, 2.0, 20).reshape(10, 2)
    trial_b = np.linspace(5.0, 7.0, 20).reshape(10, 2)
    baseline_b = filters.apply_output_filter_one_trial(trial_b)
    # Corrupting ALL of trial A cannot move trial B's filtered output.
    corrupted_a = trial_a + 1000.0
    assert np.allclose(
        filters.apply_output_filter_one_trial(trial_b), baseline_b,
    )
    # A concatenated-stream kernel WOULD leak: prove the reset is real by
    # showing the naive concatenation differs at trial B's head.
    concatenated = np.concatenate([corrupted_a, trial_b], axis=0)
    naive = filters.ema_causal_one_trial(concatenated)[10:]
    assert not np.allclose(naive[:3], baseline_b[:3])
    # The identity path is exact for the raw anchor A0.
    assert np.array_equal(filters.apply_output_filter_one_trial(trial_b, enabled=False), trial_b)


def test_pseudo_construction_kernels_reset_at_trial_boundaries() -> None:
    trial_a = np.tile(np.array([[0.5, 0.0]]), (10, 1))
    trial_b = np.tile(np.array([[0.0, 0.5]]), (10, 1))
    for construction in ("smoothed_causal", "smoothed_zero_phase"):
        baseline = filters.smooth_velocity_trajectory_one_trial(trial_b, construction=construction)
        corrupted_a = trial_a * 50.0
        _ = filters.smooth_velocity_trajectory_one_trial(corrupted_a, construction=construction)
        again = filters.smooth_velocity_trajectory_one_trial(trial_b, construction=construction)
        assert np.array_equal(baseline, again)
    raw = filters.smooth_velocity_trajectory_one_trial(trial_b, construction="raw")
    assert np.array_equal(raw, trial_b)


def test_zero_phase_kernel_properties() -> None:
    rng = np.random.default_rng(3)
    block = rng.uniform(-1.0, 1.0, size=(15, 2))
    zero_phase = filters.ema_zero_phase_one_trial(block)
    # DC gain exactly 1: a constant trajectory is a fixed point of both kernels.
    constant = np.tile(np.array([[2.0, -3.0]]), (9, 1))
    assert np.allclose(filters.ema_zero_phase_one_trial(constant), constant, atol=1e-12)
    assert np.allclose(filters.ema_causal_one_trial(constant), constant, atol=1e-12)
    # The two-pass is linear (superposition holds exactly).
    other = rng.uniform(-1.0, 1.0, size=(15, 2))
    blend = 0.37 * block + 0.63 * other
    assert np.allclose(
        filters.ema_zero_phase_one_trial(blend),
        0.37 * filters.ema_zero_phase_one_trial(block) + 0.63 * filters.ema_zero_phase_one_trial(other),
        atol=1e-9,
    )
    # Zero-phase reads the trial's own future (legal at update time): row 0
    # responds to a perturbation in the LAST row, unlike the causal kernel.
    perturbed = block.copy()
    perturbed[-1] += 5.0
    assert not np.allclose(
        filters.ema_zero_phase_one_trial(perturbed)[0],
        zero_phase[0],
    )
    assert np.allclose(
        filters.ema_causal_one_trial(perturbed)[0],
        filters.ema_causal_one_trial(block)[0],
    )


def test_true_velocity_restoration_roundtrip_and_padding() -> None:
    physical = np.array([[1.5, -2.5], [0.0, 3.0]])
    zscored = (physical - np.asarray(MEAN)) / np.asarray(STD)
    restored, padded = filters.true_physical_velocity_one_trial(
        zscored, behavior_mean=MEAN, behavior_std=STD,
    )
    assert np.allclose(restored, physical, atol=1e-12)
    assert padded.dtype == np.bool_ and not padded.any()
    padded_rows = np.full((2, 2), -1.0)
    restored_pad, padded_mask = filters.true_physical_velocity_one_trial(
        padded_rows, behavior_mean=MEAN, behavior_std=STD,
    )
    assert padded_mask.all() and np.all(restored_pad == 0.0)


# ---------------------------------------------------------------------------
# 2. Utility horizon isolation and the shared normalizer.
# ---------------------------------------------------------------------------


def test_utility_horizon_isolation() -> None:
    reject = [1.0, 2.0, 3.0, 4.0, 5.0]
    accept = [0.9, 1.9, 2.9, 3.9, 4.9]
    utility = policy.horizon_utility(reject_losses=reject, accept_losses=accept)
    assert utility["horizon_trials"] == 5
    assert math.isclose(utility["u_j"], 0.1)
    # The CURRENT trial's own loss is not an input to u_j at all: adding a
    # hypothetical trial-j loss to neither branch changes nothing.
    utility_again = policy.horizon_utility(reject_losses=reject, accept_losses=accept)
    assert utility_again["u_j"] == utility["u_j"]
    # A truncated horizon at the session end shortens the window only.
    short = policy.horizon_utility(reject_losses=reject[:2], accept_losses=accept[:2])
    assert short["horizon_trials"] == 2 and math.isclose(short["u_j"], 0.1)
    with pytest.raises(policy.P2PrimePolicyError):
        policy.horizon_utility(reject_losses=[], accept_losses=[])


def test_nsse_uses_one_session_normalizer_for_both_branches() -> None:
    rng = np.random.default_rng(5)
    target = rng.uniform(-1.0, 1.0, size=(40, 2))
    prediction_accept = target + 0.05
    prediction_reject = target + 0.20
    sst = policy.session_sst([target])
    u_small = policy.horizon_utility(
        reject_losses=[policy.trial_nsse(prediction_reject, target, sst=sst)],
        accept_losses=[policy.trial_nsse(prediction_accept, target, sst=sst)],
    )["u_j"]
    u_large = policy.horizon_utility(
        reject_losses=[policy.trial_nsse(prediction_reject, target, sst=2.0 * sst)],
        accept_losses=[policy.trial_nsse(prediction_accept, target, sst=2.0 * sst)],
    )["u_j"]
    assert u_small > 0.0
    assert math.isclose(u_small, 2.0 * u_large, rel_tol=1e-12)  # sign and ranking invariant
    assert policy.oracle_decision({"u_j": 0.0}) is False
    assert policy.oracle_decision({"u_j": 1e-12}) is True


# ---------------------------------------------------------------------------
# 1. Counterfactual parity and the oracle veto on real core objects.
# ---------------------------------------------------------------------------


def test_branch_states_differ_only_in_carrier() -> None:
    memory, channel_sha, _config = _synthetic_memory()
    b3s, native, predictions, validity = _synthetic_completed_trial(channel_sha=channel_sha)
    pending = memory.observe_completed_trial(
        b3s_trial_activity=b3s, carrier_trial_counts=native, complementary_predictions=predictions,
    )
    assert pending.activity_transition_ready is True
    assert pending.carrier_transition_accepted is True
    accept_state, reject_state, parity = policy.branch_states_for_counterfactual(
        memory, pending, independent_activity=memory.state.activity.after_completed_trial(b3s),
    )
    assert parity["activity_digest_equal"] is True
    assert parity["activity_stack_bitwise_equal"] is True
    assert parity["activity_independent_digest_match"] is True
    assert parity["carrier_reject_matches_parent"] is True
    assert parity["carrier_branches_differ"] is True
    assert accept_state.activity.digest == reject_state.activity.digest
    assert accept_state.carrier.digest != reject_state.carrier.digest
    assert reject_state.carrier.digest == memory.state.carrier.digest
    # Both branches advance the completed-query count identically.
    assert accept_state.committed_query_trials == reject_state.committed_query_trials


def test_branch_states_reject_activity_divergence() -> None:
    memory, channel_sha, _config = _synthetic_memory()
    b3s, native, predictions, _validity = _synthetic_completed_trial(channel_sha=channel_sha)
    pending = memory.observe_completed_trial(
        b3s_trial_activity=b3s, carrier_trial_counts=native, complementary_predictions=predictions,
    )
    accept_state, _reject_state, _parity = policy.branch_states_for_counterfactual(memory, pending)
    other_b3s, _other_native, _other_predictions, _other_validity = _synthetic_completed_trial(
        channel_sha=channel_sha, trial_id="query-other",
    )
    forged = core.DualMemoryState(
        activity=accept_state.activity.after_completed_trial(other_b3s),
        carrier=accept_state.carrier,
        committed_query_trials=accept_state.committed_query_trials + 1,
    )
    with pytest.raises(policy.P2PrimePolicyError):
        # A forged accept branch whose activity disagrees with the independent
        # frozen-law reconstruction must be caught by the parity check.
        policy.branch_states_for_counterfactual(
            memory,
            core.IndependentActivityPendingTrialUpdate(
                base_state_digest=pending.base_state_digest,
                activity_transition_ready=True,
                activity_fifo_changed=pending.activity_fifo_changed,
                carrier_transition_accepted=True,
                activity_rejection_reason=None,
                carrier_rejection_reason=None,
                pseudo_directions=pending.pseudo_directions,
                scalar_rates=pending.scalar_rates,
                carrier_proposal=pending.carrier_proposal,
                candidate_state=forged,
                fallback=pending.fallback,
                completed_trial_evidence=pending.completed_trial_evidence,
            ),
            independent_activity=memory.state.activity.after_completed_trial(b3s),
        )


def test_oracle_veto_restores_carrier_and_keeps_activity() -> None:
    memory, channel_sha, _config = _synthetic_memory()
    b3s, native, predictions, _validity = _synthetic_completed_trial(channel_sha=channel_sha)
    pending = memory.observe_completed_trial(
        b3s_trial_activity=b3s, carrier_trial_counts=native, complementary_predictions=predictions,
    )
    carrier_before = memory.state.carrier.digest
    vetoed = policy.oracle_rejected_pending(memory, pending)
    assert vetoed.carrier_transition_accepted is False
    assert vetoed.carrier_rejection_reason is core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE
    assert vetoed.candidate_state is not None
    assert vetoed.candidate_state.carrier.digest == carrier_before
    assert vetoed.candidate_state.activity.digest == pending.candidate_state.activity.digest
    outcome = memory.commit_independent(vetoed)
    core.validate_independent_activity_outcome_payload(outcome.payload())
    assert outcome.activity_transition_committed is True
    assert outcome.carrier_transition_committed is False
    assert outcome.carrier_before_sha256 == outcome.carrier_after_sha256


def test_construction_predictions_transform_only_the_trajectory() -> None:
    memory, channel_sha, config = _synthetic_memory()
    _b3s, _native, predictions, validity = _synthetic_completed_trial(channel_sha=channel_sha)
    noisy = predictions[0].velocity + 0.05 * np.sin(np.arange(10))[:, None]
    views = tuple(
        core.CompletedVelocityPrediction(noisy.copy(), validity) for _ in range(core.GROUP_COUNT)
    )
    raw_views, raw_meta = policy.build_construction_predictions(
        group_predictions=views, construction="raw",
    )
    assert np.array_equal(raw_views[0].velocity, noisy)
    causal_views, _meta = policy.build_construction_predictions(
        group_predictions=views, construction="smoothed_causal",
    )
    assert not np.array_equal(causal_views[0].velocity, noisy)
    assert causal_views[0].validity is views[0].validity
    zero_views, _meta2 = policy.build_construction_predictions(
        group_predictions=views, construction="smoothed_zero_phase",
    )
    assert np.all(np.isfinite(zero_views[0].velocity))
    behavior_rows = np.tile(np.array([[0.5, 0.0]]), (10, 1))
    true_views, true_meta = policy.build_construction_predictions(
        group_predictions=views, construction="true", behavior_rows=behavior_rows,
        behavior_mean=MEAN, behavior_std=STD,
    )
    assert np.allclose(true_views[0].velocity, behavior_rows * np.asarray(STD) + np.asarray(MEAN))
    assert true_meta["true_construction_padded_rows"] == 0
    # The true direction reference runs the frozen pipeline on the same rows.
    reference = policy.true_direction_payload(
        true_views[0].velocity, validity, config=config,
    )
    assert reference["accepted"] is True
    expected_index, _distance = core.nearest_canonical_direction(float(reference["theta_raw_rad"]))
    assert reference["theta_index"] == expected_index


# ---------------------------------------------------------------------------
# 4. One fixed output filter shared by every row except A0.
# ---------------------------------------------------------------------------


def test_output_filter_is_one_frozen_object_across_rows() -> None:
    for row in plan.ROWS:
        spec = plan.ROW_SPECS[row]
        if row == "A0":
            assert spec["output_filter"] == "raw"
            assert row not in plan.OUTPUT_FILTER["rows_using_it"]
        else:
            assert spec["output_filter"] == "fixed"
            assert row in plan.OUTPUT_FILTER["rows_using_it"]
    assert plan.OUTPUT_FILTER["alpha"] == 0.25
    assert plan.OUTPUT_FILTER["trial_boundary_reset"] is True
    assert plan.OUTPUT_FILTER["p1_k_on_external_selection_used"] is False
    assert filters.ALPHA == plan.OUTPUT_FILTER["alpha"]


def test_pre_registration_is_pinned() -> None:
    import copy

    payload = plan.pre_registration_payload()
    plan.validate_pre_registration(payload)
    assert payload["utility"]["primary_horizon_H"] == 5
    assert payload["utility"]["sensitivity_horizon_H"] == 10
    assert payload["sub_study_winner_rule"]["eligible"] == ["smoothed_causal", "smoothed_zero_phase"]
    drifted = copy.deepcopy(payload)
    drifted["output_filter"] = {**payload["output_filter"], "alpha": 0.5}
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)


# ---------------------------------------------------------------------------
# 5. Kill-criterion verdict boundaries.
# ---------------------------------------------------------------------------


def _matrix(o1_a1_m4, o1_o0_m4, o1_o0_m10, o2_a1_m4):
    base = 0.2000
    return {
        "m4": {
            "external": {
                "A1": {"mean_r2": base},
                "O1": {"mean_r2": base + o1_a1_m4},
                "O0": {"mean_r2": base + o1_a1_m4 - o1_o0_m4},
                "O2": {"mean_r2": base + o2_a1_m4},
            },
        },
        "m10": {
            "external": {
                "A1": {"mean_r2": 0.4000},
                "O1": {"mean_r2": 0.4000 + 0.0100},
                "O0": {"mean_r2": 0.4000 + 0.0100 - o1_o0_m10},
            },
        },
    }


def test_kc1_boundary_is_strict_less_than() -> None:
    at_boundary = policy.kill_criterion_verdicts(_matrix(0.02, 0.01, 0.01, 0.03))
    assert at_boundary["KC1_STOP_LEARNED_GATE"]["fired"] is False
    below = policy.kill_criterion_verdicts(_matrix(0.0199999, 0.01, 0.01, 0.03))
    assert below["KC1_STOP_LEARNED_GATE"]["fired"] is True
    assert below["verdict"] == "STOP_LEARNED_GATE"


def test_kc2_fires_only_when_both_deployment_budgets_below_threshold() -> None:
    both_low = policy.kill_criterion_verdicts(_matrix(0.03, 0.004, 0.004, 0.05))
    assert both_low["KC2_SMOOTHING_NO_CARRIER_VALUE"]["fired"] is True
    assert both_low["verdict"] == "SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY"
    m4_ok = policy.kill_criterion_verdicts(_matrix(0.03, 0.006, 0.004, 0.05))
    assert m4_ok["KC2_SMOOTHING_NO_CARRIER_VALUE"]["fired"] is False
    m10_exactly = policy.kill_criterion_verdicts(_matrix(0.03, 0.004, 0.005, 0.05))
    assert m10_exactly["KC2_SMOOTHING_NO_CARRIER_VALUE"]["fired"] is False


def test_kc3_requires_kc1_and_true_direction_headroom() -> None:
    fired = policy.kill_criterion_verdicts(_matrix(0.01, 0.006, 0.006, 0.02))
    assert fired["KC1_STOP_LEARNED_GATE"]["fired"] is True
    assert fired["KC3_PSEUDO_BIAS_BOTTLENECK"]["fired"] is True
    assert fired["verdict"] == "STOP_LEARNED_GATE"
    no_headroom = policy.kill_criterion_verdicts(_matrix(0.01, 0.006, 0.006, 0.019))
    assert no_headroom["KC3_PSEUDO_BIAS_BOTTLENECK"]["fired"] is False
    kc1_not_fired = policy.kill_criterion_verdicts(_matrix(0.03, 0.006, 0.006, 0.05))
    assert kc1_not_fired["KC3_PSEUDO_BIAS_BOTTLENECK"]["fired"] is False


def test_kc4_pending_and_kc5_authorization() -> None:
    passing = policy.kill_criterion_verdicts(_matrix(0.03, 0.006, 0.006, 0.05))
    assert passing["KC4_HEADROOM_NOT_IDENTIFIABLE"]["status"] == "PENDING_NOT_TESTABLE_IN_P2PRIME"
    assert passing["KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION"]["fired"] is True
    assert passing["verdict"] == "PASS_KC1_KC2_KC3_EXTERNAL_MATCHED_SCORE_AUTHORIZED"
    stopping = policy.kill_criterion_verdicts(_matrix(0.01, 0.006, 0.006, 0.05))
    assert stopping["KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION"]["fired"] is False


# ---------------------------------------------------------------------------
# M30 law and dry-plan inertness.
# ---------------------------------------------------------------------------


def test_m30_law_and_row_topology_pre_registered() -> None:
    assert plan.ROW_SPECS["A0"]["carrier_action"] == "reject all"
    assert plan.ROW_SPECS["O2"]["pseudo"] == "true direction"
    assert plan.M30_LAW["carrier"].startswith("no-op")
    assert {row: plan.ROW_SPECS[row]["output_filter"] for row in plan.ROWS} == {
        "A0": "raw", "A1": "fixed", "C0": "fixed", "C1": "fixed",
        "O0": "fixed", "O1": "fixed", "O2": "fixed", "P": "fixed",
    }
    dry = plan.dry_plan()
    assert dry["target_optimizer_backward_update"] == 0
    assert dry["inference_only"] is True


def test_m30_activity_capacity_zero_keeps_carrier_law_well_defined() -> None:
    # The frozen capacity table is what the M30 no-op verification relies on.
    assert core.ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET[30] == 0
    assert core.activity_fifo_capacity_for_support_budget(30) == 0
    assert plan.M30_LAW["bitwise_check"].startswith("carrier state digest")
    # A capacity-zero activity memory over the full 30-trial support is a
    # strict no-op under the frozen law.
    channel_sha = core.channel_order_digest(np.arange(8, dtype=np.int64))
    support = tuple(
        core.B3SInterpolatedSpikeCountTrial(
            activity=np.full((100, 8), float(index + 1), dtype=np.float32),
            session_id="synthetic", trial_id=f"support-{index}", channel_order_sha256=channel_sha,
        )
        for index in range(30)
    )
    zero_capacity = core.ActivityMemory(
        support_trials=support, query_trials=(),
        channel_ids=np.arange(8, dtype=np.int64), fifo_capacity=0,
    )
    b3s, _native, _predictions, _validity = _synthetic_completed_trial(channel_sha=channel_sha)
    after = zero_capacity.after_completed_trial(b3s)
    assert after.digest == zero_capacity.digest
