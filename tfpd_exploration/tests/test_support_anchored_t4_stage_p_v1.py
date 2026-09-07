"""Synthetic no-data/no-CUDA tests for Support-Anchored T4 Stage P (V1).

Everything runs on constructed numpy blocks, the real imported frozen
``src.causal_dual_memory_cell_d_v1.core`` capability objects, the frozen P2'
filters and the REAL Stage-O anchor/trust-region modules this route reuses.
The review-critical properties:

1. the measurement laws (design §5): complementary-group exclusion, the P1
   circular ensemble vs the P2 trajectory-level aggregation, the predeclared
   confidence weights, the P4 deterministic shuffle binding and the P5
   self-referential diagnostic;
2. the three-factor commit gate (§6): coverage warmup/repetition/mass caps,
   logdet/min-eigenvalue evidence, the Stage-O trust-region rejection and the
   exact pre-block restoration of a rejected block;
3. the anchored refit recomputation law and per-group optional labels;
4. the §8 gate boundary semantics (1e-12 as a disclosed band only), the
   promotion/continuity/confidence gates, the §8.4 stop conditions and the
   §6.4 movement ordering;
5. the whole state machine on toy carriers: causality, the alpha_M = 0 and M30
   exact no-ops bit-equal to P0, anchor immutability, leakage labelling;
6. the source-only selection law (within-6 folds, first-maximum tie-break);
7. pre-registration pinning and the stage guards.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.learned_gate_p2prime_v1 import filters as p2filters

from src.support_anchored_t4_stage_o_v1 import anchor as anchor_module
from src.support_anchored_t4_stage_o_v1 import replay as stage_o_replay
from src.support_anchored_t4_stage_o_v1 import trust_region

from src.support_anchored_t4_stage_p_v1 import direction_estimator, gate, gates, plan, replay

CANON = cdm_core.CANONICAL_DIRECTIONS_RAD
N_UNITS = 8
SESSION_ID = "toy-session"
DT = 0.02
DEFAULT_BIAS = (0.02, -0.03, 0.01, -0.01)
DEFAULT_SPEED = (9.0, 12.0, 6.0, 15.0)


# ---------------------------------------------------------------------------
# Toy session: real frozen capability objects, constructed blocks.
# ---------------------------------------------------------------------------


def _channel_digest() -> str:
    return cdm_core.channel_order_digest(np.arange(N_UNITS, dtype=np.int64))


def _b3s(rng, trial_id, channel_digest):
    activity = (rng.random((100, N_UNITS), dtype=np.float32) * 3.0)
    return cdm_core.B3SInterpolatedSpikeCountTrial(
        activity=activity, session_id=SESSION_ID, trial_id=trial_id,
        channel_order_sha256=channel_digest,
    )


def _native(rng, trial_id, channel_digest, start, length):
    counts = rng.poisson(2.0, size=(length, N_UNITS)).astype(np.float64)
    return cdm_core.NativeRewardedTrialSpikeCounts(
        counts=counts, session_id=SESSION_ID, trial_id=trial_id,
        channel_order_sha256=channel_digest,
        rewarded_interval_start_bin=start, rewarded_interval_stop_bin=start + length,
    )


def _behavior_rows(windows: int, direction_index: int) -> np.ndarray:
    theta = CANON[int(direction_index)]
    physical = np.stack(
        [np.full(windows, 9.0 * math.cos(theta)), np.full(windows, 9.0 * math.sin(theta))],
        axis=1,
    )
    mean = np.asarray(v1plan.SEALED_BEHAVIOR_MEAN, dtype=np.float64)
    std = np.asarray(v1plan.SEALED_BEHAVIOR_STD, dtype=np.float64)
    return ((physical - mean[None, :]) / std[None, :]).astype(np.float32)


@dataclass
class ToyTrial:
    trial_id: str
    chronology_position: int
    b3s_activity: Any
    native_counts: Any
    velocity_validity: Any
    neural_windows: Any
    endpoint_bins: Any
    group_bias: tuple = DEFAULT_BIAS
    group_speed: tuple = DEFAULT_SPEED


@dataclass
class ToySession:
    surface: str
    session: str
    behavior: np.ndarray
    channel_ids: np.ndarray
    trials_by_id: Mapping[str, ToyTrial]
    query_trial_ids: Mapping[int, tuple]
    support_trial_ids: Mapping[int, tuple]
    support_rates: Mapping[str, np.ndarray]
    support_direction_indices: Mapping[str, int]


def _tuning_rates(direction_index: int, rng) -> np.ndarray:
    theta = CANON[int(direction_index)]
    preferred = np.asarray(CANON, dtype=np.float64)[np.arange(N_UNITS) % len(CANON)]
    gain = 40.0 + 8.0 * np.arange(N_UNITS) / N_UNITS
    return 45.0 + gain * np.cos(theta - preferred) + rng.normal(0.0, 1.0, size=N_UNITS)


def build_toy_session(
    seed: int = 7, *, n_query: int = 6, surface: str = "within",
    overrides: Optional[Mapping[int, Mapping[str, tuple]]] = None,
) -> ToySession:
    """A toy grid; ``overrides[index] = {"group_bias": (...), "group_speed": (...)}``."""
    rng = np.random.default_rng(seed)
    channel_digest = _channel_digest()
    channel_ids = np.arange(N_UNITS, dtype=np.int64)
    support_directions = [0, 2, 5, 0, 1, 3, 6, 4, 7, 2, 0, 5, 1, 4, 6, 3, 7, 0, 2, 5, 1, 3, 4, 6, 7, 0, 2, 5, 1, 3]
    query_directions = [1, 3, 6, 4, 7, 2, 1, 0][:n_query]
    trials: dict[str, ToyTrial] = {}
    behavior = np.zeros((40 * 60, 2), dtype=np.float32)
    support_rates: dict[str, np.ndarray] = {}
    support_direction_indices: dict[str, int] = {}
    position = 0

    def _make(trial_id: str, direction: int, windows: int, length: int,
              bias=None, speed=None) -> ToyTrial:
        nonlocal position
        start = position * 40
        endpoints = np.arange(start, start + windows, dtype=np.int64)
        behavior[endpoints] = _behavior_rows(windows, direction)
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(windows, dtype=np.bool_), session_id=SESSION_ID,
            trial_id=trial_id, prediction_interval_start_bin=int(endpoints[0]),
            prediction_interval_stop_bin=int(endpoints[-1]) + 1,
        )
        trial = ToyTrial(
            trial_id=trial_id, chronology_position=position,
            b3s_activity=_b3s(rng, trial_id, channel_digest),
            native_counts=_native(rng, trial_id, channel_digest, start, length),
            velocity_validity=validity,
            neural_windows=rng.random((windows, 50, N_UNITS), dtype=np.float32),
            endpoint_bins=endpoints,
            group_bias=DEFAULT_BIAS if bias is None else tuple(bias),
            group_speed=DEFAULT_SPEED if speed is None else tuple(speed),
        )
        position += 1
        return trial

    for index in range(30):
        trial_id = f"{SESSION_ID}:trial:{index}"
        trials[trial_id] = _make(trial_id, support_directions[index], 30, 40)
        support_rates[trial_id] = _tuning_rates(support_directions[index], rng)
        support_direction_indices[trial_id] = int(support_directions[index])
    for offset, direction in enumerate(query_directions):
        trial_id = f"{SESSION_ID}:trial:{30 + offset}"
        options = dict((overrides or {}).get(offset, {}))
        trials[trial_id] = _make(
            trial_id, direction, 25, 36,
            bias=options.get("group_bias"), speed=options.get("group_speed"),
        )
    support_ids = {budget: tuple(f"{SESSION_ID}:trial:{index}" for index in range(budget))
                   for budget in (4, 10, 30)}
    query_ids = {budget: tuple(f"{SESSION_ID}:trial:{index}" for index in range(30, 30 + n_query))
                 for budget in (4, 10, 30)}
    return ToySession(
        surface=surface, session=SESSION_ID, behavior=behavior, channel_ids=channel_ids,
        trials_by_id=trials, query_trial_ids=query_ids, support_trial_ids=support_ids,
        support_rates=support_rates, support_direction_indices=support_direction_indices,
    )


@dataclass
class ToyTransition:
    trial_id: str
    activity_transition_committed: bool
    activity_fifo_changed: bool
    carrier_transition_committed: bool
    activity_rejection_reason: Optional[str]
    carrier_rejection_reason: Optional[str]
    state_before_sha256: str
    state_after_sha256: str
    activity_before_sha256: str
    activity_after_sha256: str
    carrier_before_sha256: str
    carrier_after_sha256: str


class ToyActivityOnlyMemory(cdm_core.IndependentActivityCausalDualMemory):
    """The toy mirror of the sealed activity-only law (never a carrier proposal)."""

    def observe_completed_trial(self, *, b3s_trial_activity, carrier_trial_counts, complementary_predictions):
        del carrier_trial_counts, complementary_predictions
        reason = self.state.activity.validate_complete_trial(b3s_trial_activity)
        if reason is not None:
            return self._activity_invalid_pending(reason)
        candidate, changed = self._activity_candidate(b3s_trial_activity)
        return self._carrier_rejected_pending(
            activity_candidate=candidate, activity_fifo_changed=changed,
            reason=cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
        )


class ToyRuntime:
    """The minimal runtime surface the Stage-P rollouts consume (toy forward)."""

    def __init__(self) -> None:
        self._memory_mode = "cdm"
        self.forward_log: list[tuple[str, str, np.ndarray]] = []
        self.completed_trials = 0

    def _make_memory(self, *, activity, carrier):
        if self._memory_mode == "activity_only":
            return ToyActivityOnlyMemory(activity=activity, carrier=carrier)
        return cdm_core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)

    def _initial_memory(self, *, session, budget):
        support_ids = tuple(session.support_trial_ids[budget])
        rates = np.ascontiguousarray(
            np.stack([session.support_rates[item] for item in support_ids]), dtype=np.float64,
        )
        directions = np.ascontiguousarray(
            np.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np.int64),
        )
        initial = cdm_core.fit_carriers_from_trial_table(
            rates, directions, mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        config = cdm_core.CDMDConfig(
            support_budget_m=budget, active_fit_mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = cdm_core.CarrierMemory.from_support_trials(
            initial_raw_t4=initial, channel_ids=session.channel_ids,
            support_trial_rates=rates, support_direction_indices=directions,
            config=config, valid_mask=np.ones(N_UNITS, dtype=np.bool_),
        )
        activity = cdm_core.ActivityMemory.initialize(
            [session.trials_by_id[item].b3s_activity for item in support_ids],
            channel_ids=session.channel_ids, fifo_capacity=int(config.activity_fifo_capacity),
        )
        memory = self._make_memory(activity=activity, carrier=carrier)
        evidence = {
            "budget": budget,
            "support_trial_ids": list(support_ids),
            "initial_carrier_sha256": cdm_core.array_digest(initial),
            "initial_activity_sha256": memory.state.activity.digest,
            "group_assignment_sha256": carrier.groups.digest,
        }
        return memory, evidence

    def _governing_forward(self, *, trial, inputs):
        side = np.asarray(inputs.active_t4, dtype=np.float64)
        stack = np.asarray(inputs.activity_trials, dtype=np.float64)
        neural = np.asarray(trial.neural_windows, dtype=np.float64)
        context = stack.mean(axis=(0, 1))
        drive = neural.mean(axis=1) + 0.1 * context[None, :]
        velocity = np.column_stack((drive @ side[:, 0], drive @ side[:, 1]))
        self.forward_log.append((trial.trial_id, inputs.state_digest, velocity.copy()))
        return velocity

    def _truth_angle(self, session, trial) -> float:
        restored, _padded = p2filters.true_physical_velocity_one_trial(
            np.asarray(session.behavior[trial.endpoint_bins]),
            behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
        )
        displacement = restored.sum(axis=0) * DT
        return float(math.atan2(displacement[1], displacement[0]))

    def _group_predictions_dispatch(self, *, trial, memory):
        """The toy held-group forward: the trial's true direction plus a fixed
        per-group angular bias and per-group speed (deterministic, carrier-free
        measurement noise model)."""
        del memory  # the toy measurement is independent of the carrier
        session = self._session
        truth = self._truth_angle(session, trial)
        windows = int(np.asarray(trial.endpoint_bins).size)
        views = []
        for group in range(cdm_core.GROUP_COUNT):
            theta = truth + float(trial.group_bias[group])
            speed = float(trial.group_speed[group])
            velocity = np.column_stack((
                np.full(windows, speed * math.cos(theta)),
                np.full(windows, speed * math.sin(theta)),
            ))
            views.append(cdm_core.CompletedVelocityPrediction(velocity, trial.velocity_validity))
        evidence = [{"forward_chunk_count": 1, "held_group": group} for group in range(cdm_core.GROUP_COUNT)]
        return tuple(views), evidence

    def _commit_dispatch(self, *, memory, pending, trial_id):
        outcome = memory.commit_independent(pending)
        cdm_core.validate_independent_activity_outcome_payload(outcome.payload())
        return ToyTransition(
            trial_id=trial_id,
            activity_transition_committed=outcome.activity_transition_committed,
            activity_fifo_changed=outcome.activity_fifo_changed,
            carrier_transition_committed=outcome.carrier_transition_committed,
            activity_rejection_reason=None if outcome.activity_rejection_reason is None
            else outcome.activity_rejection_reason.value,
            carrier_rejection_reason=None if outcome.carrier_rejection_reason is None
            else outcome.carrier_rejection_reason.value,
            state_before_sha256=outcome.state_before_sha256,
            state_after_sha256=outcome.state_after_sha256,
            activity_before_sha256=outcome.activity_before_sha256,
            activity_after_sha256=outcome.activity_after_sha256,
            carrier_before_sha256=outcome.carrier_before_sha256,
            carrier_after_sha256=outcome.carrier_after_sha256,
        )

    def _session_target_views(self, session, budget):
        targets, masks = [], []
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            targets.append(np.ascontiguousarray(session.behavior[trial.endpoint_bins], dtype=np.float32))
            masks.append(np.ones(trial.endpoint_bins.size, dtype=np.uint8))
        return targets, masks, 1.0

    def _matrix_r2(self, per_trial_prediction, targets, masks):
        joined = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float64)[m.astype(bool)]
            for item, m in zip(per_trial_prediction, masks)
        ], axis=0))
        target = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float64)[m.astype(bool)]
            for item, m in zip(targets, masks)
        ], axis=0))
        sse = float(np.sum((joined - target) ** 2))
        sst = float(np.sum((target - target.mean(axis=0)) ** 2)) or 1.0
        return 1.0 - sse / sst, joined

    def _house_raw_r2(self, per_trial_raw, targets, masks):
        value, _joined = self._matrix_r2(per_trial_raw, targets, masks)
        return float(value)

    @staticmethod
    def _raw_prediction_digest(per_trial_raw, masks):
        joined = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float32)[m.astype(bool)]
            for item, m in zip(per_trial_raw, masks)
        ], axis=0))
        return hashlib.sha256(joined.tobytes()).hexdigest()

    def _bump_trial_counters(self):
        self.completed_trials += 1

    def _rollout_activity(self, *, session, budget):
        self._memory_mode = "activity_only"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, _sst = self._session_target_views(session, budget)
        per_trial_raw: list[np.ndarray] = []
        transitions: list[ToyTransition] = []
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            inputs = memory.read_prediction_inputs()
            per_trial_raw.append(self._governing_forward(trial=trial, inputs=inputs))
            pending = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=(None,) * cdm_core.GROUP_COUNT,
            )
            transitions.append(self._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id))
            self._bump_trial_counters()
        assert all(not item.carrier_transition_committed for item in transitions)
        per_trial_filtered = [p2filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
        raw_r2 = self._house_raw_r2(per_trial_raw, targets, masks)
        filtered_r2, _joined = self._matrix_r2(per_trial_filtered, targets, masks)
        rows = {
            "A0": {"house_raw_r2": raw_r2, "matrix_r2": raw_r2,
                   "prediction_sha256_raw": self._raw_prediction_digest(per_trial_raw, masks)},
            "A1": {"house_raw_r2": raw_r2, "matrix_r2": float(filtered_r2),
                   "prediction_sha256_raw": self._raw_prediction_digest(per_trial_raw, masks)},
        }
        return {
            "rows": rows, "transitions": transitions, "initial": initial,
            "per_trial_raw": per_trial_raw, "per_trial_filtered": per_trial_filtered,
            "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_ROW["P0"]),
            "per_trial_receipts": [
                {"trial_id": item.trial_id, "fs": None, "sb": item.state_before_sha256,
                 "sa": item.state_after_sha256} for item in transitions
            ],
        }


def _run_rollout(runtime: ToyRuntime, session: ToySession, budget: int, row: str, hp, **kwargs):
    runtime._session = session
    runtime._memory_mode = "cdm"
    return replay.rollout_p(
        runtime, session=session, budget=budget, spec=replay.ROW_SPECS[row], hp=hp, **kwargs,
    )


def _default_hp(**overrides) -> replay.CellHyperparameters:
    values = dict(
        rho_M=1.0, alpha_M=0.5, c_M=None,
        thresholds=gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0),
    )
    values.update(overrides)
    return replay.CellHyperparameters(**values)


def _toy_anchor(runtime: ToyRuntime, session: ToySession, budget: int) -> anchor_module.SupportAnchor:
    memory, _initial = runtime._initial_memory(session=session, budget=budget)
    carrier = memory.state.carrier
    rates, directions, _ids = stage_o_replay._support_binding(session, budget)
    return anchor_module.SupportAnchor.from_labeled_support(
        groups=carrier.groups, support_trial_rates=rates,
        support_direction_indices=directions, support_t4=carrier.active_t4,
    )


def _trial_facts(session, trial, rates) -> gate.TrialFacts:
    return gate.TrialFacts(
        session_id=session.session, trial_id=trial.trial_id,
        chronology_position=int(trial.chronology_position), scalar_rates=rates,
        rate_sha256=cdm_core.array_digest(rates),
        native_counts_sha256=cdm_core.array_digest(trial.native_counts.counts),
        channel_order_sha256=cdm_core.channel_order_digest(session.channel_ids),
        valid_mask_sha256=cdm_core.array_digest(np.ones(N_UNITS, dtype=np.bool_)),
    )


def _measure(runtime, session, trial, *, law="cross_group_trajectory",
             weight_law="resultant_times_early_late_agreement",
             binding="complementary_exclusion", tau_d=math.pi / 4.0, config=None):
    if config is None:
        memory, _initial = runtime._initial_memory(session=session, budget=4)
        config = memory.state.carrier.config
    runtime._session = session
    views, _evidence = runtime._group_predictions_dispatch(trial=trial, memory=None)
    return direction_estimator.measure_trial(
        views, config=config, law=law, weight_law=weight_law, binding=binding, tau_d=tau_d,
    )


# ---------------------------------------------------------------------------
# 1. The measurement laws (design §5).
# ---------------------------------------------------------------------------


def test_p1_is_the_circular_ensemble_of_the_other_groups():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    measurement = _measure(runtime, session, trial, law="cross_group_circular",
                           weight_law="resultant_length")
    assert measurement.accepted
    memory, _initial = runtime._initial_memory(session=session, budget=4)
    config = memory.state.carrier.config
    evidence = [
        direction_estimator.group_trajectory_evidence(view, group=group, config=config)
        for group, view in enumerate(runtime._group_predictions_dispatch(trial=trial, memory=None)[0])
    ]
    for group, record in enumerate(measurement.per_group):
        assert record is not None
        assert group not in record.included_groups
        others = [float(evidence[h].theta_raw_rad) for h in record.included_groups]
        expected = math.atan2(
            float(np.mean(np.sin(others))), float(np.mean(np.cos(others))),
        )
        assert record.theta_raw_rad == pytest.approx(expected, abs=1.0e-12)
        assert record.weight == pytest.approx(
            float(np.hypot(np.mean(np.sin(others)), np.mean(np.cos(others)))), abs=1.0e-12,
        )
        assert record.agreement is None  # the circular law has no early/late statistic


def test_p2_integrates_displacement_before_atan2_and_weights_by_agreement():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    measurement = _measure(runtime, session, trial)
    memory, _initial = runtime._initial_memory(session=session, budget=4)
    config = memory.state.carrier.config
    views = runtime._group_predictions_dispatch(trial=trial, memory=None)[0]
    evidence = [
        direction_estimator.group_trajectory_evidence(view, group=group, config=config)
        for group, view in enumerate(views)
    ]
    for group, record in enumerate(measurement.per_group):
        assert record is not None
        assert group not in record.included_groups
        total = np.zeros(2)
        early = np.zeros(2)
        late = np.zeros(2)
        for h in record.included_groups:
            total = total + np.asarray(evidence[h].displacement)
            early = early + np.asarray(evidence[h].early_displacement)
            late = late + np.asarray(evidence[h].late_displacement)
        assert record.theta_raw_rad == pytest.approx(
            math.atan2(total[1], total[0]), abs=1.0e-12,
        )
        assert record.displacement_norm == pytest.approx(float(np.linalg.norm(total)), abs=1.0e-12)
        expected_early_late = float(cdm_core.circular_distance(
            math.atan2(early[1], early[0]), math.atan2(late[1], late[0]),
        ))
        assert record.early_late_distance_rad == pytest.approx(expected_early_late, abs=1.0e-12)
        assert record.agreement == pytest.approx(
            max(0.0, 1.0 - expected_early_late / (math.pi / 2.0)), abs=1.0e-12,
        )
        assert record.weight == pytest.approx(record.resultant_length * record.agreement, abs=1.0e-12)
    # With unequal per-group speeds the displacement-space ensemble differs from
    # the angle-space ensemble: trajectory aggregation is a different law.
    circular = _measure(runtime, session, trial, law="cross_group_circular",
                        weight_law="resultant_length")
    thetas_p2 = {record.theta_raw_rad for record in measurement.per_group if record}
    thetas_p1 = {record.theta_raw_rad for record in circular.per_group if record}
    assert thetas_p2 != thetas_p1


def test_dispersion_gate_rejects_only_the_groups_that_include_the_biased_view():
    runtime = ToyRuntime()
    # A 0.392 rad bias on view 0 stays inside the frozen per-group canonical
    # margin (pi/8 = 0.3927) so all four frozen directions remain acceptable,
    # but any ensemble CONTAINING view 0 exceeds the pi/8 dispersion threshold.
    session = build_toy_session(overrides={0: {"group_bias": (0.392, -0.03, 0.01, -0.01)}})
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    measurement = _measure(runtime, session, trial, tau_d=math.pi / 8.0)
    assert measurement.accepted
    # Group 0's own label excludes the biased view 0 -> it survives; groups
    # 1..3 include it -> their ensembles are too dispersed.
    assert measurement.per_group[0] is not None
    for group in (1, 2, 3):
        assert measurement.per_group[group] is None


def test_a_frozen_per_group_gate_failure_rejects_the_whole_trial():
    runtime = ToyRuntime()
    session = build_toy_session(overrides={1: {"group_speed": (9.0, 12.0, 0.01, 15.0)}})
    trial = session.trials_by_id[session.query_trial_ids[4][1]]
    measurement = _measure(runtime, session, trial)
    assert not measurement.accepted
    assert measurement.reason == "low_displacement"
    assert all(item is None for item in measurement.per_group)


def test_early_late_incoherence_fails_the_trajectory_coherence_pass():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    memory, _initial = runtime._initial_memory(session=session, budget=4)
    config = memory.state.carrier.config
    runtime._session = session
    views = list(runtime._group_predictions_dispatch(trial=trial, memory=None)[0])
    windows = int(np.asarray(trial.endpoint_bins).size)
    half = windows // 2
    skewed = []
    for view in views:
        velocity = np.asarray(view.velocity, dtype=np.float64).copy()
        direction = velocity[0] / np.linalg.norm(velocity[0])
        rotated = np.asarray([-direction[1], direction[0]])  # +pi/2 rotation
        # Early half drives the true direction; the late half is a tiny
        # perpendicular drift, so the FULL displacement stays near canonical
        # while the early/late ensemble directions disagree by pi/2.
        velocity[half:] = 0.01 * rotated[None, :]
        skewed.append(cdm_core.CompletedVelocityPrediction(velocity, view.validity))
    measurement = direction_estimator.measure_trial(
        tuple(skewed), config=config, law="cross_group_trajectory",
        weight_law="resultant_times_early_late_agreement", binding="complementary_exclusion",
        tau_d=math.pi / 4.0,
    )
    assert not measurement.accepted
    assert all(item is None for item in measurement.per_group)
    circular = direction_estimator.measure_trial(
        tuple(skewed), config=config, law="cross_group_circular",
        weight_law="resultant_length", binding="complementary_exclusion",
        tau_d=math.pi / 4.0,
    )
    # The circular law has no coherence factor, so the same skewed trial passes.
    assert circular.accepted and all(item is not None for item in circular.per_group)


def test_p3_keeps_the_direction_law_but_fixes_the_confidence():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    p2 = _measure(runtime, session, trial)
    p3 = _measure(runtime, session, trial, weight_law="constant_one")
    for left, right in zip(p2.per_group, p3.per_group):
        assert left is not None and right is not None
        assert left.theta_raw_rad == right.theta_raw_rad
        assert left.theta_index == right.theta_index
        assert right.weight == 1.0
        assert left.weight <= 1.0


def test_p4_rotation_rebinds_groups_and_confidence_deterministically():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    p2 = _measure(runtime, session, trial)
    p4 = _measure(runtime, session, trial, binding="rotated_group_and_confidence")
    count = cdm_core.GROUP_COUNT
    for group in range(count):
        source = p2.per_group[(group + 1) % count]
        weight_source = p2.per_group[(group + 2) % count]
        rotated = p4.per_group[group]
        assert source is not None and weight_source is not None and rotated is not None
        assert rotated.theta_raw_rad == source.theta_raw_rad
        assert rotated.included_groups == source.included_groups
        assert rotated.weight == weight_source.weight
        # The rotation restores the self-labeling binding: group g now receives
        # a label whose ensemble included group g.
        assert group in rotated.included_groups


def test_p5_labels_each_group_with_its_own_view_and_is_self_referential():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    memory, _initial = runtime._initial_memory(session=session, budget=4)
    config = memory.state.carrier.config
    runtime._session = session
    views = runtime._group_predictions_dispatch(trial=trial, memory=None)[0]
    measurement = _measure(runtime, session, trial, law="same_group", weight_law="constant_one")
    assert measurement.accepted
    for group, record in enumerate(measurement.per_group):
        assert record is not None
        assert record.included_groups == (group,)
        own = cdm_core.pseudo_direction_from_velocity(
            views[group].velocity, views[group].validity.valid_mask, config=config,
        )
        assert record.theta_index == own.theta_index
        assert record.weight == 1.0


# ---------------------------------------------------------------------------
# 2. The anchored bank and the three-factor gate (design §4.2/§6).
# ---------------------------------------------------------------------------


def _toy_labels(runtime, session, budget, law="cross_group_trajectory", n_rows=3,
                tau_d=math.pi / 4.0):
    anchor = _toy_anchor(runtime, session, budget)
    labels_per_trial = []
    for trial_id in list(session.query_trial_ids[budget])[:n_rows]:
        trial = session.trials_by_id[trial_id]
        measurement = _measure(runtime, session, trial, law=law, tau_d=tau_d)
        rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        labels_per_trial.append((trial, measurement, rates))
    return anchor, labels_per_trial


def _commit_rows(runtime, session, budget, thresholds, *, rho_M=1.0, c_M=None, n_rows=3,
                 law="cross_group_trajectory"):
    anchor, items = _toy_labels(runtime, session, budget, law=law, n_rows=n_rows,
                                tau_d=thresholds.tau_d)
    bank = gate.EvidenceBankP.empty()
    decisions = []
    for trial, measurement, rates in items:
        decision = gate.evaluate_block(
            anchor, bank, measurement.per_group,
            facts=_trial_facts(session, trial, rates), measurement_sha256=measurement.digest,
            rho_M=rho_M, c_M=c_M, thresholds=thresholds, block_index=len(bank),
        )
        decisions.append(decision)
        if decision.committed:
            bank = bank.with_row(decision.row)
    return anchor, bank, decisions


def test_refit_uses_only_the_labeled_rows_of_each_group():
    runtime = ToyRuntime()
    session = build_toy_session(overrides={0: {"group_bias": (0.392, -0.03, 0.01, -0.01)}})
    thresholds = gate.GateThresholds(tau_d=math.pi / 8.0, r_max=8, d_min=1, max_mass_relative=2.0)
    anchor, bank, decisions = _commit_rows(runtime, session, 4, thresholds, n_rows=1)
    assert decisions[0].committed
    counts = bank.direction_counts(0)
    # The biased view means groups 1..3 carry no label on trial 0 (measurement
    # factor) -- but group 0 does, and its moments must reflect only that row.
    assert int(counts.sum()) == 1
    S, t = bank.evidence_moments(anchor, 0)
    label = bank.rows[0].labels[0]
    design = np.asarray([math.cos(CANON[int(label.theta_index)]),
                         math.sin(CANON[int(label.theta_index)]), 1.0])
    expected_S = float(label.weight) * np.outer(design, design)
    assert np.allclose(S, expected_S, atol=1.0e-12, rtol=0.0)
    units = anchor.groups.unit_indices(0)
    expected_t = float(label.weight) * design[:, None] * bank.rows[0].facts.scalar_rates[units][None, :]
    assert np.allclose(t, expected_t, atol=1.0e-12, rtol=0.0)
    for group in (1, 2, 3):
        assert int(bank.direction_counts(group).sum()) == 0


def test_coverage_warmup_repetition_and_mass_caps_reject_blocks():
    runtime = ToyRuntime()
    session = build_toy_session()
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0)
    anchor, bank, decisions = _commit_rows(runtime, session, 4, thresholds, n_rows=3)
    # The default grid offers three DISTINCT directions: every block is novel,
    # so the no-deadlock warmup lets all three commit and the logdet/min-eig
    # evidence is recorded on every candidate row.
    assert all(decision.committed for decision in decisions)
    coverage = decisions[0].factor_evidence["coverage_rows"][0]
    assert coverage["direction_novel"] is True
    assert coverage["distinct_directions_with_row"] == 1
    assert coverage["distinct_ok"] is True
    assert coverage["logdet_increase"] > 0.0
    assert coverage["min_eigenvalue_with_row"] >= coverage["min_eigenvalue_before"] - 1.0e-9
    assert coverage["coverage_ok"] is True

    # Warmup: when every trial repeats ONE direction, only the novel first
    # block commits (d_min = 3 can never be reached by repeats).
    repeated = build_toy_session()
    for index in range(6):
        trial_id = repeated.query_trial_ids[4][index]
        rows = _behavior_rows(int(repeated.trials_by_id[trial_id].endpoint_bins.size), 1)
        repeated.behavior[repeated.trials_by_id[trial_id].endpoint_bins] = rows
    warmup = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0)
    _anchor_w, bank_w, decisions_w = _commit_rows(runtime, repeated, 4, warmup, n_rows=3)
    assert decisions_w[0].committed
    assert not decisions_w[1].committed
    assert decisions_w[1].block_reason == "direction_coverage_rejected_all_groups"
    blocked = decisions_w[1].factor_evidence["coverage_rows"][0]
    assert blocked["direction_novel"] is False
    assert blocked["distinct_directions_with_row"] == 1
    assert blocked["distinct_ok"] is False
    assert int(bank_w.direction_counts(0).sum()) == 1

    # Repetition cap: with r_max = 1 the second same-direction block is
    # rejected even after the warmup is satisfied (d_min = 1).
    strict = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=1, d_min=1, max_mass_relative=2.0)
    _anchor2, bank2, decisions2 = _commit_rows(runtime, repeated, 4, strict, n_rows=3)
    assert decisions2[0].committed
    assert not decisions2[1].committed
    repetition = decisions2[1].factor_evidence["coverage_rows"][0]
    assert repetition["repetition_count"] == 2 and repetition["repetition_ok"] is False
    assert int(bank2.direction_counts(0).sum()) == 1

    # Mass cap: rho * (mass + w) <= max_mass_relative * support rows (4 rows).
    tight = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=0.05)
    _anchor3, _bank3, decisions3 = _commit_rows(runtime, session, 4, tight, n_rows=1)
    assert not decisions3[0].committed
    mass = decisions3[0].factor_evidence["coverage_rows"][0]
    assert mass["mass_ok"] is False


def test_trust_region_rejection_restores_the_pre_block_state_exactly():
    runtime = ToyRuntime()
    session = build_toy_session()
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=2.0)
    # Calibrate an honest c_M from an unbounded pass, then shrink it below every
    # proposal: every block must be rejected whole and the state must not move.
    anchor, bank, decisions = _commit_rows(runtime, session, 4, thresholds, n_rows=4, c_M=None)
    committed_d2 = [decision.d2_unprojected for decision in decisions if decision.committed]
    assert committed_d2
    tiny = 1.0e-9 * min(committed_d2)
    _anchor2, bank2, decisions2 = _commit_rows(runtime, session, 4, thresholds, n_rows=4, c_M=tiny)
    assert not any(decision.committed for decision in decisions2)
    assert all(decision.block_reason == "support_trust_region_exceeded" for decision in decisions2
               if decision.d2_unprojected is not None)
    assert len(bank2) == 0
    assert bank2.digest == gate.EvidenceBankP.empty().digest
    # The drop-last-block inverse of the committed bank: removing and re-adding
    # the same block reproduces the exact refit (the recomputation law).
    assert len(bank) >= 1
    refit_full = gate.refit_from_anchor_p(anchor, bank, rho_M=1.0)
    dropped = bank.drop_last_block(1)
    assert len(dropped) == len(bank) - 1
    assert not np.array_equal(
        refit_full.candidate_t4, gate.refit_from_anchor_p(anchor, dropped, rho_M=1.0).candidate_t4,
    )
    restored = dropped.with_row(bank.rows[-1])
    assert restored.digest == bank.digest
    assert np.array_equal(
        gate.refit_from_anchor_p(anchor, restored, rho_M=1.0).candidate_t4,
        refit_full.candidate_t4,
    )


def test_empty_bank_parity_and_rho_scaling():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor = _toy_anchor(runtime, session, 4)
    parity = gate.empty_bank_refit_parity_p(anchor, rho_M=1.0)
    assert parity["candidate_t4_sha256"] == parity["support_t4_sha256"]
    # The published empty-bank carrier is the sealed support carrier bitwise;
    # the raw float64 solve differs only by the production float32 quantization.
    assert max(parity["max_abs_coefficient_difference_by_group"]) <= 1.0e-5
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=2.0)
    _anchor, bank, _decisions = _commit_rows(runtime, session, 4, thresholds, n_rows=2)
    rho_half = gate.refit_from_anchor_p(anchor, bank, rho_M=0.5)
    rho_one = gate.refit_from_anchor_p(anchor, bank, rho_M=1.0)
    for group in range(cdm_core.GROUP_COUNT):
        assert np.allclose(
            rho_half.At[group], anchor.per_group[group].A0
            + 0.5 * (rho_one.At[group] - anchor.per_group[group].A0),
            atol=1.0e-12, rtol=0.0,
        )
    with pytest.raises(gate.StagePGateError):
        gate.refit_from_anchor_p(anchor, bank, rho_M=-0.1)


def test_bank_rejects_duplicate_trials_and_index_drift():
    runtime = ToyRuntime()
    session = build_toy_session()
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=2.0)
    anchor, bank, _decisions = _commit_rows(runtime, session, 4, thresholds, n_rows=2)
    row = bank.rows[-1]
    with pytest.raises(gate.StagePGateError):
        bank.with_row(row)  # duplicate trial
    with pytest.raises(gate.StagePGateError):
        gate.evaluate_block(
            anchor, bank, row.labels,
            facts=row.facts, measurement_sha256=row.measurement_sha256,
            rho_M=1.0, c_M=None, thresholds=thresholds, block_index=len(bank) + 5,
        )


# ---------------------------------------------------------------------------
# 3. The toy state machine: causality, no-ops, leakage, receipts.
# ---------------------------------------------------------------------------


def _assert_causality(runtime, rollout, session, budget):
    assert stage_o_replay.verify_causality_receipts(rollout["receipts"])
    forward_states = {trial_id: digest for trial_id, digest, _vel in runtime.forward_log}
    for receipt in rollout["receipts"]:
        assert forward_states[receipt["trial_id"]] == receipt["sb"]


def test_p2_rollout_commits_and_never_touches_its_own_trial():
    runtime = ToyRuntime()
    session = build_toy_session()
    budget = 4
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0)
    # Calibrate c_M from an unbounded pass (the Stage-O law, within-session toy).
    unbounded = _run_rollout(runtime, session, budget, "P2", _default_hp(thresholds=thresholds, c_M=None))
    c_M = trust_region.calibrate_c_M([float(item) for item in unbounded["d2_values"]])
    hp = _default_hp(thresholds=thresholds, c_M=c_M)
    rollout = _run_rollout(runtime, session, budget, "P2", hp)
    anchor_digest_before = rollout["anchor"].a0_b0_digest

    _assert_causality(runtime, rollout, session, budget)
    committed = [item for item in rollout["receipts"] if item["acc"]]
    assert committed, "the toy P2 run must commit carrier updates"
    for receipt in committed:
        assert receipt["ca"] != receipt["cb"]
        assert receipt["sa"] != receipt["sb"]
        assert receipt["d2"] <= c_M + 1.0e-9
    rejected = [item for item in rollout["receipts"] if not item["acc"]]
    assert all(item["cb"] == item["ca"] for item in rejected)
    assert any(item["reason"].startswith("measurement_rejected:")
               or item["reason"] in ("direction_coverage_rejected_all_groups",
                                     "support_trust_region_exceeded",
                                     "block_committed_zero_movement")
               for item in rejected) or rejected == []

    # The committed update provably did not affect its own supplying trial: the
    # recorded forward read exactly the pre-reveal state digest, and re-running
    # the same trial's forward under the POST-commit carrier differs.
    reference = ToyRuntime()
    reference._session = session
    memory, _initial = reference._initial_memory(session=session, budget=budget)
    config = memory.state.carrier.config
    bank = gate.EvidenceBankP.empty()
    reference_raws: list[np.ndarray] = []
    moved_carrier = None
    for trial_id in session.query_trial_ids[budget]:
        trial = session.trials_by_id[trial_id]
        inputs = memory.read_prediction_inputs()
        raw = reference._governing_forward(trial=trial, inputs=inputs)
        reference_raws.append(raw)
        moved_carrier = None
        measurement = _measure(reference, session, trial)
        rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        if measurement.accepted:
            decision = gate.evaluate_block(
                rollout["anchor"], bank, measurement.per_group,
                facts=_trial_facts(session, trial, rates), measurement_sha256=measurement.digest,
                rho_M=hp.rho_M, c_M=c_M, thresholds=thresholds, block_index=len(bank),
            )
            if decision.committed:
                bank = bank.with_row(decision.row)
                outcome = trust_region.active_carrier(
                    rollout["anchor"], decision.coefficients, alpha_M=hp.alpha_M, c_M=c_M,
                )
                carrier = cdm_core.CarrierMemory(
                    groups=memory.state.carrier.groups,
                    initial_raw_t4=memory.state.carrier.initial_raw_t4,
                    statistics=rollout["anchor"].statistics,
                    ordinary_t4=outcome.active_t4, fixed_ridge_t4=outcome.active_t4,
                    config=config,
                )
                if carrier.digest != memory.state.carrier.digest:
                    stepped = cdm_core.IndependentActivityCausalDualMemory(
                        activity=memory.state.activity, carrier=carrier,
                    )
                    post_inputs = stepped.read_prediction_inputs()
                    assert post_inputs.state_digest != inputs.state_digest
                    post_raw = reference._governing_forward(trial=trial, inputs=post_inputs)
                    assert not np.array_equal(np.asarray(post_raw), np.asarray(raw))
                    moved_carrier = carrier
        activity_candidate, changed = memory._activity_candidate(trial.b3s_activity)
        carrier_after = memory.state.carrier
        if moved_carrier is not None:
            carrier_after = moved_carrier
        candidate_state = cdm_core.DualMemoryState(
            activity=activity_candidate.activity, carrier=carrier_after,
            committed_query_trials=activity_candidate.committed_query_trials,
        )
        memory.commit_independent(cdm_core.IndependentActivityPendingTrialUpdate(
            base_state_digest=inputs.state_digest, activity_transition_ready=True,
            activity_fifo_changed=changed,
            carrier_transition_accepted=moved_carrier is not None,
            activity_rejection_reason=None,
            carrier_rejection_reason=None if moved_carrier is not None
            else cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
            pseudo_directions=(), scalar_rates=None,
            carrier_proposal=None if moved_carrier is None
            else cdm_core.CarrierProposal(True, None, moved_carrier, (), None),
            candidate_state=candidate_state, fallback=memory.read_prediction_inputs(),
        ))
    # The reference trace reproduces the rollout's predictions bit-for-bit.
    assert len(reference_raws) == len(rollout["per_trial_raw"])
    for left, right in zip(reference_raws, rollout["per_trial_raw"]):
        assert np.array_equal(np.asarray(left), np.asarray(right))

    # Anchor immutability across every commit.
    assert _toy_anchor(ToyRuntime(), session, budget).a0_b0_digest == anchor_digest_before


def test_alpha_zero_and_m30_are_the_bit_exact_p0_no_op():
    runtime = ToyRuntime()
    session = build_toy_session(seed=11)
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0)
    runtime._session = session
    p0 = stage_o_replay.rollout_o0(runtime, session=session, budget=4)
    for row in ("P1", "P2", "P3", "P4", "P5"):
        zero = _run_rollout(
            ToyRuntime(), session, 4, row,
            _default_hp(thresholds=thresholds, alpha_M=0.0, c_M=None),
        )
        for left, right in zip(zero["per_trial_raw"], p0["per_trial_raw"]):
            assert np.array_equal(np.asarray(left), np.asarray(right))
        assert all(not item["acc"] for item in zero["receipts"])
        assert all(item["cb"] == item["ca"] for item in zero["receipts"])
        # Blocks still commit their labels into the bank at zero movement.
        assert any(item["reason"] == "block_committed_zero_movement" for item in zero["receipts"])
        assert zero["committed_rows"] > 0
        assert all(item["mv"] == 0.0 for item in zero["receipts"] if "mv" in item)
    # M30: the capacity-zero FIFO is an activity no-op and alpha_M = 0 keeps the
    # carrier byte-identical, so P2@M30 == P0@M30 bit-exactly.
    runtime._session = session
    p0_m30 = stage_o_replay.rollout_o0(runtime, session=session, budget=30)
    m30 = _run_rollout(
        ToyRuntime(), session, 30, "P2",
        _default_hp(thresholds=thresholds, alpha_M=0.0, c_M=None),
    )
    for left, right in zip(m30["per_trial_raw"], p0_m30["per_trial_raw"]):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert all(item["ab"] == item["aa"] for item in m30["receipts"])


def test_leakage_flags_and_session_row_receipts():
    runtime = ToyRuntime()
    session = build_toy_session()
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=2.0)
    hp = _default_hp(thresholds=thresholds)
    rollout = _run_rollout(runtime, session, 4, "P2", hp)
    assert rollout["leakage_flags"] == {
        "target_label_leakage": False, "self_referential_leakage": False,
        "checkpoint_selection_from_target": False, "hyperparameter_selection_from_target": False,
    }
    p5 = _run_rollout(ToyRuntime(), session, 4, "P5", hp)
    assert p5["leakage_flags"]["self_referential_leakage"] is True
    assert p5["leakage_flags"]["target_label_leakage"] is False
    targets, masks, _sst = runtime._session_target_views(session, 4)
    row = replay._session_row(
        runtime, row_id="P2", budget=4, session=session, targets=targets, masks=masks,
        rollout=rollout, hp=hp,
    )
    assert row["causality_state_chain_verified"] is True
    assert row["schema"] == "support_anchored_t4_stage_p_row_session_v1"
    assert row["carrier_transitions_committed"] == sum(1 for item in rollout["receipts"] if item["acc"])
    assert row["movement"]["rho_M"] == 1.0
    assert row["support_anchor"]["a0_b0_sha256"] == rollout["anchor"].a0_b0_digest
    assert len(row["per_trial_receipts"]) == len(session.query_trial_ids[4])
    assert row["committed_label_counts"] == rollout["committed_label_counts"]
    assert sum(row["committed_label_counts"]) >= 1
    p5_row = replay._session_row(
        runtime, row_id="P5", budget=4, session=session, targets=targets, masks=masks,
        rollout=p5, hp=hp, governing=False,
    )
    assert p5_row["governing"] is False


def test_p4_and_p3_controls_run_the_same_gate_law():
    runtime = ToyRuntime()
    session = build_toy_session()
    thresholds = gate.GateThresholds(tau_d=math.pi / 4.0, r_max=8, d_min=1, max_mass_relative=2.0)
    hp = _default_hp(thresholds=thresholds)
    rows = {row: _run_rollout(ToyRuntime(), session, 4, row, hp) for row in ("P2", "P3", "P4")}
    for row, rollout in rows.items():
        assert stage_o_replay.verify_causality_receipts(rollout["receipts"])
    # All three controls commit evidence under the same three-factor law.
    assert all(rollout["committed_rows"] > 0 for rollout in rows.values())
    # The shuffled binding differs from the correct one somewhere.
    assert rows["P2"]["bank_digest_chain"][-1] != rows["P4"]["bank_digest_chain"][-1]


# ---------------------------------------------------------------------------
# 4. The §8 gates (epsilon band as disclosure only) and diagnostics.
# ---------------------------------------------------------------------------


def _gate_view(
    *,
    p0: float = 0.40,
    m4_p2_delta: float = 0.0, m4_p2_positive: int = 0,
    m4_p1_delta: float = 0.0,
    m4_p3_delta: float = 0.0, m4_p4_delta: float = 0.0, m4_p5_delta: float = 0.0,
    m10_p2_delta: float = 0.0, m10_p2_positive: int = 0,
    m10_p1_delta: float = 0.0, m10_p3_delta: float = 0.0, m10_p4_delta: float = 0.0,
    within_delta: float = 0.0, m30_delta: float = 0.0,
    n_external: int = 15, n_within: int = 6,
) -> dict:
    def external_per_session(delta: float, positive: int) -> dict[str, float]:
        negative = -0.05
        if positive <= 0:
            return {f"e{i}": delta for i in range(n_external)}
        bumped = (n_external * delta - (n_external - positive) * negative) / positive
        return {f"e{i}": (bumped if i < positive else negative) for i in range(n_external)}

    def row_values(delta: float, positive: int, roster: list[str], within: bool):
        if within:
            return {name: delta for name in roster}
        return external_per_session(delta, positive)

    view: dict[str, Any] = {}
    for budget, (p2_delta, p2_positive, p1, p3, p4, p5) in (
        (4, (m4_p2_delta, m4_p2_positive, m4_p1_delta, m4_p3_delta, m4_p4_delta, m4_p5_delta)),
        (10, (m10_p2_delta, m10_p2_positive, m10_p1_delta, m10_p3_delta, m10_p4_delta, 0.0)),
    ):
        external_roster = [f"e{i}" for i in range(n_external)]
        within_roster = [f"w{i}" for i in range(n_within)]
        view[f"m{budget}"] = {
            "external": {
                "P0": {"mean_r2": p0, "per_session": {s: p0 for s in external_roster}},
                "P1": {"mean_r2": p0 + p1, "per_session": {s: p0 + p1 for s in external_roster}},
                "P2": {"mean_r2": p0 + p2_delta,
                       "per_session": {s: p0 + v for s, v in row_values(p2_delta, p2_positive, external_roster, False).items()}},
                "P3": {"mean_r2": p0 + p3, "per_session": {s: p0 + p3 for s in external_roster}},
                "P4": {"mean_r2": p0 + p4, "per_session": {s: p0 + p4 for s in external_roster}},
                "P5": {"mean_r2": p0 + p5, "per_session": {s: p0 + p5 for s in external_roster}},
            },
            "within": {
                row: {"mean_r2": p0 + (0.0 if row == "P0" else within_delta),
                      "per_session": {s: p0 + (0.0 if row == "P0" else within_delta)
                                      for s in within_roster}}
                for row in ("P0", "P1", "P2", "P3", "P4", "P5")
            },
        }
    view["m30"] = {
        surface: {
            row: {"mean_r2": p0 + m30_delta,
                  "per_session": ({f"e{i}": p0 + m30_delta for i in range(n_external)} if surface == "external"
                                  else {f"w{i}": p0 + m30_delta for i in range(n_within)})}
            for row in ("P0", "P1", "P2", "P3", "P4", "P5")
        }
        for surface in ("within", "external")
    }
    return view


_SAFETY = {key: True for key in gates.SAFETY_CONDITIONS}


def test_margin_boundary_is_exact_and_epsilon_is_only_a_band():
    verdict = gates.margin_verdict(0.01, 0.01)
    assert verdict["meets_margin"] is True
    assert verdict["within_epsilon_band_of_boundary"] is False
    below = gates.margin_verdict(0.01 - 1.0e-13, 0.01)
    assert below["meets_margin"] is False
    assert below["within_epsilon_band_of_boundary"] is True
    assert below["boundary_epsilon"] == 1.0e-12
    assert gates.margin_verdict(0.001, 0.01)["within_epsilon_band_of_boundary"] is False


def test_promotion_gate_requires_all_conditions_and_the_controls():
    view = _gate_view(m4_p2_delta=0.05, m4_p2_positive=10, m4_p3_delta=0.01, m4_p4_delta=0.0)
    result = gates.evaluate_promotion_gate(view, safety=_SAFETY)
    assert result["per_budget"]["m4"]["P2"]["promoted"] is True
    assert result["decision"] == plan.DISPOSITION_PROMOTE
    assert result["driving_cell"] == "P2@m4"
    # Losing to the constant-confidence control blocks the promotion.
    constant_wins = _gate_view(m4_p2_delta=0.05, m4_p2_positive=10, m4_p3_delta=0.06)
    promotion = gates.evaluate_promotion_gate(constant_wins, safety=_SAFETY)
    assert promotion["per_budget"]["m4"]["P2"]["promoted"] is False
    assert promotion["decision"] == plan.DISPOSITION_STOP
    assert gates.stop_conditions(constant_wins, promotion, safety=promotion["safety"])[
        "conditions"]["gains_disappear_under_constant_or_shuffle_controls"]["fired"] is True
    # Breadth below 10/15 blocks the promotion and fires the §8.4 stop row.
    narrow = _gate_view(m4_p2_delta=0.05, m4_p2_positive=8)
    promotion = gates.evaluate_promotion_gate(narrow, safety=_SAFETY)
    assert promotion["decision"] == plan.DISPOSITION_STOP
    stops = gates.stop_conditions(narrow, promotion, safety=promotion["safety"])
    assert stops["conditions"]["p1_p2_fail_to_beat_activity_only"]["fired"] is True
    assert stops["conditions"]["positive_gain_carried_by_fewer_than_10_of_15"]["fired"] is True


def test_promotion_within_regression_is_a_safety_condition():
    view = _gate_view(m4_p2_delta=0.05, m4_p2_positive=15, within_delta=-0.03)
    result = gates.evaluate_promotion_gate(view, safety=_SAFETY)
    assert result["safety"]["within_regression_bound"] is False
    assert result["decision"] == plan.DISPOSITION_STOP
    safe = gates.evaluate_promotion_gate(
        _gate_view(m4_p2_delta=0.05, m4_p2_positive=15, within_delta=-0.0199), safety=_SAFETY,
    )
    assert safe["decision"] == plan.DISPOSITION_PROMOTE
    for failing in ("p0_bit_anchor", "m30_exact_noop", "trust_region_no_committed_drift",
                    "hyperparameters_source_selected_only"):
        unsafe = dict(_SAFETY)
        unsafe[failing] = False
        assert gates.evaluate_promotion_gate(
            _gate_view(m4_p2_delta=0.05, m4_p2_positive=15), safety=unsafe,
        )["decision"] == plan.DISPOSITION_STOP


def test_other_low_budget_floor_and_attribution_rows():
    view = _gate_view(m4_p2_delta=0.05, m4_p2_positive=15, m10_p2_delta=-0.02)
    result = gates.evaluate_promotion_gate(view, safety=_SAFETY)
    assert result["per_budget"]["m4"]["P2"]["promoted"] is False
    continuity = gates.continuity_attribution(_gate_view(m4_p1_delta=0.02, m4_p2_delta=0.05,
                                                         m4_p2_positive=12))
    assert continuity["per_budget"]["m4"]["claim_supported"] is True
    null = gates.continuity_attribution(_gate_view(m4_p1_delta=0.05, m4_p2_delta=0.055,
                                                   m4_p2_positive=12))
    assert null["per_budget"]["m4"]["claim_supported"] is False
    confidence = gates.confidence_attribution(_gate_view(m4_p2_delta=0.05, m4_p3_delta=0.01,
                                                         m4_p4_delta=0.0))
    assert confidence["per_budget"]["m4"]["claim_supported"] is True
    tied = gates.confidence_attribution(_gate_view(m4_p2_delta=0.05, m4_p3_delta=0.05))
    assert tied["per_budget"]["m4"]["claim_supported"] is False


def test_interpretation_rows_and_movement_ordering():
    view = _gate_view(m4_p1_delta=0.05, m4_p2_delta=0.0)
    promotion = gates.evaluate_promotion_gate(view, safety=_SAFETY)
    rows = gates.interpretation_rows(view, promotion)
    assert rows["p1_positive_p2_null"]["value"] is True
    assert rows["confidence_null"]["value"] is False
    stopped = _gate_view()
    stopped_promotion = gates.evaluate_promotion_gate(stopped, safety=_SAFETY)
    assert gates.interpretation_rows(stopped, stopped_promotion)[
        "headroom_without_deployable_measurement"]["value"] is True
    assert gates.movement_ordering({"m4": 1.0, "m10": 0.5, "m30": 0.0})["ordering_holds"] is True
    assert gates.movement_ordering({"m4": 0.1, "m10": 0.5, "m30": 0.0})["ordering_holds"] is False
    ordering = gates.movement_ordering({"m4": 1.0, "m10": 0.5, "m30": 0.0})
    assert ordering["m30_zero_by_construction"] is True
    with pytest.raises(gates.StagePGateError):
        gates.movement_ordering({"m4": 1.0, "m10": 0.5})


def test_stop_conditions_cover_the_design_8_4_rows():
    view = _gate_view(m4_p2_delta=0.05, m4_p2_positive=15)
    promotion = gates.evaluate_promotion_gate(view, safety=_SAFETY)
    healthy = gates.stop_conditions(view, promotion, safety=promotion["safety"])
    assert healthy["any_fired"] is False
    null = _gate_view()
    promotion = gates.evaluate_promotion_gate(null, safety=_SAFETY)
    stopped = gates.stop_conditions(null, promotion, safety=promotion["safety"])
    assert stopped["conditions"]["p1_p2_fail_to_beat_activity_only"]["fired"] is True
    assert stopped["conditions"]["o2_stop_after_initial_run_or_permitted_hold"]["fired"] is False
    unsafe = dict(promotion["safety"])
    unsafe["zero_target_updates"] = False
    unsafe["causality_state_chains"] = False
    broken = gates.stop_conditions(null, promotion, safety=unsafe)
    assert broken["conditions"]["updates_require_target_gradients_or_decoder_changes"]["fired"] is True
    assert broken["conditions"]["oracle_refit_causality_leak"]["fired"] is True


# ---------------------------------------------------------------------------
# 5. The source-only selection law.
# ---------------------------------------------------------------------------


def test_selection_grid_enumeration_and_first_maximum_tie_break():
    assert len(replay.enumerate_gate_grid()) == 16
    assert len(replay.enumerate_mass_grid()) == 8
    assert replay.enumerate_mass_grid()[0] == (0.5, 0.0)  # less movement first
    scored = [
        {"vector": "a", "mean_r2": 0.30, "per_session": {"s": 0.30}},
        {"vector": "b", "mean_r2": 0.32, "per_session": {"s": 0.32}},
        {"vector": "c", "mean_r2": 0.32, "per_session": {"s": 0.32}},
    ]
    assert replay.select_vector(scored)["vector"] == "b"  # first maximum wins
    with pytest.raises(replay.StagePReplayError):
        replay.select_vector([])


def test_selection_pass_selects_from_the_grid_on_within_sessions_only():
    runtime = ToyRuntime()
    session = build_toy_session()
    runtime._session = session
    selection = replay._selection_pass(
        runtime, budget=4, within_sessions=[session], started=time.monotonic(),
    )
    assert selection["budget"] == 4
    assert len(selection["stage1_scored"]) == 16
    assert len(selection["stage2_scored"]) == 8
    assert selection["stage1_selected"] == replay.select_vector(selection["stage1_scored"])["vector"]
    assert selection["stage2_selected"] == replay.select_vector(selection["stage2_scored"])["vector"]
    selected = selection["selected"]
    assert selected["alpha_M"] in plan.ALPHA_M_CANDIDATES
    assert selected["rho_M"] in plan.RHO_M_CANDIDATES
    assert selected["thresholds"]["r_max_repetition_per_direction"] in plan.R_MAX_CANDIDATES
    assert selected["thresholds"]["d_min_distinct_directions"] in plan.D_MIN_CANDIDATES
    assert selected["thresholds"]["tau_d_rad"] in plan.TAU_D_CANDIDATES
    assert selected["thresholds"]["max_pseudo_mass_relative_to_support_rows"] in (
        plan.MAX_PSEUDO_MASS_RELATIVE_CANDIDATES
    )
    assert selection["c_M_calibration"]["c_M"] > 0.0
    assert selection["c_M_calibration"]["law"]["selection_surface"] == (
        "within-6 ONLY; external-15 never selects"
    )


# ---------------------------------------------------------------------------
# 6. Pre-registration pinning, the GO anchor and the stage guards.
# ---------------------------------------------------------------------------


def test_pre_registration_payload_is_pinned():
    payload = plan.pre_registration_payload()
    assert payload["row_order"] == ["P0", "P1", "P2", "P3", "P4", "P5"]
    assert payload["commit_law"]["block"] == "1 completed trial (the binding work-order law)"
    assert payload["hyperparameters"]["fixed_predeclared"]["block_size_completed_trials"] == 1
    assert payload["hyperparameters"]["m30"]["alpha_M"] == 0.0
    assert payload["hyperparameter_grid"]["alpha_M"] == [0.0, 0.125, 0.25, 0.5]
    assert payload["hyperparameter_grid"]["rho_M"] == [0.5, 1.0]
    assert len(payload["hyperparameter_grid"]["tau_d_rad"]) == 2
    assert payload["selection"]["surface"] == "within-6 ONLY; external-15 never selects"
    assert payload["c_m_calibration"]["selection_surface"] == "within-6 ONLY; external-15 never selects"
    assert payload["gates"]["boundary_epsilon"] == 1.0e-12
    assert payload["gates"]["promotion"]["conditions"] == [
        "candidate - activity_only >= +0.01 equal-session external R2",
        "positive external sessions >= 10/15",
        "other low budget >= activity_only - 0.01",
        "within every budget >= activity_only - 0.02",
        "M30 carrier state is exact no-op",
        "target optimizer/backward/model/normalizer updates = 0",
    ]
    assert payload["gates"]["stop_conditions"]["stop_conditions"] == [
        "O2 receives a STOP decision after the initial run or the one permitted HOLD sensitivity check",
        "oracle refit improves only the current trial through a causality leak",
        "P1/P2 fail to beat activity-only CDM",
        "gains disappear under constant/shuffle controls",
        "positive aggregate gain is carried by fewer than 10/15 external sessions",
        "M4/M10 gains require target-selected thresholds",
        "carrier updates require target gradients or decoder changes",
        "repeated carrier proposals drift outside the support trust region",
    ]
    assert payload["leakage_flags_by_row"]["P5"]["self_referential_leakage"] is True
    assert payload["leakage_flags_by_row"]["P2"]["target_label_leakage"] is False
    assert payload["cells"]["P5"]["governing"] is False
    plan.validate_pre_registration(payload)
    for mutation in ("alpha_M", "rho_M"):
        drifted = json.loads(json.dumps(payload))
        drifted["hyperparameter_grid"][mutation] = [0.123456]
        with pytest.raises(ValueError):
            plan.validate_pre_registration(drifted)
    drifted = json.loads(json.dumps(payload))
    drifted["hyperparameters"]["m30"]["alpha_M"] = 0.5
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)


def test_stage_o_go_anchor_verification():
    anchor = replay.verify_stage_o_go_anchor(ROOT)
    assert anchor["verified"] is True
    assert anchor["decision"] == plan.STAGE_O_GO_ANCHOR["decision"]
    assert anchor["m4_external_o2_minus_o0"] == plan.STAGE_O_GO_ANCHOR["m4_external_o2_minus_o0"]
    assert anchor["m10_external_o2_minus_o0"] == plan.STAGE_O_GO_ANCHOR["m10_external_o2_minus_o0"]


def test_run_stage_p_replay_refuses_without_its_prerequisites(tmp_path):
    empty = tmp_path / "stage-p"
    empty.mkdir()
    with pytest.raises(replay.StagePReplayError):
        replay.run_stage_p_replay(ROOT, gpu_index=1, output_root=empty)
    (empty / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StagePReplayError):
        replay.run_stage_p_replay(ROOT, gpu_index=1, output_root=empty)
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "attempt.json").write_text("{}", encoding="utf-8")
    (closed / "replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StagePReplayError):
        replay.run_stage_p_replay(ROOT, gpu_index=1, output_root=closed)
    terminal = tmp_path / "terminalized"
    terminal.mkdir()
    (terminal / "attempt.json").write_text("{}", encoding="utf-8")
    (terminal / "terminal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StagePReplayError):
        replay.run_stage_p_replay(ROOT, gpu_index=1, output_root=terminal)


def test_terminal_builder_end_to_end_on_a_synthetic_matrix():
    within_roster = [f"w{i}" for i in range(6)]
    external_roster = [f"e{i}" for i in range(15)]
    matrix: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        key = f"m{budget}"
        matrix[key] = {}
        for surface, roster in (("within", within_roster), ("external", external_roster)):
            rows = {}
            for row, delta in (("P0", 0.0), ("P1", 0.005 if budget == 4 else -0.002),
                               ("P2", 0.05 if budget == 4 else -0.002),
                               ("P3", 0.01 if budget == 4 else -0.002),
                               ("P4", 0.0 if budget == 4 else -0.002),
                               ("P5", -0.02 if budget == 4 else -0.002)):
                per_session = {}
                for index, session in enumerate(roster):
                    bump = delta
                    if row == "P2" and budget == 4:
                        bump = 0.051 if index < 11 else -0.06
                    per_session[session] = 0.40 + bump
                rows[row] = {"mean_r2": sum(per_session.values()) / len(roster),
                             "per_session": per_session}
            matrix[key][surface] = rows
    replay_payload = {
        "matrix": matrix,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
        "model_state_digest_unchanged": True,
        "causality_state_chains_all_rows": True,
        "trust_region_no_committed_drift": True,
        "movement_by_budget_medians": {"m4": 2.0, "m10": 1.0, "m30": 0.0},
        "hyperparameter_selection": {"law": {"surface": plan.SELECTION["surface"]}},
        "anchors": {
            "sealed_activity_only_all_exact": True,
            "m30_noop_all_rows": True,
            "stage_o_go": {"decision": plan.STAGE_O_GO_ANCHOR["decision"]},
        },
    }
    body = replay.build_terminal(replay_payload=replay_payload)
    gate = body["gate"]
    assert gate["safety_all_pass"] is True
    assert gate["driving_cell"] == "P2@m4"
    assert gate["per_budget"]["m4"]["P2"]["positive_external_sessions"] == 11
    assert body["continuity_attribution"]["per_budget"]["m4"]["claim_supported"] is True
    assert body["confidence_attribution"]["per_budget"]["m4"]["claim_supported"] is True
    assert body["movement_ordering"]["ordering_holds"] is True
    assert body["paired_contrasts"]["P2_minus_P0_m4_external"]["n_total"] == 15
    assert body["stop_conditions"]["any_fired"] is False
    drifted = dict(replay_payload, matrix={"m4": matrix["m4"]})
    with pytest.raises(gates.StagePGateError):
        replay.build_terminal(replay_payload=drifted)
