"""Synthetic no-data/no-CUDA tests for Support-Anchored T4 Stage O (V1).

Everything runs on constructed numpy blocks and the real imported frozen
``src.causal_dual_memory_cell_d_v1.core`` evidence objects plus the real
frozen P2' filters/policy construction law.  The review-critical properties:

1. causality -- an update committed from trial i provably cannot affect trial
   i's own predictions (the forward's state digest, the receipt chain, and a
   post-commit recomputation that differs);
2. anchor immutability -- A0/b0/support digests unchanged across commits;
3. the block-refit recomputation law -- always from A0/b0 + bank, and removing
   a rejected block reproduces the pre-block state exactly;
4. trust-region projection math -- D2 after projection <= c_M, alpha_M = 0 is
   the bit-exact identity, norm monotonicity, movement bound;
5. O0 bit-anchor comparison logic;
6. §3.5 gate boundary semantics (epsilon 1e-12 as a disclosed band only) and
   the interpretation / movement-ordering rows;
7. the whole state machine on toy carriers, including O2(alpha=0) == O0
   bit-exactly, the c_M calibration invariance and the direction-table
   double-entry check.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from src.support_anchored_t4_stage_o_v1 import block_refit, gates, plan, replay, trust_region

CANON = cdm_core.CANONICAL_DIRECTIONS_RAD
N_UNITS = 8
SESSION_ID = "toy-session"


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


def build_toy_session(seed: int = 7, *, n_query: int = 6, surface: str = "within") -> ToySession:
    rng = np.random.default_rng(seed)
    channel_digest = _channel_digest()
    channel_ids = np.arange(N_UNITS, dtype=np.int64)
    support_directions = [0, 2, 5, 0, 1, 3, 6, 4, 7, 2, 0, 5, 1, 4, 6, 3, 7, 0, 2, 5, 1, 3, 4, 6, 7, 0, 2, 5, 1, 3]
    query_directions = [1, 3, 6, 4, 7, 2][:n_query]
    trials: dict[str, ToyTrial] = {}
    behavior = np.zeros((40 * 40, 2), dtype=np.float32)
    support_rates: dict[str, np.ndarray] = {}
    support_direction_indices: dict[str, int] = {}
    position = 0

    def _make(trial_id: str, direction: int, windows: int, length: int) -> ToyTrial:
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
        trials[trial_id] = _make(trial_id, direction, 25, 36)
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
    """The minimal runtime surface the Stage-O rollouts consume (toy forward)."""

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
        """The toy sealed activity-only loop (the O0 arm)."""
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
        }


def _toy_table_cell(runtime: ToyRuntime, session: ToySession, budget: int) -> dict[str, Any]:
    """The digest-pinned direction cell for the toy session (the table double)."""
    memory, initial = runtime._initial_memory(session=session, budget=budget)
    config = memory.state.carrier.config
    rows = []
    for trial_id in session.query_trial_ids[budget]:
        trial = session.trials_by_id[trial_id]
        direction, meta = replay.true_direction_for_trial(session, trial, config=config)
        rows.append(replay._direction_row_payload(trial, direction, meta))
    binding = replay._binding_payload(budget, list(session.support_trial_ids[budget]), initial, config)
    digest = hashlib.sha256(json.dumps(
        {"rows": rows, "binding": dict(binding)}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {"cell_digest": digest, "rows": rows}


def _toy_anchor(runtime: ToyRuntime, session: ToySession, budget: int) -> anchor_module.SupportAnchor:
    memory, _initial = runtime._initial_memory(session=session, budget=budget)
    carrier = memory.state.carrier
    rates, directions, _ids = replay._support_binding(session, budget)
    return anchor_module.SupportAnchor.from_labeled_support(
        groups=carrier.groups, support_trial_rates=rates,
        support_direction_indices=directions, support_t4=carrier.active_t4,
    )


def _evidence_row(session, trial, direction, rates, *, block_index: int) -> block_refit.EvidenceRow:
    return block_refit.EvidenceRow(
        session_id=session.session, trial_id=trial.trial_id,
        chronology_position=int(trial.chronology_position), block_index=block_index,
        direction_indices=tuple(int(direction.theta_index) for _ in range(cdm_core.GROUP_COUNT)),
        theta_raw_rad=tuple(float(direction.theta_raw_rad) for _ in range(cdm_core.GROUP_COUNT)),
        canonical_distance_rad=tuple(float(direction.canonical_distance_rad) for _ in range(cdm_core.GROUP_COUNT)),
        movement_bins=tuple(int(direction.movement_bins) for _ in range(cdm_core.GROUP_COUNT)),
        displacement_norm=tuple(float(direction.displacement_norm) for _ in range(cdm_core.GROUP_COUNT)),
        mean_speed=tuple(float(direction.mean_speed) for _ in range(cdm_core.GROUP_COUNT)),
        scalar_rates=rates,
        rate_sha256=cdm_core.array_digest(rates),
        native_counts_sha256=cdm_core.array_digest(trial.native_counts.counts),
        channel_order_sha256=cdm_core.channel_order_digest(session.channel_ids),
        valid_mask_sha256=cdm_core.array_digest(np.ones(N_UNITS, dtype=np.bool_)),
    )


def _toy_bank(runtime, session, budget, n_rows):
    anchor = _toy_anchor(runtime, session, budget)
    memory, _initial = runtime._initial_memory(session=session, budget=budget)
    config = memory.state.carrier.config
    bank = block_refit.EvidenceBank.empty()
    for trial_id in list(session.query_trial_ids[budget])[:n_rows]:
        trial = session.trials_by_id[trial_id]
        direction, _meta = replay.true_direction_for_trial(session, trial, config=config)
        assert direction.accepted
        rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        bank = bank.with_row(_evidence_row(session, trial, direction, rates, block_index=len(bank)))
    return anchor, bank


# ---------------------------------------------------------------------------
# 1. The support anchor.
# ---------------------------------------------------------------------------


def test_anchor_reproduces_the_production_convention():
    runtime = ToyRuntime()
    session = build_toy_session()
    memory, _initial = runtime._initial_memory(session=session, budget=10)
    carrier = memory.state.carrier
    rates, directions, _ids = replay._support_binding(session, 10)
    anchor = anchor_module.SupportAnchor.from_labeled_support(
        groups=carrier.groups, support_trial_rates=rates,
        support_direction_indices=directions, support_t4=carrier.active_t4,
    )
    parity = anchor.support_coefficient_parity
    assert parity["rebuilt_rows_bitwise_equal"] is True
    assert parity["max_relative_coefficient_difference"] <= 1.0e-6
    reference = cdm_core.CarrierSufficientStatistics.from_labeled_support(
        carrier.groups, rates, directions,
    )
    assert anchor.statistics.digest == reference.digest
    theta = np.asarray([CANON[int(i)] for i in directions])
    design = np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size)))
    expected = design.T @ design + np.diag((directions.size * 0.1, directions.size * 0.1, 0.0))
    assert np.allclose(anchor.per_group[0].A0, expected, atol=1.0e-12, rtol=0.0)
    units = carrier.groups.unit_indices(0)
    assert np.allclose(anchor.per_group[0].b0, design.T @ rates[:, units], atol=1.0e-12, rtol=0.0)


def test_anchor_rebuild_from_support_coefficients_is_bit_exact():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor = _toy_anchor(runtime, session, 4)
    support_blocks = anchor.coefficients(anchor.support_t4)
    rebuilt = anchor.rebuild_t4(support_blocks)
    support = np.asarray(anchor.support_t4)
    assert rebuilt.dtype == support.dtype
    # The coefficient columns round-trip exactly; the derived hypot column is
    # recomputed from the quantized coefficients and can move by one float32 ulp
    # (the production _reconstruct law recomputes it the same way).  The exact
    # alpha_M = 0 identity is the short-circuit that returns the support array
    # itself, asserted in the trust-region tests.
    assert np.array_equal(rebuilt[:, [0, 1, 3]], support[:, [0, 1, 3]])
    hypot_gap = np.abs(rebuilt[:, 2] - support[:, 2])
    assert float(np.max(hypot_gap)) <= float(np.finfo(np.float32).eps * np.abs(support[:, 2]).max())


def test_anchor_is_immutable():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor = _toy_anchor(runtime, session, 4)
    digest_before = anchor.digest
    a0_before = anchor.a0_b0_digest
    with pytest.raises(ValueError):
        anchor.per_group[0].A0[0, 0] = 1.0  # type: ignore[index]
    assert anchor.digest == digest_before and anchor.a0_b0_digest == a0_before


# ---------------------------------------------------------------------------
# 2. Block refit: recomputation law and block removal.
# ---------------------------------------------------------------------------


def test_refit_is_recomputed_from_anchor_and_bank_only():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 3)
    first = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    second = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    assert first.digest == second.digest
    assert np.array_equal(first.candidate_t4, second.candidate_t4)
    incremental = block_refit.EvidenceBank.empty()
    for row in bank.rows:
        incremental = incremental.with_row(row)
    assert incremental.digest == bank.digest
    assert block_refit.refit_from_anchor(anchor, incremental, rho_M=1.0).digest == first.digest


def test_empty_bank_refit_returns_the_support_solve():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor = _toy_anchor(runtime, session, 4)
    parity = block_refit.empty_bank_refit_parity(anchor, rho_M=1.0)
    # The empty-bank candidate IS the sealed support carrier, bitwise.
    assert block_refit.refit_from_anchor(
        anchor, block_refit.EvidenceBank.empty(), rho_M=1.0,
    ).candidate_t4 is not None
    assert anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"] is True


def test_dropping_a_block_reproduces_the_pre_block_state_exactly():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 3)
    pre_block = block_refit.refit_from_anchor(anchor, bank.drop_last_block(1), rho_M=1.0)
    restored = bank.drop_last_block(1)
    assert restored.digest == _toy_bank(runtime, session, 4, 2)[1].digest
    after_restore = block_refit.refit_from_anchor(anchor, restored, rho_M=1.0)
    assert after_restore.digest == pre_block.digest
    assert np.array_equal(after_restore.candidate_t4, pre_block.candidate_t4)


def test_bank_rejects_duplicate_trials_and_non_monotone_chronology():
    runtime = ToyRuntime()
    session = build_toy_session()
    _anchor, bank = _toy_bank(runtime, session, 4, 2)
    row = bank.rows[-1]
    with pytest.raises(block_refit.BlockRefitError):
        bank.with_row(row)
    bad = block_refit.EvidenceRow(
        session_id=row.session_id, trial_id="other", chronology_position=0,
        block_index=99, direction_indices=row.direction_indices,
        theta_raw_rad=row.theta_raw_rad, canonical_distance_rad=row.canonical_distance_rad,
        movement_bins=row.movement_bins, displacement_norm=row.displacement_norm,
        mean_speed=row.mean_speed, scalar_rates=row.scalar_rates,
        rate_sha256=row.rate_sha256, native_counts_sha256=row.native_counts_sha256,
        channel_order_sha256=row.channel_order_sha256, valid_mask_sha256=row.valid_mask_sha256,
    )
    with pytest.raises(block_refit.BlockRefitError):
        bank.with_row(bad)


def test_rho_scales_the_evidence_moment_not_the_anchor():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 2)
    rho_one = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    rho_zero = block_refit.refit_from_anchor(anchor, bank, rho_M=0.0)
    empty = block_refit.refit_from_anchor(anchor, block_refit.EvidenceBank.empty(), rho_M=1.0)
    assert np.array_equal(rho_zero.candidate_t4, empty.candidate_t4)
    assert not np.array_equal(rho_one.candidate_t4, empty.candidate_t4)
    for group in range(cdm_core.GROUP_COUNT):
        assert np.array_equal(rho_zero.At[group], anchor.per_group[group].A0)
        assert np.array_equal(rho_zero.bt[group], anchor.per_group[group].b0)


# ---------------------------------------------------------------------------
# 3. Trust-region projection math.
# ---------------------------------------------------------------------------


def test_projection_restores_the_bound_and_is_monotone():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 2)
    refit = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    support_blocks = anchor.coefficients(anchor.support_t4)
    delta = [
        np.asarray(refit.coefficients[g]) - np.asarray(support_blocks[g])
        for g in range(cdm_core.GROUP_COUNT)
    ]
    d2, _per_group = trust_region.aggregate_d2(anchor, delta)
    assert d2 > 0.0
    tight = 0.25 * d2
    outcome = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=0.5, c_M=tight)
    assert outcome.projected is True
    assert outcome.d2_projected_scaled <= tight * (1.0 + 1.0e-9)
    assert outcome.d2_projected_scaled < outcome.d2_unprojected
    assert outcome.delta_projected_frobenius <= outcome.delta_unprojected_frobenius + 1.0e-12
    loose = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=0.5, c_M=4.0 * d2)
    assert loose.projected is False
    assert loose.d2_projected_scaled == loose.d2_unprojected


def test_alpha_zero_is_the_bit_exact_identity():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 2)
    refit = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    outcome = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=0.0, c_M=1.0e-12)
    assert np.array_equal(outcome.active_t4, np.asarray(anchor.support_t4))
    assert outcome.movement_frobenius == 0.0
    full = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=1.0, c_M=None)
    assert full.projected is False
    assert full.movement_frobenius > 0.0


def test_projected_movement_is_bounded_by_alpha_and_c_M():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 10, 3)
    refit = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    support_blocks = anchor.coefficients(anchor.support_t4)
    delta = [
        np.asarray(refit.coefficients[g]) - np.asarray(support_blocks[g])
        for g in range(cdm_core.GROUP_COUNT)
    ]
    for alpha in (0.125, 0.25, 0.5, 1.0):
        for c_M in (1.0e-6, 1.0e-2, 1.0):
            # Exact coefficient-space bound: D2(alpha * project(delta)) <= alpha^2 c_M.
            projected, _before, _after, _flag = trust_region.project_delta(anchor, delta, c_M=c_M)
            d2_moved, _per = trust_region.aggregate_d2(
                anchor, [float(alpha) * block for block in projected],
            )
            assert d2_moved <= alpha * alpha * c_M * (1.0 + 1.0e-12)
            # The published carrier additionally passes the production float32
            # quantization; its distance respects the bound up to that noise.
            outcome = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=alpha, c_M=c_M)
            moved = [
                np.asarray(outcome.active_t4, dtype=np.float64)[anchor.groups.unit_indices(g)][:, [0, 1, 3]].T
                - np.asarray(support_blocks[g])
                for g in range(cdm_core.GROUP_COUNT)
            ]
            d2_quantized, _per2 = trust_region.aggregate_d2(anchor, moved)
            # The published carrier is the production float32-quantized estimand:
            # below the quantization floor (a c_M of order 1e-6 at coefficient
            # magnitudes of order 50) the published D2 is bounded by the floor,
            # not by c_M.  The receipt records the exact pre-quantization D2.
            assert d2_quantized <= alpha * alpha * c_M * (1.0 + 1.0e-3) + 1.0e-5


def test_projection_and_alpha_domain_checks():
    runtime = ToyRuntime()
    session = build_toy_session()
    anchor, bank = _toy_bank(runtime, session, 4, 1)
    refit = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
    with pytest.raises(trust_region.TrustRegionError):
        trust_region.active_carrier(anchor, refit.coefficients, alpha_M=0.5, c_M=0.0)
    with pytest.raises(trust_region.TrustRegionError):
        trust_region.active_carrier(anchor, refit.coefficients, alpha_M=1.5, c_M=1.0)


def test_calibrate_c_M_is_the_median_of_d2():
    assert trust_region.calibrate_c_M([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(3.0)
    with pytest.raises(trust_region.TrustRegionError):
        trust_region.calibrate_c_M([])


# ---------------------------------------------------------------------------
# 4. Gate boundary semantics (§3.5) and diagnostics.
# ---------------------------------------------------------------------------


def _gate_view(
    *,
    o0: float = 0.40,
    m4_delta: float = 0.0,
    m4_positive: int = 0,
    m10_delta: float = 0.0,
    m10_positive: int = 0,
    m30_delta: float = 0.0,
    o1_delta: float = 0.0,
    within_delta: float = 0.0,
    n_external: int = 15,
    n_within: int = 6,
) -> dict:
    """Build a gate view with exact control of each budget's external delta."""
    def external_per_session(delta: float, positive: int) -> dict[str, float]:
        negative = -0.05
        if positive <= 0:
            return {f"e{i}": delta for i in range(n_external)}
        bumped = (n_external * delta - (n_external - positive) * negative) / positive
        return {
            f"e{i}": (bumped if i < positive else negative) for i in range(n_external)
        }

    view: dict[str, Any] = {}
    for budget, (delta, positive) in (
        (4, (m4_delta, m4_positive)), (10, (m10_delta, m10_positive)), (30, (m30_delta, 0)),
    ):
        key = f"m{budget}"
        view[key] = {}
        external_values = external_per_session(delta, positive)
        view[key]["external"] = {
            "O0": {"mean_r2": o0, "per_session": {s: o0 for s in external_values}},
            "O1": {"mean_r2": o0 + o1_delta,
                   "per_session": {s: o0 + o1_delta for s in external_values}},
            "O2": {"mean_r2": o0 + sum(external_values.values()) / n_external,
                   "per_session": {s: o0 + v for s, v in external_values.items()}},
        }
        view[key]["within"] = {
            "O0": {"mean_r2": o0, "per_session": {f"w{i}": o0 for i in range(n_within)}},
            "O1": {"mean_r2": o0 + o1_delta,
                   "per_session": {f"w{i}": o0 + o1_delta for i in range(n_within)}},
            "O2": {"mean_r2": o0 + within_delta,
                   "per_session": {f"w{i}": o0 + within_delta for i in range(n_within)}},
        }
    return view


_SAFETY = {key: True for key in plan.GATES["safety_conditions"]}


def test_margin_boundary_is_exact_and_epsilon_is_only_a_band():
    verdict = gates.margin_verdict(0.03, 0.03)
    assert verdict["meets_margin"] is True
    assert verdict["within_epsilon_band_of_boundary"] is False
    below = gates.margin_verdict(0.03 - 1.0e-13, 0.03)
    assert below["meets_margin"] is False
    assert below["within_epsilon_band_of_boundary"] is True
    assert below["boundary_epsilon"] == 1.0e-12
    clearly = gates.margin_verdict(0.02, 0.03)
    assert clearly["meets_margin"] is False
    assert clearly["within_epsilon_band_of_boundary"] is False


def test_gate_go_requires_all_conditions_on_a_low_budget():
    view = _gate_view(m4_delta=0.05, m4_positive=10, o1_delta=-0.01)
    result = gates.evaluate_oracle_gate(view, safety=_SAFETY)
    assert result["per_budget"]["m4"]["external"]["positive_sessions"] == 10
    assert result["per_budget"]["m4"]["decision_row"]["go_eligible"] is True
    assert result["decision"] == plan.DISPOSITION_GO
    assert result["driving_budget"] == "m4"


def test_gate_decision_follows_the_achieved_mean_exactly_near_the_boundary():
    # The exact->= semantics are checked relative to the ACHIEVED float64 mean,
    # so no ulp of the roster mean can flip the verdict either way.
    for target in (0.03, 0.03 + 1.0e-13, 0.03 - 1.0e-13):
        view = _gate_view(m4_delta=target, m4_positive=15)
        result = gates.evaluate_oracle_gate(view, safety=_SAFETY)
        achieved = result["per_budget"]["m4"]["decision_row"]["external_delta"]
        assert (result["decision"] == plan.DISPOSITION_GO) == bool(achieved >= 0.03)
        margin = result["per_budget"]["m4"]["decision_row"]["external_margin"]
        assert margin["meets_margin"] == bool(achieved >= 0.03)
        if target < 0.03:
            assert result["decision"] == plan.DISPOSITION_HOLD


def test_gate_hold_band_and_breadth():
    hold = gates.evaluate_oracle_gate(_gate_view(m4_delta=0.02, m4_positive=10), safety=_SAFETY)
    assert hold["decision"] == plan.DISPOSITION_HOLD
    assert plan.GATES["HOLD"]["authorizes"]
    narrow = gates.evaluate_oracle_gate(_gate_view(m4_delta=0.02, m4_positive=8), safety=_SAFETY)
    assert narrow["decision"] == plan.DISPOSITION_STOP


def test_gate_stop_below_the_hold_margin_and_on_any_safety_failure():
    stop = gates.evaluate_oracle_gate(_gate_view(m4_delta=0.005, m4_positive=15), safety=_SAFETY)
    assert stop["decision"] == plan.DISPOSITION_STOP
    go_view = _gate_view(m4_delta=0.05, m4_positive=15)
    for failing in ("o0_bit_anchor", "causality_state_chains", "zero_target_updates"):
        unsafe = dict(_SAFETY)
        unsafe[failing] = False
        assert gates.evaluate_oracle_gate(go_view, safety=unsafe)["decision"] == plan.DISPOSITION_STOP


def test_gate_within_regression_is_a_safety_condition():
    view = _gate_view(m4_delta=0.05, m4_positive=15, within_delta=-0.03)
    result = gates.evaluate_oracle_gate(view, safety=_SAFETY)
    assert result["within_regression_bound_pass"] is False
    assert result["decision"] == plan.DISPOSITION_STOP
    assert result["safety"]["within_regression_bound"] is False
    # A within delta that sits exactly at -0.02 in decimal lands one roster-mean
    # ulp below it in float64: the miss is disclosed inside the epsilon band and
    # still fails the exact >= (the band never widens the margin).  A clean pass
    # needs a delta strictly inside the bound in float64.
    at_band = gates.evaluate_oracle_gate(
        _gate_view(m4_delta=0.05, m4_positive=15, within_delta=-0.02), safety=_SAFETY,
    )
    band_verdict = at_band["within_bounds"]["m10"]
    assert at_band["within_regression_bound_pass"] == band_verdict["meets_margin"]
    safe = gates.evaluate_oracle_gate(
        _gate_view(m4_delta=0.05, m4_positive=15, within_delta=-0.0199), safety=_SAFETY,
    )
    assert safe["within_regression_bound_pass"] is True
    assert safe["decision"] == plan.DISPOSITION_GO


def test_other_low_budget_floor_blocks_go():
    result = gates.evaluate_oracle_gate(
        _gate_view(m4_delta=0.05, m4_positive=15, m10_delta=-0.02, m10_positive=0),
        safety=_SAFETY,
    )
    assert result["per_budget"]["m4"]["decision_row"]["go_eligible"] is False
    assert result["decision"] != plan.DISPOSITION_GO


def test_interpretation_rows_and_movement_ordering():
    rows = gates.interpretation_rows(_gate_view(o1_delta=-0.01, m4_delta=0.05, m4_positive=15))
    assert rows["recursive_estimator_at_fault"]["per_budget"]["m4"] is True
    assert rows["both_help"]["per_budget"]["m4"] is False
    both = gates.interpretation_rows(_gate_view(o1_delta=0.02, m4_delta=0.05, m4_positive=15))
    assert both["both_help"]["per_budget"]["m4"] is True
    closed = gates.interpretation_rows(_gate_view(o1_delta=-0.001, m4_delta=-0.001))
    assert closed["close_continuous_t4"]["per_budget"]["m4"] is True
    assert gates.movement_ordering({"m4": 1.0, "m10": 0.5, "m30": 0.1})["ordering_holds"] is True
    assert gates.movement_ordering({"m4": 0.1, "m10": 0.5, "m30": 1.0})["ordering_holds"] is False
    with pytest.raises(gates.StageOGateError):
        gates.movement_ordering({"m4": 1.0, "m10": 0.5})


# ---------------------------------------------------------------------------
# 5. O0 bit-anchor comparison logic and the P2' cross-reference.
# ---------------------------------------------------------------------------


def test_sealed_o0_anchor_sources_cover_the_grid():
    cells = replay._sealed_activity_cells(ROOT)
    assert set(cells) == {(budget, surface) for budget in (4, 10, 30) for surface in ("within", "external")}
    assert len(cells[(4, "within")]) == 6
    assert len(cells[(10, "external")]) == 15
    # M30 anchors against the V8 matched score's sealed STATIC cells.
    assert len(cells[(30, "within")]) == 6
    assert len(cells[(30, "external")]) == 15
    row = next(iter(cells[(30, "within")].values()))
    assert "governing_r2" in row and "prediction_sha256" in row and "valid_last_bin_count" in row


def test_anchor_o0_vs_sealed_logic():
    sealed = {
        "prediction_sha256": "a" * 64,
        "governing_r2": 0.38768261671066284,
        "valid_last_bin_count": 1206,
    }
    row = {"prediction_sha256_raw": "a" * 64, "house_raw_r2": 0.38768261671066284, "n_windows": 1206}
    assert replay.anchor_o0_vs_sealed(row=row, sealed_row=sealed, label="toy")["exact_match"] is True
    drifted = dict(row, prediction_sha256_raw="b" * 64)
    assert replay.anchor_o0_vs_sealed(row=drifted, sealed_row=sealed, label="toy")["exact_match"] is False
    r2_drift = dict(row, house_raw_r2=0.387682616710663)
    r2_report = replay.anchor_o0_vs_sealed(row=r2_drift, sealed_row=sealed, label="toy")
    assert r2_report["exact_match"] is False
    assert r2_report["house_raw_r2_gap"] > 0.0
    windows = dict(row, n_windows=1205)
    assert replay.anchor_o0_vs_sealed(row=windows, sealed_row=sealed, label="toy")["exact_match"] is False


def test_p2prime_cross_reference_quotes_the_sealed_upper_bound():
    reference = replay.p2prime_cross_reference(ROOT)
    assert reference["read_only"] is True
    m4 = reference["rows"]["m4_external"]
    assert m4["O2_minus_O0"] == pytest.approx(0.0919665819214975, abs=1.0e-9)
    assert abs(m4["O2_minus_O0"] - reference["m4_external_o2_minus_o0_expected_approx"]) < 0.01


# ---------------------------------------------------------------------------
# 6. Causality receipt chain helper.
# ---------------------------------------------------------------------------


def test_verify_causality_receipts_detects_a_broken_chain():
    good = [{"fs": "s0", "sb": "s0", "sa": "s1"}, {"fs": "s1", "sb": "s1", "sa": "s2"}]
    assert replay.verify_causality_receipts(good) is True
    assert replay.verify_causality_receipts([
        {"fs": "s0", "sb": "s0", "sa": "s1"},
        {"fs": "s0", "sb": "s0", "sa": "s2"},  # reused the pre-update state
    ]) is False
    assert replay.verify_causality_receipts([{"sb": "s0", "sa": "s1"}]) is False


# ---------------------------------------------------------------------------
# 7. The whole state machine on toy carriers.
# ---------------------------------------------------------------------------


def _reference_o2_trace(runtime: ToyRuntime, session: ToySession, budget: int,
                        anchor: anchor_module.SupportAnchor, c_M, alpha: float = 0.5):
    """A second, independent implementation of the O2 state law for the proof.

    Returns one record per trial: the pre-reveal forward state digest, the
    recorded prediction, and (for committed trials) the prediction the SAME
    trial would have received from the post-commit carrier.
    """
    memory, _initial = runtime._initial_memory(session=session, budget=budget)
    config = memory.state.carrier.config
    bank = block_refit.EvidenceBank.empty()
    active = np.asarray(memory.state.carrier.active_t4)
    records = []
    for trial_id in session.query_trial_ids[budget]:
        trial = session.trials_by_id[trial_id]
        inputs = memory.read_prediction_inputs()
        state_before = memory.state.digest
        raw = runtime._governing_forward(trial=trial, inputs=inputs)
        direction, _meta = replay.true_direction_for_trial(session, trial, config=config)
        rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        moved = False
        post_prediction = None
        if direction.accepted:
            bank = bank.with_row(_evidence_row(session, trial, direction, rates, block_index=len(bank)))
            refit = block_refit.refit_from_anchor(anchor, bank, rho_M=1.0)
            outcome = trust_region.active_carrier(anchor, refit.coefficients, alpha_M=alpha, c_M=c_M)
            candidate = np.asarray(outcome.active_t4)
            if not np.array_equal(candidate, active):
                # Same activity, NEW carrier: what trial i would have seen had
                # its own update been visible to it (it must not be).
                swap = cdm_core.CarrierMemory(
                    groups=memory.state.carrier.groups,
                    initial_raw_t4=memory.state.carrier.initial_raw_t4,
                    statistics=memory.state.carrier.statistics,
                    ordinary_t4=candidate, fixed_ridge_t4=candidate,
                    config=memory.state.carrier.config,
                )
                stepped = cdm_core.IndependentActivityCausalDualMemory(
                    activity=memory.state.activity, carrier=swap,
                )
                post_inputs = stepped.read_prediction_inputs()
                assert post_inputs.state_digest != state_before
                post_prediction = runtime._governing_forward(trial=trial, inputs=post_inputs)
                moved = True
                active = candidate
        records.append({
            "trial_id": trial_id, "state_before": state_before, "raw": raw,
            "moved": moved, "post_prediction": post_prediction,
        })
        # Advance the frozen independent-activity law exactly like the rollout.
        activity_candidate, changed = memory._activity_candidate(trial.b3s_activity)
        pseudo = tuple(
            cdm_core.pseudo_direction_from_velocity(
                view.velocity, view.validity.valid_mask, config=config,
            )
            for view in replay.true_views_for_trial(session, trial)[0]
        )
        if moved:
            next_carrier = cdm_core.CarrierMemory(
                groups=memory.state.carrier.groups,
                initial_raw_t4=memory.state.carrier.initial_raw_t4,
                statistics=memory.state.carrier.statistics,
                ordinary_t4=active, fixed_ridge_t4=active,
                config=memory.state.carrier.config,
            )
            candidate_state = cdm_core.DualMemoryState(
                activity=activity_candidate.activity, carrier=next_carrier,
                committed_query_trials=activity_candidate.committed_query_trials,
            )
            pending = cdm_core.IndependentActivityPendingTrialUpdate(
                base_state_digest=memory.state.digest, activity_transition_ready=True,
                activity_fifo_changed=changed, carrier_transition_accepted=True,
                activity_rejection_reason=None, carrier_rejection_reason=None,
                pseudo_directions=pseudo, scalar_rates=rates,
                carrier_proposal=cdm_core.CarrierProposal(True, None, next_carrier, (), None),
                candidate_state=candidate_state, fallback=memory.read_prediction_inputs(),
            )
        else:
            reason = (
                cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE
                if direction.accepted else direction.reason
            )
            candidate_state = cdm_core.DualMemoryState(
                activity=activity_candidate.activity, carrier=memory.state.carrier,
                committed_query_trials=activity_candidate.committed_query_trials,
            )
            pending = cdm_core.IndependentActivityPendingTrialUpdate(
                base_state_digest=memory.state.digest, activity_transition_ready=True,
                activity_fifo_changed=changed, carrier_transition_accepted=False,
                activity_rejection_reason=None, carrier_rejection_reason=reason,
                pseudo_directions=pseudo, scalar_rates=rates, carrier_proposal=None,
                candidate_state=candidate_state, fallback=memory.read_prediction_inputs(),
            )
        memory.commit_independent(pending)
    return records


def test_stage_o_rollouts_on_toy_carriers():
    runtime = ToyRuntime()
    session = build_toy_session()
    budget = 4
    cell = _toy_table_cell(runtime, session, budget)
    anchor = _toy_anchor(runtime, session, budget)
    a0_digest_before = anchor.a0_b0_digest

    o0 = replay.rollout_o0(runtime, session=session, budget=budget)
    o1 = replay.rollout_o1(runtime, session=session, budget=budget, table_cell=cell)
    calibration = replay.rollout_o2(
        runtime, session=session, budget=budget, alpha_M=0.5, c_M=None,
        table_cell=cell,
    )
    c_M = trust_region.calibrate_c_M([float(item) for item in calibration["d2_values"]])
    o2 = replay.rollout_o2(
        runtime, session=session, budget=budget, alpha_M=0.5, c_M=c_M,
        table_cell=cell,
    )

    # The evidence bank never depends on the active carrier: the D2 sequence and
    # the bank-digest chain are invariant to the projection (c_M calibration law).
    assert o2["bank_digest_chain"] == calibration["bank_digest_chain"]
    assert o2["d2_values"] == calibration["d2_values"]

    # (a) Causality: the receipt chains close for O1 and O2, and every forward
    # read exactly the state digest recorded before the reveal.
    for result in (o1, o2):
        assert replay.verify_causality_receipts(result["receipts"])
    forward_states = {trial_id: digest for trial_id, digest, _vel in runtime.forward_log}
    for receipt in o2["receipts"]:
        assert forward_states[receipt["trial_id"]] == receipt["sb"]
    committed = [item for item in o2["receipts"] if item["acc"]]
    assert committed, "the toy O2 run must commit carrier updates"
    for receipt in committed:
        assert receipt["ca"] != receipt["cb"]
        assert receipt["sa"] != receipt["sb"]

    # (b) Direct proof via the independent reference trace: it reproduces the
    # rollout's predictions bit-for-bit, and for every committed trial the
    # prediction from the post-commit carrier DIFFERS from the recorded one --
    # the committed update did not (and cannot) touch its own supplying trial.
    reference_runtime = ToyRuntime()
    records = _reference_o2_trace(reference_runtime, session, budget, anchor, c_M)
    for record, raw in zip(records, o2["per_trial_raw"]):
        assert np.array_equal(np.asarray(record["raw"]), np.asarray(raw))
    moved_records = [record for record in records if record["moved"]]
    assert moved_records
    for record in moved_records:
        assert not np.array_equal(np.asarray(record["post_prediction"]), np.asarray(record["raw"]))

    # (c) O2 with alpha_M = 0 is the bit-exact O0 no-op.
    zero = replay.rollout_o2(
        runtime, session=session, budget=budget, alpha_M=0.0, c_M=c_M,
        table_cell=cell,
    )
    for left, right in zip(zero["per_trial_raw"], o0["per_trial_raw"]):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert all(not item["acc"] for item in zero["receipts"])
    assert all(item["reason"] == "always_commit_zero_movement" for item in zero["receipts"])

    # (d) Anchor immutability across every commit above.
    assert _toy_anchor(runtime, session, budget).a0_b0_digest == a0_digest_before

    # (e) O1 commits through the frozen recursive estimator with true directions.
    assert any(item["acc"] for item in o1["receipts"])
    assert all(item["d_acc"] for item in o1["receipts"])
    o1_carriers = {item["cb"] for item in o1["receipts"]}
    assert len(o1_carriers) > 1, "the recursive estimator must accumulate state"

    # (f) Leakage labelling and the O0 no-op law.
    assert o0["leakage_flags"]["target_label_leakage"] is False
    assert o1["leakage_flags"]["target_label_leakage"] is True
    assert o2["leakage_flags"]["target_label_leakage"] is True
    assert o2["leakage_flags"]["checkpoint_selection_from_target"] is False
    assert o2["leakage_flags"]["hyperparameter_selection_from_target"] is False
    assert all(not item["acc"] for item in o0["per_trial_receipts"])

    # (g) Session-row assembly carries the §9 receipt facts.
    targets, masks, _sst = runtime._session_target_views(session, budget)
    row = replay._session_row(
        runtime, row_id="O2", budget=budget, session=session, targets=targets, masks=masks,
        rollout=o2, alpha_M=0.5, c_M=c_M,
    )
    assert row["causality_state_chain_verified"] is True
    assert row["carrier_transitions_committed"] == len(committed)
    assert row["movement"]["rho_M"] == 1.0
    assert row["support_anchor"]["a0_b0_sha256"] == a0_digest_before
    assert row["leakage_flags"]["target_label_leakage"] is True
    assert len(row["per_trial_receipts"]) == len(session.query_trial_ids[budget])


def test_stage_o_rollouts_m10_and_the_m30_activity_noop_diagnostic():
    for budget in (10, 30):
        runtime = ToyRuntime()
        session = build_toy_session(seed=11)
        cell = _toy_table_cell(runtime, session, budget)
        o2 = replay.rollout_o2(
            runtime, session=session, budget=budget, alpha_M=0.5, c_M=1.0,
            table_cell=cell,
        )
        assert replay.verify_causality_receipts(o2["receipts"])
        if budget == 30:
            # M30's capacity-zero FIFO is an exact activity no-op while the
            # carrier still moves (the mandatory diagnostic column).
            assert all(item["ab"] == item["aa"] for item in o2["receipts"])
            assert any(item["acc"] for item in o2["receipts"])


def test_o2_direction_table_drift_is_detected():
    runtime = ToyRuntime()
    session = build_toy_session()
    cell = dict(_toy_table_cell(runtime, session, 4))
    cell["cell_digest"] = "0" * 64
    with pytest.raises(replay.StageOReplayError):
        replay.rollout_o2(
            runtime, session=session, budget=4, alpha_M=0.5, c_M=None,
            table_cell=cell,
        )


# ---------------------------------------------------------------------------
# 8. Pre-registration pinning and the frozen construction law.
# ---------------------------------------------------------------------------


def test_pre_registration_payload_is_pinned():
    payload = plan.pre_registration_payload()
    assert payload["hyperparameters"]["rho_M"] == 1.0
    assert payload["hyperparameters"]["block_size_completed_trials"] == 1
    assert payload["hyperparameters"]["alpha_M_primary"] == 0.5
    assert tuple(payload["hyperparameters"]["alpha_M_sensitivity"]) == (0.125, 0.25, 1.0)
    assert payload["hyperparameters"]["alpha_M_sensitivity_governs"] is False
    assert payload["o2_commit_law"]["commit_rule"] == "always-commit"
    assert payload["o2_commit_law"]["movement_rule"] == "trust-region projection only"
    assert payload["c_m_calibration"]["selection_surface"] == "within-6 ONLY; external-15 never selects"
    assert payload["gates"]["boundary_epsilon"] == 1.0e-12
    assert payload["row_order"] == ["O0", "O1", "O2", "O2A0125", "O2A025", "O2A100"]
    assert payload["leakage_flags_by_row"]["O1"]["target_label_leakage"] is True
    assert payload["leakage_flags_by_row"]["O1"]["checkpoint_selection_from_target"] is False
    assert payload["leakage_flags_by_row"]["O2"]["hyperparameter_selection_from_target"] is False
    assert payload["leakage_flags_by_row"]["O0"]["target_label_leakage"] is False
    plan.validate_pre_registration(payload)
    for mutation in ("rho_M", "alpha_M_primary", "block_size_completed_trials"):
        drifted = json.loads(json.dumps(payload))
        drifted["hyperparameters"][mutation] = 0.123456
        with pytest.raises(ValueError):
            plan.validate_pre_registration(drifted)


def test_true_views_use_the_frozen_construction_law():
    runtime = ToyRuntime()
    session = build_toy_session()
    trial = session.trials_by_id[session.query_trial_ids[4][0]]
    views, meta = replay.true_views_for_trial(session, trial)
    assert meta["construction"] == "true"
    assert len(views) == cdm_core.GROUP_COUNT
    restored, padded = p2filters.true_physical_velocity_one_trial(
        np.asarray(session.behavior[trial.endpoint_bins]),
        behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
    )
    for view in views:
        assert np.array_equal(np.asarray(view.velocity), restored)
        assert view.validity is trial.velocity_validity
    assert int(np.sum(padded)) == 0
    memory, _initial = runtime._initial_memory(session=session, budget=4)
    direction = cdm_core.pseudo_direction_from_velocity(
        views[0].velocity, views[0].validity.valid_mask, config=memory.state.carrier.config,
    )
    assert direction.accepted
    assert direction.canonical_distance_rad == pytest.approx(0.0, abs=1.0e-6)


# ---------------------------------------------------------------------------
# 9. Stage guards (no data, no CUDA: the refusals fire before any access).
# ---------------------------------------------------------------------------


def test_run_stage_o_replay_refuses_without_its_prerequisites(tmp_path):
    empty = tmp_path / "stage-o"
    empty.mkdir()
    with pytest.raises(replay.StageOReplayError):
        replay.run_stage_o_replay(ROOT, gpu_index=1, output_root=empty)
    (empty / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StageOReplayError):
        replay.run_stage_o_replay(ROOT, gpu_index=1, output_root=empty)
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "attempt.json").write_text("{}", encoding="utf-8")
    (closed / "directions.json").write_text("{}", encoding="utf-8")
    (closed / "replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StageOReplayError):
        replay.run_stage_o_replay(ROOT, gpu_index=1, output_root=closed)
    terminal = tmp_path / "terminalized"
    terminal.mkdir()
    (terminal / "terminal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.StageOReplayError):
        replay.run_stage_o_replay(ROOT, gpu_index=1, output_root=terminal)


def test_terminal_builder_end_to_end_on_a_synthetic_matrix():
    within_roster = [f"w{i}" for i in range(6)]
    external_roster = [f"e{i}" for i in range(15)]
    matrix: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        key = f"m{budget}"
        matrix[key] = {}
        for surface, roster in (("within", within_roster), ("external", external_roster)):
            rows = {}
            for row, delta in (("O0", 0.0), ("O1", -0.005 if budget == 4 else 0.002),
                               ("O2", 0.05 if budget == 4 else -0.002)):
                per_session = {}
                for index, session in enumerate(roster):
                    bump = delta
                    if row == "O2" and budget == 4:
                        bump = 0.051 if index < 11 else -0.06
                    per_session[session] = 0.40 + bump
                rows[row] = {
                    "mean_r2": sum(per_session.values()) / len(roster),
                    "per_session": per_session,
                }
            matrix[key][surface] = rows
    replay_payload = {
        "matrix": matrix,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
        "model_state_digest_unchanged": True,
        "causality_state_chains_all_rows": True,
        "movement_by_budget_medians": {"m4": 2.0, "m10": 1.0, "m30": 0.5},
        "anchors": {
            "sealed_activity_only_all_exact": True,
            "p2prime_cross_reference": {"rows": {"m4_external": {"O2_minus_O0": 0.092}}},
        },
        "sensitivity_means_non_governing": {"O2A025_m4_external": 0.43},
        "c_M_calibration": {"m4": {"c_M": 1.5}},
    }
    body = replay.build_terminal(replay_payload=replay_payload)
    gate = body["gate"]
    assert gate["safety_all_pass"] is True
    assert gate["driving_budget"] == "m4"
    assert gate["per_budget"]["m4"]["external"]["positive_sessions"] == 11
    assert body["movement_ordering"]["ordering_holds"] is True
    assert body["paired_contrasts"]["O2_minus_O0_m4_external"]["n_total"] == 15
    assert body["p2prime_cross_reference"]["rows"]["m4_external"]["O2_minus_O0"] == 0.092
    drifted = dict(replay_payload, matrix={"m4": matrix["m4"]})
    with pytest.raises(gates.StageOGateError):
        replay.build_terminal(replay_payload=drifted)
