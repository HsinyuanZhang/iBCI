"""Focused synthetic CPU gates for CDM-D Stage 0.

No test opens an NWB, checkpoint tensor, result root, or CUDA device.  The
only production imports are the two CPU estimator helpers used to independently
check the deliberately distinct OLS and fixed-ridge formulas.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "tfpd_exploration", ROOT / "sua_exploration"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import plan  # noqa: E402
from src.calibration_budget_comparators_v1 import fit_ridge_t4  # noqa: E402
from mc_maze.d_optimal_calibration_design import fit_carriers_from_selected_trials  # noqa: E402


SESSION_ID = "synthetic-session"


def _table(*, units: int = 8) -> dict[str, np.ndarray]:
    """Exact synthetic cosine table with deliberately repeated directions."""
    directions = np.asarray((0, 2, 4, 3, 3, 3, 5, 7, 1, 6, 3), dtype=np.int64)
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[index] for index in directions], dtype=np.float64)
    a = np.linspace(0.7, 1.4, units, dtype=np.float64)
    c = np.linspace(-0.8, 0.5, units, dtype=np.float64)
    b = np.linspace(4.0, 5.0, units, dtype=np.float64)
    rates = b[None, :] + np.cos(theta)[:, None] * a[None, :] + np.sin(theta)[:, None] * c[None, :]
    raw_t4 = np.column_stack((a, c, np.hypot(a, c), b)).astype(np.float64)
    channels = np.arange(1000, 1000 + units, dtype=np.int64)
    return {
        "directions": directions,
        "theta": theta,
        "rates": rates,
        "raw_t4": raw_t4,
        "channels": channels,
        "channel_digest": np.asarray([core.channel_order_digest(channels)]),
    }


def _channel_digest(table: dict[str, np.ndarray]) -> str:
    return str(table["channel_digest"][0])


def _config(
    *,
    support_budget_m: int = 4,
    active_mode: core.CarrierFitMode = core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION,
    **overrides: object,
) -> core.CDMDConfig:
    values: dict[str, object] = {
        "support_budget_m": support_budget_m,
        "dt": 1.0,
        "minimum_movement_bins": 2,
        "minimum_displacement": 0.01,
        "minimum_mean_speed": 0.01,
        "max_canonical_distance_rad": math.pi / 8.0,
        "max_group_direction_disagreement_rad": math.pi / 8.0,
        "minimum_accepted_evidence": 3,
        "active_fit_mode": active_mode,
    }
    values.update(overrides)
    return core.CDMDConfig(**values)


def _b3s_trial(
    table: dict[str, np.ndarray],
    values: np.ndarray,
    *,
    trial_id: str,
) -> core.B3SInterpolatedSpikeCountTrial:
    activity = np.repeat(np.asarray(values, dtype=np.float64)[None, :], 100, axis=0)
    return core.B3SInterpolatedSpikeCountTrial(
        activity=activity,
        session_id=SESSION_ID,
        trial_id=trial_id,
        channel_order_sha256=_channel_digest(table),
    )


def _native_counts(
    table: dict[str, np.ndarray],
    rates: np.ndarray,
    *,
    trial_id: str,
    bins: int = 7,
    start: int = 1000,
    channel_digest: str | None = None,
    session_id: str = SESSION_ID,
) -> core.NativeRewardedTrialSpikeCounts:
    # Synthetic values are a faithful floating representation of binned counts;
    # the formula remains exact: mean(counts) / 0.020 == supplied rate.
    counts = np.repeat((np.asarray(rates, dtype=np.float64) * 0.020)[None, :], bins, axis=0)
    return core.NativeRewardedTrialSpikeCounts(
        counts=counts,
        session_id=session_id,
        trial_id=trial_id,
        channel_order_sha256=_channel_digest(table) if channel_digest is None else channel_digest,
        rewarded_interval_start_bin=start,
        rewarded_interval_stop_bin=start + bins,
    )


def _validity(
    *,
    trial_id: str,
    bins: int = 6,
    start: int = 3000,
    session_id: str = SESSION_ID,
    mask: np.ndarray | None = None,
) -> core.VelocityValidityEvidence:
    return core.VelocityValidityEvidence(
        valid_mask=np.ones(bins, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_),
        session_id=session_id,
        trial_id=trial_id,
        prediction_interval_start_bin=start,
        prediction_interval_stop_bin=start + bins,
    )


def _velocity(direction_index: int = 3, *, bins: int = 6, speed: float = 1.0) -> np.ndarray:
    theta = core.CANONICAL_DIRECTIONS_RAD[direction_index]
    return np.repeat(
        np.asarray([[speed * math.cos(theta), speed * math.sin(theta)]], dtype=np.float64),
        bins,
        axis=0,
    )


def _predictions(
    validity: core.VelocityValidityEvidence,
    *,
    direction_index: int = 3,
    count: int = core.GROUP_COUNT,
) -> tuple[core.CompletedVelocityPrediction, ...]:
    return tuple(
        core.CompletedVelocityPrediction(_velocity(direction_index, bins=validity.valid_mask.size), validity)
        for _ in range(count)
    )


def _query_views(
    table: dict[str, np.ndarray],
    *,
    trial_id: str,
    direction_index: int = 3,
    scale: float = 1.0,
    native_bins: int = 7,
    prediction_bins: int = 6,
) -> tuple[core.B3SInterpolatedSpikeCountTrial, core.NativeRewardedTrialSpikeCounts, core.VelocityValidityEvidence]:
    matching = np.flatnonzero(table["directions"] == direction_index)
    assert matching.size
    rates = table["rates"][matching[0]] * scale
    return (
        _b3s_trial(table, rates, trial_id=trial_id),
        _native_counts(table, rates, trial_id=trial_id, bins=native_bins),
        _validity(trial_id=trial_id, bins=prediction_bins),
    )


def _make_memory(
    *,
    support_budget_m: int = 4,
    active_mode: core.CarrierFitMode = core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION,
) -> tuple[core.CausalDualMemory, dict[str, np.ndarray]]:
    table = _table()
    config = _config(support_budget_m=support_budget_m, active_mode=active_mode)
    # Three distinct support directions give every group a rank-3 support design
    # before the first completed pseudo-labelled trial.
    support_indices = np.resize(np.arange(table["rates"].shape[0], dtype=np.int64), config.support_budget_m)
    support_rates = table["rates"][support_indices]
    support_directions = table["directions"][support_indices]
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=table["raw_t4"],
        channel_ids=table["channels"],
        support_trial_rates=support_rates,
        support_direction_indices=support_directions,
        config=config,
    )
    support_activity = tuple(
        _b3s_trial(table, rate, trial_id=f"support-{position}")
        for position, rate in enumerate(support_rates)
    )
    activity = core.ActivityMemory.initialize(
        support_activity,
        channel_ids=table["channels"],
        fifo_capacity=config.activity_fifo_capacity,
    )
    return core.CausalDualMemory(activity=activity, carrier=carrier), table


def _observe(
    machine: core.CausalDualMemory,
    table: dict[str, np.ndarray],
    *,
    trial_id: str,
    direction_index: int = 3,
    scale: float = 1.0,
    validity: core.VelocityValidityEvidence | None = None,
    b3s: core.B3SInterpolatedSpikeCountTrial | None = None,
    native: core.NativeRewardedTrialSpikeCounts | None = None,
    predictions: tuple[core.CompletedVelocityPrediction, ...] | None = None,
) -> core.PendingTrialUpdate:
    default_b3s, default_native, default_validity = _query_views(
        table,
        trial_id=trial_id,
        direction_index=direction_index,
        scale=scale,
    )
    selected_b3s = default_b3s if b3s is None else b3s
    selected_native = default_native if native is None else native
    selected_validity = default_validity if validity is None else validity
    selected_predictions = _predictions(selected_validity, direction_index=direction_index) if predictions is None else predictions
    return machine.observe_completed_trial(
        b3s_trial_activity=selected_b3s,
        carrier_trial_counts=selected_native,
        complementary_predictions=selected_predictions,
    )


def _accepted_pending(machine: core.CausalDualMemory, table: dict[str, np.ndarray], *, trial_id: str = "accepted-query") -> core.PendingTrialUpdate:
    return _observe(machine, table, trial_id=trial_id)


class _FrozenFakeDecoder:
    """Stateful-looking injected callable with no mutation and held-mask tracing."""

    def __init__(self) -> None:
        self._weight = np.asarray([3.0, -2.0], dtype=np.float64)
        self.held_masks: list[np.ndarray | None] = []

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": self._weight.copy()}

    def __call__(
        self,
        neural_activity: np.ndarray,
        *,
        activity_trials: np.ndarray,
        raw_t4: np.ndarray,
        held_unit_mask: np.ndarray | None,
    ) -> np.ndarray:
        neural = np.asarray(neural_activity, dtype=np.float64)
        assert activity_trials.ndim == 3 and activity_trials.shape[1] == 100
        assert raw_t4.shape == (neural.shape[1], 4)
        self.held_masks.append(None if held_unit_mask is None else held_unit_mask.copy())
        available = np.ones(neural.shape[1], dtype=np.bool_) if held_unit_mask is None else ~held_unit_mask
        x = neural[:, available].sum(axis=1)
        return np.column_stack((np.maximum(x, 1.0), np.zeros(neural.shape[0], dtype=np.float64)))


class _TorchFrozenDecoder(torch.nn.Module):
    """CPU-only injected decoder proving the wrapper adds no Torch parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(
        self,
        neural_activity: np.ndarray,
        *,
        activity_trials: np.ndarray,
        raw_t4: np.ndarray,
        held_unit_mask: np.ndarray | None,
    ) -> np.ndarray:
        del activity_trials, raw_t4, held_unit_mask
        neural = np.asarray(neural_activity, dtype=np.float64)
        return np.column_stack((np.maximum(neural.sum(axis=1), 1.0), np.zeros(neural.shape[0], dtype=np.float64)))


def _inputs_equal(left: core.PredictionInputs, right: core.PredictionInputs) -> bool:
    return (
        left.state_digest == right.state_digest
        and left.query_trials_completed == right.query_trials_completed
        and np.array_equal(left.activity_trials, right.activity_trials)
        and np.array_equal(left.support_activity_trials, right.support_activity_trials)
        and np.array_equal(left.active_t4, right.active_t4)
        and np.array_equal(left.ordinary_point_t4, right.ordinary_point_t4)
        and np.array_equal(left.fixed_ridge_t4, right.fixed_ridge_t4)
    )


def test_completed_trial_prefix_causality_and_future_mutation_invariance() -> None:
    machine, table = _make_memory()
    wrapper = core.CausalDualMemoryCellDWrapper(machine, _FrozenFakeDecoder())
    b3s, native, validity = _query_views(table, trial_id="query-0")
    neural = np.repeat(table["rates"][3][None, :], validity.valid_mask.size, axis=0) + 0.1
    future = _b3s_trial(table, table["rates"][3] + 99.0, trial_id="future")

    for prefix in range(4):
        trial_id = f"query-{prefix}"
        b3s, native, validity = _query_views(table, trial_id=trial_id)
        before = machine.read_prediction_inputs()
        _ = wrapper.predict_current(neural)
        pending = wrapper.observe_completed_trial(
            b3s_trial_activity=b3s,
            carrier_trial_counts=native,
            neural_activity_for_complementary_prediction=neural,
            velocity_validity=validity,
        )
        assert _inputs_equal(before, machine.read_prediction_inputs())
        assert pending.accepted
        outcome = wrapper.commit(pending)
        assert outcome.committed
        after = machine.read_prediction_inputs()
        assert after.query_trials_completed == prefix + 1
        assert after.state_digest != before.state_digest
        assert np.array_equal(after.activity_trials[-1], b3s.activity)
        future_mutated = future.activity.copy()
        future_mutated[:, :] = 9_999.0 + prefix
        assert machine.read_prediction_inputs().state_digest == after.state_digest

    assert machine.state.activity.query_count == 4
    assert machine.state.committed_query_trials == 4


def test_activity_fifo_support_immutability_and_rejection_state_digest() -> None:
    table = _table()
    support = tuple(_b3s_trial(table, np.full(8, float(index + 1)), trial_id=f"support-{index}") for index in range(4))
    memory = core.ActivityMemory.initialize(support, channel_ids=table["channels"], fifo_capacity=26)
    assert np.all(memory.stack()[:2] > 0.0)
    first_digest = memory.digest
    for value in range(5, 31):
        memory = memory.after_accepted_trial(_b3s_trial(table, np.full(8, float(value)), trial_id=f"query-{value}"))
    stack = memory.stack()
    assert stack.shape == (30, 100, 8)
    assert np.array_equal(stack[0], np.ones((100, 8)))
    assert np.array_equal(stack[1], np.full((100, 8), 2.0))
    assert np.array_equal(stack[-1], np.full((100, 8), 30.0))
    assert memory.query_count == 26
    assert first_digest != memory.digest

    with pytest.raises(core.CDMDStage0Error, match="bare"):
        core.ActivityMemory.initialize(np.ones((4, 100, 8)), channel_ids=table["channels"], fifo_capacity=26)

    machine, table = _make_memory()
    before = machine.state.digest
    rejected = machine.observe_completed_trial(
        b3s_trial_activity=np.full((100, 8), np.nan),
        carrier_trial_counts=np.full((7, 8), 1.0),
        complementary_predictions=(),
    )
    assert rejected.reason is core.UpdateRejectionReason.TRIAL_CAPABILITY
    result = machine.commit(rejected)
    assert not result.committed and result.state_digest_before == result.state_digest_after == before
    assert machine.state.digest == before


def test_budget_derived_activity_capacities_and_m30_zero_memory_carrier_commit() -> None:
    table = _table()
    expected = {4: 26, 10: 20, 30: 0}
    for budget, capacity in expected.items():
        config = _config(support_budget_m=budget)
        assert config.activity_fifo_capacity == capacity
        assert core.activity_fifo_capacity_for_support_budget(budget) == capacity
        support = tuple(_b3s_trial(table, np.full(8, float(index)), trial_id=f"s-{budget}-{index}") for index in range(budget))
        activity = core.ActivityMemory.initialize(support, channel_ids=table["channels"], fifo_capacity=capacity)
        assert activity.stack().shape[0] == budget
        assert activity.stack().shape[0] + capacity == 30
        after = activity.after_accepted_trial(_b3s_trial(table, np.full(8, 99.0), trial_id=f"q-{budget}"))
        assert after.stack().shape[0] == budget + (0 if capacity == 0 else 1)
        assert after.stack().shape[0] <= 30
        if capacity == 0:
            assert after.query_count == 0
            assert after.digest == activity.digest
            assert np.array_equal(after.stack(), activity.stack())
        else:
            assert after.query_count == 1

    with pytest.raises(core.CDMDStage0Error, match="30 minus"):
        _ = core.CDMDConfig(support_budget_m=4, activity_fifo_capacity=20)

    machine, table = _make_memory(support_budget_m=30)
    before_activity_digest = machine.state.activity.digest
    before_stack = machine.state.activity.stack()
    before_carrier_digest = machine.state.carrier.statistics.digest
    pending = _accepted_pending(machine, table, trial_id="m30-query")
    assert pending.accepted
    outcome = machine.commit(pending)
    assert outcome.committed
    assert machine.state.activity.query_count == 0
    assert machine.state.activity.digest == before_activity_digest
    assert np.array_equal(machine.state.activity.stack(), before_stack)
    assert machine.state.activity.stack().shape[0] == 30
    assert machine.state.carrier.statistics.digest != before_carrier_digest
    assert machine.state.committed_query_trials == 1


def test_group_assignment_is_complete_balanced_and_permutation_equivariant() -> None:
    table = _table(units=12)
    first = core.build_complementary_groups(table["raw_t4"], table["channels"])
    assert set(first.assignment.tolist()) == {0, 1, 2, 3}
    sizes = [first.unit_indices(group).size for group in range(4)]
    assert max(sizes) - min(sizes) <= 1
    assert len(first.channel_to_group()) == 12

    permutation = np.asarray((8, 1, 10, 3, 0, 11, 5, 6, 2, 9, 7, 4), dtype=np.int64)
    second = core.build_complementary_groups(table["raw_t4"][permutation], table["channels"][permutation])
    assert first.channel_to_group() == second.channel_to_group()
    for group in range(4):
        held = first.held_mask(group)
        assert held.dtype == np.bool_ and int(held.sum()) == sizes[group]


def test_wrapper_uses_held_group_exclusion_for_its_pseudo_trajectory() -> None:
    first, table = _make_memory()
    first_decoder = _FrozenFakeDecoder()
    first_wrapper = core.CausalDualMemoryCellDWrapper(first, first_decoder)
    b3s, native, validity = _query_views(table, trial_id="wrapper-query")
    neural = np.repeat(table["rates"][3][None, :], validity.valid_mask.size, axis=0) + 0.25
    pending_one = first_wrapper.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=native,
        neural_activity_for_complementary_prediction=neural,
        velocity_validity=validity,
    )
    assert pending_one.accepted

    second, _ = _make_memory()
    second_decoder = _FrozenFakeDecoder()
    second_wrapper = core.CausalDualMemoryCellDWrapper(second, second_decoder)
    altered = neural.copy()
    held0 = second.state.carrier.groups.held_mask(0)
    altered[:, held0] += 1_000.0
    pending_two = second_wrapper.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=native,
        neural_activity_for_complementary_prediction=altered,
        velocity_validity=validity,
    )
    assert pending_two.accepted
    assert pending_one.pseudo_directions[0].payload() == pending_two.pseudo_directions[0].payload()
    assert len(first_decoder.held_masks) == len(second_decoder.held_masks) == 4
    for group, mask in enumerate(first_decoder.held_masks):
        assert mask is not None and np.array_equal(mask, first.state.carrier.groups.held_mask(group))


def test_velocity_integration_circular_nearest_direction_and_tie_break() -> None:
    config = _config(dt=0.5)
    midpoint = -5.0 * math.pi / 8.0
    velocity = np.repeat(np.asarray([[2.0 * math.cos(midpoint), 2.0 * math.sin(midpoint)]], dtype=np.float64), 4, axis=0)
    result = core.pseudo_direction_from_velocity(velocity, np.ones(4, dtype=np.bool_), config=config)
    assert result.accepted
    assert result.theta_index == 0
    assert result.canonical_distance_rad == pytest.approx(math.pi / 8.0)
    assert result.displacement_norm == pytest.approx(4.0)
    assert result.mean_speed == pytest.approx(2.0)
    assert core.nearest_canonical_direction(midpoint) == (0, pytest.approx(math.pi / 8.0))


@pytest.mark.parametrize(
    ("velocity", "mask", "config", "reason"),
    [
        (np.asarray([[np.nan, 0.0], [1.0, 0.0]], dtype=np.float64), np.ones(2, dtype=np.bool_), _config(), core.UpdateRejectionReason.VELOCITY_NONFINITE),
        (_velocity(bins=6, speed=1.0), np.asarray([True, False, False, False, False, False]), _config(), core.UpdateRejectionReason.MOVEMENT_TOO_SHORT),
        (_velocity(bins=3, speed=1.0e-4), np.ones(3, dtype=np.bool_), _config(), core.UpdateRejectionReason.LOW_DISPLACEMENT),
        (_velocity(bins=3, speed=0.1), np.ones(3, dtype=np.bool_), _config(minimum_displacement=0.01, minimum_mean_speed=0.5), core.UpdateRejectionReason.LOW_MEAN_SPEED),
        (
            np.repeat(np.asarray([[math.cos(-5.0 * math.pi / 8.0), math.sin(-5.0 * math.pi / 8.0)]], dtype=np.float64), 3, axis=0),
            np.ones(3, dtype=np.bool_),
            _config(max_canonical_distance_rad=0.1),
            core.UpdateRejectionReason.CANONICAL_DIRECTION_TOO_FAR,
        ),
    ],
)
def test_trajectory_quality_rejections_fail_closed(
    velocity: np.ndarray,
    mask: np.ndarray,
    config: core.CDMDConfig,
    reason: core.UpdateRejectionReason,
) -> None:
    result = core.pseudo_direction_from_velocity(velocity, mask, config=config)
    assert not result.accepted and result.reason is reason


def test_native_scalar_rate_uses_full_native_interval_not_b3s_or_velocity_mask() -> None:
    table = _table(units=2)
    native = core.NativeRewardedTrialSpikeCounts(
        counts=np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]], dtype=np.float64),
        session_id=SESSION_ID,
        trial_id="rate",
        channel_order_sha256=_channel_digest(table),
        rewarded_interval_start_bin=40,
        rewarded_interval_stop_bin=44,
    )
    rates = core.scalar_rates_from_native_rewarded_counts(native)
    assert np.array_equal(rates, np.asarray([200.0, 2000.0]))
    b3s = core.B3SInterpolatedSpikeCountTrial(
        activity=np.linspace(0.0, 100.0, 200, dtype=np.float64).reshape(100, 2),
        session_id=SESSION_ID,
        trial_id="rate",
        channel_order_sha256=_channel_digest(table),
    )
    assert not np.allclose(rates, b3s.activity.mean(axis=0) / 0.020)
    with pytest.raises(core.CDMDStage0Error, match="native"):
        core.scalar_rates_from_native_rewarded_counts(b3s)  # type: ignore[arg-type]

    groups = core.build_complementary_groups(_table()["raw_t4"], _table()["channels"])
    stats = core.CarrierSufficientStatistics.empty(groups)
    scalar = np.arange(groups.units, dtype=np.float64) + 0.5
    directions = (0, 1, 2, 3)
    next_stats = stats.after_pseudo_trial(directions, scalar)
    for group, direction in enumerate(directions):
        units = groups.unit_indices(group)
        assert next_stats.counts[group, direction] == 1
        assert np.array_equal(next_stats.rate_sums[group, direction, units], scalar[units])
        assert np.array_equal(next_stats.rate_sq_sums[group, direction, units], scalar[units] ** 2)


def test_trial_view_capabilities_reject_bare_swapped_and_mismatched_inputs() -> None:
    machine, table = _make_memory()
    b3s, native, validity = _query_views(table, trial_id="typed")
    before = machine.state.digest
    raw = machine.observe_completed_trial(
        b3s_trial_activity=b3s.activity,
        carrier_trial_counts=native.counts,
        complementary_predictions=_predictions(validity),
    )
    assert raw.reason is core.UpdateRejectionReason.TRIAL_CAPABILITY
    swapped = machine.observe_completed_trial(
        b3s_trial_activity=native,
        carrier_trial_counts=b3s,
        complementary_predictions=_predictions(validity),
    )
    assert swapped.reason is core.UpdateRejectionReason.TRIAL_CAPABILITY
    wrong_trial = _native_counts(table, table["rates"][3], trial_id="other")
    mismatch = machine.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=wrong_trial,
        complementary_predictions=_predictions(validity),
    )
    assert mismatch.reason is core.UpdateRejectionReason.TRIAL_BINDING
    wrong_channel = _native_counts(
        table,
        table["rates"][3],
        trial_id="typed",
        channel_digest="a" * 64,
    )
    channel_mismatch = machine.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=wrong_channel,
        complementary_predictions=_predictions(validity),
    )
    assert channel_mismatch.reason is core.UpdateRejectionReason.TRIAL_BINDING
    wrong_validity = _validity(trial_id="other")
    validity_mismatch = machine.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=native,
        complementary_predictions=_predictions(wrong_validity),
    )
    assert validity_mismatch.reason is core.UpdateRejectionReason.VELOCITY_VALIDITY
    assert machine.state.digest == before


def test_distinct_b3s_native_and_velocity_intervals_bind_one_trial_without_shape_conflation() -> None:
    machine, table = _make_memory()
    b3s, native, validity = _query_views(table, trial_id="different-shapes", native_bins=9, prediction_bins=6)
    assert b3s.activity.shape == (100, 8)
    assert native.counts.shape == (9, 8)
    assert validity.valid_mask.shape == (6,)
    pending = machine.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=native,
        complementary_predictions=_predictions(validity),
    )
    assert pending.accepted
    evidence = pending.payload()["completed_trial_evidence"]
    assert evidence["b3s_interpolated_trial"]["trial_id"] == evidence["native_rewarded_counts"]["trial_id"] == "different-shapes"
    assert evidence["native_rewarded_counts"]["counts"]["shape"] == [9, 8]
    assert evidence["velocity_validity"][0]["valid_mask"]["shape"] == [6]
    assert evidence["velocity_validity"][0]["target_behavior_used"] is False
    with pytest.raises(core.CDMDStage0Error, match="target-label-free"):
        core.VelocityValidityEvidence(
            valid_mask=np.ones(6, dtype=np.bool_),
            session_id=SESSION_ID,
            trial_id="different-shapes",
            prediction_interval_start_bin=0,
            prediction_interval_stop_bin=6,
            source="behavior",
        )


def test_clean_label_parity_against_production_direction_mean_ols_with_repeats() -> None:
    table = _table()
    selected = np.arange(table["rates"].shape[0], dtype=np.int64)
    expected = fit_carriers_from_selected_trials(table["rates"], table["directions"], selected)
    observed = core.fit_carriers_from_trial_table(
        table["rates"], table["directions"], mode=core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION
    )
    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    evidence = core.assert_production_parity(observed, table["rates"], table["directions"], mode=core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION)
    assert evidence["mode"] == "ordinary_ols_by_direction"


def test_clean_label_parity_against_production_fixed_ridge_with_unequal_counts() -> None:
    table = _table()
    expected, details = fit_ridge_t4(table["rates"], table["theta"], normalized_lambda=0.1)
    observed = core.fit_carriers_from_trial_table(
        table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL
    )
    assert details["normalized_lambda"] == 0.1
    assert np.allclose(observed, expected, atol=1e-7, rtol=1e-7)
    evidence = core.assert_production_parity(observed, table["rates"], table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL)
    assert evidence["mode"] == "fixed_ridge_by_trial"


def test_carrier_bank_reconstructs_both_production_modes_from_direction_statistics() -> None:
    table = _table()
    config = _config(active_mode=core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION)
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=table["raw_t4"],
        channel_ids=table["channels"],
        support_trial_rates=table["rates"][:4],
        support_direction_indices=table["directions"][:4],
        config=config,
    )
    support_activity = tuple(_b3s_trial(table, rate, trial_id=f"bank-support-{index}") for index, rate in enumerate(table["rates"][:4]))
    activity = core.ActivityMemory.initialize(support_activity, channel_ids=table["channels"], fifo_capacity=config.activity_fifo_capacity)
    machine = core.CausalDualMemory(activity=activity, carrier=carrier)
    for index, (rate, direction) in enumerate(zip(table["rates"][4:], table["directions"][4:])):
        trial_id = f"bank-query-{index}"
        b3s = _b3s_trial(table, rate, trial_id=trial_id)
        native = _native_counts(table, rate, trial_id=trial_id)
        validity = _validity(trial_id=trial_id)
        pending = machine.observe_completed_trial(
            b3s_trial_activity=b3s,
            carrier_trial_counts=native,
            complementary_predictions=_predictions(validity, direction_index=int(direction)),
        )
        assert pending.accepted
        assert machine.commit(pending).committed
    expected_ols = fit_carriers_from_selected_trials(table["rates"], table["directions"], np.arange(table["rates"].shape[0], dtype=np.int64))
    expected_ridge, _ = fit_ridge_t4(table["rates"], table["theta"], normalized_lambda=0.1)
    assert np.allclose(machine.state.carrier.ordinary_t4, expected_ols, atol=1e-12, rtol=1e-12)
    assert np.allclose(machine.state.carrier.fixed_ridge_t4, expected_ridge, atol=1e-7, rtol=1e-7)


def test_ols_direction_weighting_and_ridge_trial_weighting_never_collapse() -> None:
    table = _table()
    noisy = table["rates"].copy()
    noisy[table["directions"] == 3] += np.linspace(-2.0, 3.0, int((table["directions"] == 3).sum()))[:, None]
    ordinary = core.fit_carriers_from_trial_table(noisy, table["directions"], mode=core.CarrierFitMode.ORDINARY_OLS_BY_DIRECTION)
    ridge = core.fit_carriers_from_trial_table(noisy, table["directions"], mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL)
    assert not np.allclose(ordinary, ridge, atol=1e-4, rtol=1e-4)


def test_both_production_parity_modes_reject_deliberate_perturbation() -> None:
    table = _table()
    for mode in core.CarrierFitMode:
        fitted = core.fit_carriers_from_trial_table(table["rates"], table["directions"], mode=mode).copy()
        fitted[0, 0] += 0.2
        with pytest.raises(core.ProductionParityError, match=mode.value):
            core.assert_production_parity(fitted, table["rates"], table["directions"], mode=mode)


def test_departure_freeze_exact_fallback_and_no_cross_group_same_label_update() -> None:
    machine, table = _make_memory()
    before = machine.state
    bad_rate = np.full(table["rates"].shape[1], 10_000.0, dtype=np.float64)
    b3s = _b3s_trial(table, bad_rate, trial_id="bad")
    native = _native_counts(table, bad_rate, trial_id="bad")
    validity = _validity(trial_id="bad")
    frozen = machine.observe_completed_trial(
        b3s_trial_activity=b3s,
        carrier_trial_counts=native,
        complementary_predictions=_predictions(validity),
    )
    assert not frozen.accepted and frozen.reason is core.UpdateRejectionReason.DEPARTURE_FREEZE
    assert np.array_equal(frozen.fallback.ordinary_point_t4, before.carrier.ordinary_t4)
    assert np.array_equal(frozen.fallback.support_activity_trials, np.stack(tuple(item.activity for item in before.activity.support_trials), axis=0))
    outcome = machine.commit(frozen)
    assert not outcome.committed and machine.state.digest == before.digest

    accepted = _accepted_pending(machine, table, trial_id="good")
    assert accepted.accepted and accepted.candidate_state is not None
    old = before.carrier.statistics
    new = accepted.candidate_state.carrier.statistics
    for group, pseudo in enumerate(accepted.pseudo_directions):
        direction = int(pseudo.theta_index)
        own = new.groups.unit_indices(group)
        other = new.groups.assignment != group
        assert new.counts[group, direction] == old.counts[group, direction] + 1
        assert np.array_equal(new.rate_sums[group, direction, other], old.rate_sums[group, direction, other])
        assert np.any(new.rate_sums[group, direction, own] != old.rate_sums[group, direction, own])


def test_parameter_free_wrapper_and_injected_decoder_state_equality() -> None:
    machine, table = _make_memory()
    decoder = _TorchFrozenDecoder()
    wrapper = core.CausalDualMemoryCellDWrapper(machine, decoder)
    audit = wrapper.parameter_audit()
    assert audit.wrapper_new_trainable_parameters == 0
    assert audit.wrapper_new_trainable_parameter_names == ()
    assert audit.injected_decoder_trainable_parameters == 1
    before = core.decoder_state_digest(decoder)
    prediction = wrapper.predict_current(np.repeat(table["rates"][3][None, :], 6, axis=0) + 0.2)
    after = core.decoder_state_digest(decoder)
    assert prediction.shape == (6, 2)
    assert before == after


def test_global_python_numpy_and_torch_rng_invariance_for_deterministic_core() -> None:
    table = _table()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    groups = core.build_complementary_groups(table["raw_t4"], table["channels"])
    _ = core.CarrierSufficientStatistics.empty(groups)
    _ = core.nearest_canonical_direction(-5.0 * math.pi / 8.0)
    _ = core.pseudo_direction_from_velocity(_velocity(), np.ones(6, dtype=np.bool_), config=_config())
    _ = core.scalar_rates_from_native_rewarded_counts(_native_counts(table, table["rates"][3], trial_id="rng"))
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and numpy_after[2:] == numpy_before[2:]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_static_cli_imports_no_torch_writes_nothing_and_closure_is_explicit(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py"
    code = (
        "import importlib.util,sys; "
        f"spec=importlib.util.spec_from_file_location('cdm_cli',{str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.main(['--dry-run']); print('TORCH_IMPORTED='+str('torch' in sys.modules))"
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=tmp_path, env=env, text=True,
        capture_output=True, check=True,
    )
    assert "TORCH_IMPORTED=False" in completed.stdout
    assert list(tmp_path.iterdir()) == []
    closure = plan.closure_payload(ROOT)
    assert [row["path"] for row in closure["paths"]] == list(plan.STAGE0_CLOSURE_PATHS)
    assert len(closure["closure_sha256"]) == 64
    assert plan.dry_plan(ROOT)["execution_authorized"] is False
