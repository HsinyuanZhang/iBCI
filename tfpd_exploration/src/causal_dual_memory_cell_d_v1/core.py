"""Synthetic CPU core for Causal Dual-Memory Cell D (CDM-D) Stage 0.

This module deliberately has no data discovery, raw-NWB preprocessing,
checkpoint loading, scorer import, CUDA initialization, result publication, or
training route.  It consumes caller-supplied *trialized* activity and completed
decoder velocity predictions, then makes the causal transition explicit:

``read/predict -> observe completed trial -> commit accepted update``.

The two T4 fitting modes are intentionally separate:

* ``ordinary_ols_by_direction`` matches production's equal-direction mean OLS;
* ``fixed_ridge_by_trial`` matches the M4/M10 fixed-ridge system, retaining
  per-trial count weights and its ``n_trials * 0.1`` directional penalty.

Neither mode is evidence for the other.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import plan


GROUP_COUNT = 4
CANONICAL_DIRECTIONS_RAD: tuple[float, ...] = tuple(
    -3.0 * math.pi / 4.0 + index * (math.pi / 4.0) for index in range(8)
)
RIDGE_NORMALIZED_LAMBDA = 0.1
B8_DEPARTURE_THRESHOLD = 2.0
NUMERIC_EPSILON = 1.0e-12
ACTIVITY_STACK_LIMIT = plan.B3S_ACTIVITY_STACK_LIMIT
ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET = dict(plan.ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET)


class CDMDStage0Error(ValueError):
    """Fail-closed error for a malformed Stage-0 interface or invariant."""


class ProductionParityError(CDMDStage0Error):
    """Raised when a synthetic production-estimator parity check fails."""


class CarrierFitMode(str, Enum):
    """The two frozen, deliberately non-interchangeable carrier estimands."""

    ORDINARY_OLS_BY_DIRECTION = "ordinary_ols_by_direction"
    FIXED_RIDGE_BY_TRIAL = "fixed_ridge_by_trial"


class UpdateRejectionReason(str, Enum):
    """Typed reasons for a no-update transition.

    A rejection is not an exception from the state-machine perspective: it is
    a valid, auditable fallback to the exact preceding memory state.
    """

    ACTIVITY_SHAPE = "activity_shape"
    ACTIVITY_NONFINITE = "activity_nonfinite"
    TRIAL_CAPABILITY = "trial_capability"
    TRIAL_BINDING = "trial_binding"
    CARRIER_COUNTS = "carrier_counts"
    VELOCITY_VALIDITY = "velocity_validity"
    VELOCITY_SHAPE = "velocity_shape"
    VELOCITY_NONFINITE = "velocity_nonfinite"
    MOVEMENT_MASK = "movement_mask"
    MOVEMENT_TOO_SHORT = "movement_too_short"
    LOW_DISPLACEMENT = "low_displacement"
    LOW_MEAN_SPEED = "low_mean_speed"
    CANONICAL_DIRECTION_TOO_FAR = "canonical_direction_too_far"
    COMPLEMENTARY_DISAGREEMENT = "complementary_disagreement"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INSUFFICIENT_DESIGN_RANK = "insufficient_design_rank"
    ILL_CONDITIONED_DESIGN = "ill_conditioned_design"
    NONFINITE_CARRIER = "nonfinite_carrier"
    DEPARTURE_FREEZE = "departure_freeze"
    # Reserved for additive successor routes which apply a separately
    # receipt-bound support-precision credible-region gate *after* every
    # historical design/departure proposal check.  No V1/V3/V5/V8 proposal or
    # commit path selects this value by default.
    PRECISION_CREDIBLE_REGION = "precision_credible_region"
    STALE_PENDING_UPDATE = "stale_pending_update"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CDMDStage0Error(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _immutable_array(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def array_digest(value: Any) -> str:
    """Exact dtype/shape/byte digest without mutating the supplied array."""
    array = np.ascontiguousarray(np.asarray(value))
    return sha256_bytes(
        canonical_json_bytes(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "bytes_sha256": sha256_bytes(array.tobytes()),
            }
        )
    )


def channel_order_digest(channel_ids: Any) -> str:
    """Return the exact immutable channel-order identity for a trial view.

    Stage 0 receives two physically different neural views of one completed
    trial.  The caller may not join them merely by array width: both carry this
    digest and the live memory recomputes it from its frozen channel order.
    """
    channels = np.asarray(channel_ids)
    _require(channels.ndim == 1 and channels.size >= 1 and np.issubdtype(channels.dtype, np.integer),
             "channel order must be nonempty integer [N]")
    canonical = np.asarray(channels, dtype=np.int64)
    _require(len(set(int(value) for value in canonical.tolist())) == canonical.size,
             "channel order must be unique")
    return array_digest(canonical)


def _require_sha256(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
             f"{label} must be a lowercase SHA256")
    return value


def _require_nonempty_identifier(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def _frozen_array_payload(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": array_digest(value),
    }


def _wrap_angle(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def circular_distance(left: float, right: float) -> float:
    return abs(_wrap_angle(float(left) - float(right)))


def activity_fifo_capacity_for_support_budget(support_budget_m: int) -> int:
    """Return the only allowed FIFO capacity for a B3S support budget.

    The B3S calibration stack must never exceed its fixed training shape of 30
    trials.  In particular, M30 is an intentional zero-query-memory mode, not
    an empty-slice accident.
    """
    budget = int(support_budget_m)
    if budget not in ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET:
        raise CDMDStage0Error("CDM-D support budget must be exactly one of M4/M10/M30")
    return int(ACTIVITY_FIFO_CAPACITY_BY_SUPPORT_BUDGET[budget])


@dataclass(frozen=True)
class CDMDConfig:
    """Immutable Stage-0 thresholds, all disclosed in the state digest."""

    support_budget_m: int = 30
    activity_fifo_capacity: int | None = None
    dt: float = 0.02
    minimum_movement_bins: int = 3
    minimum_displacement: float = 0.05
    minimum_mean_speed: float = 0.05
    max_canonical_distance_rad: float = math.pi / 8.0
    max_group_direction_disagreement_rad: float = math.pi / 4.0
    minimum_design_rank: int = 3
    max_design_condition: float = 1.0e8
    minimum_accepted_evidence: int = 3
    departure_threshold: float = B8_DEPARTURE_THRESHOLD
    ridge_normalized_lambda: float = RIDGE_NORMALIZED_LAMBDA
    canonical_directions_rad: tuple[float, ...] = CANONICAL_DIRECTIONS_RAD
    group_count: int = GROUP_COUNT
    active_fit_mode: CarrierFitMode = CarrierFitMode.FIXED_RIDGE_BY_TRIAL

    def __post_init__(self) -> None:
        _require(self.group_count == GROUP_COUNT, "CDM-D freezes exactly K=4 complementary groups")
        expected_fifo_capacity = activity_fifo_capacity_for_support_budget(self.support_budget_m)
        if self.activity_fifo_capacity is None:
            object.__setattr__(self, "activity_fifo_capacity", expected_fifo_capacity)
        _require(self.activity_fifo_capacity == expected_fifo_capacity,
                 "activity FIFO capacity must be exactly 30 minus the frozen B3S support budget")
        _require(self.minimum_movement_bins >= 1, "minimum movement bins must be positive")
        _require(self.minimum_design_rank == 3, "CDM-D production carrier needs rank-3 cosine design")
        _require(self.minimum_accepted_evidence >= self.minimum_design_rank, "evidence minimum must support rank")
        for label, value in (
            ("dt", self.dt),
            ("minimum_displacement", self.minimum_displacement),
            ("minimum_mean_speed", self.minimum_mean_speed),
            ("max_canonical_distance_rad", self.max_canonical_distance_rad),
            ("max_group_direction_disagreement_rad", self.max_group_direction_disagreement_rad),
            ("max_design_condition", self.max_design_condition),
            ("departure_threshold", self.departure_threshold),
            ("ridge_normalized_lambda", self.ridge_normalized_lambda),
        ):
            _require(isinstance(value, (float, int)) and math.isfinite(float(value)), f"{label} must be finite")
        _require(float(self.dt) > 0.0, "dt must be positive")
        _require(float(self.minimum_displacement) > 0.0, "minimum displacement must be positive")
        _require(float(self.minimum_mean_speed) > 0.0, "minimum mean speed must be positive")
        _require(0.0 < float(self.max_canonical_distance_rad) <= math.pi / 4.0,
                 "canonical-direction threshold must be in (0, pi/4]")
        _require(0.0 <= float(self.max_group_direction_disagreement_rad) <= math.pi,
                 "group-direction threshold must be in [0, pi]")
        _require(float(self.max_design_condition) >= 1.0, "maximum condition must be >= 1")
        _require(float(self.departure_threshold) == B8_DEPARTURE_THRESHOLD,
                 "CDM-D must reuse the B8-compatible departure threshold 2.0")
        _require(float(self.ridge_normalized_lambda) == RIDGE_NORMALIZED_LAMBDA,
                 "CDM-D fixed ridge must use normalized lambda 0.1")
        _require(tuple(float(item) for item in self.canonical_directions_rad) == CANONICAL_DIRECTIONS_RAD,
                 "canonical direction literal drift")
        _require(isinstance(self.active_fit_mode, CarrierFitMode), "active fit mode must be explicit")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_cell_d_stage0_config_v1",
            "support_budget_m": int(self.support_budget_m),
            "activity_fifo_capacity": int(self.activity_fifo_capacity),
            "b3s_activity_stack_limit": ACTIVITY_STACK_LIMIT,
            "dt": float(self.dt),
            "minimum_movement_bins": int(self.minimum_movement_bins),
            "minimum_displacement": float(self.minimum_displacement),
            "minimum_mean_speed": float(self.minimum_mean_speed),
            "max_canonical_distance_rad": float(self.max_canonical_distance_rad),
            "max_group_direction_disagreement_rad": float(self.max_group_direction_disagreement_rad),
            "minimum_design_rank": int(self.minimum_design_rank),
            "max_design_condition": float(self.max_design_condition),
            "minimum_accepted_evidence": int(self.minimum_accepted_evidence),
            "departure_threshold": float(self.departure_threshold),
            "departure_threshold_source": "B8_carrier_recursive_estimator.DEFAULT_DEPARTURE_THRESHOLD",
            "ridge_normalized_lambda": float(self.ridge_normalized_lambda),
            "canonical_directions_rad": [float(item) for item in self.canonical_directions_rad],
            "group_count": int(self.group_count),
            "active_fit_mode": self.active_fit_mode.value,
        }


@dataclass(frozen=True)
class ComplementaryGroups:
    """Deterministic group assignment indexed in the supplied channel order."""

    channel_ids: np.ndarray = field(repr=False, compare=False)
    assignment: np.ndarray = field(repr=False, compare=False)
    valid_mask: np.ndarray = field(repr=False, compare=False)
    group_count: int = GROUP_COUNT

    def __post_init__(self) -> None:
        channels = np.asarray(self.channel_ids)
        assignment = np.asarray(self.assignment)
        valid = np.asarray(self.valid_mask)
        _require(channels.ndim == assignment.ndim == valid.ndim == 1, "group vectors must be one-dimensional")
        _require(channels.size >= self.group_count and assignment.size == channels.size == valid.size,
                 "group/channel count drift")
        _require(np.issubdtype(channels.dtype, np.integer), "channel IDs must be stable integers")
        _require(len(set(int(item) for item in channels.tolist())) == channels.size, "channel IDs must be unique")
        _require(valid.dtype == np.bool_, "valid mask must be boolean")
        _require(int(valid.sum()) >= self.group_count, "at least four valid units are required")
        _require(np.all((assignment[valid] >= 0) & (assignment[valid] < self.group_count)),
                 "each valid unit must have one complementary group")
        _require(np.all(assignment[~valid] == -1), "invalid units must carry group -1")
        counts = [int(np.sum(assignment == group)) for group in range(self.group_count)]
        _require(min(counts) >= 1, "no complementary group may be empty")
        _require(max(counts) - min(counts) <= 1, "complementary groups must be balanced")
        object.__setattr__(self, "channel_ids", _immutable_array(channels, dtype=np.int64))
        object.__setattr__(self, "assignment", _immutable_array(assignment, dtype=np.int64))
        object.__setattr__(self, "valid_mask", _immutable_array(valid, dtype=np.bool_))

    @property
    def units(self) -> int:
        return int(self.channel_ids.size)

    def unit_indices(self, group: int) -> np.ndarray:
        _require(0 <= int(group) < self.group_count, "invalid complementary group")
        return np.flatnonzero(self.assignment == int(group)).astype(np.int64, copy=False)

    def held_mask(self, group: int) -> np.ndarray:
        mask = np.asarray(self.assignment == int(group), dtype=np.bool_)
        mask.setflags(write=False)
        return mask

    def channel_to_group(self) -> dict[int, int]:
        return {int(channel): int(group) for channel, group in zip(self.channel_ids, self.assignment) if group >= 0}

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_complementary_groups_v1",
            "group_count": int(self.group_count),
            "channel_ids": _frozen_array_payload(self.channel_ids),
            "assignment": _frozen_array_payload(self.assignment),
            "valid_mask": _frozen_array_payload(self.valid_mask),
            "group_sizes": [int(self.unit_indices(group).size) for group in range(self.group_count)],
            "channel_to_group": {str(key): value for key, value in sorted(self.channel_to_group().items())},
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def build_complementary_groups(
    initial_raw_t4: Any,
    channel_ids: Any,
    *,
    valid_mask: Any | None = None,
    group_count: int = GROUP_COUNT,
) -> ComplementaryGroups:
    """Assign valid units by `(atan2(c,a), magnitude, channel_id)` then round robin.

    Sorting by stable channel ID only breaks exact directional/magnitude ties.
    Consequently, permuting the input rows while carrying the same channel IDs
    leaves the channel-to-group mapping invariant.
    """
    _require(group_count == GROUP_COUNT, "CDM-D freezes exactly K=4 groups")
    raw = np.asarray(initial_raw_t4, dtype=np.float64)
    channels = np.asarray(channel_ids)
    _require(raw.ndim == 2 and raw.shape[1] == 4 and raw.shape[0] >= group_count,
             "initial raw T4 must be [N,4] with N>=4")
    _require(channels.ndim == 1 and channels.size == raw.shape[0] and np.issubdtype(channels.dtype, np.integer),
             "channel IDs must be integer [N] matching raw T4")
    _require(len(set(int(item) for item in channels.tolist())) == channels.size, "channel IDs must be unique")
    if valid_mask is None:
        valid = np.isfinite(raw).all(axis=1)
    else:
        valid = np.asarray(valid_mask)
        _require(valid.dtype == np.bool_ and valid.shape == (raw.shape[0],), "valid mask must be bool [N]")
        _require(np.isfinite(raw[valid]).all(), "valid raw T4 rows must be finite")
    _require(int(valid.sum()) >= group_count, "need at least four valid units")
    theta = np.arctan2(raw[:, 1], raw[:, 0])
    magnitude = raw[:, 2]
    valid_indices = [int(index) for index in np.flatnonzero(valid)]
    ordered = sorted(valid_indices, key=lambda index: (float(theta[index]), float(magnitude[index]), int(channels[index])))
    assignment = np.full(raw.shape[0], -1, dtype=np.int64)
    for rank, index in enumerate(ordered):
        assignment[index] = int(rank % group_count)
    return ComplementaryGroups(
        channel_ids=_immutable_array(channels, dtype=np.int64),
        assignment=_immutable_array(assignment, dtype=np.int64),
        valid_mask=_immutable_array(valid, dtype=np.bool_),
        group_count=group_count,
    )


@dataclass(frozen=True)
class PseudoDirection:
    """Quality-audited displacement direction from one completed trial."""

    accepted: bool
    reason: UpdateRejectionReason | None
    theta_raw_rad: float | None
    theta_index: int | None
    theta_canonical_rad: float | None
    canonical_distance_rad: float | None
    movement_bins: int
    displacement_norm: float | None
    mean_speed: float | None

    def payload(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": None if self.reason is None else self.reason.value,
            "theta_raw_rad": self.theta_raw_rad,
            "theta_index": self.theta_index,
            "theta_canonical_rad": self.theta_canonical_rad,
            "canonical_distance_rad": self.canonical_distance_rad,
            "movement_bins": int(self.movement_bins),
            "displacement_norm": self.displacement_norm,
            "mean_speed": self.mean_speed,
        }


def nearest_canonical_direction(theta_rad: float, *, canonical_directions_rad: Sequence[float] = CANONICAL_DIRECTIONS_RAD) -> tuple[int, float]:
    """Return nearest direction with a lowest-index deterministic circular tie break."""
    theta = float(theta_rad)
    _require(math.isfinite(theta), "theta must be finite")
    canonical = tuple(float(value) for value in canonical_directions_rad)
    _require(canonical == CANONICAL_DIRECTIONS_RAD, "canonical direction literal drift")
    distances = np.asarray([circular_distance(theta, value) for value in canonical], dtype=np.float64)
    minimum = float(np.min(distances))
    # The tolerance is solely for numerical representation around an exact
    # halfway direction.  The deterministic rule itself is lower index.
    tied = np.flatnonzero(np.abs(distances - minimum) <= 16.0 * np.finfo(np.float64).eps)
    index = int(tied[0])
    return index, minimum


def pseudo_direction_from_velocity(
    predicted_velocity: Any,
    movement_mask: Any,
    *,
    config: CDMDConfig,
) -> PseudoDirection:
    """Integrate completed predicted velocity and apply all local quality gates."""
    velocity = np.asarray(predicted_velocity)
    mask = np.asarray(movement_mask)
    if velocity.ndim != 2 or velocity.shape[1] != 2 or velocity.shape[0] < 1:
        return PseudoDirection(False, UpdateRejectionReason.VELOCITY_SHAPE, None, None, None, None, 0, None, None)
    if mask.dtype != np.bool_ or mask.ndim != 1 or mask.shape[0] != velocity.shape[0]:
        return PseudoDirection(False, UpdateRejectionReason.MOVEMENT_MASK, None, None, None, None, 0, None, None)
    if not np.issubdtype(velocity.dtype, np.floating) or not np.isfinite(velocity).all():
        return PseudoDirection(False, UpdateRejectionReason.VELOCITY_NONFINITE, None, None, None, None, int(mask.sum()), None, None)
    movement_bins = int(mask.sum())
    if movement_bins < config.minimum_movement_bins:
        return PseudoDirection(False, UpdateRejectionReason.MOVEMENT_TOO_SHORT, None, None, None, None, movement_bins, None, None)
    selected = np.asarray(velocity[mask], dtype=np.float64)
    displacement = selected.sum(axis=0) * float(config.dt)
    displacement_norm = float(np.linalg.norm(displacement))
    speeds = np.linalg.norm(selected, axis=1)
    mean_speed = float(speeds.mean())
    if not math.isfinite(displacement_norm) or displacement_norm < float(config.minimum_displacement):
        return PseudoDirection(False, UpdateRejectionReason.LOW_DISPLACEMENT, None, None, None, None, movement_bins, displacement_norm, mean_speed)
    if not math.isfinite(mean_speed) or mean_speed < float(config.minimum_mean_speed):
        return PseudoDirection(False, UpdateRejectionReason.LOW_MEAN_SPEED, None, None, None, None, movement_bins, displacement_norm, mean_speed)
    theta_raw = float(math.atan2(float(displacement[1]), float(displacement[0])))
    index, distance = nearest_canonical_direction(theta_raw, canonical_directions_rad=config.canonical_directions_rad)
    if distance > float(config.max_canonical_distance_rad):
        return PseudoDirection(False, UpdateRejectionReason.CANONICAL_DIRECTION_TOO_FAR, theta_raw, index,
                               float(config.canonical_directions_rad[index]), distance, movement_bins, displacement_norm, mean_speed)
    return PseudoDirection(True, None, theta_raw, index, float(config.canonical_directions_rad[index]), distance,
                           movement_bins, displacement_norm, mean_speed)


@dataclass(frozen=True)
class B3SInterpolatedSpikeCountTrial:
    """The exact 100-bin interpolated/padded spike-count view used by B3S.

    This capability is deliberately *not* a rate-estimation view.  It is the
    only type accepted by :class:`ActivityMemory`, so a caller cannot silently
    substitute native rewarded-trial counts or a bare ``ndarray``.
    """

    activity: np.ndarray = field(repr=False, compare=False)
    session_id: str
    trial_id: str
    channel_order_sha256: str

    def __post_init__(self) -> None:
        activity = np.asarray(self.activity)
        _require(activity.ndim == 2 and activity.shape[0] == 100 and activity.shape[1] >= 1,
                 "B3S trial activity must be exact interpolated/padded [100,N]")
        _require(np.issubdtype(activity.dtype, np.floating) and np.isfinite(activity).all(),
                 "B3S trial activity must be finite floating spike-count view")
        _require_nonempty_identifier(self.session_id, label="B3S session_id")
        _require_nonempty_identifier(self.trial_id, label="B3S trial_id")
        _require_sha256(self.channel_order_sha256, label="B3S channel_order_sha256")
        object.__setattr__(self, "activity", _immutable_array(activity))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "cdmd_b3s_interpolated_spike_count_trial_v1",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "channel_order_sha256": self.channel_order_sha256,
            "activity": _frozen_array_payload(self.activity),
            "b3s_shape": [100, int(self.activity.shape[1])],
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class NativeRewardedTrialSpikeCounts:
    """Native binned counts over the full rewarded-trial interval.

    ``counts`` intentionally remains a distinct view from B3S's cubic
    interpolated/padded tensor.  Its full declared interval, not the velocity
    validity mask, defines the scalar rate used by the carrier estimator.
    """

    counts: np.ndarray = field(repr=False, compare=False)
    session_id: str
    trial_id: str
    channel_order_sha256: str
    rewarded_interval_start_bin: int
    rewarded_interval_stop_bin: int
    bin_width_seconds: float = 0.020

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        _require(counts.ndim == 2 and counts.shape[0] >= 1 and counts.shape[1] >= 1,
                 "native rewarded-trial counts must be [T,N]")
        _require(np.issubdtype(counts.dtype, np.number) and np.isfinite(counts).all() and np.all(counts >= 0.0),
                 "native rewarded-trial counts must be finite nonnegative numeric values")
        _require_nonempty_identifier(self.session_id, label="native-count session_id")
        _require_nonempty_identifier(self.trial_id, label="native-count trial_id")
        _require_sha256(self.channel_order_sha256, label="native-count channel_order_sha256")
        _require(isinstance(self.rewarded_interval_start_bin, int) and isinstance(self.rewarded_interval_stop_bin, int),
                 "native rewarded interval bounds must be integers")
        _require(self.rewarded_interval_stop_bin > self.rewarded_interval_start_bin,
                 "native rewarded interval must be nonempty")
        _require(self.rewarded_interval_stop_bin - self.rewarded_interval_start_bin == counts.shape[0],
                 "native count rows must span the full declared rewarded interval")
        _require(float(self.bin_width_seconds) == 0.020,
                 "carrier rate is frozen to 20-ms native bins")
        object.__setattr__(self, "counts", _immutable_array(counts))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "cdmd_native_rewarded_trial_spike_counts_v1",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "channel_order_sha256": self.channel_order_sha256,
            "rewarded_interval_start_bin": int(self.rewarded_interval_start_bin),
            "rewarded_interval_stop_bin": int(self.rewarded_interval_stop_bin),
            "bin_width_seconds": float(self.bin_width_seconds),
            "counts": _frozen_array_payload(self.counts),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class VelocityValidityEvidence:
    """Target-label-free validity evidence for a completed velocity prediction.

    This evidence has its own prediction interval and mask length.  It must
    bind the completed trial, but it is not asserted to share the B3S shape or
    native-count interval used for the carrier scalar rate.
    """

    valid_mask: np.ndarray = field(repr=False, compare=False)
    session_id: str
    trial_id: str
    prediction_interval_start_bin: int
    prediction_interval_stop_bin: int
    source: str = "neural_window_availability_and_trial_bounds"
    target_behavior_used: bool = False

    def __post_init__(self) -> None:
        mask = np.asarray(self.valid_mask)
        _require(mask.ndim == 1 and mask.size >= 1 and mask.dtype == np.bool_,
                 "velocity validity mask must be nonempty bool [P]")
        _require_nonempty_identifier(self.session_id, label="velocity-validity session_id")
        _require_nonempty_identifier(self.trial_id, label="velocity-validity trial_id")
        _require(isinstance(self.prediction_interval_start_bin, int) and isinstance(self.prediction_interval_stop_bin, int),
                 "velocity validity interval bounds must be integers")
        _require(self.prediction_interval_stop_bin > self.prediction_interval_start_bin,
                 "velocity validity interval must be nonempty")
        _require(self.prediction_interval_stop_bin - self.prediction_interval_start_bin == mask.size,
                 "velocity validity mask must span its declared prediction interval")
        _require(self.source == "neural_window_availability_and_trial_bounds",
                 "velocity validity source must remain target-label-free")
        _require(self.target_behavior_used is False,
                 "velocity validity must not use target behavior labels")
        object.__setattr__(self, "valid_mask", _immutable_array(mask, dtype=np.bool_))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "cdmd_velocity_validity_evidence_v1",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "prediction_interval_start_bin": int(self.prediction_interval_start_bin),
            "prediction_interval_stop_bin": int(self.prediction_interval_stop_bin),
            "source": self.source,
            "target_behavior_used": bool(self.target_behavior_used),
            "valid_mask": _frozen_array_payload(self.valid_mask),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class CompletedVelocityPrediction:
    """A decoder velocity output coupled to independent validity evidence."""

    velocity: np.ndarray = field(repr=False, compare=False)
    validity: VelocityValidityEvidence = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        velocity = np.asarray(self.velocity)
        _require(isinstance(self.validity, VelocityValidityEvidence),
                 "completed velocity requires typed validity evidence")
        _require(velocity.ndim == 2 and velocity.shape == (self.validity.valid_mask.size, 2),
                 "completed velocity must be [P,2] for its own validity interval")
        _require(np.issubdtype(velocity.dtype, np.floating) and np.isfinite(velocity).all(),
                 "completed velocity must be finite floating [P,2]")
        object.__setattr__(self, "velocity", _immutable_array(velocity))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "cdmd_completed_velocity_prediction_v1",
            "velocity": _frozen_array_payload(self.velocity),
            "validity": self.validity.payload(),
        }


@dataclass(frozen=True)
class ActivityMemory:
    """Immutable B3S support plus a bounded FIFO of typed completed trials."""

    support_trials: tuple[B3SInterpolatedSpikeCountTrial, ...] = field(repr=False, compare=False)
    query_trials: tuple[B3SInterpolatedSpikeCountTrial, ...] = field(repr=False, compare=False)
    channel_ids: np.ndarray = field(repr=False, compare=False)
    fifo_capacity: int = 0

    def __post_init__(self) -> None:
        _require(self.fifo_capacity >= 0, "activity FIFO capacity must be nonnegative")
        channels = np.asarray(self.channel_ids)
        _require(channels.ndim == 1 and channels.size >= 1 and np.issubdtype(channels.dtype, np.integer),
                 "activity channel IDs must be nonempty integer [N]")
        _require(len(set(int(value) for value in channels.tolist())) == channels.size, "activity channel IDs must be unique")
        _require(bool(self.support_trials), "activity memory requires at least one immutable support trial")
        _require(len(self.query_trials) <= self.fifo_capacity, "activity FIFO exceeds configured capacity")
        _require(len(self.support_trials) + self.fifo_capacity == ACTIVITY_STACK_LIMIT,
                 "activity support plus FIFO capacity must exactly preserve the B3S 30-trial stack")
        expected_channel_digest = channel_order_digest(channels)
        expected_session: str | None = None
        trial_ids: set[str] = set()
        for collection in (self.support_trials, self.query_trials):
            for trial in collection:
                _require(isinstance(trial, B3SInterpolatedSpikeCountTrial),
                         "activity memory accepts only typed B3S interpolated trial capabilities")
                _require(trial.activity.shape == (100, channels.size),
                         "B3S activity trial shape must be exact [100,N]")
                _require(trial.channel_order_sha256 == expected_channel_digest,
                         "B3S trial channel-order capability drift")
                if expected_session is None:
                    expected_session = trial.session_id
                _require(trial.session_id == expected_session, "B3S activity memory cannot mix sessions")
                _require(trial.trial_id not in trial_ids, "B3S activity memory cannot duplicate trial IDs")
                trial_ids.add(trial.trial_id)
        object.__setattr__(self, "channel_ids", _immutable_array(channels, dtype=np.int64))
        object.__setattr__(self, "support_trials", tuple(self.support_trials))
        object.__setattr__(self, "query_trials", tuple(self.query_trials))

    @classmethod
    def initialize(
        cls,
        support_trials: Sequence[B3SInterpolatedSpikeCountTrial],
        *,
        channel_ids: Any,
        fifo_capacity: int,
    ) -> "ActivityMemory":
        _require(not isinstance(support_trials, np.ndarray),
                 "activity memory rejects bare [trials,100,N] arrays; use B3S trial capabilities")
        _require(isinstance(support_trials, Sequence) and not isinstance(support_trials, (str, bytes)),
                 "support activity must be an ordered B3S trial capability sequence")
        support = tuple(support_trials)
        _require(bool(support), "support activity must be nonempty")
        return cls(support_trials=support, query_trials=(), channel_ids=_immutable_array(channel_ids, dtype=np.int64),
                   fifo_capacity=int(fifo_capacity))

    @property
    def session_id(self) -> str:
        return self.support_trials[0].session_id

    @property
    def channel_order_sha256(self) -> str:
        return channel_order_digest(self.channel_ids)

    @property
    def trial_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.support_trials[0].activity.shape)

    @property
    def query_count(self) -> int:
        return len(self.query_trials)

    def stack(self) -> np.ndarray:
        """Return B3S support followed by completed queries without mutable state."""
        result = np.stack(tuple(item.activity for item in self.support_trials + self.query_trials), axis=0)
        result.setflags(write=False)
        return result

    def validate_complete_trial(self, trial_activity: Any) -> UpdateRejectionReason | None:
        if not isinstance(trial_activity, B3SInterpolatedSpikeCountTrial):
            return UpdateRejectionReason.TRIAL_CAPABILITY
        if trial_activity.activity.shape != self.trial_shape:
            return UpdateRejectionReason.ACTIVITY_SHAPE
        if trial_activity.channel_order_sha256 != self.channel_order_sha256 or trial_activity.session_id != self.session_id:
            return UpdateRejectionReason.TRIAL_BINDING
        if trial_activity.trial_id in {item.trial_id for item in self.support_trials + self.query_trials}:
            return UpdateRejectionReason.TRIAL_BINDING
        return None

    def after_accepted_trial(self, trial_activity: B3SInterpolatedSpikeCountTrial) -> "ActivityMemory":
        reason = self.validate_complete_trial(trial_activity)
        _require(reason is None, f"cannot append invalid B3S activity trial: {reason}")
        # ``[-0:]`` is the entire tuple in Python.  M30 is a real no-query-
        # memory mode, so spell out the capacity-zero transition explicitly.
        next_queries = () if self.fifo_capacity == 0 else (self.query_trials + (trial_activity,))[-self.fifo_capacity :]
        return ActivityMemory(
            support_trials=self.support_trials,
            query_trials=next_queries,
            channel_ids=self.channel_ids,
            fifo_capacity=self.fifo_capacity,
        )

    def after_completed_trial(self, trial_activity: B3SInterpolatedSpikeCountTrial) -> "ActivityMemory":
        """Append a valid unlabeled completed-trial B3S capability.

        ``after_accepted_trial`` is the frozen V1/V2 spelling and remains
        byte-for-byte behaviorally authoritative for those routes.  The V3
        independent-activity successor needs a name that makes its distinct
        causal rule explicit: a valid *activity* view advances regardless of
        whether a separate pseudo-label/carrier proposal is later accepted.
        This is deliberately a narrow alias rather than a change to the old
        transition API.
        """
        return self.after_accepted_trial(trial_activity)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_activity_fifo_v2",
            "channel_ids": _frozen_array_payload(self.channel_ids),
            "channel_order_sha256": self.channel_order_sha256,
            "session_id": self.session_id,
            "fifo_capacity": int(self.fifo_capacity),
            "b3s_activity_stack_limit": ACTIVITY_STACK_LIMIT,
            "support_trials": [item.payload() for item in self.support_trials],
            "query_trials": [item.payload() for item in self.query_trials],
            "support_precedes_query": True,
            "query_count": int(self.query_count),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class CarrierSufficientStatistics:
    """Direction count/sum/squared-sum bank for four complementary groups."""

    counts: np.ndarray = field(repr=False, compare=False)
    rate_sums: np.ndarray = field(repr=False, compare=False)
    rate_sq_sums: np.ndarray = field(repr=False, compare=False)
    groups: ComplementaryGroups = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        sums = np.asarray(self.rate_sums)
        squares = np.asarray(self.rate_sq_sums)
        expected_counts = (self.groups.group_count, len(CANONICAL_DIRECTIONS_RAD))
        expected_sums = expected_counts + (self.groups.units,)
        _require(counts.shape == expected_counts and np.issubdtype(counts.dtype, np.integer), "carrier count shape/dtype drift")
        _require(sums.shape == expected_sums and squares.shape == expected_sums, "carrier sum shape drift")
        _require(np.issubdtype(sums.dtype, np.floating) and np.issubdtype(squares.dtype, np.floating),
                 "carrier sums must be floating")
        _require(np.all(counts >= 0) and np.isfinite(sums).all() and np.isfinite(squares).all() and np.all(squares >= 0.0),
                 "carrier statistics must be finite and nonnegative where required")
        for group in range(self.groups.group_count):
            unheld = self.groups.assignment != group
            _require(np.array_equal(sums[group, :, unheld], np.zeros_like(sums[group, :, unheld])),
                     "cross-group rate sum contamination")
            _require(np.array_equal(squares[group, :, unheld], np.zeros_like(squares[group, :, unheld])),
                     "cross-group squared-rate contamination")
        object.__setattr__(self, "counts", _immutable_array(counts, dtype=np.int64))
        object.__setattr__(self, "rate_sums", _immutable_array(sums, dtype=np.float64))
        object.__setattr__(self, "rate_sq_sums", _immutable_array(squares, dtype=np.float64))

    @classmethod
    def empty(cls, groups: ComplementaryGroups) -> "CarrierSufficientStatistics":
        shape = (groups.group_count, len(CANONICAL_DIRECTIONS_RAD), groups.units)
        return cls(
            counts=np.zeros(shape[:2], dtype=np.int64),
            rate_sums=np.zeros(shape, dtype=np.float64),
            rate_sq_sums=np.zeros(shape, dtype=np.float64),
            groups=groups,
        )

    @classmethod
    def from_labeled_support(
        cls,
        groups: ComplementaryGroups,
        trial_rates: Any,
        direction_indices: Any,
    ) -> "CarrierSufficientStatistics":
        rates = np.asarray(trial_rates, dtype=np.float64)
        directions = np.asarray(direction_indices)
        _require(rates.ndim == 2 and rates.shape[1] == groups.units and rates.shape[0] >= 1,
                 "support rates must be [trials,units]")
        _require(np.isfinite(rates).all(), "support rates must be finite")
        _require(directions.ndim == 1 and directions.size == rates.shape[0] and np.issubdtype(directions.dtype, np.integer),
                 "support directions must be integer [trials]")
        _require(np.all((directions >= 0) & (directions < len(CANONICAL_DIRECTIONS_RAD))),
                 "support direction index out of range")
        counts = np.zeros((groups.group_count, len(CANONICAL_DIRECTIONS_RAD)), dtype=np.int64)
        sums = np.zeros((groups.group_count, len(CANONICAL_DIRECTIONS_RAD), groups.units), dtype=np.float64)
        squares = np.zeros_like(sums)
        for trial_index, direction in enumerate(directions.tolist()):
            for group in range(groups.group_count):
                units = groups.unit_indices(group)
                counts[group, int(direction)] += 1
                values = rates[trial_index, units]
                sums[group, int(direction), units] += values
                squares[group, int(direction), units] += values * values
        return cls(counts=counts, rate_sums=sums, rate_sq_sums=squares, groups=groups)

    def after_pseudo_trial(self, direction_indices: Sequence[int], scalar_rates: Any) -> "CarrierSufficientStatistics":
        directions = tuple(int(value) for value in direction_indices)
        rates = np.asarray(scalar_rates, dtype=np.float64)
        _require(len(directions) == self.groups.group_count, "need one pseudo direction for every complementary group")
        _require(rates.shape == (self.groups.units,) and np.isfinite(rates).all(), "scalar rates must be finite [units]")
        _require(all(0 <= value < len(CANONICAL_DIRECTIONS_RAD) for value in directions), "pseudo direction out of range")
        counts = self.counts.copy()
        sums = self.rate_sums.copy()
        squares = self.rate_sq_sums.copy()
        for group, direction in enumerate(directions):
            units = self.groups.unit_indices(group)
            counts[group, direction] += 1
            sums[group, direction, units] += rates[units]
            squares[group, direction, units] += rates[units] * rates[units]
        return CarrierSufficientStatistics(counts=counts, rate_sums=sums, rate_sq_sums=squares, groups=self.groups)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_carrier_sufficient_statistics_v1",
            "counts": _frozen_array_payload(self.counts),
            "rate_sums": _frozen_array_payload(self.rate_sums),
            "rate_sq_sums": _frozen_array_payload(self.rate_sq_sums),
            "groups_sha256": self.groups.digest,
            "directions": len(CANONICAL_DIRECTIONS_RAD),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def scalar_rates_from_native_rewarded_counts(native_counts: NativeRewardedTrialSpikeCounts) -> np.ndarray:
    """Compute scalar carrier rates from all native rewarded-trial bins only.

    The denominator is the frozen 20-ms bin width.  In particular, this
    function never accepts an interpolated B3S tensor and never takes a
    velocity/movement mask: direction validity and rate estimation are
    deliberately separate estimands.
    """
    _require(isinstance(native_counts, NativeRewardedTrialSpikeCounts),
             "carrier scalar rates require typed native rewarded-trial counts")
    counts = native_counts.counts
    result = np.asarray(counts.mean(axis=0, dtype=np.float64) / float(native_counts.bin_width_seconds), dtype=np.float64)
    _require(result.shape == (counts.shape[1],) and np.isfinite(result).all(), "native scalar rate construction failed")
    result.setflags(write=False)
    return result


def _design_by_direction(present_direction_indices: np.ndarray) -> np.ndarray:
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in present_direction_indices], dtype=np.float64)
    return np.column_stack((np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)))


def fit_carriers_from_trial_table(
    trial_rates: Any,
    direction_indices: Any,
    *,
    mode: CarrierFitMode,
    normalized_lambda: float = RIDGE_NORMALIZED_LAMBDA,
) -> np.ndarray:
    """Production-compatible direct reference for one complete trial table.

    This function exists for synthetic parity evidence.  It does not select
    trials, discover labels, or accept any sequential/RLS observation mode.
    """
    rates = np.asarray(trial_rates, dtype=np.float64)
    directions = np.asarray(direction_indices)
    _require(rates.ndim == 2 and rates.shape[0] >= 3 and rates.shape[1] >= 1, "trial rates must be [trials,units]")
    _require(np.isfinite(rates).all(), "trial rates must be finite")
    _require(directions.ndim == 1 and directions.size == rates.shape[0] and np.issubdtype(directions.dtype, np.integer),
             "direction indices must be integer [trials]")
    _require(np.all((directions >= 0) & (directions < len(CANONICAL_DIRECTIONS_RAD))), "direction index out of range")
    _require(isinstance(mode, CarrierFitMode), "fit mode must be explicit")
    if mode is CarrierFitMode.ORDINARY_OLS_BY_DIRECTION:
        present = np.unique(directions).astype(np.int64, copy=False)
        design = _design_by_direction(present)
        _require(int(np.linalg.matrix_rank(design)) == 3, "ordinary OLS requires rank-3 present directions")
        means = np.stack([rates[directions == direction].mean(axis=0) for direction in present], axis=0)
        coefficients, *_ = np.linalg.lstsq(design, means, rcond=None)
        b, a, c = coefficients
        result = np.column_stack((a, c, np.hypot(a, c), b)).astype(np.float64, copy=False)
    else:
        _require(float(normalized_lambda) == RIDGE_NORMALIZED_LAMBDA, "fixed ridge lambda must be 0.1")
        theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions], dtype=np.float64)
        design = np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size, dtype=np.float64)))
        penalty = np.diag((theta.size * float(normalized_lambda), theta.size * float(normalized_lambda), 0.0))
        try:
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ rates)
        except np.linalg.LinAlgError as error:
            raise CDMDStage0Error("fixed ridge normal equations are singular") from error
        a, c, b = coefficients
        result = np.ascontiguousarray(np.column_stack((a, c, np.hypot(a, c), b)), dtype=np.float32)
    _require(result.shape == (rates.shape[1], 4) and np.isfinite(result).all(), "carrier fit output drift")
    result.setflags(write=False)
    return result


def _fit_group_from_statistics(
    statistics: CarrierSufficientStatistics,
    group: int,
    *,
    mode: CarrierFitMode,
    normalized_lambda: float,
) -> np.ndarray:
    """Fit only a held group's units from its count/sum sufficient statistics."""
    units = statistics.groups.unit_indices(group)
    counts = np.asarray(statistics.counts[group], dtype=np.int64)
    present = np.flatnonzero(counts > 0).astype(np.int64, copy=False)
    _require(present.size >= 1, "carrier group has no accepted evidence")
    sums = statistics.rate_sums[group, present][:, units]
    if mode is CarrierFitMode.ORDINARY_OLS_BY_DIRECTION:
        design = _design_by_direction(present)
        means = sums / counts[present, None]
        coefficients, *_ = np.linalg.lstsq(design, means, rcond=None)
        b, a, c = coefficients
        fitted = np.column_stack((a, c, np.hypot(a, c), b)).astype(np.float64, copy=False)
    else:
        theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in present], dtype=np.float64)
        rows = np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size, dtype=np.float64)))
        weight = counts[present].astype(np.float64, copy=False)
        system = rows.T @ (weight[:, None] * rows)
        penalty = np.diag((float(weight.sum()) * normalized_lambda, float(weight.sum()) * normalized_lambda, 0.0))
        right = rows.T @ sums
        try:
            coefficients = np.linalg.solve(system + penalty, right)
        except np.linalg.LinAlgError as error:
            raise CDMDStage0Error("fixed-ridge statistics normal equations are singular") from error
        a, c, b = coefficients
        fitted = np.ascontiguousarray(np.column_stack((a, c, np.hypot(a, c), b)), dtype=np.float32)
    _require(fitted.shape == (units.size, 4) and np.isfinite(fitted).all(), "group carrier fit drift")
    return fitted


def _design_evidence(statistics: CarrierSufficientStatistics, config: CDMDConfig) -> tuple[UpdateRejectionReason | None, tuple[dict[str, Any], ...]]:
    rows: list[dict[str, Any]] = []
    reason: UpdateRejectionReason | None = None
    for group in range(statistics.groups.group_count):
        counts = statistics.counts[group]
        present = np.flatnonzero(counts > 0).astype(np.int64, copy=False)
        evidence_count = int(counts.sum())
        design = _design_by_direction(present) if present.size else np.empty((0, 3), dtype=np.float64)
        rank = int(np.linalg.matrix_rank(design)) if present.size else 0
        condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
        rows.append(
            {
                "group": group,
                "accepted_evidence": evidence_count,
                "present_directions": [int(item) for item in present],
                "design_rank": rank,
                "design_condition": condition,
            }
        )
        if evidence_count < config.minimum_accepted_evidence:
            reason = reason or UpdateRejectionReason.INSUFFICIENT_EVIDENCE
        elif rank < config.minimum_design_rank:
            reason = reason or UpdateRejectionReason.INSUFFICIENT_DESIGN_RANK
        elif not math.isfinite(condition) or condition > config.max_design_condition:
            reason = reason or UpdateRejectionReason.ILL_CONDITIONED_DESIGN
    return reason, tuple(rows)


def _departure_ratios(initial_t4: np.ndarray, candidate_t4: np.ndarray, groups: ComplementaryGroups) -> np.ndarray:
    initial = np.asarray(initial_t4, dtype=np.float64)
    candidate = np.asarray(candidate_t4, dtype=np.float64)
    _require(initial.shape == candidate.shape == (groups.units, 4), "T4 departure shape drift")
    baseline = np.linalg.norm(initial[:, :2], axis=1)
    delta = np.linalg.norm(candidate[:, :2] - initial[:, :2], axis=1)
    ratios = np.where(baseline > NUMERIC_EPSILON, delta / baseline, np.where(delta > NUMERIC_EPSILON, delta, 0.0))
    ratios[~groups.valid_mask] = 0.0
    return ratios


@dataclass(frozen=True)
class CarrierMemory:
    """Immutable carrier bank with current OLS and fixed-ridge reconstructions."""

    groups: ComplementaryGroups = field(repr=False, compare=False)
    initial_raw_t4: np.ndarray = field(repr=False, compare=False)
    statistics: CarrierSufficientStatistics = field(repr=False, compare=False)
    ordinary_t4: np.ndarray = field(repr=False, compare=False)
    fixed_ridge_t4: np.ndarray = field(repr=False, compare=False)
    config: CDMDConfig = field(default_factory=CDMDConfig)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_raw_t4)
        ordinary = np.asarray(self.ordinary_t4)
        ridge = np.asarray(self.fixed_ridge_t4)
        _require(initial.shape == ordinary.shape == ridge.shape == (self.groups.units, 4), "carrier T4 shape drift")
        _require(np.issubdtype(initial.dtype, np.floating) and np.isfinite(initial).all(), "initial raw T4 must be finite")
        _require(np.isfinite(ordinary).all() and np.isfinite(ridge).all(), "carrier T4 must be finite")
        _require(self.statistics.groups.digest == self.groups.digest, "carrier statistics/group binding drift")
        object.__setattr__(self, "initial_raw_t4", _immutable_array(initial, dtype=np.float64))
        object.__setattr__(self, "ordinary_t4", _immutable_array(ordinary))
        object.__setattr__(self, "fixed_ridge_t4", _immutable_array(ridge))

    @classmethod
    def from_support_trials(
        cls,
        *,
        initial_raw_t4: Any,
        channel_ids: Any,
        support_trial_rates: Any,
        support_direction_indices: Any,
        config: CDMDConfig,
        valid_mask: Any | None = None,
    ) -> "CarrierMemory":
        initial = np.asarray(initial_raw_t4, dtype=np.float64)
        support_rates = np.asarray(support_trial_rates)
        support_directions = np.asarray(support_direction_indices)
        _require(support_rates.ndim == 2 and support_rates.shape[0] == config.support_budget_m,
                 "carrier support-rate rows must exactly equal the frozen M4/M10/M30 budget")
        _require(support_directions.ndim == 1 and support_directions.size == config.support_budget_m,
                 "carrier support-direction rows must exactly equal the frozen M4/M10/M30 budget")
        groups = build_complementary_groups(initial, channel_ids, valid_mask=valid_mask, group_count=config.group_count)
        stats = CarrierSufficientStatistics.from_labeled_support(groups, support_rates, support_directions)
        # The initial ordinary point T4 remains the exact no-update fallback.
        # Online accepted updates reconstruct both estimands from these support
        # statistics plus completed pseudo-labelled trials.
        return cls(groups=groups, initial_raw_t4=initial, statistics=stats, ordinary_t4=initial,
                   fixed_ridge_t4=initial, config=config)

    @property
    def active_t4(self) -> np.ndarray:
        return self.ordinary_t4 if self.config.active_fit_mode is CarrierFitMode.ORDINARY_OLS_BY_DIRECTION else self.fixed_ridge_t4

    def _reconstruct(self, statistics: CarrierSufficientStatistics, mode: CarrierFitMode) -> np.ndarray:
        seed = self.initial_raw_t4.copy()
        for group in range(self.groups.group_count):
            units = self.groups.unit_indices(group)
            fitted = _fit_group_from_statistics(statistics, group, mode=mode,
                                                normalized_lambda=self.config.ridge_normalized_lambda)
            seed[units] = fitted
        _require(np.isfinite(seed).all(), "reconstructed carrier is nonfinite")
        result = np.ascontiguousarray(seed if mode is CarrierFitMode.ORDINARY_OLS_BY_DIRECTION else seed.astype(np.float32))
        result.setflags(write=False)
        return result

    def propose_pseudo_trial(self, direction_indices: Sequence[int], scalar_rates: Any) -> "CarrierProposal":
        candidate_stats = self.statistics.after_pseudo_trial(direction_indices, scalar_rates)
        design_reason, design_rows = _design_evidence(candidate_stats, self.config)
        if design_reason is not None:
            return CarrierProposal(False, design_reason, None, design_rows, None)
        try:
            ordinary = self._reconstruct(candidate_stats, CarrierFitMode.ORDINARY_OLS_BY_DIRECTION)
            ridge = self._reconstruct(candidate_stats, CarrierFitMode.FIXED_RIDGE_BY_TRIAL)
        except CDMDStage0Error:
            return CarrierProposal(False, UpdateRejectionReason.NONFINITE_CARRIER, None, design_rows, None)
        candidate = CarrierMemory(
            groups=self.groups,
            initial_raw_t4=self.initial_raw_t4,
            statistics=candidate_stats,
            ordinary_t4=ordinary,
            fixed_ridge_t4=ridge,
            config=self.config,
        )
        ratios = _departure_ratios(self.initial_raw_t4, candidate.active_t4, self.groups)
        if not np.isfinite(ratios).all():
            return CarrierProposal(False, UpdateRejectionReason.NONFINITE_CARRIER, None, design_rows, ratios)
        if float(ratios.max()) > self.config.departure_threshold:
            return CarrierProposal(False, UpdateRejectionReason.DEPARTURE_FREEZE, None, design_rows, ratios)
        return CarrierProposal(True, None, candidate, design_rows, ratios)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_carrier_memory_v1",
            "groups_sha256": self.groups.digest,
            "initial_raw_t4": _frozen_array_payload(self.initial_raw_t4),
            "ordinary_t4": _frozen_array_payload(self.ordinary_t4),
            "fixed_ridge_t4": _frozen_array_payload(self.fixed_ridge_t4),
            "active_fit_mode": self.config.active_fit_mode.value,
            "statistics_sha256": self.statistics.digest,
            "config": self.config.payload(),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class CarrierProposal:
    accepted: bool
    reason: UpdateRejectionReason | None
    candidate: CarrierMemory | None = field(repr=False, compare=False)
    design_rows: tuple[dict[str, Any], ...]
    departure_ratios: np.ndarray | None = field(repr=False, compare=False)

    def payload(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": None if self.reason is None else self.reason.value,
            "design_rows": [dict(row) for row in self.design_rows],
            "departure_ratios": None if self.departure_ratios is None else _frozen_array_payload(np.asarray(self.departure_ratios)),
            "candidate_carrier_sha256": None if self.candidate is None else self.candidate.digest,
        }


def assert_production_parity(
    candidate_t4: Any,
    trial_rates: Any,
    direction_indices: Any,
    *,
    mode: CarrierFitMode,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> dict[str, Any]:
    """Fail closed if a supplied carrier differs from the explicit mode reference."""
    candidate = np.asarray(candidate_t4)
    expected = fit_carriers_from_trial_table(trial_rates, direction_indices, mode=mode)
    _require(candidate.shape == expected.shape, "carrier parity shape drift")
    difference = float(np.max(np.abs(candidate.astype(np.float64) - expected.astype(np.float64))))
    if not np.allclose(candidate, expected, atol=atol, rtol=rtol):
        raise ProductionParityError(f"{mode.value} production parity failed: max_abs_difference={difference}")
    return {
        "mode": mode.value,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_difference": difference,
        "reference_sha256": array_digest(expected),
        "candidate_sha256": array_digest(candidate),
    }


@dataclass(frozen=True)
class PredictionInputs:
    """Exact no-update snapshot handed to a future B3S/Cell-D adapter."""

    activity_trials: np.ndarray = field(repr=False, compare=False)
    support_activity_trials: np.ndarray = field(repr=False, compare=False)
    active_t4: np.ndarray = field(repr=False, compare=False)
    ordinary_point_t4: np.ndarray = field(repr=False, compare=False)
    fixed_ridge_t4: np.ndarray = field(repr=False, compare=False)
    state_digest: str = ""
    query_trials_completed: int = 0

    def __post_init__(self) -> None:
        activity = np.asarray(self.activity_trials)
        support = np.asarray(self.support_activity_trials)
        active = np.asarray(self.active_t4)
        ordinary = np.asarray(self.ordinary_point_t4)
        ridge = np.asarray(self.fixed_ridge_t4)
        _require(activity.ndim == 3 and np.isfinite(activity).all(), "prediction activity stack drift")
        _require(support.ndim == 3 and support.shape[1:] == activity.shape[1:] and np.isfinite(support).all(),
                 "prediction support activity stack drift")
        _require(support.shape[0] <= activity.shape[0] and np.array_equal(activity[: support.shape[0]], support),
                 "prediction support activity must be the immutable stack prefix")
        _require(active.shape == ordinary.shape == ridge.shape and active.ndim == 2 and active.shape[1] == 4,
                 "prediction T4 shape drift")
        _require(np.isfinite(active).all() and np.isfinite(ordinary).all() and np.isfinite(ridge).all(),
                 "prediction T4 must be finite")
        _require(isinstance(self.state_digest, str) and len(self.state_digest) == 64, "prediction state digest drift")
        object.__setattr__(self, "activity_trials", _immutable_array(activity))
        object.__setattr__(self, "support_activity_trials", _immutable_array(support))
        object.__setattr__(self, "active_t4", _immutable_array(active))
        object.__setattr__(self, "ordinary_point_t4", _immutable_array(ordinary))
        object.__setattr__(self, "fixed_ridge_t4", _immutable_array(ridge))

    def payload(self) -> dict[str, Any]:
        return {
            "activity_trials": _frozen_array_payload(self.activity_trials),
            "support_activity_trials": _frozen_array_payload(self.support_activity_trials),
            "active_t4": _frozen_array_payload(self.active_t4),
            "ordinary_point_t4": _frozen_array_payload(self.ordinary_point_t4),
            "fixed_ridge_t4": _frozen_array_payload(self.fixed_ridge_t4),
            "state_digest": self.state_digest,
            "query_trials_completed": int(self.query_trials_completed),
        }


@dataclass(frozen=True)
class DualMemoryState:
    activity: ActivityMemory = field(repr=False, compare=False)
    carrier: CarrierMemory = field(repr=False, compare=False)
    committed_query_trials: int = 0

    def __post_init__(self) -> None:
        _require(self.activity.channel_ids.shape == self.carrier.groups.channel_ids.shape,
                 "activity/carrier channel-count drift")
        _require(np.array_equal(self.activity.channel_ids, self.carrier.groups.channel_ids),
                 "activity/carrier channel order drift")
        _require(
            self.activity.query_count == min(self.committed_query_trials, self.activity.fifo_capacity),
            "activity FIFO/query-transition count drift",
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_state_v1",
            "activity_sha256": self.activity.digest,
            "carrier_sha256": self.carrier.digest,
            "committed_query_trials": int(self.committed_query_trials),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class PendingTrialUpdate:
    """A proposed transition that cannot become visible until ``commit``."""

    base_state_digest: str
    accepted: bool
    reason: UpdateRejectionReason | None
    pseudo_directions: tuple[PseudoDirection, ...]
    scalar_rates: np.ndarray | None = field(repr=False, compare=False)
    carrier_proposal: CarrierProposal | None = field(repr=False, compare=False)
    candidate_state: DualMemoryState | None = field(repr=False, compare=False)
    fallback: PredictionInputs = field(repr=False, compare=False)
    completed_trial_evidence: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    def payload(self) -> dict[str, Any]:
        return {
            "base_state_digest": self.base_state_digest,
            "accepted": bool(self.accepted),
            "reason": None if self.reason is None else self.reason.value,
            "pseudo_directions": [item.payload() for item in self.pseudo_directions],
            "scalar_rates": None if self.scalar_rates is None else _frozen_array_payload(np.asarray(self.scalar_rates)),
            "carrier_proposal": None if self.carrier_proposal is None else self.carrier_proposal.payload(),
            "candidate_state_digest": None if self.candidate_state is None else self.candidate_state.digest,
            "fallback": self.fallback.payload(),
            "completed_trial_evidence": None if self.completed_trial_evidence is None else dict(self.completed_trial_evidence),
        }


@dataclass(frozen=True)
class UpdateOutcome:
    committed: bool
    reason: UpdateRejectionReason | None
    state_digest_before: str
    state_digest_after: str
    fallback: PredictionInputs = field(repr=False, compare=False)

    def payload(self) -> dict[str, Any]:
        return {
            "committed": bool(self.committed),
            "reason": None if self.reason is None else self.reason.value,
            "state_digest_before": self.state_digest_before,
            "state_digest_after": self.state_digest_after,
            "fallback": self.fallback.payload(),
        }


class CausalDualMemory:
    """Mutable owner of immutable snapshots with an explicit commit boundary."""

    def __init__(self, *, activity: ActivityMemory, carrier: CarrierMemory) -> None:
        _require(activity.fifo_capacity == carrier.config.activity_fifo_capacity,
                 "activity FIFO capacity must be the frozen carrier config value")
        _require(len(activity.support_trials) == carrier.config.support_budget_m,
                 "activity support rows must exactly equal the frozen M4/M10/M30 budget")
        _require(len(activity.support_trials) + activity.fifo_capacity == ACTIVITY_STACK_LIMIT,
                 "CDM-D activity stack must remain exactly bounded at 30 trials")
        self._state = DualMemoryState(activity=activity, carrier=carrier, committed_query_trials=activity.query_count)

    @property
    def state(self) -> DualMemoryState:
        return self._state

    def read_prediction_inputs(self) -> PredictionInputs:
        state = self._state
        return PredictionInputs(
            activity_trials=state.activity.stack(),
            support_activity_trials=np.stack(tuple(item.activity for item in state.activity.support_trials), axis=0),
            active_t4=state.carrier.active_t4,
            ordinary_point_t4=state.carrier.ordinary_t4,
            fixed_ridge_t4=state.carrier.fixed_ridge_t4,
            state_digest=state.digest,
            query_trials_completed=state.committed_query_trials,
        )

    def _rejected_pending(
        self,
        reason: UpdateRejectionReason,
        *,
        pseudo: tuple[PseudoDirection, ...] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> PendingTrialUpdate:
        state = self._state
        return PendingTrialUpdate(state.digest, False, reason, pseudo, None, None, None, self.read_prediction_inputs(), evidence)

    def validate_completed_trial_views(
        self,
        *,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        complementary_predictions: Sequence[Any] | None = None,
    ) -> UpdateRejectionReason | None:
        """Validate the three non-interchangeable completed-trial capabilities.

        This deliberately precedes direction integration so that raw arrays,
        swapped neural views, and trial/channel joins cannot enter either the
        B3S FIFO or carrier sufficient statistics.
        """
        state = self._state
        activity_reason = state.activity.validate_complete_trial(b3s_trial_activity)
        if activity_reason is not None:
            return activity_reason
        if not isinstance(carrier_trial_counts, NativeRewardedTrialSpikeCounts):
            return UpdateRejectionReason.TRIAL_CAPABILITY
        if carrier_trial_counts.counts.shape[1] != state.carrier.groups.units:
            return UpdateRejectionReason.CARRIER_COUNTS
        if (
            carrier_trial_counts.session_id != b3s_trial_activity.session_id
            or carrier_trial_counts.trial_id != b3s_trial_activity.trial_id
            or carrier_trial_counts.channel_order_sha256 != b3s_trial_activity.channel_order_sha256
        ):
            return UpdateRejectionReason.TRIAL_BINDING
        if complementary_predictions is None:
            return None
        rows = tuple(complementary_predictions)
        if len(rows) != state.carrier.groups.group_count:
            return UpdateRejectionReason.VELOCITY_SHAPE
        for prediction in rows:
            if not isinstance(prediction, CompletedVelocityPrediction):
                return UpdateRejectionReason.TRIAL_CAPABILITY
            validity = prediction.validity
            if validity.session_id != b3s_trial_activity.session_id or validity.trial_id != b3s_trial_activity.trial_id:
                return UpdateRejectionReason.VELOCITY_VALIDITY
        return None

    @staticmethod
    def _completed_trial_evidence(
        b3s_trial_activity: B3SInterpolatedSpikeCountTrial,
        carrier_trial_counts: NativeRewardedTrialSpikeCounts,
        complementary_predictions: Sequence[CompletedVelocityPrediction],
    ) -> dict[str, Any]:
        return {
            "b3s_interpolated_trial": b3s_trial_activity.payload(),
            "native_rewarded_counts": carrier_trial_counts.payload(),
            "velocity_validity": [item.validity.payload() for item in complementary_predictions],
            "same_session_trial_channel_binding": {
                "session_id": b3s_trial_activity.session_id,
                "trial_id": b3s_trial_activity.trial_id,
                "channel_order_sha256": b3s_trial_activity.channel_order_sha256,
            },
            "carrier_rate_rule": "mean(native_rewarded_counts_over_full_declared_interval)/0.020",
        }

    def observe_completed_trial(
        self,
        *,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        complementary_predictions: Sequence[Any],
    ) -> PendingTrialUpdate:
        """Propose a causal update without changing the visible state.

        The B3S interpolation view, native rate view, and velocity-validity
        evidence are intentionally independent capabilities.  Only a validated
        triple may advance state.
        """
        state = self._state
        view_reason = self.validate_completed_trial_views(
            b3s_trial_activity=b3s_trial_activity,
            carrier_trial_counts=carrier_trial_counts,
            complementary_predictions=complementary_predictions,
        )
        if view_reason is not None:
            return self._rejected_pending(view_reason)
        # Narrowing after the typed boundary keeps all following operations
        # incapable of interpreting an interpolated B3S tensor as counts.
        typed_b3s = b3s_trial_activity
        typed_counts = carrier_trial_counts
        typed_predictions = tuple(complementary_predictions)
        _require(isinstance(typed_b3s, B3SInterpolatedSpikeCountTrial),
                 "validated B3S capability narrowing drift")
        _require(isinstance(typed_counts, NativeRewardedTrialSpikeCounts),
                 "validated native-count capability narrowing drift")
        _require(all(isinstance(item, CompletedVelocityPrediction) for item in typed_predictions),
                 "validated velocity capability narrowing drift")
        evidence = self._completed_trial_evidence(typed_b3s, typed_counts, typed_predictions)
        pseudo = tuple(
            pseudo_direction_from_velocity(item.velocity, item.validity.valid_mask, config=state.carrier.config)
            for item in typed_predictions
        )
        failed = next((item.reason for item in pseudo if not item.accepted), None)
        if failed is not None:
            return self._rejected_pending(failed, pseudo=pseudo, evidence=evidence)
        theta = tuple(float(item.theta_raw_rad) for item in pseudo)
        max_disagreement = max(circular_distance(left, right) for position, left in enumerate(theta) for right in theta[position + 1 :])
        if max_disagreement > state.carrier.config.max_group_direction_disagreement_rad:
            return self._rejected_pending(UpdateRejectionReason.COMPLEMENTARY_DISAGREEMENT, pseudo=pseudo, evidence=evidence)
        try:
            scalar_rates = scalar_rates_from_native_rewarded_counts(typed_counts)
        except CDMDStage0Error:
            return self._rejected_pending(UpdateRejectionReason.CARRIER_COUNTS, pseudo=pseudo, evidence=evidence)
        direction_indices = tuple(int(item.theta_index) for item in pseudo)
        carrier_proposal = state.carrier.propose_pseudo_trial(direction_indices, scalar_rates)
        if not carrier_proposal.accepted:
            return PendingTrialUpdate(state.digest, False, carrier_proposal.reason, pseudo, scalar_rates,
                                      carrier_proposal, None, self.read_prediction_inputs(), evidence)
        _require(carrier_proposal.candidate is not None, "accepted carrier proposal has no candidate")
        next_activity = state.activity.after_accepted_trial(typed_b3s)
        candidate = DualMemoryState(
            activity=next_activity,
            carrier=carrier_proposal.candidate,
            committed_query_trials=state.committed_query_trials + 1,
        )
        return PendingTrialUpdate(state.digest, True, None, pseudo, scalar_rates, carrier_proposal, candidate,
                                  self.read_prediction_inputs(), evidence)

    def commit(self, pending: PendingTrialUpdate) -> UpdateOutcome:
        """Commit exactly one accepted pending state; reject stale observations."""
        before = self._state.digest
        if pending.base_state_digest != before:
            raise CDMDStage0Error(UpdateRejectionReason.STALE_PENDING_UPDATE.value)
        if not pending.accepted:
            return UpdateOutcome(False, pending.reason, before, before, pending.fallback)
        _require(pending.candidate_state is not None, "accepted pending update lacks candidate state")
        self._state = pending.candidate_state
        return UpdateOutcome(True, None, before, self._state.digest, pending.fallback)


@dataclass(frozen=True)
class IndependentActivityPendingTrialUpdate:
    """V3 pending transition with separate activity and carrier facts.

    Frozen V1/V2 ``PendingTrialUpdate.accepted`` means that *both* memories
    may commit.  That is intentionally left untouched for historical receipt
    replay.  The V3 successor instead records whether a valid unlabeled B3S
    activity row is eligible to advance independently from whether the
    pseudo-label-gated carrier proposal is eligible to advance.
    """

    base_state_digest: str
    activity_transition_ready: bool
    activity_fifo_changed: bool
    carrier_transition_accepted: bool
    activity_rejection_reason: UpdateRejectionReason | None
    carrier_rejection_reason: UpdateRejectionReason | None
    pseudo_directions: tuple[PseudoDirection, ...]
    scalar_rates: np.ndarray | None = field(repr=False, compare=False)
    carrier_proposal: CarrierProposal | None = field(repr=False, compare=False)
    candidate_state: DualMemoryState | None = field(repr=False, compare=False)
    fallback: PredictionInputs = field(repr=False, compare=False)
    completed_trial_evidence: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256(self.base_state_digest, label="V3 pending base-state digest")
        _require(type(self.activity_transition_ready) is bool
                 and type(self.activity_fifo_changed) is bool
                 and type(self.carrier_transition_accepted) is bool,
                 "V3 pending transition flags must be exact bools")
        _require(isinstance(self.fallback, PredictionInputs), "V3 pending fallback type drift")
        _require(all(isinstance(item, PseudoDirection) for item in self.pseudo_directions),
                 "V3 pending pseudo-direction type drift")
        _require(self.activity_rejection_reason is None
                 or isinstance(self.activity_rejection_reason, UpdateRejectionReason),
                 "V3 pending activity rejection reason drift")
        _require(self.carrier_rejection_reason is None
                 or isinstance(self.carrier_rejection_reason, UpdateRejectionReason),
                 "V3 pending carrier rejection reason drift")
        if not self.activity_transition_ready:
            _require(self.activity_fifo_changed is False
                     and self.carrier_transition_accepted is False
                     and self.activity_rejection_reason is not None
                     and self.carrier_rejection_reason is None
                     and self.candidate_state is None,
                     "V3 invalid activity may not synthesize a carrier or state transition")
        else:
            _require(self.activity_rejection_reason is None and isinstance(self.candidate_state, DualMemoryState),
                     "V3 valid activity requires exactly one candidate state")
            if self.carrier_transition_accepted:
                _require(self.carrier_rejection_reason is None and self.carrier_proposal is not None
                         and self.carrier_proposal.accepted,
                         "V3 accepted carrier transition proof drift")
            else:
                _require(self.carrier_rejection_reason is not None,
                         "V3 rejected carrier must disclose a typed rejection reason")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_independent_activity_pending_v3",
            "base_state_sha256": self.base_state_digest,
            "activity_transition_ready": self.activity_transition_ready,
            "activity_fifo_changed": self.activity_fifo_changed,
            "carrier_transition_accepted": self.carrier_transition_accepted,
            "activity_rejection_reason_or_null": (
                None if self.activity_rejection_reason is None else self.activity_rejection_reason.value
            ),
            "carrier_rejection_reason_or_null": (
                None if self.carrier_rejection_reason is None else self.carrier_rejection_reason.value
            ),
            "pseudo_directions": [item.payload() for item in self.pseudo_directions],
            "scalar_rates": None if self.scalar_rates is None else _frozen_array_payload(np.asarray(self.scalar_rates)),
            "carrier_proposal": None if self.carrier_proposal is None else self.carrier_proposal.payload(),
            "candidate_state_sha256": None if self.candidate_state is None else self.candidate_state.digest,
            "fallback": self.fallback.payload(),
            "completed_trial_evidence": None if self.completed_trial_evidence is None else dict(self.completed_trial_evidence),
        }


@dataclass(frozen=True)
class IndependentActivityUpdateOutcome:
    """Committed V3 transition facts suitable for durable receipt evidence."""

    activity_transition_committed: bool
    activity_fifo_changed: bool
    carrier_transition_committed: bool
    carrier_rejection_reason: UpdateRejectionReason | None
    activity_rejection_reason: UpdateRejectionReason | None
    state_before_sha256: str
    state_after_sha256: str
    activity_before_sha256: str
    activity_after_sha256: str
    carrier_before_sha256: str
    carrier_after_sha256: str
    fallback: PredictionInputs = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(type(self.activity_transition_committed) is bool
                 and type(self.activity_fifo_changed) is bool
                 and type(self.carrier_transition_committed) is bool,
                 "V3 outcome transition flags must be exact bools")
        for label, value in (
            ("V3 outcome state-before", self.state_before_sha256),
            ("V3 outcome state-after", self.state_after_sha256),
            ("V3 outcome activity-before", self.activity_before_sha256),
            ("V3 outcome activity-after", self.activity_after_sha256),
            ("V3 outcome carrier-before", self.carrier_before_sha256),
            ("V3 outcome carrier-after", self.carrier_after_sha256),
        ):
            _require_sha256(value, label=label)
        _require(isinstance(self.fallback, PredictionInputs), "V3 outcome fallback type drift")
        _require(self.carrier_rejection_reason is None
                 or isinstance(self.carrier_rejection_reason, UpdateRejectionReason),
                 "V3 outcome carrier rejection reason drift")
        _require(self.activity_rejection_reason is None
                 or isinstance(self.activity_rejection_reason, UpdateRejectionReason),
                 "V3 outcome activity rejection reason drift")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "causal_dual_memory_independent_activity_outcome_v3",
            "activity_transition_committed": self.activity_transition_committed,
            "activity_fifo_changed": self.activity_fifo_changed,
            "carrier_transition_committed": self.carrier_transition_committed,
            "carrier_rejection_reason_or_null": (
                None if self.carrier_rejection_reason is None else self.carrier_rejection_reason.value
            ),
            "activity_rejection_reason_or_null": (
                None if self.activity_rejection_reason is None else self.activity_rejection_reason.value
            ),
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "activity_before_sha256": self.activity_before_sha256,
            "activity_after_sha256": self.activity_after_sha256,
            "carrier_before_sha256": self.carrier_before_sha256,
            "carrier_after_sha256": self.carrier_after_sha256,
            "fallback": self.fallback.payload(),
        }


def validate_independent_activity_outcome_payload(value: object) -> dict[str, Any]:
    """Validate that a V3 receipt did not collapse the two memory booleans.

    The validator intentionally derives the digest-equality consequences from
    the independent transition facts.  A forged receipt that merely copies
    one old ``accepted``/``committed`` bit into both fields cannot represent a
    valid carrier rejection with a successful activity append.
    """

    _require(isinstance(value, Mapping), "V3 independent-activity outcome must be a mapping")
    result = dict(value)
    _require(result.get("schema") == "causal_dual_memory_independent_activity_outcome_v3",
             "V3 independent-activity outcome schema drift")
    for key in (
        "activity_transition_committed", "activity_fifo_changed", "carrier_transition_committed",
    ):
        _require(type(result.get(key)) is bool, f"V3 outcome {key} must be an exact bool")
    for key in (
        "state_before_sha256", "state_after_sha256", "activity_before_sha256",
        "activity_after_sha256", "carrier_before_sha256", "carrier_after_sha256",
    ):
        _require_sha256(result.get(key), label=f"V3 outcome {key}")
    valid_reasons = {item.value for item in UpdateRejectionReason}
    activity_reason = result.get("activity_rejection_reason_or_null")
    carrier_reason = result.get("carrier_rejection_reason_or_null")
    _require(activity_reason is None or activity_reason in valid_reasons,
             "V3 outcome activity rejection reason drift")
    _require(carrier_reason is None or carrier_reason in valid_reasons,
             "V3 outcome carrier rejection reason drift")
    activity_committed = result["activity_transition_committed"]
    fifo_changed = result["activity_fifo_changed"]
    carrier_committed = result["carrier_transition_committed"]
    state_before, state_after = result["state_before_sha256"], result["state_after_sha256"]
    activity_before, activity_after = result["activity_before_sha256"], result["activity_after_sha256"]
    carrier_before, carrier_after = result["carrier_before_sha256"], result["carrier_after_sha256"]
    if not activity_committed:
        _require(
            fifo_changed is False and carrier_committed is False
            and activity_reason is not None and carrier_reason is None
            and state_before == state_after and activity_before == activity_after
            and carrier_before == carrier_after,
            "V3 invalid activity outcome must leave both memories exact",
        )
        return result
    _require(activity_reason is None and state_before != state_after,
             "V3 valid activity outcome state transition drift")
    _require(fifo_changed == (activity_before != activity_after),
             "V3 activity FIFO change/digest relation drift")
    if carrier_committed:
        _require(carrier_reason is None and carrier_before != carrier_after,
                 "V3 committed carrier outcome digest/reason drift")
    else:
        _require(carrier_reason in valid_reasons and carrier_before == carrier_after,
                 "V3 rejected carrier must preserve carrier bytes while activity advances")
    return result


class IndependentActivityCausalDualMemory(CausalDualMemory):
    """V3 CDM-D state machine with an independent unlabeled activity FIFO.

    This additive successor intentionally does not override or reinterpret
    ``CausalDualMemory``.  Its state snapshots, carrier estimators, gates,
    and prediction-input API are inherited, while the completion boundary is
    split into an activity transition and a carrier transition.
    """

    def _activity_invalid_pending(
        self,
        reason: UpdateRejectionReason,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> IndependentActivityPendingTrialUpdate:
        state = self._state
        return IndependentActivityPendingTrialUpdate(
            base_state_digest=state.digest,
            activity_transition_ready=False,
            activity_fifo_changed=False,
            carrier_transition_accepted=False,
            activity_rejection_reason=reason,
            carrier_rejection_reason=None,
            pseudo_directions=(),
            scalar_rates=None,
            carrier_proposal=None,
            candidate_state=None,
            fallback=self.read_prediction_inputs(),
            completed_trial_evidence=evidence,
        )

    @staticmethod
    def _activity_only_evidence(
        trial: B3SInterpolatedSpikeCountTrial,
        *,
        carrier_rejection_reason: UpdateRejectionReason | None,
    ) -> dict[str, Any]:
        return {
            "b3s_interpolated_trial": trial.payload(),
            "activity_transition_rule": "valid_unlabeled_completed_b3s_trial_advances_independently_v3",
            "carrier_side_attempted": True,
            "carrier_rejection_reason_or_null": (
                None if carrier_rejection_reason is None else carrier_rejection_reason.value
            ),
        }

    def _activity_candidate(self, trial: B3SInterpolatedSpikeCountTrial) -> tuple[DualMemoryState, bool]:
        state = self._state
        next_activity = state.activity.after_completed_trial(trial)
        candidate = DualMemoryState(
            activity=next_activity,
            carrier=state.carrier,
            committed_query_trials=state.committed_query_trials + 1,
        )
        return candidate, next_activity.digest != state.activity.digest

    def _carrier_rejected_pending(
        self,
        *,
        activity_candidate: DualMemoryState,
        activity_fifo_changed: bool,
        reason: UpdateRejectionReason,
        pseudo: tuple[PseudoDirection, ...] = (),
        scalar_rates: np.ndarray | None = None,
        carrier_proposal: CarrierProposal | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> IndependentActivityPendingTrialUpdate:
        return IndependentActivityPendingTrialUpdate(
            base_state_digest=self._state.digest,
            activity_transition_ready=True,
            activity_fifo_changed=activity_fifo_changed,
            carrier_transition_accepted=False,
            activity_rejection_reason=None,
            carrier_rejection_reason=reason,
            pseudo_directions=pseudo,
            scalar_rates=scalar_rates,
            carrier_proposal=carrier_proposal,
            candidate_state=activity_candidate,
            fallback=self.read_prediction_inputs(),
            completed_trial_evidence=evidence,
        )

    def observe_completed_trial(
        self,
        *,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        complementary_predictions: Sequence[Any],
    ) -> IndependentActivityPendingTrialUpdate:
        """Propose a V3 transition without exposing it until ``commit``.

        B3S capability validation intentionally occurs before *all* carrier
        inputs.  Once that unlabeled activity capability is valid, an invalid
        native-count/velocity join or a later pseudo-label rejection may not
        suppress the activity-state transition.
        """

        state = self._state
        activity_reason = state.activity.validate_complete_trial(b3s_trial_activity)
        if activity_reason is not None:
            return self._activity_invalid_pending(activity_reason)
        typed_b3s = b3s_trial_activity
        _require(isinstance(typed_b3s, B3SInterpolatedSpikeCountTrial),
                 "V3 validated B3S capability narrowing drift")
        activity_candidate, fifo_changed = self._activity_candidate(typed_b3s)
        carrier_view_reason = self.validate_completed_trial_views(
            b3s_trial_activity=typed_b3s,
            carrier_trial_counts=carrier_trial_counts,
            complementary_predictions=complementary_predictions,
        )
        if carrier_view_reason is not None:
            return self._carrier_rejected_pending(
                activity_candidate=activity_candidate,
                activity_fifo_changed=fifo_changed,
                reason=carrier_view_reason,
                evidence=self._activity_only_evidence(typed_b3s, carrier_rejection_reason=carrier_view_reason),
            )
        typed_counts = carrier_trial_counts
        typed_predictions = tuple(complementary_predictions)
        _require(isinstance(typed_counts, NativeRewardedTrialSpikeCounts),
                 "V3 validated native-count capability narrowing drift")
        _require(all(isinstance(item, CompletedVelocityPrediction) for item in typed_predictions),
                 "V3 validated velocity capability narrowing drift")
        evidence = self._completed_trial_evidence(typed_b3s, typed_counts, typed_predictions)
        evidence["activity_transition_rule"] = "valid_unlabeled_completed_b3s_trial_advances_independently_v3"
        pseudo = tuple(
            pseudo_direction_from_velocity(item.velocity, item.validity.valid_mask, config=state.carrier.config)
            for item in typed_predictions
        )
        failed = next((item.reason for item in pseudo if not item.accepted), None)
        if failed is not None:
            _require(isinstance(failed, UpdateRejectionReason), "V3 pseudo rejection reason drift")
            return self._carrier_rejected_pending(
                activity_candidate=activity_candidate, activity_fifo_changed=fifo_changed,
                reason=failed, pseudo=pseudo, evidence=evidence,
            )
        theta = tuple(float(item.theta_raw_rad) for item in pseudo)
        max_disagreement = max(
            circular_distance(left, right)
            for position, left in enumerate(theta) for right in theta[position + 1 :]
        )
        if max_disagreement > state.carrier.config.max_group_direction_disagreement_rad:
            return self._carrier_rejected_pending(
                activity_candidate=activity_candidate, activity_fifo_changed=fifo_changed,
                reason=UpdateRejectionReason.COMPLEMENTARY_DISAGREEMENT, pseudo=pseudo, evidence=evidence,
            )
        try:
            scalar_rates = scalar_rates_from_native_rewarded_counts(typed_counts)
        except CDMDStage0Error:
            return self._carrier_rejected_pending(
                activity_candidate=activity_candidate, activity_fifo_changed=fifo_changed,
                reason=UpdateRejectionReason.CARRIER_COUNTS, pseudo=pseudo, evidence=evidence,
            )
        direction_indices = tuple(int(item.theta_index) for item in pseudo)
        carrier_proposal = state.carrier.propose_pseudo_trial(direction_indices, scalar_rates)
        if not carrier_proposal.accepted:
            _require(isinstance(carrier_proposal.reason, UpdateRejectionReason),
                     "V3 rejected carrier proposal lacks a typed reason")
            return self._carrier_rejected_pending(
                activity_candidate=activity_candidate, activity_fifo_changed=fifo_changed,
                reason=carrier_proposal.reason, pseudo=pseudo, scalar_rates=scalar_rates,
                carrier_proposal=carrier_proposal, evidence=evidence,
            )
        _require(carrier_proposal.candidate is not None, "V3 accepted carrier proposal has no candidate")
        final_candidate = DualMemoryState(
            activity=activity_candidate.activity,
            carrier=carrier_proposal.candidate,
            committed_query_trials=activity_candidate.committed_query_trials,
        )
        return IndependentActivityPendingTrialUpdate(
            base_state_digest=state.digest,
            activity_transition_ready=True,
            activity_fifo_changed=fifo_changed,
            carrier_transition_accepted=True,
            activity_rejection_reason=None,
            carrier_rejection_reason=None,
            pseudo_directions=pseudo,
            scalar_rates=scalar_rates,
            carrier_proposal=carrier_proposal,
            candidate_state=final_candidate,
            fallback=self.read_prediction_inputs(),
            completed_trial_evidence=evidence,
        )

    def commit_independent(
        self, pending: IndependentActivityPendingTrialUpdate,
    ) -> IndependentActivityUpdateOutcome:
        """Commit a V3 pending transition and disclose both memory outcomes."""

        _require(isinstance(pending, IndependentActivityPendingTrialUpdate),
                 "V3 independent commit requires a typed pending transition")
        before_state = self._state
        before_state_sha = before_state.digest
        if pending.base_state_digest != before_state_sha:
            raise CDMDStage0Error(UpdateRejectionReason.STALE_PENDING_UPDATE.value)
        before_activity = before_state.activity.digest
        before_carrier = before_state.carrier.digest
        if not pending.activity_transition_ready:
            outcome = IndependentActivityUpdateOutcome(
                activity_transition_committed=False,
                activity_fifo_changed=False,
                carrier_transition_committed=False,
                carrier_rejection_reason=None,
                activity_rejection_reason=pending.activity_rejection_reason,
                state_before_sha256=before_state_sha,
                state_after_sha256=before_state_sha,
                activity_before_sha256=before_activity,
                activity_after_sha256=before_activity,
                carrier_before_sha256=before_carrier,
                carrier_after_sha256=before_carrier,
                fallback=pending.fallback,
            )
            validate_independent_activity_outcome_payload(outcome.payload())
            return outcome
        candidate = pending.candidate_state
        _require(isinstance(candidate, DualMemoryState), "V3 valid activity pending lacks candidate state")
        _require(candidate.committed_query_trials == before_state.committed_query_trials + 1,
                 "V3 activity transition must increment the completed-query count exactly once")
        _require(
            (candidate.activity.digest != before_activity) == pending.activity_fifo_changed,
            "V3 pending activity FIFO digest relation drift",
        )
        if pending.carrier_transition_accepted:
            _require(candidate.carrier.digest != before_carrier,
                     "V3 accepted carrier transition must change carrier state")
        else:
            _require(candidate.carrier.digest == before_carrier,
                     "V3 rejected carrier transition changed carrier state")
        self._state = candidate
        outcome = IndependentActivityUpdateOutcome(
            activity_transition_committed=True,
            activity_fifo_changed=pending.activity_fifo_changed,
            carrier_transition_committed=pending.carrier_transition_accepted,
            carrier_rejection_reason=pending.carrier_rejection_reason,
            activity_rejection_reason=None,
            state_before_sha256=before_state_sha,
            state_after_sha256=candidate.digest,
            activity_before_sha256=before_activity,
            activity_after_sha256=candidate.activity.digest,
            carrier_before_sha256=before_carrier,
            carrier_after_sha256=candidate.carrier.digest,
            fallback=pending.fallback,
        )
        validate_independent_activity_outcome_payload(outcome.payload())
        return outcome


def _encoder_state_value(value: Any) -> Any:
    """Canonicalize common immutable decoder-state leaves without importing Torch."""
    if isinstance(value, np.ndarray):
        return _frozen_array_payload(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "contiguous"):
        # Torch tensor duck type; do not import Torch in this additive CPU core.
        try:
            detached = value.detach().cpu().contiguous()
            if getattr(detached, "numel")() == 0:
                return {"tensor": "empty", "dtype": str(getattr(detached, "dtype")), "shape": list(getattr(detached, "shape"))}
            return {
                "tensor": True,
                "dtype": str(getattr(detached, "dtype")),
                "shape": list(getattr(detached, "shape")),
                "sha256": sha256_bytes(np.asarray(detached.numpy()).tobytes()),
            }
        except Exception:
            # Explicit sentinel preserves lazy topology without materialising it.
            return {"tensor": "unavailable_or_lazy", "type": type(value).__qualname__}
    if isinstance(value, Mapping):
        return {str(key): _encoder_state_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


def decoder_state_digest(decoder: Any) -> str | None:
    """Digest injected decoder state when it exposes ``state_dict`` or payload."""
    if hasattr(decoder, "state_dict") and callable(decoder.state_dict):
        return sha256_bytes(canonical_json_bytes(_encoder_state_value(decoder.state_dict())))
    if hasattr(decoder, "state_payload") and callable(decoder.state_payload):
        return sha256_bytes(canonical_json_bytes(_encoder_state_value(decoder.state_payload())))
    return None


@dataclass(frozen=True)
class WrapperParameterAudit:
    wrapper_new_trainable_parameters: int
    wrapper_new_trainable_parameter_names: tuple[str, ...]
    injected_decoder_trainable_parameters: int | None

    def payload(self) -> dict[str, Any]:
        return {
            "wrapper_new_trainable_parameters": int(self.wrapper_new_trainable_parameters),
            "wrapper_new_trainable_parameter_names": list(self.wrapper_new_trainable_parameter_names),
            "injected_decoder_trainable_parameters": self.injected_decoder_trainable_parameters,
        }


class CausalDualMemoryCellDWrapper:
    """Parameter-free composition around a sealed Cell-D-like callable.

    The callable contract is intentionally small and explicit:

    ``decoder(neural_activity, *, activity_trials, raw_t4, held_unit_mask)``

    where ``held_unit_mask`` is ``None`` for the normal current prediction and
    a bool ``[N]`` mask for one complementary pseudo-trajectory request.
    No activity/T4 normalization occurs in this wrapper.
    """

    def __init__(self, memory: CausalDualMemory, decoder: Callable[..., Any]) -> None:
        _require(callable(decoder), "CDM-D wrapper requires an injected callable decoder")
        self.memory = memory
        self.decoder = decoder

    def parameter_audit(self) -> WrapperParameterAudit:
        decoder_count: int | None = None
        if hasattr(self.decoder, "named_parameters") and callable(self.decoder.named_parameters):
            total = 0
            for _, parameter in self.decoder.named_parameters():
                if bool(getattr(parameter, "requires_grad", False)):
                    try:
                        total += int(parameter.numel())
                    except Exception:
                        # A lazy parameter should remain lazy; it is not a new
                        # wrapper parameter and therefore need not be materialised.
                        continue
            decoder_count = total
        return WrapperParameterAudit(
            wrapper_new_trainable_parameters=0,
            wrapper_new_trainable_parameter_names=(),
            injected_decoder_trainable_parameters=decoder_count,
        )

    def _decode(self, neural_activity: Any, *, held_unit_mask: np.ndarray | None) -> np.ndarray:
        inputs = self.memory.read_prediction_inputs()
        before = decoder_state_digest(self.decoder)
        prediction = self.decoder(
            neural_activity,
            activity_trials=inputs.activity_trials,
            raw_t4=inputs.active_t4,
            held_unit_mask=held_unit_mask,
        )
        after = decoder_state_digest(self.decoder)
        _require(before == after, "injected decoder state changed during parameter-free CDM-D forward")
        result = np.asarray(prediction)
        _require(result.ndim == 2 and result.shape[1] == 2 and np.issubdtype(result.dtype, np.floating)
                 and np.isfinite(result).all(), "injected decoder must return finite [T,2] velocity")
        result = np.ascontiguousarray(result)
        result.setflags(write=False)
        return result

    def predict_current(self, neural_activity: Any) -> np.ndarray:
        """Current prediction uses only the pre-observation state."""
        return self._decode(neural_activity, held_unit_mask=None)

    def observe_completed_trial(
        self,
        *,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        neural_activity_for_complementary_prediction: Any,
        velocity_validity: Any,
    ) -> PendingTrialUpdate:
        """Request held-group pseudo trajectories under typed trial evidence.

        The B3S and native-count capabilities are checked before any decoder
        forward.  The decoder receives no target behavior and only produces
        velocity values that are then coupled to the separate validity proof.
        """
        pre_decode_reason = self.memory.validate_completed_trial_views(
            b3s_trial_activity=b3s_trial_activity,
            carrier_trial_counts=carrier_trial_counts,
        )
        if pre_decode_reason is not None:
            return self.memory._rejected_pending(pre_decode_reason)
        if not isinstance(velocity_validity, VelocityValidityEvidence):
            return self.memory._rejected_pending(UpdateRejectionReason.TRIAL_CAPABILITY)
        if (
            velocity_validity.session_id != b3s_trial_activity.session_id
            or velocity_validity.trial_id != b3s_trial_activity.trial_id
        ):
            return self.memory._rejected_pending(UpdateRejectionReason.VELOCITY_VALIDITY)
        groups = self.memory.state.carrier.groups
        try:
            predictions = tuple(
                CompletedVelocityPrediction(
                    self._decode(neural_activity_for_complementary_prediction, held_unit_mask=groups.held_mask(group)),
                    velocity_validity,
                )
                for group in range(groups.group_count)
            )
        except CDMDStage0Error:
            return self.memory._rejected_pending(UpdateRejectionReason.VELOCITY_VALIDITY)
        return self.memory.observe_completed_trial(
            b3s_trial_activity=b3s_trial_activity,
            carrier_trial_counts=carrier_trial_counts,
            complementary_predictions=predictions,
        )

    def commit(self, pending: PendingTrialUpdate) -> UpdateOutcome:
        return self.memory.commit(pending)


def stage0_closure(root: Any) -> dict[str, Any]:
    """Delegate explicit static closure computation without a runtime side effect."""
    return plan.closure_payload(root)
