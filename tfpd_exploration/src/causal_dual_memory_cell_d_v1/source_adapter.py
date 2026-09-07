"""Dependency-injected strict-source adapter for the CDM-D safety audit.

The module contains no path discovery or data opening.  A future reviewed
caller must provide a held source record through :class:`HeldSourceTrialRecord`.
That record exposes two independently reconstructed neural views: the
datamodule-compatible B3S cubic/padded activity view and the native binned
count view used for carrier rates.  A caller cannot authorize a native view by
simply attaching a friendly label to an arbitrary array.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import core


STRICT_SOURCE_SESSION_COUNT = 27
WINDOW_BINS = 50
BIN_WIDTH_SECONDS = 0.020
B3S_REBUILDER_SEMANTICS = "datamodule_cubic_interpolate_pad_spike_counts_v1"
FROZEN_B3S_PARITY_ATOL = 1.0e-6


class SourceAdapterError(ValueError):
    """Fail closed for a source scope, view, chronology, or provenance drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAdapterError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
             f"{label} must be exact lowercase SHA256")
    return value


def _immutable(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


def raw_trial_binding_sha256(
    *,
    session_id: str,
    trial_id: str,
    source_record_sha256: str,
    channel_order_sha256: str,
    full_native_counts_sha256: str,
    rewarded_interval_start_bin: int,
    rewarded_interval_stop_bin: int,
    accepted_b3s_activity_sha256: str,
    window_endpoint_bins_sha256: str,
    neural_endpoint_available_sha256: str,
    b3s_rebuilder_semantics: str,
) -> str:
    """Canonical raw trial binding across all held source-array identities."""
    return _sha(_json_bytes({
        "session_id": session_id,
        "trial_id": trial_id,
        "source_record_sha256": source_record_sha256,
        "channel_order_sha256": channel_order_sha256,
        "full_native_counts_sha256": full_native_counts_sha256,
        "rewarded_interval_start_bin": rewarded_interval_start_bin,
        "rewarded_interval_stop_bin": rewarded_interval_stop_bin,
        "accepted_b3s_activity_sha256": accepted_b3s_activity_sha256,
        "window_endpoint_bins_sha256": window_endpoint_bins_sha256,
        "neural_endpoint_available_sha256": neural_endpoint_available_sha256,
        "b3s_rebuilder_semantics": b3s_rebuilder_semantics,
    }))


class Surface(str, Enum):
    SOURCE = "source"
    WITHIN = "within"
    EXTERNAL = "external"
    FORMAL = "formal"
    TARGET = "target"


def require_source_only_surface(surface: Surface | str) -> Surface:
    try:
        parsed = Surface(surface)
    except ValueError as error:
        raise SourceAdapterError("unknown source-audit surface") from error
    _require(parsed is Surface.SOURCE, "CDM-D source adapter refuses non-source surface resolution")
    return parsed


@dataclass(frozen=True)
class StrictSourceRoster:
    """Exact source-only session ordering, injected from future authority."""

    session_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(len(self.session_ids) == STRICT_SOURCE_SESSION_COUNT,
                 "CDM-D source roster must contain exactly strict-27 sessions")
        _require(all(isinstance(item, str) and item.startswith("sub-C") for item in self.session_ids),
                 "CDM-D source roster must contain sub-C sessions only")
        _require(len(set(self.session_ids)) == STRICT_SOURCE_SESSION_COUNT,
                 "CDM-D source roster contains duplicate sessions")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_strict_source_roster_v1",
            "scope": Surface.SOURCE.value,
            "session_ids": list(self.session_ids),
            "count": STRICT_SOURCE_SESSION_COUNT,
        }

    @property
    def digest(self) -> str:
        return _sha(_json_bytes(self.payload()))


@dataclass(frozen=True)
class SourceTrialDescriptor:
    """Descriptor-bound identity for one held source trial."""

    session_id: str
    trial_id: str
    source_record_sha256: str
    channel_order_sha256: str
    full_native_counts_sha256: str
    accepted_b3s_activity_sha256: str
    raw_trial_binding_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and self.session_id.startswith("sub-C"),
                 "source descriptor session must be sub-C")
        _require(isinstance(self.trial_id, str) and bool(self.trial_id), "source descriptor trial ID missing")
        for label in (
            "source_record_sha256", "channel_order_sha256", "full_native_counts_sha256", "accepted_b3s_activity_sha256",
            "raw_trial_binding_sha256",
        ):
            _sha_string(getattr(self, label), label)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_source_trial_descriptor_v1",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "source_record_sha256": self.source_record_sha256,
            "channel_order_sha256": self.channel_order_sha256,
            "full_native_counts_sha256": self.full_native_counts_sha256,
            "accepted_b3s_activity_sha256": self.accepted_b3s_activity_sha256,
            "raw_trial_binding_sha256": self.raw_trial_binding_sha256,
        }


@dataclass(frozen=True)
class HeldSourceTrialRecord:
    """A physically held source record, never an arbitrary caller view.

    ``full_native_binned_counts`` is the source record's native bin grid.  The
    adapter derives the rewarded slice itself.  ``rebuild_b3s_activity`` is an
    injected future datamodule helper and must reproduce the independently
    accepted calibration activity bytes exactly.
    """

    descriptor: SourceTrialDescriptor
    channel_ids: np.ndarray = field(repr=False, compare=False)
    full_native_binned_counts: np.ndarray = field(repr=False, compare=False)
    rewarded_interval_start_bin: int = 0
    rewarded_interval_stop_bin: int = 0
    accepted_calibration_b3s_activity: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty((0, 0)))
    rebuild_b3s_activity: Callable[["HeldSourceTrialRecord"], Any] = field(repr=False, compare=False, default=lambda _: None)
    window_endpoint_bins: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0, dtype=np.int64))
    neural_endpoint_available: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0, dtype=np.bool_))
    b3s_rebuilder_semantics: str = B3S_REBUILDER_SEMANTICS

    def __post_init__(self) -> None:
        channels = np.asarray(self.channel_ids)
        counts = np.asarray(self.full_native_binned_counts)
        accepted = np.asarray(self.accepted_calibration_b3s_activity)
        endpoints = np.asarray(self.window_endpoint_bins)
        available = np.asarray(self.neural_endpoint_available)
        _require(channels.ndim == 1 and channels.size >= 1 and np.issubdtype(channels.dtype, np.integer),
                 "held source record channel IDs must be integer [N]")
        _require(len(set(int(value) for value in channels.tolist())) == channels.size,
                 "held source record channels must be unique")
        _require(core.channel_order_digest(channels) == self.descriptor.channel_order_sha256,
                 "held source record channel-order descriptor drift")
        _require(counts.ndim == 2 and counts.shape[0] >= 1 and counts.shape[1] == channels.size,
                 "held source native count matrix must be [fullT,N]")
        _require(np.issubdtype(counts.dtype, np.number) and np.isfinite(counts).all() and np.all(counts >= 0.0),
                 "held source native count matrix must be finite nonnegative")
        _require(np.array_equal(counts, np.rint(counts)),
                 "held source native count matrix must be integer-valued spike counts")
        _require(core.array_digest(counts) == self.descriptor.full_native_counts_sha256,
                 "held source native count bytes descriptor drift")
        _require(isinstance(self.rewarded_interval_start_bin, int) and isinstance(self.rewarded_interval_stop_bin, int),
                 "held source rewarded interval bounds must be integers")
        _require(0 <= self.rewarded_interval_start_bin < self.rewarded_interval_stop_bin <= counts.shape[0],
                 "held source rewarded interval leaves native record bounds")
        _require(accepted.shape == (100, channels.size) and np.issubdtype(accepted.dtype, np.floating)
                 and np.isfinite(accepted).all(), "accepted B3S activity must be finite [100,N]")
        _require(core.array_digest(accepted) == self.descriptor.accepted_b3s_activity_sha256,
                 "accepted B3S activity bytes descriptor drift")
        _require(callable(self.rebuild_b3s_activity), "held source record needs the datamodule B3S rebuilder")
        _require(self.b3s_rebuilder_semantics == B3S_REBUILDER_SEMANTICS,
                 "held source B3S rebuilder semantics drift")
        _require(endpoints.ndim == available.ndim == 1 and endpoints.size >= 1 and endpoints.size == available.size,
                 "held source endpoint availability shape drift")
        _require(np.issubdtype(endpoints.dtype, np.integer) and available.dtype == np.bool_,
                 "held source endpoints/mask dtype drift")
        _require(np.array_equal(endpoints, np.arange(int(endpoints[0]), int(endpoints[0]) + endpoints.size)),
                 "held source endpoints must be ordered contiguous W=50 endpoints")
        expected_binding = raw_trial_binding_sha256(
            session_id=self.descriptor.session_id,
            trial_id=self.descriptor.trial_id,
            source_record_sha256=self.descriptor.source_record_sha256,
            channel_order_sha256=self.descriptor.channel_order_sha256,
            full_native_counts_sha256=self.descriptor.full_native_counts_sha256,
            rewarded_interval_start_bin=self.rewarded_interval_start_bin,
            rewarded_interval_stop_bin=self.rewarded_interval_stop_bin,
            accepted_b3s_activity_sha256=self.descriptor.accepted_b3s_activity_sha256,
            window_endpoint_bins_sha256=core.array_digest(endpoints),
            neural_endpoint_available_sha256=core.array_digest(available),
            b3s_rebuilder_semantics=self.b3s_rebuilder_semantics,
        )
        _require(self.descriptor.raw_trial_binding_sha256 == expected_binding,
                 "held source record raw session/trial/channel binding drift")
        object.__setattr__(self, "channel_ids", _immutable(channels, dtype=np.int64))
        object.__setattr__(self, "full_native_binned_counts", _immutable(counts))
        object.__setattr__(self, "accepted_calibration_b3s_activity", _immutable(accepted))
        object.__setattr__(self, "window_endpoint_bins", _immutable(endpoints, dtype=np.int64))
        object.__setattr__(self, "neural_endpoint_available", _immutable(available, dtype=np.bool_))

    def native_rewarded_slice(self) -> np.ndarray:
        """Reconstruct the sole native carrier view from held source counts."""
        result = self.full_native_binned_counts[
            self.rewarded_interval_start_bin:self.rewarded_interval_stop_bin
        ]
        result = np.ascontiguousarray(result).copy()
        result.setflags(write=False)
        return result

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_held_source_trial_record_v1",
            "descriptor": self.descriptor.payload(),
            "channel_ids": {"sha256": core.array_digest(self.channel_ids), "shape": list(self.channel_ids.shape)},
            "full_native_binned_counts": {"sha256": core.array_digest(self.full_native_binned_counts), "shape": list(self.full_native_binned_counts.shape)},
            "rewarded_interval_start_bin": self.rewarded_interval_start_bin,
            "rewarded_interval_stop_bin": self.rewarded_interval_stop_bin,
            "accepted_calibration_b3s_activity": {"sha256": core.array_digest(self.accepted_calibration_b3s_activity), "shape": list(self.accepted_calibration_b3s_activity.shape)},
            "window_endpoint_bins": {"sha256": core.array_digest(self.window_endpoint_bins), "shape": list(self.window_endpoint_bins.shape)},
            "neural_endpoint_available": {"sha256": core.array_digest(self.neural_endpoint_available), "shape": list(self.neural_endpoint_available.shape)},
            "b3s_rebuilder_semantics": self.b3s_rebuilder_semantics,
        }


@dataclass(frozen=True)
class SourceTrialViews:
    """The only Stage-0 input triple emitted by the source adapter."""

    b3s_activity: core.B3SInterpolatedSpikeCountTrial = field(repr=False, compare=False)
    carrier_counts: core.NativeRewardedTrialSpikeCounts = field(repr=False, compare=False)
    velocity_validity: core.VelocityValidityEvidence = field(repr=False, compare=False)
    held_record_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.b3s_activity, core.B3SInterpolatedSpikeCountTrial), "source views B3S capability drift")
        _require(isinstance(self.carrier_counts, core.NativeRewardedTrialSpikeCounts), "source views native-count capability drift")
        _require(isinstance(self.velocity_validity, core.VelocityValidityEvidence), "source views validity capability drift")
        _sha_string(self.held_record_sha256, "source view held record SHA")
        _require(
            self.b3s_activity.session_id == self.carrier_counts.session_id == self.velocity_validity.session_id
            and self.b3s_activity.trial_id == self.carrier_counts.trial_id == self.velocity_validity.trial_id
            and self.b3s_activity.channel_order_sha256 == self.carrier_counts.channel_order_sha256,
            "source views must bind one session/trial/channel identity",
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_source_trial_views_v1",
            "held_record_sha256": self.held_record_sha256,
            "b3s_activity": self.b3s_activity.payload(),
            "carrier_counts": self.carrier_counts.payload(),
            "velocity_validity": self.velocity_validity.payload(),
        }


@dataclass(frozen=True)
class AuditOnlyTrueDirection:
    """True source label that is intentionally absent from ``SourceTrialViews``."""

    session_id: str
    trial_id: str
    direction_radians: float

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and self.session_id.startswith("sub-C"), "audit label session drift")
        _require(isinstance(self.trial_id, str) and bool(self.trial_id), "audit label trial drift")
        _require(np.isfinite(float(self.direction_radians)), "audit label direction must be finite")


def materialize_source_trial_views(
    record: HeldSourceTrialRecord,
    *,
    roster: StrictSourceRoster,
    surface: Surface | str = Surface.SOURCE,
) -> SourceTrialViews:
    """Rebuild all three physical views from one held source record.

    No caller-supplied native/b3s/validity arrays are accepted.  This is the
    source-boundary guarantee that prevents an interpolated or rewrapped array
    from being relabelled as native counts.
    """
    require_source_only_surface(surface)
    _require(isinstance(record, HeldSourceTrialRecord), "source adapter requires a held source trial record")
    _require(isinstance(roster, StrictSourceRoster) and record.descriptor.session_id in roster.session_ids,
             "source record is absent from the exact strict-27 source roster")
    native = record.native_rewarded_slice()
    _require(np.array_equal(native, np.rint(native)) and np.all(native >= 0.0),
             "reconstructed native rewarded slice must remain integer-valued")
    rebuilt = np.asarray(record.rebuild_b3s_activity(record))
    _require(rebuilt.shape == record.accepted_calibration_b3s_activity.shape and np.issubdtype(rebuilt.dtype, np.floating)
             and np.isfinite(rebuilt).all(), "datamodule B3S rebuild shape/finite drift")
    _require(core.array_digest(rebuilt) == record.descriptor.accepted_b3s_activity_sha256,
             "datamodule B3S rebuild differs from accepted calibration view")
    endpoints = record.window_endpoint_bins
    lower_bounds = endpoints - (WINDOW_BINS - 1)
    valid = np.asarray(
        record.neural_endpoint_available
        & (lower_bounds >= record.rewarded_interval_start_bin)
        & (endpoints < record.rewarded_interval_stop_bin),
        dtype=np.bool_,
    )
    descriptor = record.descriptor
    b3s = core.B3SInterpolatedSpikeCountTrial(
        activity=rebuilt,
        session_id=descriptor.session_id,
        trial_id=descriptor.trial_id,
        channel_order_sha256=descriptor.channel_order_sha256,
    )
    counts = core.NativeRewardedTrialSpikeCounts(
        counts=native,
        session_id=descriptor.session_id,
        trial_id=descriptor.trial_id,
        channel_order_sha256=descriptor.channel_order_sha256,
        rewarded_interval_start_bin=record.rewarded_interval_start_bin,
        rewarded_interval_stop_bin=record.rewarded_interval_stop_bin,
        bin_width_seconds=BIN_WIDTH_SECONDS,
    )
    validity = core.VelocityValidityEvidence(
        valid_mask=valid,
        session_id=descriptor.session_id,
        trial_id=descriptor.trial_id,
        prediction_interval_start_bin=int(endpoints[0]),
        prediction_interval_stop_bin=int(endpoints[-1]) + 1,
    )
    held_record_sha = _sha(_json_bytes(record.payload()))
    return SourceTrialViews(b3s, counts, validity, held_record_sha)


@dataclass(frozen=True)
class SealedBudgetAuthority:
    """Support/pseudo identity is finalized before any pseudo processing."""

    budget: int
    first30_trial_ids: tuple[str, ...]
    support_indices: tuple[int, ...]
    pseudo_indices: tuple[int, ...]
    support_sealed: bool = True

    def __post_init__(self) -> None:
        _require(self.budget in (4, 10, 30), "CDM-D budget must be M4/M10/M30")
        _require(len(self.first30_trial_ids) == 30 and len(set(self.first30_trial_ids)) == 30,
                 "CDM-D budget authority requires ordered unique first-30 trial IDs")
        _require(len(self.support_indices) == self.budget and len(set(self.support_indices)) == self.budget,
                 "support index count/uniqueness drift")
        _require(all(0 <= index < 30 for index in self.support_indices), "support index leaves first-30 pool")
        expected_pseudo = tuple(index for index in range(30) if index not in set(self.support_indices))
        _require(self.pseudo_indices == expected_pseudo, "unlabelled pseudo trials must be chronological complement")
        _require(self.support_sealed is True, "support labels must be sealed before pseudo processing")
        if self.budget in (10, 30):
            _require(self.support_indices == tuple(range(self.budget)), "M10/M30 support must be chronological")

    @property
    def support_trial_ids(self) -> tuple[str, ...]:
        return tuple(self.first30_trial_ids[index] for index in self.support_indices)

    @property
    def pseudo_trial_ids(self) -> tuple[str, ...]:
        return tuple(self.first30_trial_ids[index] for index in self.pseudo_indices)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_sealed_budget_authority_v1",
            "budget": self.budget,
            "first30_trial_ids": list(self.first30_trial_ids),
            "support_indices": list(self.support_indices),
            "pseudo_indices": list(self.pseudo_indices),
            "support_trial_ids": list(self.support_trial_ids),
            "pseudo_trial_ids": list(self.pseudo_trial_ids),
            "support_sealed_before_pseudo": True,
        }

    @property
    def digest(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_budget_authorities(
    first30_trial_ids: Sequence[str],
    *,
    m4_d_optimal_indices: Sequence[int],
) -> dict[int, SealedBudgetAuthority]:
    """Freeze M4 D-opt and M10/M30 chronological protocol identities."""
    first30 = tuple(first30_trial_ids)
    _require(len(first30) == 30 and all(isinstance(item, str) and item for item in first30),
             "budget protocol requires 30 ordered nonempty trial IDs")
    m4 = tuple(int(index) for index in m4_d_optimal_indices)
    _require(len(m4) == 4 and len(set(m4)) == 4 and all(0 <= index < 30 for index in m4),
             "M4 D-opt support must be four unique first-30 indices")
    return {
        4: SealedBudgetAuthority(4, first30, m4, tuple(index for index in range(30) if index not in set(m4))),
        10: SealedBudgetAuthority(10, first30, tuple(range(10)), tuple(range(10, 30))),
        30: SealedBudgetAuthority(30, first30, tuple(range(30)), ()),
    }


@dataclass(frozen=True)
class SourceAuditBudgetAuthority:
    """Freeze one budget's *audit* rows separately from deployment memory.

    ``deployment`` is the sealed Cell-D support/pseudo authority for the
    first rewarded-30 history.  The source constructibility audit is not
    allowed to replay M30's labelled support rows as pseudo inputs: its M30
    screen instead receives the next 30 rewarded source trials.  This type
    makes that distinction durable rather than relying on a caller comment.
    """

    deployment: SealedBudgetAuthority
    audit_trial_ids: tuple[str, ...]
    audit_source_chronology_positions: tuple[int, ...]
    audit_population: str
    audit_enters_deployment_memory: bool

    def __post_init__(self) -> None:
        _require(isinstance(self.deployment, SealedBudgetAuthority),
                 "source audit needs a sealed deployment budget authority")
        trial_ids = tuple(self.audit_trial_ids)
        positions = tuple(int(value) for value in self.audit_source_chronology_positions)
        _require(trial_ids and all(isinstance(item, str) and item for item in trial_ids)
                 and len(set(trial_ids)) == len(trial_ids),
                 "source audit rows must be ordered unique nonempty trial IDs")
        _require(len(positions) == len(trial_ids) and len(set(positions)) == len(positions),
                 "source audit chronology positions must align one-to-one with audit rows")
        budget = self.deployment.budget
        first30 = self.deployment.first30_trial_ids
        if budget == 4:
            _require(self.audit_population == "first30_dopt_complement_26",
                     "M4 audit identity must be the sealed D-opt complement")
            _require(trial_ids == self.deployment.pseudo_trial_ids and len(trial_ids) == 26,
                     "M4 audit must contain exactly the first-30 D-opt complement")
            _require(positions == self.deployment.pseudo_indices,
                     "M4 audit chronology positions must equal the sealed D-opt complement")
            _require(self.audit_enters_deployment_memory is True,
                     "M4 first-30 audit history must remain in deployment activity memory")
        elif budget == 10:
            _require(self.audit_population == "first30_chronological_10_29",
                     "M10 audit identity must be chronological first30 indices 10..29")
            _require(trial_ids == first30[10:30] and len(trial_ids) == 20,
                     "M10 audit must contain exactly first30 indices 10..29")
            _require(positions == tuple(range(10, 30)),
                     "M10 audit chronology positions must be exactly 10..29")
            _require(self.audit_enters_deployment_memory is True,
                     "M10 first-30 audit history must remain in deployment activity memory")
        else:
            _require(self.audit_population == "post_first30_next30_rewarded_source_trials",
                     "M30 audit identity must be the next rewarded-30 source pool")
            _require(len(trial_ids) == 30 and not (set(trial_ids) & set(first30)),
                     "M30 audit rows must be a distinct post-support rewarded-30 pool")
            _require(positions == tuple(range(30, 60)),
                     "M30 audit chronology positions must be exactly the next rewarded 30 after first30")
            _require(self.audit_enters_deployment_memory is False,
                     "M30 post-support audit rows may never enter deployment memory/state")
        object.__setattr__(self, "audit_trial_ids", trial_ids)
        object.__setattr__(self, "audit_source_chronology_positions", positions)

    @property
    def budget(self) -> int:
        return self.deployment.budget

    @property
    def deployment_support_trial_ids(self) -> tuple[str, ...]:
        return self.deployment.support_trial_ids

    @property
    def deployment_memory_trial_ids(self) -> tuple[str, ...]:
        """The sole first-30 B3S/deployment history, never post-support M30."""
        return self.deployment.first30_trial_ids

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_source_audit_budget_authority_v2",
            "budget": self.budget,
            "deployment_authority_sha256": self.deployment.digest,
            "deployment_support_trial_ids": list(self.deployment_support_trial_ids),
            "deployment_memory_trial_ids": list(self.deployment_memory_trial_ids),
            "audit_trial_ids": list(self.audit_trial_ids),
            "audit_source_chronology_positions": list(self.audit_source_chronology_positions),
            "audit_population": self.audit_population,
            "audit_enters_deployment_memory": self.audit_enters_deployment_memory,
            "m30_current_pseudo_trials_disjoint_from_labelled_support": (
                self.budget != 30
                or not (set(self.audit_trial_ids) & set(self.deployment_support_trial_ids))
            ),
        }

    @property
    def digest(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_source_audit_authorities(
    ordered_first60_rewarded_source_trial_ids: Sequence[str],
    *,
    m4_d_optimal_indices: Sequence[int],
) -> dict[int, SourceAuditBudgetAuthority]:
    """Bind fixed M4/M10/M30 audit pools without changing deployment support.

    The caller must provide the first 60 *chronological rewarded* source-trial
    IDs from the held source adapter.  This API derives rather than accepts an
    independently caller-labelled ``next30`` pool, making M30's position
    30--59 identity explicit and proving it cannot replay first-30 support.
    """
    ordered_first60 = tuple(ordered_first60_rewarded_source_trial_ids)
    _require(len(ordered_first60) == 60
             and all(isinstance(item, str) and item for item in ordered_first60)
             and len(set(ordered_first60)) == 60,
             "source audit needs exactly ordered unique first-60 rewarded source trial IDs")
    first30 = ordered_first60[:30]
    next30 = ordered_first60[30:60]
    deployment = build_budget_authorities(
        first30,
        m4_d_optimal_indices=m4_d_optimal_indices,
    )
    _require(not (set(next30) & set(deployment[30].first30_trial_ids)),
             "M30 source audit next-30 rows may not overlap sealed first-30 support")
    return {
        4: SourceAuditBudgetAuthority(
            deployment[4], deployment[4].pseudo_trial_ids, deployment[4].pseudo_indices,
            "first30_dopt_complement_26", True,
        ),
        10: SourceAuditBudgetAuthority(
            deployment[10], deployment[10].pseudo_trial_ids, tuple(range(10, 30)),
            "first30_chronological_10_29", True,
        ),
        30: SourceAuditBudgetAuthority(
            deployment[30], next30, tuple(range(30, 60)),
            "post_first30_next30_rewarded_source_trials", False,
        ),
    }


def require_m30_post_support_audit_only(authority: SourceAuditBudgetAuthority) -> tuple[str, ...]:
    """Return M30 audit rows only after proving they cannot alter M30 state."""
    _require(isinstance(authority, SourceAuditBudgetAuthority) and authority.budget == 30,
             "M30 post-support audit requires the M30 typed authority")
    _require(authority.audit_enters_deployment_memory is False,
             "M30 audit rows must never enter deployment activity memory/state")
    _require(not (set(authority.audit_trial_ids) & set(authority.deployment_support_trial_ids)),
             "M30 audit rows may not replay labelled support trials as pseudo inputs")
    return authority.audit_trial_ids


def require_sealed_before_pseudo(authority: SealedBudgetAuthority) -> tuple[str, ...]:
    _require(isinstance(authority, SealedBudgetAuthority) and authority.support_sealed,
             "pseudo processing requires sealed support authority")
    return authority.pseudo_trial_ids


def first30_b3s_parity(
    authority: SealedBudgetAuthority,
    ordered_views: Sequence[SourceTrialViews],
) -> dict[str, object]:
    """Prove support-plus-pseudo B3S stack has the M30 activity multiset."""
    _require(isinstance(authority, SealedBudgetAuthority), "B3S parity requires a sealed budget authority")
    _require(len(ordered_views) == 30, "B3S parity requires exactly first-30 source views")
    by_trial = {item.b3s_activity.trial_id: item for item in ordered_views}
    _require(len(by_trial) == 30 and tuple(item.b3s_activity.trial_id for item in ordered_views) == authority.first30_trial_ids,
             "B3S parity ordered first-30 identity drift")
    _require(len({item.b3s_activity.session_id for item in ordered_views}) == 1
             and len({item.b3s_activity.channel_order_sha256 for item in ordered_views}) == 1,
             "B3S parity may not mix sessions or channel orders")
    support_then_pseudo = tuple(by_trial[trial_id] for trial_id in authority.support_trial_ids + authority.pseudo_trial_ids)
    chronological = tuple(ordered_views)
    left_stack = np.stack(tuple(item.b3s_activity.activity for item in support_then_pseudo), axis=0)
    right_stack = np.stack(tuple(item.b3s_activity.activity for item in chronological), axis=0)
    multiset_left = tuple(sorted(item.b3s_activity.digest for item in support_then_pseudo))
    multiset_right = tuple(sorted(item.b3s_activity.digest for item in chronological))
    exact_mean = np.array_equal(left_stack.mean(axis=0), right_stack.mean(axis=0))
    tolerance_mean = bool(np.allclose(left_stack.mean(axis=0), right_stack.mean(axis=0), rtol=0.0, atol=FROZEN_B3S_PARITY_ATOL))
    _require(multiset_left == multiset_right and (exact_mean or tolerance_mean),
             "support/pseudo B3S stack does not reproduce first-30 M30 activity multiset")
    return {
        "schema": "causal_dual_memory_first30_b3s_parity_v1",
        "budget": authority.budget,
        "authority_sha256": authority.digest,
        "chronological_trial_ids": list(authority.first30_trial_ids),
        "support_then_pseudo_trial_ids": list(authority.support_trial_ids + authority.pseudo_trial_ids),
        "multiset_sha256": _sha(_json_bytes(list(multiset_right))),
        "exact_mean_parity": exact_mean,
        "frozen_tolerance": FROZEN_B3S_PARITY_ATOL,
        "tolerance_mean_parity": tolerance_mean,
    }
