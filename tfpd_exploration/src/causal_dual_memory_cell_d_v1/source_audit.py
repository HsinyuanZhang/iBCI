"""Source-only CDM-D constructibility and B8 safety-audit logic.

This is deliberately an analysis of already-produced pseudo directions.  It
does not construct Stage-0 updates and therefore cannot leak audit-only true
source directions into carrier fitting or the model forward path.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from . import core
from . import source_adapter


B8_MEDIAN_COSINE_MIN = 0.50
B8_CORRECT_MINUS_SHUFFLE_MIN = 0.01
B8_FRACTION_COSINE_040_MIN = 0.50
FAIL_FAST_BUDGET_ORDER = (30, 10, 4)
FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_SEED = 42
FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_NAMESPACE = "pseudo-label-carrier-gate-v1"
FROZEN_PSEUDO_LABEL_GATE_SEMANTICS = (
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py:"
    "evaluate_pseudo_label_gate"
)


class SourceAuditError(ValueError):
    """Fail closed for malformed source-only audit evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _wrap(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PseudoAuditRow:
    """Post-update audit row; true direction is absent from update constructors."""

    budget: int
    session_id: str
    trial_id: str
    accepted: bool
    rejection_reason: core.UpdateRejectionReason | None
    pseudo_direction_radians: float | None
    true_direction: source_adapter.AuditOnlyTrueDirection
    complementary_disagreement_radians: float | None
    canonical_distance_radians: float | None
    displacement: float | None
    mean_speed: float | None
    design_rank: int | None
    design_condition: float | None
    departure: float | None
    prefix_digest: str

    def __post_init__(self) -> None:
        _require(self.budget in (4, 10, 30), "source audit budget drift")
        _require(isinstance(self.true_direction, source_adapter.AuditOnlyTrueDirection),
                 "source audit requires separately typed true direction")
        _require(self.session_id == self.true_direction.session_id and self.trial_id == self.true_direction.trial_id,
                 "source audit true direction trial binding drift")
        _require(isinstance(self.prefix_digest, str) and len(self.prefix_digest) == 64,
                 "source audit prefix digest drift")
        if self.accepted:
            _require(self.rejection_reason is None and self.pseudo_direction_radians is not None,
                     "accepted pseudo audit row must have a pseudo direction")
            _require(np.isfinite(float(self.pseudo_direction_radians)), "accepted pseudo direction must be finite")
        else:
            _require(self.rejection_reason is not None and self.pseudo_direction_radians is None,
                     "rejected pseudo audit row must expose exactly one typed reason")

    def cosine(self) -> float | None:
        if not self.accepted:
            return None
        return float(math.cos(_wrap(float(self.pseudo_direction_radians) - self.true_direction.direction_radians)))

    def circular_error(self) -> float | None:
        if not self.accepted:
            return None
        return abs(_wrap(float(self.pseudo_direction_radians) - self.true_direction.direction_radians))


def deterministic_shuffle_trial_identities(rows: Sequence[PseudoAuditRow]) -> tuple[tuple[str, str], ...]:
    """Stable no-global-RNG permutation used only for post hoc audit pairing."""
    identities = [(row.session_id, row.trial_id) for row in rows]
    seed = int.from_bytes(hashlib.sha256(_json_bytes(identities)).digest()[:8], "big")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if len(order) > 1 and order == list(range(len(rows))):
        # A shuffled control that happens to be the identity is not a control.
        # This deterministic rotation neither uses global RNG nor chooses a
        # more favorable pairing after inspecting scores.
        order = order[1:] + order[:1]
    return tuple((rows[index].session_id, rows[index].trial_id) for index in order)


def descriptive_trial_direction_aggregate(rows: Sequence[PseudoAuditRow]) -> dict[str, object]:
    """Describe raw trial-direction agreement without licensing B8.

    This remains useful for diagnosis, but its trial-level cosines are a
    different estimand from frozen B8.  In particular it deliberately does
    not contain B8 thresholds or a ``pass`` field, so callers cannot mistake
    a mean pairing statistic for carrier-refit constructibility evidence.
    """
    _require(bool(rows), "B8 aggregate requires source audit rows")
    accepted = tuple(row for row in rows if row.accepted)
    budgets = {row.budget for row in rows}
    _require(len(budgets) == 1, "trial-direction diagnostics must cover one budget")
    correct = [float(row.cosine()) for row in accepted]
    shuffled_identities = deterministic_shuffle_trial_identities(accepted)
    labels = {(row.session_id, row.trial_id): row.true_direction.direction_radians for row in accepted}
    shuffled = [
        float(math.cos(_wrap(float(row.pseudo_direction_radians) - labels[other_identity])))
        for row, other_identity in zip(accepted, shuffled_identities, strict=True)
    ]
    median = None if not correct else float(np.median(np.asarray(correct, dtype=np.float64)))
    correct_minus_shuffle = None if not correct else float(np.mean(correct) - np.mean(shuffled))
    fraction = 0.0 if not correct else float(np.mean(np.asarray(correct, dtype=np.float64) >= 0.40))
    rejections: dict[str, int] = {}
    for row in rows:
        if row.rejection_reason is not None:
            rejections[row.rejection_reason.value] = rejections.get(row.rejection_reason.value, 0) + 1
    return {
        "schema": "causal_dual_memory_trial_direction_descriptive_v2",
        "non_governing": True,
        "budget": next(iter(budgets)),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "fallback": len(rows) - len(accepted),
        "rejection_counts": dict(sorted(rejections.items())),
        "trial_cosine_median": median,
        "trial_cosine_mean_correct_minus_deterministic_shuffle": correct_minus_shuffle,
        "trial_cosine_fraction_ge_040": fraction,
        "shuffled_trial_identities": [
            {"session_id": session_id, "trial_id": trial_id}
            for session_id, trial_id in shuffled_identities
        ],
    }


def descriptive_group_trial_direction_aggregate(
    rows: Sequence[GroupedPseudoAuditRow],
) -> dict[str, object]:
    """Record raw per-group direction agreement without instantiating B8.

    This is deliberately separate from :func:`carrier_b8_aggregate`: raw
    group-direction cosines can explain failures, but never supply its carrier
    refit thresholds or pass/fail decision.
    """
    audit_rows = tuple(rows)
    _require(audit_rows and all(isinstance(row, GroupedPseudoAuditRow) for row in audit_rows),
             "group trial-direction diagnostics require typed grouped audit rows")
    budgets = {row.budget for row in audit_rows}
    _require(len(budgets) == 1, "group trial-direction diagnostics must cover one budget")
    groups: list[dict[str, object]] = []
    for group in range(core.GROUP_COUNT):
        cosines: list[float] = []
        reasons: dict[str, int] = {}
        for row in audit_rows:
            index = row.pseudo_direction_indices[group]
            reason = row.rejection_reasons[group]
            if index is None:
                _require(reason is not None, "group rejection diagnostic lost its typed reason")
                reasons[reason.value] = reasons.get(reason.value, 0) + 1
                continue
            theta = core.CANONICAL_DIRECTIONS_RAD[int(index)]
            cosines.append(float(math.cos(_wrap(theta - row.true_direction.direction_radians))))
        groups.append({
            "group": group,
            "accepted": len(cosines),
            "rejected": sum(reasons.values()),
            "trial_cosine_median": None if not cosines else float(np.median(np.asarray(cosines, dtype=np.float64))),
            "rejection_counts": dict(sorted(reasons.items())),
        })
    return {
        "schema": "causal_dual_memory_group_trial_direction_descriptive_v1",
        "non_governing": True,
        "budget": next(iter(budgets)),
        "groups": groups,
    }


def _immutable_int64(
    value: Any,
    *,
    label: str,
    ndim: int,
    allow_rejection_sentinel: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    _require(np.issubdtype(raw.dtype, np.integer), f"{label} must have an integer dtype")
    array = np.ascontiguousarray(np.asarray(raw, dtype=np.int64)).copy()
    _require(array.ndim == ndim and array.size >= 2, f"{label} has invalid integer shape")
    low = -1 if allow_rejection_sentinel else 0
    _require(np.all((array >= low) & (array < len(core.CANONICAL_DIRECTIONS_RAD))),
             f"{label} leaves the frozen 8-direction grid")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GroupedPseudoAuditRow:
    """Final per-group pseudo labels for one completed source audit trial.

    A group is either finalized at one canonical direction index or carries a
    typed rejection reason.  Neither state may be omitted or silently removed
    from the fixed B8 audit pool.
    """

    budget: int
    session_id: str
    trial_id: str
    true_direction: source_adapter.AuditOnlyTrueDirection
    pseudo_direction_indices: tuple[int | None, ...]
    rejection_reasons: tuple[core.UpdateRejectionReason | None, ...]
    prefix_digest: str

    def __post_init__(self) -> None:
        _require(self.budget in (4, 10, 30), "grouped carrier B8 budget drift")
        _require(isinstance(self.true_direction, source_adapter.AuditOnlyTrueDirection),
                 "grouped carrier B8 true direction must remain audit-only")
        _require(self.session_id == self.true_direction.session_id and self.trial_id == self.true_direction.trial_id,
                 "grouped carrier B8 true direction trial binding drift")
        _require(isinstance(self.prefix_digest, str) and len(self.prefix_digest) == 64,
                 "grouped carrier B8 prefix digest drift")
        pseudo = tuple(self.pseudo_direction_indices)
        reasons = tuple(self.rejection_reasons)
        _require(len(pseudo) == len(reasons) == core.GROUP_COUNT,
                 "grouped carrier B8 needs exactly one finalized result for each complementary group")
        for group, (index, reason) in enumerate(zip(pseudo, reasons, strict=True)):
            if index is None:
                _require(isinstance(reason, core.UpdateRejectionReason),
                         f"group {group} rejection must carry a typed reason")
            else:
                _require(isinstance(index, int) and 0 <= index < len(core.CANONICAL_DIRECTIONS_RAD),
                         f"group {group} pseudo direction must be a finalized canonical index")
                _require(reason is None, f"group {group} accepted pseudo direction may not carry a rejection")
        object.__setattr__(self, "pseudo_direction_indices", pseudo)
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def accepted(self) -> bool:
        return all(index is not None for index in self.pseudo_direction_indices)


@dataclass(frozen=True)
class FinalizedGroupedCarrierDirectionIndices:
    """Fixed-pool true labels and K=4 finalized/rejected pseudo-label columns."""

    budget: int
    session_id: str
    trial_ids: tuple[str, ...]
    true_direction_indices: np.ndarray = field(repr=False, compare=False)
    pseudo_direction_indices: np.ndarray = field(repr=False, compare=False)
    rejection_reasons: tuple[tuple[core.UpdateRejectionReason | None, ...], ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(self.budget in (4, 10, 30), "grouped carrier B8 budget drift")
        _require(isinstance(self.session_id, str) and self.session_id.startswith("sub-C"),
                 "grouped carrier B8 must remain source-session scoped")
        ids = tuple(self.trial_ids)
        _require(len(ids) >= 2 and len(set(ids)) == len(ids)
                 and all(isinstance(item, str) and item for item in ids),
                 "grouped carrier B8 needs ordered unique audit trial IDs")
        true = _immutable_int64(self.true_direction_indices, label="grouped B8 true direction indices", ndim=1)
        pseudo = _immutable_int64(
            self.pseudo_direction_indices,
            label="grouped B8 pseudo direction indices",
            ndim=2,
            allow_rejection_sentinel=True,
        )
        _require(true.size == len(ids) and pseudo.shape == (len(ids), core.GROUP_COUNT),
                 "grouped B8 trial IDs/direction matrix shape drift")
        reasons = tuple(tuple(row) for row in self.rejection_reasons)
        _require(len(reasons) == len(ids) and all(len(row) == core.GROUP_COUNT for row in reasons),
                 "grouped B8 rejection matrix shape drift")
        for trial, row in enumerate(reasons):
            for group, reason in enumerate(row):
                if pseudo[trial, group] == -1:
                    _require(isinstance(reason, core.UpdateRejectionReason),
                             "rejected grouped B8 cell needs a typed reason")
                else:
                    _require(reason is None, "accepted grouped B8 cell may not carry a rejection")
        object.__setattr__(self, "trial_ids", ids)
        object.__setattr__(self, "true_direction_indices", true)
        object.__setattr__(self, "pseudo_direction_indices", pseudo)
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def has_rejections(self) -> bool:
        return bool(np.any(self.pseudo_direction_indices == -1))

    @property
    def rejected_group_count(self) -> int:
        return int(np.sum(self.pseudo_direction_indices == -1))

    @property
    def rejected_trial_count(self) -> int:
        return int(np.sum(np.any(self.pseudo_direction_indices == -1, axis=1)))

    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rejection_reasons:
            for reason in row:
                if reason is not None:
                    counts[reason.value] = counts.get(reason.value, 0) + 1
        return dict(sorted(counts.items()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_finalized_grouped_carrier_direction_indices_v1",
            "budget": self.budget,
            "session_id": self.session_id,
            "trial_ids": list(self.trial_ids),
            "true_direction_indices_sha256": core.array_digest(self.true_direction_indices),
            "pseudo_direction_indices_sha256": core.array_digest(self.pseudo_direction_indices),
            "group_count": core.GROUP_COUNT,
            "rejected_group_count": self.rejected_group_count,
            "rejected_trial_count": self.rejected_trial_count,
            "rejection_counts": self.rejection_counts(),
            "pseudo_directions_already_finalized": True,
        }

    @property
    def digest(self) -> str:
        return _sha(_json_bytes(self.payload()))


@dataclass(frozen=True)
class CarrierB8Inputs:
    """Typed native rates and the exact CDM-D complementary-group consumer map."""

    audit_authority: source_adapter.SourceAuditBudgetAuthority
    groups: core.ComplementaryGroups
    directions: FinalizedGroupedCarrierDirectionIndices
    native_rate_views: tuple[source_adapter.SourceTrialViews, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.audit_authority, source_adapter.SourceAuditBudgetAuthority),
                 "carrier B8 requires the fixed typed source-audit identity")
        _require(isinstance(self.groups, core.ComplementaryGroups),
                 "carrier B8 requires the exact complementary unit/group mapping")
        _require(isinstance(self.directions, FinalizedGroupedCarrierDirectionIndices),
                 "carrier B8 requires finalized grouped direction-index authority")
        _require(self.audit_authority.budget == self.directions.budget,
                 "carrier B8 audit identity/direction budget drift")
        _require(self.groups.group_count == core.GROUP_COUNT,
                 "carrier B8 must retain the frozen four complementary groups")
        valid_mask = np.asarray(self.groups.valid_mask, dtype=np.bool_)
        assignment = np.asarray(self.groups.assignment, dtype=np.int64)
        assigned_valid = np.zeros(self.groups.units, dtype=np.bool_)
        for group in range(core.GROUP_COUNT):
            group_units = self.groups.unit_indices(group)
            _require(group_units.size >= 1 and bool(np.all(valid_mask[group_units])),
                     "carrier B8 group assignment contains an invalid or empty valid group")
            _require(not bool(np.any(assigned_valid[group_units])),
                     "carrier B8 valid unit was assigned to more than one group")
            assigned_valid[group_units] = True
        _require(np.array_equal(assigned_valid, valid_mask),
                 "carrier B8 must assign every and only authority-valid rows exactly once")
        _require(bool(np.all(assignment[~valid_mask] == -1)),
                 "carrier B8 authority-invalid rows must retain complementary group -1")
        views = tuple(self.native_rate_views)
        _require(len(views) == len(self.directions.trial_ids),
                 "carrier B8 native view/direction row count drift")
        _require(all(isinstance(view, source_adapter.SourceTrialViews) for view in views),
                 "carrier B8 accepts typed source trial views only")
        _require(self.directions.trial_ids == self.audit_authority.audit_trial_ids,
                 "carrier B8 finalized directions must cover the fixed complete audit pool")
        _require(tuple(view.carrier_counts.trial_id for view in views) == self.directions.trial_ids,
                 "carrier B8 native rate views/trial IDs drift")
        _require(all(view.carrier_counts.session_id == self.directions.session_id for view in views),
                 "carrier B8 cannot mix source sessions")
        channel_digests = {view.carrier_counts.channel_order_sha256 for view in views}
        widths = {view.carrier_counts.counts.shape[1] for view in views}
        _require(len(channel_digests) == len(widths) == 1,
                 "carrier B8 native rate rows must preserve one exact unit order")
        _require(next(iter(channel_digests)) == core.channel_order_digest(self.groups.channel_ids),
                 "carrier B8 complementary groups/channel order binding drift")
        _require(next(iter(widths)) == self.groups.units,
                 "carrier B8 complementary groups/unit width drift")
        object.__setattr__(self, "native_rate_views", views)

    @property
    def budget(self) -> int:
        return self.directions.budget

    @property
    def session_id(self) -> str:
        return self.directions.session_id

    def native_trial_rates(self) -> np.ndarray:
        """Derive rates only through the typed native-count Stage-0 primitive."""
        rates = np.stack(
            tuple(core.scalar_rates_from_native_rewarded_counts(view.carrier_counts) for view in self.native_rate_views),
            axis=0,
        )
        _require(rates.ndim == 2 and rates.shape == (len(self.native_rate_views), self.groups.units)
                 and np.isfinite(rates).all(), "carrier B8 native rate construction drift")
        rates = np.ascontiguousarray(rates, dtype=np.float64)
        rates.setflags(write=False)
        return rates

    def unit_topology_payload(self) -> dict[str, object]:
        """Bind all physical rows while exposing the valid cosine estimand.

        The sealed theta authority can contain rows without a defined tuning
        direction.  Those physical channel rows stay in the immutable native
        rate and channel-order evidence, but their assignment is exactly -1
        and they are never observations in the B8 cosine distribution.
        """
        valid_mask = np.asarray(self.groups.valid_mask, dtype=np.bool_)
        assignment = np.asarray(self.groups.assignment, dtype=np.int64)
        return {
            "total_unit_count": int(self.groups.units),
            "valid_unit_count": int(valid_mask.sum()),
            "invalid_unit_count": int((~valid_mask).sum()),
            "valid_mask_sha256": core.array_digest(valid_mask),
            "complementary_assignment_sha256": core.array_digest(assignment),
            "invalid_assignment_is_minus_one": bool(np.all(assignment[~valid_mask] == -1)),
        }

    def payload(self) -> dict[str, object]:
        rates = self.native_trial_rates()
        return {
            "schema": "causal_dual_memory_grouped_carrier_b8_inputs_v3",
            "audit_authority_sha256": self.audit_authority.digest,
            "groups_sha256": self.groups.digest,
            "directions_sha256": self.directions.digest,
            "budget": self.budget,
            "session_id": self.session_id,
            "native_rate_trial_ids": list(self.directions.trial_ids),
            "native_rate_rows_sha256": core.array_digest(rates),
            "native_channel_order_sha256": self.native_rate_views[0].carrier_counts.channel_order_sha256,
            "native_rate_shape": list(rates.shape),
            "unit_topology": self.unit_topology_payload(),
        }


def finalized_grouped_carrier_directions_from_rows(
    rows: Sequence[GroupedPseudoAuditRow],
) -> FinalizedGroupedCarrierDirectionIndices:
    """Join true audit labels after all K=4 pseudo outcomes are final."""
    audit_rows = tuple(rows)
    _require(len(audit_rows) >= 2 and all(isinstance(row, GroupedPseudoAuditRow) for row in audit_rows),
             "grouped carrier B8 needs completed typed audit rows")
    budgets = {row.budget for row in audit_rows}
    sessions = {row.session_id for row in audit_rows}
    _require(len(budgets) == len(sessions) == 1, "grouped carrier B8 rows must cover one source budget/session")
    trial_ids = tuple(row.trial_id for row in audit_rows)
    _require(len(set(trial_ids)) == len(trial_ids), "grouped carrier B8 audit row identities must be unique")
    true_indices: list[int] = []
    pseudo_rows: list[list[int]] = []
    reasons: list[tuple[core.UpdateRejectionReason | None, ...]] = []
    for row in audit_rows:
        true_index, _ = core.nearest_canonical_direction(float(row.true_direction.direction_radians))
        true_indices.append(true_index)
        pseudo_rows.append([-1 if item is None else int(item) for item in row.pseudo_direction_indices])
        reasons.append(tuple(row.rejection_reasons))
    return FinalizedGroupedCarrierDirectionIndices(
        next(iter(budgets)), next(iter(sessions)), trial_ids,
        np.asarray(true_indices, dtype=np.int64), np.asarray(pseudo_rows, dtype=np.int64), tuple(reasons),
    )


def carrier_b8_inputs_from_finalized_rows(
    rows: Sequence[GroupedPseudoAuditRow],
    native_rate_views: Sequence[source_adapter.SourceTrialViews],
    *,
    audit_authority: source_adapter.SourceAuditBudgetAuthority,
    groups: core.ComplementaryGroups,
) -> CarrierB8Inputs:
    """Construct the governing CDM-D grouped B8 inputs from typed native views."""
    directions = finalized_grouped_carrier_directions_from_rows(rows)
    return CarrierB8Inputs(audit_authority, groups, directions, tuple(native_rate_views))


def _frozen_gate_fit_carriers(trial_rates: np.ndarray, direction_indices: np.ndarray) -> np.ndarray:
    """Byte-for-byte algebraic twin of frozen pseudo-label carrier refitting.

    This intentionally does *not* call Stage-0 fixed ridge: frozen B8 compares
    equal-direction ordinary OLS carriers.  The no-data test independently
    checks this routine against ``pseudo_label_carrier_gate.py``.
    """
    rates = np.asarray(trial_rates, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    _require(rates.ndim == 2 and directions.size == rates.shape[0],
             "frozen carrier B8 rates/directions alignment drift")
    units = int(rates.shape[1])
    carriers = np.zeros((units, 4), dtype=np.float64)
    present = sorted({int(index) for index in directions})
    thetas = np.asarray([core.CANONICAL_DIRECTIONS_RAD[index] for index in present], dtype=np.float64)
    for unit in range(units):
        per_direction = []
        for direction_index in present:
            mask = directions == direction_index
            per_direction.append(float(rates[mask, unit].mean()))
        design = np.stack(
            [np.ones_like(thetas), np.cos(thetas), np.sin(thetas)], axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(per_direction, dtype=np.float64), rcond=None)
        b, a, c = (float(value) for value in coefficients)
        carriers[unit] = [a, c, math.hypot(a, c), b]
    return carriers


def _frozen_gate_ac_cosines(first_ac: np.ndarray, second_ac: np.ndarray) -> np.ndarray:
    left = np.asarray(first_ac, dtype=np.float64)
    right = np.asarray(second_ac, dtype=np.float64)
    _require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == 2,
             "frozen carrier B8 [a,c] shape drift")
    norm_left = np.linalg.norm(left, axis=1)
    norm_right = np.linalg.norm(right, axis=1)
    defined = (norm_left > 1.0e-12) & (norm_right > 1.0e-12)
    cosines = np.full(left.shape[0], np.nan, dtype=np.float64)
    cosines[defined] = np.sum(left[defined] * right[defined], axis=1) / (
        norm_left[defined] * norm_right[defined]
    )
    return cosines


def _frozen_gate_shuffle_order(n_trials: int, *, session_id: str, seed: int) -> tuple[np.ndarray, int]:
    _require(n_trials >= 2, "frozen carrier B8 shuffle needs at least two trials")
    digest = hashlib.sha256(
        f"{FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_NAMESPACE}:{session_id}:{seed}".encode()
    ).digest()
    shift = 1 + int.from_bytes(digest[:8], "little") % (n_trials - 1)
    order = np.roll(np.arange(n_trials, dtype=np.int64), shift)
    _require(not np.array_equal(order, np.arange(n_trials, dtype=np.int64)),
             "frozen carrier B8 shuffle retained identity")
    return order, int(shift)


def _frozen_b8_gates(
    median_correct: float | None,
    correct_minus_shuffle: float | None,
    fraction_ge_040: float | None,
) -> dict[str, bool]:
    gates = {
        "median_cosine_ge_050": bool(median_correct is not None and median_correct >= B8_MEDIAN_COSINE_MIN),
        "correct_minus_shuffle_ge_001": bool(
            correct_minus_shuffle is not None and correct_minus_shuffle >= B8_CORRECT_MINUS_SHUFFLE_MIN
        ),
        "fraction_ge_040_ge_050": bool(
            fraction_ge_040 is not None and fraction_ge_040 >= B8_FRACTION_COSINE_040_MIN
        ),
    }
    gates["all_predeclared_gates"] = all(gates.values())
    return gates


def _rejected_fixed_pool_b8(inputs: CarrierB8Inputs) -> dict[str, object]:
    directions = inputs.directions
    gates = _frozen_b8_gates(None, None, None)
    topology = inputs.unit_topology_payload()
    return {
        "schema": "causal_dual_memory_grouped_carrier_b8_evidence_v4",
        "status": "STOP_FIXED_POOL_REJECTION",
        "governing_estimand": "grouped_carrier_refit_per_unit_ac_cosine",
        "frozen_reference": FROZEN_PSEUDO_LABEL_GATE_SEMANTICS,
        "inputs": inputs.payload(),
        "budget": inputs.budget,
        "session_id": inputs.session_id,
        "fixed_pool_trial_count": len(directions.trial_ids),
        "fixed_pool_group_cells": len(directions.trial_ids) * core.GROUP_COUNT,
        "unit_topology": topology,
        "rejected_group_count": directions.rejected_group_count,
        "rejected_trial_count": directions.rejected_trial_count,
        "rejection_counts": directions.rejection_counts(),
        "median_cosine_correct": None,
        "median_cosine_shuffled": None,
        "correct_minus_shuffle": None,
        "fraction_ge_040_correct": None,
        "defined_units_correct": 0,
        "defined_units_shuffled": 0,
        "shuffle_seed": FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_SEED,
        "shuffle_shift": None,
        "common_shuffle_order_sha256": None,
        "group_evidence": [],
        "thresholds": {
            "median_cosine_ge": B8_MEDIAN_COSINE_MIN,
            "correct_minus_shuffle_ge": B8_CORRECT_MINUS_SHUFFLE_MIN,
            "fraction_ge_040_ge": B8_FRACTION_COSINE_040_MIN,
        },
        "gates": gates,
        "pass": False,
    }


def carrier_b8_aggregate(
    inputs: CarrierB8Inputs,
    *,
    shuffle_seed: int = FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_SEED,
) -> dict[str, object]:
    """Apply frozen B8 to CDM-D's actual four-group carrier consumer.

    Each held group gets its own final pseudo-label column.  We fit three full
    carrier tables per group, retain only that group's rows, and reassemble the
    result in the immutable source channel order before taking frozen B8
    statistics.  Physical channel rows with an authority-undefined tuning
    direction stay as NaN/unassigned evidence and are excluded; every valid
    row is assigned exactly once.  A single deterministic trial shuffle is
    shared by all four groups.
    """
    _require(isinstance(inputs, CarrierB8Inputs),
             "governing B8 requires typed grouped native-rate carrier inputs")
    _require(int(shuffle_seed) == FROZEN_PSEUDO_LABEL_GATE_SHUFFLE_SEED,
             "carrier B8 shuffle seed is frozen")
    if inputs.directions.has_rejections:
        return _rejected_fixed_pool_b8(inputs)
    rates = inputs.native_trial_rates()
    directions = inputs.directions
    order, shift = _frozen_gate_shuffle_order(
        rates.shape[0], session_id=inputs.session_id, seed=int(shuffle_seed),
    )
    correct_by_channel = np.full(inputs.groups.units, np.nan, dtype=np.float64)
    shuffled_by_channel = np.full(inputs.groups.units, np.nan, dtype=np.float64)
    assigned_by_group = np.zeros(inputs.groups.units, dtype=np.bool_)
    valid_mask = np.asarray(inputs.groups.valid_mask, dtype=np.bool_)
    topology = inputs.unit_topology_payload()
    group_evidence: list[dict[str, object]] = []
    for group in range(core.GROUP_COUNT):
        units = inputs.groups.unit_indices(group)
        true_carriers = _frozen_gate_fit_carriers(rates, directions.true_direction_indices)
        pseudo_carriers = _frozen_gate_fit_carriers(rates, directions.pseudo_direction_indices[:, group])
        shuffled_carriers = _frozen_gate_fit_carriers(rates, directions.pseudo_direction_indices[order, group])
        correct = _frozen_gate_ac_cosines(true_carriers[:, :2], pseudo_carriers[:, :2])
        shuffled = _frozen_gate_ac_cosines(true_carriers[:, :2], shuffled_carriers[:, :2])
        correct_by_channel[units] = correct[units]
        shuffled_by_channel[units] = shuffled[units]
        assigned_by_group[units] = True
        group_evidence.append({
            "group": group,
            "unit_indices_sha256": core.array_digest(units),
            "unit_count": int(units.size),
            "valid_unit_count": int(units.size),
            "total_unit_count": topology["total_unit_count"],
            "invalid_unit_count": topology["invalid_unit_count"],
            "valid_mask_sha256": topology["valid_mask_sha256"],
            "pseudo_direction_column_sha256": core.array_digest(directions.pseudo_direction_indices[:, group]),
            "common_shuffle_order_sha256": core.array_digest(order),
            "correct_cosines_sha256": core.array_digest(correct[units]),
            "shuffled_cosines_sha256": core.array_digest(shuffled[units]),
        })
    _require(np.array_equal(assigned_by_group, valid_mask),
             "grouped carrier B8 failed to reassemble every and only authority-valid source-channel row")
    _require(bool(np.all(np.isnan(correct_by_channel[~valid_mask]))),
             "authority-invalid rows entered the correct carrier-cosine estimand")
    _require(bool(np.all(np.isnan(shuffled_by_channel[~valid_mask]))),
             "authority-invalid rows entered the shuffled carrier-cosine estimand")
    valid_correct = correct_by_channel[valid_mask]
    valid_shuffled = shuffled_by_channel[valid_mask]
    finite_correct = valid_correct[np.isfinite(valid_correct)]
    finite_shuffled = valid_shuffled[np.isfinite(valid_shuffled)]
    median_correct = float(np.median(finite_correct)) if finite_correct.size else None
    median_shuffled = float(np.median(finite_shuffled)) if finite_shuffled.size else None
    delta = (
        float(median_correct - median_shuffled)
        if median_correct is not None and median_shuffled is not None
        else None
    )
    fraction = float(np.mean(finite_correct >= 0.40)) if finite_correct.size else None
    gates = _frozen_b8_gates(median_correct, delta, fraction)
    return {
        "schema": "causal_dual_memory_grouped_carrier_b8_evidence_v4",
        "status": "COMPLETE_FIXED_POOL",
        "governing_estimand": "grouped_carrier_refit_per_unit_ac_cosine",
        "frozen_reference": FROZEN_PSEUDO_LABEL_GATE_SEMANTICS,
        "inputs": inputs.payload(),
        "budget": inputs.budget,
        "session_id": inputs.session_id,
        "fixed_pool_trial_count": len(directions.trial_ids),
        "fixed_pool_group_cells": len(directions.trial_ids) * core.GROUP_COUNT,
        "unit_topology": topology,
        "valid_rows_assigned_exactly_once": bool(np.array_equal(assigned_by_group, valid_mask)),
        "invalid_rows_unassigned": bool(np.all(~assigned_by_group[~valid_mask])),
        "invalid_correct_cosines_all_nan": bool(np.all(np.isnan(correct_by_channel[~valid_mask]))),
        "invalid_shuffled_cosines_all_nan": bool(np.all(np.isnan(shuffled_by_channel[~valid_mask]))),
        "assigned_valid_mask_sha256": core.array_digest(assigned_by_group),
        "rejected_group_count": 0,
        "rejected_trial_count": 0,
        "rejection_counts": {},
        "median_cosine_correct": median_correct,
        "median_cosine_shuffled": median_shuffled,
        "correct_minus_shuffle": delta,
        "fraction_ge_040_correct": fraction,
        "defined_units_correct": int(finite_correct.size),
        "defined_units_shuffled": int(finite_shuffled.size),
        "shuffle_seed": int(shuffle_seed),
        "shuffle_shift": shift,
        "common_shuffle_order_sha256": core.array_digest(order),
        "group_evidence": group_evidence,
        "correct_cosines_channel_order_sha256": core.array_digest(correct_by_channel),
        "shuffled_cosines_channel_order_sha256": core.array_digest(shuffled_by_channel),
        "thresholds": {
            "median_cosine_ge": B8_MEDIAN_COSINE_MIN,
            "correct_minus_shuffle_ge": B8_CORRECT_MINUS_SHUFFLE_MIN,
            "fraction_ge_040_ge": B8_FRACTION_COSINE_040_MIN,
        },
        "gates": gates,
        "pass": gates["all_predeclared_gates"],
    }


class SourceAuditVerdict(str, Enum):
    PASS = "PASS_SOURCE_CONSTRUCTIBLE"
    STOP = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
    NOT_REACHED = "NOT_REACHED_AFTER_HIGHER_BUDGET_STOP"


def fail_fast_b8_decision(inputs_by_budget: Mapping[int, CarrierB8Inputs]) -> dict[str, object]:
    """Evaluate exact M30 -> M10 -> M4 carrier-level B8 fail-fast viability."""
    _require(set(inputs_by_budget) == set(FAIL_FAST_BUDGET_ORDER),
             "source audit requires M30/M10/M4 carrier B8 inputs")
    results: dict[str, object] = {}
    blocked = False
    overall = SourceAuditVerdict.PASS
    for budget in FAIL_FAST_BUDGET_ORDER:
        if blocked:
            results[str(budget)] = {"budget": budget, "verdict": SourceAuditVerdict.NOT_REACHED.value}
            continue
        inputs = inputs_by_budget[budget]
        _require(isinstance(inputs, CarrierB8Inputs) and inputs.budget == budget,
                 "governing fail-fast requires carrier B8 inputs, not descriptive trial-direction rows")
        aggregate = carrier_b8_aggregate(inputs)
        if aggregate["pass"] is True:
            results[str(budget)] = {**aggregate, "verdict": SourceAuditVerdict.PASS.value}
        else:
            results[str(budget)] = {**aggregate, "verdict": SourceAuditVerdict.STOP.value}
            blocked = True
            overall = SourceAuditVerdict.STOP
    return {
        "schema": "causal_dual_memory_source_audit_fail_fast_v1",
        "order": list(FAIL_FAST_BUDGET_ORDER),
        "by_budget": results,
        "verdict": overall.value,
    }


def summarize_trial_diagnostics(rows: Sequence[PseudoAuditRow]) -> dict[str, object]:
    """Record constructibility distributions without tuning any threshold."""
    _require(bool(rows), "source diagnostics require rows")
    def finite_values(name: str) -> list[float]:
        values = [getattr(row, name) for row in rows]
        return [float(value) for value in values if value is not None and np.isfinite(float(value))]

    return {
        "accepted": sum(row.accepted for row in rows),
        "rejected": sum(not row.accepted for row in rows),
        "circular_error": [value for row in rows if (value := row.circular_error()) is not None],
        "complementary_disagreement_radians": finite_values("complementary_disagreement_radians"),
        "canonical_distance_radians": finite_values("canonical_distance_radians"),
        "displacement": finite_values("displacement"),
        "mean_speed": finite_values("mean_speed"),
        "design_rank": [row.design_rank for row in rows if row.design_rank is not None],
        "design_condition": finite_values("design_condition"),
        "departure": finite_values("departure"),
    }
