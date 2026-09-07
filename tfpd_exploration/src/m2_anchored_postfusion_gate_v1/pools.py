"""Pure causal pool laws for APFG source coordinates.

No activity tensors, labels, masks, models, or Torch objects are accepted
here.  The live route must first establish trial-completion coordinates from
the sealed source authority, then use this module to reject invalid exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

from . import plan


class PoolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoolError(message)


@dataclass(frozen=True)
class PoolState:
    """One exact causal activity membership for one supervised coordinate."""

    session_id: str
    requested_size: int
    query_trial_index: int
    query_endpoint: int
    support_trial_ids: tuple[int, ...]
    completed_non_support_trial_ids: tuple[int, ...]
    member_trial_ids: tuple[int, ...]
    ordered_activity_sha256: tuple[str, ...]
    selected_support4_carrier_hz_sha256: str
    normalized_side_sha256: str
    normalizer_sha256: str

    def digest(self) -> str:
        payload = (self.session_id, str(self.requested_size), str(self.query_trial_index), str(self.query_endpoint),
                   ",".join(map(str, self.support_trial_ids)),
                   ",".join(map(str, self.completed_non_support_trial_ids)),
                   ",".join(map(str, self.member_trial_ids)),
                   ",".join(self.ordered_activity_sha256), self.selected_support4_carrier_hz_sha256,
                   self.normalized_side_sha256, self.normalizer_sha256)
        return sha256("|".join(payload).encode("ascii")).hexdigest()

    def identity_digest(self) -> str:
        """Alpha-independent identity key; endpoint remains receipt evidence."""
        payload = (self.session_id, str(self.requested_size), str(self.query_trial_index),
                   ",".join(map(str, self.member_trial_ids)), ",".join(self.ordered_activity_sha256),
                   self.selected_support4_carrier_hz_sha256, self.normalized_side_sha256, self.normalizer_sha256)
        return sha256("|".join(payload).encode("ascii")).hexdigest()


def _sha256_literal(value: object, label: str) -> str:
    text = str(value)
    _require(len(text) == 64 and all(character in "0123456789abcdef" for character in text),
             f"{label} must be a lowercase 64-hex SHA256")
    return text


def validate_support_indices(values: Sequence[int]) -> tuple[int, ...]:
    support = tuple(int(value) for value in values)
    _require(len(support) == plan.SUPPORT_COUNT, "APFG requires exactly four D-opt support trials")
    _require(len(set(support)) == len(support), "D-opt support trials duplicate")
    _require(all(0 <= value < 30 for value in support), "D-opt support must come from first thirty trials")
    return tuple(sorted(support))


def causal_pool_state(*, session_id: str, requested_size: int, support_trial_ids: Sequence[int],
                      completed_non_support_trial_ids: Sequence[int], query_trial_index: int,
                      query_endpoint: int, ordered_activity_sha256: Sequence[str],
                      selected_support4_carrier_hz_sha256: str, normalized_side_sha256: str,
                      normalizer_sha256: str) -> PoolState:
    """Select support plus *most recent* completed non-support trials.

    ``completed_non_support_trial_ids`` is an ordered chronology supplied by
    the caller for this endpoint.  Every item must be complete strictly before
    the current query trial.  The exact requested cardinality is mandatory.
    """
    _require(requested_size in plan.POOL_CYCLE, "unknown APFG requested pool cardinality")
    session = str(session_id)
    _require(bool(session), "APFG pool state requires a nonempty session identity")
    support = validate_support_indices(support_trial_ids)
    completed = tuple(int(value) for value in completed_non_support_trial_ids)
    _require(query_trial_index >= 30, "source endpoint precedes completion of first thirty trials")
    _require(query_endpoint >= 0, "query endpoint must be nonnegative")
    _require(tuple(sorted(completed)) == completed and len(set(completed)) == len(completed),
             "completed non-support trials must be unique chronological order")
    _require(all(value >= 30 for value in completed), "completed non-support trial precedes first30 boundary")
    _require(all(value < query_trial_index for value in completed),
             "future, current, or partially completed trial entered APFG pool")
    _require(not set(completed).intersection(support), "support trial appears in non-support chronology")
    needed = plan.NON_SUPPORT_COMPLETIONS[requested_size]
    _require(len(completed) >= needed, "requested APFG pool cardinality is not yet causally available")
    selected_recent = completed[-needed:] if needed else ()
    members = support + selected_recent
    _require(len(members) == requested_size and len(set(members)) == requested_size,
             "APFG pool membership cardinality drift")
    activity = tuple(_sha256_literal(value, "ordered activity digest") for value in ordered_activity_sha256)
    _require(len(activity) == len(members), "APFG activity digest count must match causal members")
    return PoolState(session_id=session, requested_size=int(requested_size), query_trial_index=int(query_trial_index),
                     query_endpoint=int(query_endpoint), support_trial_ids=support,
                     completed_non_support_trial_ids=completed, member_trial_ids=members,
                     ordered_activity_sha256=activity,
                     selected_support4_carrier_hz_sha256=_sha256_literal(selected_support4_carrier_hz_sha256, "selected support4 carrier digest"),
                     normalized_side_sha256=_sha256_literal(normalized_side_sha256, "normalized side digest"),
                     normalizer_sha256=_sha256_literal(normalizer_sha256, "frozen normalizer digest"))


def group_coordinate_states(states: Mapping[object, PoolState]) -> dict[str, tuple[object, ...]]:
    """Group only coordinates with byte-identical causal pool state digests."""
    _require(bool(states), "APFG state batch is empty")
    grouped: dict[str, list[object]] = {}
    for coordinate, state in states.items():
        _require(isinstance(state, PoolState), "batch contains non-APFG pool state")
        grouped.setdefault(state.identity_digest(), []).append(coordinate)
    return {digest: tuple(values) for digest, values in grouped.items()}


def cycle_for_coordinates(states: Iterable[PoolState]) -> tuple[int, ...]:
    """Return the emitted cardinalities; missing sizes are never substituted."""
    output = tuple(state.requested_size for state in states)
    _require(all(value in plan.POOL_CYCLE for value in output), "APFG emitted unknown pool size")
    return output


def requested_pool_size(epoch_one_indexed: int, canonical_batch_ordinal: int) -> int:
    """Frozen epoch/batch controller; 12 epochs expose each size four times."""
    _require(1 <= int(epoch_one_indexed) <= plan.EPOCHS, "APFG epoch outside fixed 1..12 horizon")
    _require(int(canonical_batch_ordinal) >= 0, "APFG canonical batch ordinal is negative")
    return plan.POOL_CYCLE[(int(epoch_one_indexed) - 1 + int(canonical_batch_ordinal)) % len(plan.POOL_CYCLE)]


def validate_canonical_m30_batch(states: Sequence[PoolState]) -> tuple[PoolState, ...]:
    """Validate one source batch list entry before the 4/10/30 cycle begins."""
    values = tuple(states)
    _require(0 < len(values) <= plan.SOURCE_BATCH_MAX_MEMBERS, "APFG canonical batch member count drift")
    anchor = values[0]
    _require(all(value.session_id == anchor.session_id and value.query_trial_index == anchor.query_trial_index
                 and value.identity_digest() == anchor.identity_digest() for value in values),
             "APFG canonical batch must have one exact session/query/pool state")
    _require(anchor.requested_size == 30 and len(anchor.member_trial_ids) == 30,
             "APFG canonical source list must contain only M30-constructible coordinates")
    return values


__all__ = ("PoolError", "PoolState", "validate_support_indices", "causal_pool_state",
           "group_coordinate_states", "cycle_for_coordinates", "requested_pool_size", "validate_canonical_m30_batch")
