"""Pure canonical M30-ready stream and fixed 4/10/30 controller."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

from . import plan


class ControllerError(ValueError):
    pass


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ControllerError(message)


@dataclass(frozen=True, order=True)
class M30Coordinate:
    """One already M30-causal coordinate; no fallback can make it eligible."""
    session: str
    query_trial: int
    window_start: int
    completed_non_support: tuple[int, ...]
    state_digest: str

    def validate(self) -> None:
        _need(bool(self.session), "AJPF coordinate session missing")
        _need(self.query_trial >= 30 + plan.NON_SUPPORT_COMPLETIONS[30], "AJPF coordinate not M30-ready")
        _need(tuple(sorted(self.completed_non_support)) == self.completed_non_support, "AJPF completion chronology drift")
        _need(len(set(self.completed_non_support)) == len(self.completed_non_support), "AJPF duplicate completion")
        _need(all(30 <= value < self.query_trial for value in self.completed_non_support), "AJPF current/future completion")
        _need(len(self.completed_non_support) >= plan.NON_SUPPORT_COMPLETIONS[30], "AJPF runtime M30 shortage")
        _need(len(self.state_digest) == 64 and all(c in "0123456789abcdef" for c in self.state_digest),
              "AJPF coordinate digest drift")


def requested_pool(epoch_one_indexed: int, canonical_batch_ordinal: int) -> int:
    _need(1 <= epoch_one_indexed <= plan.EPOCHS and canonical_batch_ordinal >= 0, "AJPF controller domain drift")
    return plan.POOL_CYCLE[(epoch_one_indexed - 1 + canonical_batch_ordinal) % len(plan.POOL_CYCLE)]


def causal_members(coordinate: M30Coordinate, support: Sequence[int], requested: int) -> tuple[int, ...]:
    coordinate.validate()
    selected = tuple(int(x) for x in support)
    _need(len(selected) == plan.SUPPORT_COUNT and len(set(selected)) == plan.SUPPORT_COUNT
          and all(0 <= x < 30 for x in selected), "AJPF D-opt4 support drift")
    _need(requested in plan.POOL_CYCLE, "AJPF unsupported pool request")
    need = plan.NON_SUPPORT_COMPLETIONS[requested]
    members = selected + (coordinate.completed_non_support[-need:] if need else ())
    _need(len(members) == requested and len(set(members)) == requested, "AJPF causal membership shortage")
    return members


def canonical_batches(coordinates: Iterable[M30Coordinate], *, max_batch: int = plan.BATCH_SIZE) -> tuple[tuple[M30Coordinate, ...], ...]:
    """Lexical, state-homogeneous stream; it never skips a short runtime state."""
    _need(max_batch == plan.BATCH_SIZE, "AJPF batch-size law drift")
    ordered = tuple(sorted(coordinates, key=lambda x: (x.session, x.query_trial, x.window_start)))
    _need(ordered and len({(x.session, x.query_trial, x.window_start) for x in ordered}) == len(ordered),
          "AJPF coordinate empty/duplicate")
    for value in ordered:
        value.validate()
    output: list[tuple[M30Coordinate, ...]] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index]
        stop = index + 1
        while stop < len(ordered) and (ordered[stop].session, ordered[stop].query_trial, ordered[stop].state_digest) == (
            anchor.session, anchor.query_trial, anchor.state_digest):
            stop += 1
        group = ordered[index:stop]
        output.extend(tuple(group[pos:pos + max_batch]) for pos in range(0, len(group), max_batch))
        index = stop
    _need(all(0 < len(batch) <= max_batch and len({(x.session, x.query_trial, x.state_digest) for x in batch}) == 1
              for batch in output), "AJPF mixed resident batch")
    return tuple(output)


def stream_digest(batches: Sequence[Sequence[M30Coordinate]]) -> str:
    text = "|".join(
        f"{coord.session}:{coord.query_trial}:{coord.window_start}:{coord.state_digest}"
        for batch in batches for coord in batch
    )
    return sha256(text.encode("ascii")).hexdigest()


def validate_historical_stream(*, batches: Sequence[Sequence[M30Coordinate]], coordinate_count: int,
                               coordinate_sha256: str, batch_sha256: str) -> None:
    _need(coordinate_count == plan.SOURCE_COORDINATES and coordinate_sha256 == plan.SOURCE_COORDINATE_SHA256
          and batch_sha256 == plan.SOURCE_BATCH_SHA256, "AJPF held APFG source authority drift")
    _need(len(batches) == plan.SOURCE_GROUPS_PER_EPOCH, "AJPF shared batch count drift")

