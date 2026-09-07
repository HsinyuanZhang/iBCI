"""Route-local activity state and deterministic trial-free chunking.

The state intentionally does *not* reuse the sealed ``ActivityMemory``.  That
class binds a carrier support count to the activity capacity, while A0 has an
M10 carrier and an independently seeded M30 activity identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import plan


class ChunkMemoryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkMemoryError(message)


def activity_sha256(rows: Sequence[np.ndarray]) -> str:
    """Canonical digest of ordered float32 B3S rows, preserving FIFO order."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"row_count": len(rows), "bins": plan.B3S_BINS},
                             sort_keys=True, separators=(",", ":")).encode("ascii"))
    for row in rows:
        array = np.ascontiguousarray(np.asarray(row, dtype=np.float32))
        digest.update(json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                                 sort_keys=True, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _b3s_row(values: np.ndarray, *, channels: int | None = None) -> np.ndarray:
    row = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    _require(row.ndim == 2 and row.shape[0] == plan.B3S_BINS,
             "every activity row must be a B3S [100,N] row")
    if channels is not None:
        _require(row.shape[1] == channels, "activity channel count drift")
    _require(bool(np.isfinite(row).all()), "activity row is nonfinite")
    row.setflags(write=False)
    return row


@dataclass(frozen=True)
class ActivityUpdate:
    arm: str
    coordinate: int
    evicted_digest: str
    appended_digest: str
    before_activity_stack_sha256: str
    after_activity_stack_sha256: str
    retained_count: int


class SeededActivity30RollingMemory:
    """Thirty-row FIFO activity state with a separate frozen M10 authority.

    ``carrier_support_indices`` records the only legal carrier authority and
    is never updated by this label-free state.  Calls that update memory must
    happen after a corresponding prediction marker, preventing a current
    chunk/trial from leaking into its own decode.
    """

    state_name = "SeededActivity30RollingMemory"

    def __init__(self, *, first30_b3s: Sequence[np.ndarray], carrier_support_indices: Sequence[int]) -> None:
        _require(len(first30_b3s) == plan.ACTIVITY_CAPACITY,
                 "A0 initial activity identity must contain exactly first30 rows")
        indices = tuple(int(value) for value in carrier_support_indices)
        _require(len(indices) == plan.CARRIER_BUDGET and len(set(indices)) == plan.CARRIER_BUDGET,
                 "carrier authority must be exactly ten distinct M10 positions")
        _require(all(0 <= value < plan.CALIBRATION_TRIALS for value in indices),
                 "carrier authority must stay within chronological first30")
        rows = tuple(_b3s_row(row) for row in first30_b3s)
        channels = rows[0].shape[1]
        _require(all(row.shape[1] == channels for row in rows), "seed channel count drift")
        self._rows: list[np.ndarray] = list(rows)
        self._channels = int(channels)
        self._carrier_support_indices = indices
        self._pending_prediction: tuple[str, int] | None = None
        self._updates: list[ActivityUpdate] = []

    @property
    def carrier_support_indices(self) -> tuple[int, ...]:
        return self._carrier_support_indices

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def retained_count(self) -> int:
        return len(self._rows)

    @property
    def update_count(self) -> int:
        return len(self._updates)

    @property
    def updates(self) -> tuple[ActivityUpdate, ...]:
        return tuple(self._updates)

    def rows(self) -> tuple[np.ndarray, ...]:
        return tuple(self._rows)

    def identity_sha256(self) -> str:
        return activity_sha256(self._rows)

    def begin_causal_transaction(self, *, arm: str, coordinate: int) -> str:
        _require(arm in ("true_trial", "chunk_phase0", "chunk_phase50"), "unknown A0 arm")
        _require(int(coordinate) >= 0, "causal coordinate must be nonnegative")
        _require(self._pending_prediction is None, "previous causal transaction was not completed")
        self._pending_prediction = (arm, int(coordinate))
        return self.identity_sha256()

    def append_after_causal_transaction(self, *, arm: str, coordinate: int, activity: np.ndarray) -> ActivityUpdate:
        _require(self._pending_prediction == (arm, int(coordinate)),
                 "activity update must occur only after its exact causal transaction")
        row = _b3s_row(activity, channels=self._channels)
        before = self.identity_sha256()
        evicted = self._rows.pop(0)
        self._rows.append(row)
        _require(len(self._rows) == plan.ACTIVITY_CAPACITY, "activity FIFO capacity drift")
        after = self.identity_sha256()
        update = ActivityUpdate(
            arm=arm,
            coordinate=int(coordinate),
            evicted_digest=activity_sha256((evicted,)),
            appended_digest=activity_sha256((row,)),
            before_activity_stack_sha256=before,
            after_activity_stack_sha256=after,
            retained_count=len(self._rows),
        )
        self._updates.append(update)
        self._pending_prediction = None
        return update

    def assert_no_pending_prediction(self) -> None:
        _require(self._pending_prediction is None, "unfinished causal-bin/update transaction")

    def close_prediction_without_update(self, *, arm: str, coordinate: int) -> None:
        """Close an honest causal bin that did not complete an activity unit."""
        _require(self._pending_prediction == (arm, int(coordinate)),
                 "no-update closure must match its exact causal transaction")
        self._pending_prediction = None


class FixedChunkStream:
    """One causal neural stream for phase 0 or phase 50 chunk updates.

    The caller must invoke :meth:`begin_causal_bin` and then
    :meth:`append_after_causal_bin` for every post-calibration raw neural bin.
    This makes skipped/duplicated bins fail rather than silently changing the
    chunk geometry.  Metric validity and targets never appear in this API.
    """

    def __init__(self, *, memory: SeededActivity30RollingMemory, phase: int,
                 calibration_boundary_bin: int) -> None:
        _require(phase in (plan.PRIMARY_PHASE, plan.SENSITIVITY_PHASE), "unsupported chunk phase")
        _require(int(calibration_boundary_bin) >= 0, "calibration boundary must be nonnegative")
        self.memory = memory
        self.phase = int(phase)
        # This origin is the raw first-30 calibration boundary.  It is not a
        # padded model-window coordinate, and it never derives from a behavior
        # mask or target.  A future runtime must call this stream for every raw
        # bin after that boundary, even before the first scored query window.
        self.calibration_boundary_bin = int(calibration_boundary_bin)
        self._next_coordinate = int(calibration_boundary_bin)
        self._opened: int | None = None
        self._values: list[np.ndarray] = []
        self._done = False
        self.completed_chunks = 0
        self.partial_final_bins_discarded = 0

    @property
    def arm(self) -> str:
        return "chunk_phase0" if self.phase == 0 else "chunk_phase50"

    @property
    def done(self) -> bool:
        return self._done

    def begin_causal_bin(self, coordinate: int) -> str:
        """Open one raw-bin causal transaction; this is not a model decode."""
        _require(not self._done, "cannot consume after stream completion")
        _require(int(coordinate) == self._next_coordinate,
                 "raw neural stream has a skipped or duplicated prediction bin")
        _require(self._opened is None, "previous prediction has not appended its neural bin")
        self._opened = int(coordinate)
        return self.memory.begin_causal_transaction(arm=self.arm, coordinate=int(coordinate))

    def append_after_causal_bin(self, coordinate: int, neural_bin: np.ndarray) -> ActivityUpdate | None:
        _require(not self._done and self._opened == int(coordinate),
                 "append must match the immediately preceding causal-bin transaction")
        value = np.ascontiguousarray(np.asarray(neural_bin, dtype=np.float32)).reshape(-1)
        _require(value.size == self.memory.channels and bool(np.isfinite(value).all()),
                 "raw neural bin channel/finiteness drift")
        # Every bin is chronologically consumed.  Before phase offset it is
        # deliberately not placed in a chunk, never filtered by a metric mask.
        offset = int(coordinate) - self.calibration_boundary_bin
        update: ActivityUpdate | None = None
        if offset >= self.phase:
            self._values.append(value)
            if len(self._values) == plan.CHUNK_LENGTH:
                row = np.ascontiguousarray(np.stack(self._values, axis=0), dtype=np.float32)
                update = self.memory.append_after_causal_transaction(
                    arm=self.arm, coordinate=int(coordinate), activity=row,
                )
                self.completed_chunks += 1
                self._values.clear()
            else:
                # No update has occurred: close the prediction transaction
                # while retaining its no-update fact.  It cannot be carried to
                # the next bin, which would weaken causality accounting.
                self.memory.close_prediction_without_update(arm=self.arm, coordinate=int(coordinate))
        else:
            self.memory.close_prediction_without_update(arm=self.arm, coordinate=int(coordinate))
        self._opened = None
        self._next_coordinate += 1
        return update

    def finish(self) -> dict[str, int]:
        _require(not self._done, "stream already finished")
        _require(self._opened is None, "cannot finish with an open causal-bin transaction")
        self.partial_final_bins_discarded = len(self._values)
        self._values.clear()
        self._done = True
        return {
            "completed_chunks": int(self.completed_chunks),
            "partial_final_bins_discarded": int(self.partial_final_bins_discarded),
            "phase": int(self.phase),
        }

    def reset(self) -> None:
        _require(self._done, "reset is legal only after done")
        self._next_coordinate = self.calibration_boundary_bin
        self._opened = None
        self._values.clear()
        self._done = False
        self.completed_chunks = 0
        self.partial_final_bins_discarded = 0
