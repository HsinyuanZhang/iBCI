"""Boundary-free chunk pool laws shared by training, scoring, and deployment.

Both laws consume only the observed raw neural stream (no trial metadata, no
labels, no eval mask):

- ``chunk100``  : every tumbling 100-bin window commits when complete.
- ``chunk100e`` : a window commits only if its mean multi-unit rate is at
                  least the running median of past candidate rates
                  (label-free causal selectivity against idle bins).

Pool: first-30 calibration seed rows with the D-opt4 support rows protected;
capacity 30 with FIFO eviction over non-support rows.  A decode at endpoint
``t`` uses exactly the chunks whose completion bin is strictly before ``t``
(the sealed A0 causality law).
"""
from __future__ import annotations

from collections import deque

CHUNK_BINS = 100
POOL_CAPACITY = 30


class ChunkLawError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkLawError(message)


def chunk_commit_bins(neural, origin: int, *, energy_gated: bool) -> list[int]:
    """Return completion bins of committed tumbling 100-bin windows.

    A returned bin ``b`` means the window ``[b-99, b]`` committed at ``b``;
    it is visible to decodes at endpoints ``t > b`` only.
    """
    import numpy as np

    total = int(neural.shape[0])
    require(0 <= origin <= total, "chunk origin out of range")
    commits: list[int] = []
    rates: list[float] = []
    position = origin
    while position + CHUNK_BINS <= total:
        window = neural[position : position + CHUNK_BINS]
        rate = float(window.mean(dtype=np.float64))
        if energy_gated and rates:
            commit = rate >= float(np.median(np.asarray(rates, dtype=np.float64)))
        else:
            commit = True
        if commit:
            commits.append(position + CHUNK_BINS - 1)
        rates.append(rate)  # every candidate updates the running median
        position += CHUNK_BINS
    return commits


def committed_before(commits: list[int], endpoints) -> "Any":
    """Count commits strictly before each endpoint (A0 causality)."""
    import numpy as np

    array = np.asarray(commits, dtype=np.int64)
    if array.size == 0:
        return np.zeros(len(endpoints), dtype=np.int64)
    return np.searchsorted(array, np.asarray(endpoints, dtype=np.int64), side="left")


class SeededPool:
    """First-30 seed with support-protected FIFO capacity (AJPF lineage law)."""

    def __init__(self, seed_rows, support_count: int):
        self.rows = deque(seed_rows)
        self.support_count = int(support_count)
        require(0 < self.support_count <= len(self.rows) <= POOL_CAPACITY,
                "seed/support topology drift")

    def commit(self, row) -> None:
        self.rows.append(row)
        if len(self.rows) > POOL_CAPACITY:
            del self.rows[self.support_count]  # oldest NON-support row

    def stack(self):
        import numpy as np

        return np.ascontiguousarray(np.stack(list(self.rows)), dtype=np.float32)


def pool_rows_for_state(neural, seed_rows, support_count, commits, count: int):
    """Deterministically rebuild the pool stack after `count` commits."""
    pool = SeededPool(seed_rows, support_count)
    for commit_bin in commits[:count]:
        row_start = commit_bin - CHUNK_BINS + 1
        pool.commit(neural[row_start : commit_bin + 1])
    return pool.stack()
