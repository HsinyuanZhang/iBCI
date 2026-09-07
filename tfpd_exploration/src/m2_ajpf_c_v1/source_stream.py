"""Chunk-law source coordinate stream over the seven held-in M2 sessions.

Deterministic, trial-metadata-free: for each session the chunk commits derive
only from the raw neural bins (origin = session bin 0, matching deployment on
the official continual stream).  Coordinates are all valid window starts
(``s >= 49``, target defined at ``s+49``), subsampled by one per-session
stride so the per-epoch coordinate budget lands near the AJPF canonical
91,717.  Batches group coordinates by exact (session, chunk-state) identity,
capped at 32 members.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import chunk_law, plan


class SourceStreamError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceStreamError(message)


@dataclass(frozen=True)
class ChunkCoordinate:
    session: str
    state_index: int      # number of chunks committed strictly before this window's endpoint
    window_start: int
    query_trial_index: int  # = state_index; satisfies the AJPF step's group-check attributes


@dataclass
class SessionMaterial:
    session: str
    neural: Any
    targets: Any
    seed_rows: list
    support_count: int
    normalized_side: Any
    selected_indices: list
    commits: dict  # law -> commit bins


def build_session_material(dataset, session: str) -> SessionMaterial:
    import numpy as np

    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    for path in (root, root / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from laws import sealed_fit_ridge_side, select_dopt4_support, select_first30_activity_pool

    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    selected = select_dopt4_support(angles)
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    seed_rows = [np.ascontiguousarray(row, dtype=np.float32)
                 for row in select_first30_activity_pool(calibration)]
    side, _evidence = sealed_fit_ridge_side(dataset, session, selected)
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    targets = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    commits = {
        "chunk100": chunk_law.chunk_commit_bins(neural, 0, energy_gated=False),
        "chunk100e": chunk_law.chunk_commit_bins(neural, 0, energy_gated=True),
    }
    return SessionMaterial(
        session=session, neural=neural, targets=targets, seed_rows=seed_rows,
        support_count=len(selected), normalized_side=np.asarray(side, dtype=np.float32),
        selected_indices=[int(v) for v in selected], commits=commits,
    )


def session_stride(total_windows: int, budget: int) -> int:
    """Legacy helper kept for receipts: even window stride for a budget."""
    return max(1, round(total_windows / max(1, budget)))


def coordinate_stream(materials: list, law: str) -> list[list[ChunkCoordinate]]:
    """Grouped coordinates for one epoch, canonical order (session, start)."""
    import numpy as np

    require(law in plan.LAWS, f"unknown law {law}")
    valid_starts = {}
    for material in materials:
        n = material.neural.shape[0]
        starts = np.arange(49, n - 49, dtype=np.int64)
        valid_starts[material.session] = starts[
            np.isfinite(material.targets[starts + 49]).all(axis=1)
        ]
    groups: list[list[ChunkCoordinate]] = []
    # V2 exposure law: all chunk states.  The per-state window cap is a
    # dynamic allocation: 80% of total availability spread evenly over the
    # states, clamped to [32, WINDOWS_PER_STATE_MAX], rounded down to a
    # multiple of BATCH_SIZE.  Windows within a state are evenly subsampled
    # to that cap and split into <=32 groups.
    state_totals: dict[tuple[str, int], int] = {}
    for material in materials:
        valid = valid_starts[material.session]
        states = chunk_law.committed_before(material.commits[law], valid + 49)
        for value in states.tolist():
            key = (material.session, int(value))
            state_totals[key] = state_totals.get(key, 0) + 1
    n_states = max(1, len(state_totals))
    available_total_pre = sum(state_totals.values())
    allocation = -(-int(0.8 * available_total_pre) // n_states)  # ceil
    per_state_cap = min(plan.WINDOWS_PER_STATE_MAX, max(plan.BATCH_SIZE, allocation))
    per_state_cap = (per_state_cap // plan.BATCH_SIZE) * plan.BATCH_SIZE or plan.BATCH_SIZE
    for material in materials:
        valid = valid_starts[material.session]
        commits = material.commits[law]
        states = chunk_law.committed_before(commits, valid + 49)
        offset = 0
        while offset < len(valid):
            stop = offset
            while stop < len(valid) and states[stop] == states[offset]:
                stop += 1
            available = stop - offset
            multiple = min(available, per_state_cap)
            step = -(-available // multiple) if multiple else 1
            members = list(range(offset, stop, step))[:multiple]
            for chunk_offset in range(0, len(members), plan.BATCH_SIZE):
                groups.append([
                    ChunkCoordinate(session=material.session, state_index=int(states[index]),
                                    window_start=int(valid[index]), query_trial_index=int(states[index]))
                    for index in members[chunk_offset : chunk_offset + plan.BATCH_SIZE]
                ])
            offset = stop
    total = sum(len(group) for group in groups)
    available_total = sum(int(value.size) for value in valid_starts.values())
    require(len(groups) <= plan.GROUP_BUDGET_PER_EPOCH,
            f"group budget drift: {len(groups)} groups > {plan.GROUP_BUDGET_PER_EPOCH}")
    require(total >= 0.4 * available_total,
            f"unreasonable source exposure: {total} of {available_total}")
    return groups
