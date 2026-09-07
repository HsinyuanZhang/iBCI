"""Pure, default-off carrier corruptions for the B2 design contract.

The proposed intervention is applied only to an already source-standardized
``[N,4]`` T4 tensor during source training.  Deployment remains ordinary T4.
Carrier state is a source-session property: a run seed and source session
determine an exact 12-epoch schedule and one fixed RS4 row mapping.  Neither
construction has a batch, worker, rank, resume, or example-index input.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal, Mapping

import numpy as np

TrainingKind = Literal["t4", "rs4", "z4"]

LOGICAL_EPOCHS = tuple(range(1, 13))
TRAINING_SCHEDULE_COUNTS: Mapping[TrainingKind, int] = {
    "t4": 6,
    "rs4": 3,
    "z4": 3,
}


def _session_seed(*, namespace: str, run_seed: int, source_session: str) -> int:
    """Derive a process-stable NumPy seed from the only permitted key fields."""
    if not isinstance(source_session, str) or not source_session:
        raise ValueError("source_session must be a nonempty string")
    payload = f"b2-v2\0{namespace}\0{int(run_seed)}\0{source_session}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def training_schedule(*, run_seed: int, source_session: str) -> tuple[TrainingKind, ...]:
    """Return the exact deterministic T4x6/RS4x3/Z4x3 epoch schedule.

    The tuple position is ``logical_epoch - 1``.  This API intentionally has
    no batch, example, worker, rank, or resume key, so all windows from a
    source session receive the same state throughout a logical epoch.
    """
    multiset: tuple[TrainingKind, ...] = (
        ("t4",) * TRAINING_SCHEDULE_COUNTS["t4"]
        + ("rs4",) * TRAINING_SCHEDULE_COUNTS["rs4"]
        + ("z4",) * TRAINING_SCHEDULE_COUNTS["z4"]
    )
    if len(multiset) != len(LOGICAL_EPOCHS):
        raise RuntimeError("B2 schedule counts do not cover exactly 12 logical epochs")
    generator = np.random.RandomState(
        _session_seed(namespace="training-schedule", run_seed=run_seed, source_session=source_session)
    )
    schedule = tuple(multiset[index] for index in generator.permutation(len(multiset)))
    if Counter(schedule) != Counter(TRAINING_SCHEDULE_COUNTS):
        raise RuntimeError("B2 training schedule does not have the frozen exact counts")
    return schedule


def training_kind_for_epoch(
    *,
    run_seed: int,
    source_session: str,
    logical_epoch: int,
) -> TrainingKind:
    """Return one session-wide carrier state for logical epoch 1 through 12."""
    epoch = int(logical_epoch)
    if epoch not in LOGICAL_EPOCHS or epoch != logical_epoch:
        raise ValueError(f"logical_epoch must be one of {LOGICAL_EPOCHS}, got {logical_epoch!r}")
    return training_schedule(run_seed=run_seed, source_session=source_session)[epoch - 1]


def _validated_standardized_t4(carrier: np.ndarray) -> np.ndarray:
    value = np.asarray(carrier)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"standardized T4 must have shape [N,4], got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"standardized T4 must be floating point, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError("standardized T4 must be finite")
    return value


def complete_row_derangement(num_rows: int, *, seed: int) -> np.ndarray:
    """Return a deterministic complete row mapping with no fixed point.

    Sattolo's algorithm produces one cycle containing every row, and therefore
    a bijection with no fixed points for every ``num_rows >= 2``.
    """
    size = int(num_rows)
    if size < 2 or size != num_rows:
        raise ValueError("complete row derangement requires an integer number of rows >= 2")
    generator = np.random.RandomState(int(seed))
    order = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap = int(generator.randint(0, index))
        order[index], order[swap] = order[swap], order[index]
    if not np.array_equal(np.sort(order), np.arange(size)) or np.any(order == np.arange(size)):
        raise RuntimeError("failed to construct a complete B2 row derangement")
    return order


def rs4_row_mapping(
    num_rows: int,
    *,
    run_seed: int,
    source_session: str,
) -> np.ndarray:
    """Return the one fixed complete RS4 mapping for a run seed/session pair."""
    return complete_row_derangement(
        num_rows,
        seed=_session_seed(
            namespace="rs4-row-mapping",
            run_seed=run_seed,
            source_session=source_session,
        ),
    )


def _validated_row_mapping(row_mapping: np.ndarray, *, num_rows: int) -> np.ndarray:
    order = np.asarray(row_mapping)
    if order.shape != (num_rows,) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError(f"RS4 row mapping must be an integer vector of shape ({num_rows},)")
    order = order.astype(np.int64, copy=False)
    identity = np.arange(num_rows, dtype=np.int64)
    if not np.array_equal(np.sort(order), identity) or np.any(order == identity):
        raise ValueError("RS4 row mapping must be a complete bijection with no fixed points")
    return order


def apply_training_corruption(
    carrier: np.ndarray,
    *,
    kind: TrainingKind,
    row_mapping: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an explicit T4/RS4/Z4 state after standardization, without mutation."""
    value = _validated_standardized_t4(carrier)
    if kind == "t4":
        if row_mapping is not None:
            raise ValueError("row_mapping is only valid for RS4")
        return value.copy()
    if kind == "z4":
        if row_mapping is not None:
            raise ValueError("row_mapping is only valid for RS4")
        return np.zeros_like(value)
    if kind == "rs4":
        if row_mapping is None:
            raise ValueError("RS4 requires the fixed run-seed/source-session row mapping")
        order = _validated_row_mapping(row_mapping, num_rows=value.shape[0])
        return value[order].copy()
    raise ValueError(f"unsupported B2 training kind {kind!r}")


def apply_scheduled_training_corruption(
    carrier: np.ndarray,
    *,
    run_seed: int,
    source_session: str,
    logical_epoch: int,
) -> np.ndarray:
    """Apply the session-wide v2 state selected for one logical epoch."""
    kind = training_kind_for_epoch(
        run_seed=run_seed,
        source_session=source_session,
        logical_epoch=logical_epoch,
    )
    mapping = None
    if kind == "rs4":
        mapping = rs4_row_mapping(
            carrier.shape[0],
            run_seed=run_seed,
            source_session=source_session,
        )
    return apply_training_corruption(carrier, kind=kind, row_mapping=mapping)


def _same_angle(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-7) -> np.ndarray:
    difference = np.angle(np.exp(1j * (left - right)))
    return np.abs(difference) <= atol


def derange_finite_angles(angles: np.ndarray, *, seed: int) -> np.ndarray:
    """Preserve the finite label multiset while changing every finite label.

    This constructs the proposed LS4 diagnostic before the ordinary T4 fit.
    Missing/centre trials remain missing.  It fails closed when an exact
    condition-label derangement is impossible.
    """
    value = np.asarray(angles, dtype=np.float64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(value))
    if finite.size < 2:
        raise ValueError("LS4 requires at least two finite direction labels")
    original = value[finite]
    generator = np.random.RandomState(int(seed))
    candidates = generator.permutation(np.arange(1, finite.size, dtype=np.int64))
    for shift in candidates:
        candidate = np.roll(original, int(shift))
        if not np.any(_same_angle(candidate, original)):
            output = value.copy()
            output[finite] = candidate
            return output
    raise ValueError(
        "no complete finite-label derangement exists; LS4 is ineligible for this session"
    )
