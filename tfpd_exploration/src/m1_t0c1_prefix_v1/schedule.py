"""Deterministic chronological calibration-prefix schedule (no RNG)."""
from __future__ import annotations

import hashlib

from . import plan


class ScheduleError(RuntimeError):
    """Fail closed for prefix-cycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleError(message)


def validate_cycle(cycle) -> tuple[int, ...]:
    _require(isinstance(cycle, (tuple, list)) and tuple(cycle) == plan.CYCLE
             and all(type(item) is int and 1 <= item <= 10 for item in cycle),
             "m1 t0c1 prefix cycle is frozen to (10, 5, 2)")
    return tuple(int(item) for item in cycle)


def m_at(index: int, cycle=plan.CYCLE) -> int:
    _require(type(index) is int and index >= 0, "prefix index drift")
    validated = validate_cycle(cycle)
    return validated[index % len(validated)]


def effective_prefix_length(scheduled_m: int, available: int) -> int:
    _require(type(scheduled_m) is int and type(available) is int
             and 1 <= scheduled_m <= 10 and 1 <= available <= 10,
             "prefix length domain drift")
    return min(scheduled_m, available)


def sequence(count: int, cycle=plan.CYCLE) -> list[int]:
    _require(type(count) is int and count >= 0, "prefix sequence count drift")
    return [m_at(index, cycle) for index in range(count)]


def sequence_digest(values) -> str:
    encoded = ",".join(str(item) for item in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_digest(value) -> str:
    """sha256 over dtype/shape/bytes of a finite real tensor or array."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    _require(array.ndim >= 1 and np.isfinite(array).all(), "tensor digest needs finite nonscalar")
    header = plan.canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def batch_digest(sample_ids) -> str:
    """Digest of one episode's ordered row sample-id stream."""
    import hashlib

    _require(isinstance(sample_ids, (list, tuple)) and sample_ids
             and all(isinstance(item, str) for item in sample_ids),
             "batch digest needs ordered sample ids")
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def rng_state_digest(state) -> str:
    import hashlib

    _require(isinstance(state, tuple) and len(state) == 3 and isinstance(state[1], tuple),
             "python rng state topology drift")
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


__all__ = (
    "ScheduleError", "validate_cycle", "m_at", "effective_prefix_length", "sequence",
    "sequence_digest", "tensor_digest", "batch_digest", "rng_state_digest",
)
