"""Deterministic chronological calibration-prefix schedule (no RNG).

The PIT-M2 operator is pure integer arithmetic on the global student-training
forward counter, in an isolated arithmetic/hash domain: no Python, NumPy or
Torch RNG is consumed anywhere in this module (proven by monkeypatch tests in
``tests/test_pit_m2_v1.py``, the cal_aug_v1 law carried verbatim).
"""
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
             and all(type(item) is int and 1 <= item <= plan.MAX_CALIBRATION_TRIALS
                     for item in cycle),
             "pit m2 prefix cycle is frozen to (10, 5, 2)")
    return tuple(int(item) for item in cycle)


def m_at(index: int, cycle=plan.CYCLE) -> int:
    _require(type(index) is int and index >= 0, "prefix index drift")
    validated = validate_cycle(cycle)
    return validated[index % len(validated)]


def effective_prefix_length(scheduled_m: int, available: int) -> int:
    _require(type(scheduled_m) is int and type(available) is int
             and 1 <= scheduled_m <= plan.MAX_CALIBRATION_TRIALS
             and 1 <= available <= plan.MAX_CALIBRATION_TRIALS,
             "prefix length domain drift")
    return min(scheduled_m, available)


def sequence(count: int, cycle=plan.CYCLE) -> list[int]:
    _require(type(count) is int and count >= 0, "prefix sequence count drift")
    return [m_at(index, cycle) for index in range(count)]


def sequence_digest(values) -> str:
    """Digest of an ordered value stream (ints, floats by repr, or sha strings)."""
    encoded = ",".join(str(item) for item in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def int_list_digest(values) -> str:
    """Digest of a nested integer stream (e.g. sampler batched_indices)."""
    encoded = plan.canonical_json_bytes([[int(item) for item in batch] for batch in values])
    return hashlib.sha256(encoded).hexdigest()


def tensor_digest(value) -> str:
    """sha256 over dtype/shape/bytes of a finite real tensor or array."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    _require(array.ndim >= 1 and np.isfinite(array).all(), "tensor digest needs finite nonscalar")
    header = plan.canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def batch_digest(sample_ids) -> str:
    """Digest of one batch's ordered sample identities (session@window-start)."""
    _require(isinstance(sample_ids, (list, tuple)) and sample_ids
             and all(isinstance(item, str) for item in sample_ids),
             "batch digest needs ordered sample ids")
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def rng_state_digest(state) -> str:
    import hashlib

    _require(isinstance(state, tuple) and len(state) == 3 and isinstance(state[1], tuple),
             "python rng state topology drift")
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def torch_rng_state_digest(state) -> str:
    """Digest of torch.get_rng_state() bytes (a uint8 tensor)."""
    import hashlib

    import numpy as np

    array = np.ascontiguousarray(np.asarray(state, dtype=np.uint8).reshape(-1))
    _require(array.size > 0, "torch rng state digest needs bytes")
    return hashlib.sha256(array.tobytes()).hexdigest()


__all__ = (
    "ScheduleError", "validate_cycle", "m_at", "effective_prefix_length", "sequence",
    "sequence_digest", "int_list_digest", "tensor_digest", "batch_digest",
    "rng_state_digest", "torch_rng_state_digest",
)
