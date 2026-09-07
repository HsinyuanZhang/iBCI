"""CAL-AUG operator schedule: pure integer cycle arithmetic and digests.

The C1 prefix operator is DETERMINISTIC integer arithmetic on the global
training-forward counter (work order section 1):

    M_step = cycle[global_training_forward % len(cycle)]
    cycle  = (30, 10, 4)

It lives in an isolated arithmetic/hash domain: no Python, NumPy or Torch RNG
is consumed anywhere in this module (proven by monkeypatch tests), so the
Cell-D dynamic-dropout ``p`` sequence is untouched.

Digest helpers here cover every equality quantity the smoke and the full runs
record: the cycle itself, the realized prefix-length sequence, the visible
calibration rows per step, the sampler batch order + session names, the
sampled-p stream, and the query-neural bytes entering the model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from . import plan


class ScheduleError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleError(message)


# ---------------------------------------------------------------------------
# cycle parsing and arithmetic
# ---------------------------------------------------------------------------


def validate_cycle(cycle: Sequence[int]) -> tuple[int, ...]:
    """Fail closed unless every prefix length is a legal chronological prefix."""
    values = tuple(int(m) for m in cycle)
    _require(len(values) > 0, "prefix cycle must be non-empty")
    for m in values:
        _require(
            1 <= m <= plan.MAX_CALIBRATION_TRIALS,
            f"prefix length {m} outside [1, {plan.MAX_CALIBRATION_TRIALS}]",
        )
    return values


def parse_cycle(text: str) -> tuple[int, ...]:
    """Parse the CLI literal (``"30,10,4"``) into a validated cycle tuple."""
    _require(isinstance(text, str) and text.strip(), "cycle literal must be a non-empty string")
    parts = [part.strip() for part in text.split(",")]
    _require(all(parts), f"malformed cycle literal: {text!r}")
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise ScheduleError(f"non-integer prefix length in cycle literal {text!r}") from error
    return validate_cycle(values)


def m_at(index: int, cycle: Sequence[int]) -> int:
    """The scheduled prefix length at global training-forward ``index``."""
    values = validate_cycle(cycle)
    _require(int(index) >= 0, "training-forward index must be non-negative")
    return values[int(index) % len(values)]


def prefix_sequence(n_steps: int, cycle: Sequence[int]) -> list[int]:
    """The first ``n_steps`` scheduled prefix lengths (the exact M list)."""
    values = validate_cycle(cycle)
    _require(int(n_steps) >= 0, "n_steps must be non-negative")
    return [values[i % len(values)] for i in range(int(n_steps))]


def effective_prefix_length(scheduled_m: int, available_trials: int) -> int:
    """What the model actually sees: ``min(scheduled, the block's trial count)``.

    The training calibration block is always the full 30-trial block, so for
    T0 (operator disabled) this returns the full block width unchanged.
    """
    _require(int(available_trials) >= 1, "available_trials must be positive")
    return min(int(scheduled_m), int(available_trials))


# ---------------------------------------------------------------------------
# digests (deterministic bytes; never RNG)
# ---------------------------------------------------------------------------


def _json_digest(payload: object, domain: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"cal_aug_v1:{domain}:".encode("utf-8") + body.encode("utf-8")).hexdigest()


def cycle_digest(cycle: Sequence[int]) -> str:
    return _json_digest(list(validate_cycle(cycle)), "cycle")


def sequence_digest(sequence: Sequence[int]) -> str:
    return _json_digest([int(m) for m in sequence], "prefix_sequence")


def float_stream_digest(values: Sequence[float]) -> str:
    """Digest of a sampled-p prefix: full-precision repr, order-sensitive."""
    return _json_digest([repr(float(v)) for v in values], "sampled_p_stream")


def int_list_digest(values: Sequence[Sequence[int]]) -> str:
    """Digest of ``sampler.batched_indices[:N]`` (nested int lists)."""
    return _json_digest([[int(i) for i in batch] for batch in values], "batched_indices")


def string_list_digest(values: Sequence[str]) -> str:
    return _json_digest([str(v) for v in values], "session_order")


def mapping_digest(mapping: dict) -> str:
    return _json_digest(mapping, "mapping")


def tensor_digest(tensor) -> str:
    """Byte digest of a tensor slice: dtype + shape + IEEE-normalized bytes.

    The same law as ``tfpd_lane.arm_common.tensor_sha256`` (signed-zero
    normalized: ``-0.0 + 0.0 == +0.0``, identity otherwise).  Pure hashing —
    no RNG, no autograd, no in-place mutation of the input.
    """
    import torch

    digest = hashlib.sha256()
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.is_floating_point():
        flat = flat + 0
    digest.update(str(flat.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    if flat.numel():
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def visible_slice_digest(visible_calib) -> str:
    """Digest of the calib rows the model is about to see at one step."""
    return tensor_digest(visible_calib)


def batch_order_digest(batched_indices: Sequence[Sequence[int]], session_names: Sequence[str]):
    """(indices digest, session-order digest, combined digest) for the smoke."""
    indices = int_list_digest(batched_indices)
    sessions = string_list_digest(session_names)
    combined = _json_digest({"indices": indices, "sessions": sessions}, "batch_order")
    return {
        "batched_indices_sha256": indices,
        "session_order_sha256": sessions,
        "combined_sha256": combined,
        "n_batches": len(list(batched_indices)),
    }
