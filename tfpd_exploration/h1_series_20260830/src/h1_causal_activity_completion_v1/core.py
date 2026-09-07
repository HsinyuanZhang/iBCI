"""Pure activity-state primitives for H1-CAC V1."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .plan import IDENTITY_LENGTH, MAX_MEMBERS, UNITS


class H1CACError(RuntimeError):
    """Fail closed on an H1-CAC state or causal-law drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1CACError(message)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    require(array.ndim >= 1 and np.isfinite(array).all(), "digest input must be finite and non-scalar")
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def resample_activity_member(activity: Any, *, output_length: int = IDENTITY_LENGTH) -> np.ndarray:
    """Cubic position resampling used by true-trial and fixed-chunk members."""

    from scipy.interpolate import interp1d

    value = np.ascontiguousarray(np.asarray(activity), dtype=np.float32)
    require(value.ndim == 2 and value.shape[0] >= 4 and value.shape[1] == UNITS,
            "H1 activity member must be [T>=4,176]")
    require(np.isfinite(value).all() and type(output_length) is int and output_length >= 4,
            "H1 activity member/output length drift")
    original = np.linspace(0.0, 1.0, value.shape[0])
    target = np.linspace(0.0, 1.0, output_length)
    result = interp1d(original, value, axis=0, kind="cubic", fill_value="extrapolate")(target).astype(np.float32)
    require(result.shape == (output_length, UNITS) and np.isfinite(result).all(),
            "resampled H1 activity member drift")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class CandidateSchedule:
    """Accepted members and complete audit trace for one causal chunk law."""

    members: tuple[np.ndarray, ...]
    completion_end_exclusive: tuple[int, ...]
    candidate_end_exclusive: tuple[int, ...]
    candidate_energies: tuple[float, ...]
    accepted: tuple[bool, ...]

    def accepted_before(self, endpoint_inclusive: int) -> int:
        require(type(endpoint_inclusive) is int and endpoint_inclusive >= 0, "endpoint drift")
        # A chunk ending at exclusive coordinate e was committed after decoding
        # bin e-1, and may first affect a prediction whose endpoint is e.
        return bisect_right(self.completion_end_exclusive, endpoint_inclusive)

    def receipt(self) -> dict[str, Any]:
        return {
            "accepted_members": len(self.members),
            "candidate_count_until_freeze": len(self.accepted),
            "accepted_bits": [bool(value) for value in self.accepted],
            "candidate_end_exclusive": list(self.candidate_end_exclusive),
            "accepted_completion_end_exclusive": list(self.completion_end_exclusive),
            "candidate_energies": list(self.candidate_energies),
            "candidate_energies_sha256": array_sha256(np.asarray(self.candidate_energies, dtype=np.float64)),
            "accepted_bits_sha256": array_sha256(np.asarray(self.accepted, dtype=np.uint8)),
            "commit_bins_sha256": array_sha256(np.asarray(self.completion_end_exclusive, dtype=np.int64)),
        }


def build_chunk_schedule(
    neural: Any,
    *,
    origin: int,
    chunk_length: int,
    initial_members: int,
    energy_gated: bool,
) -> CandidateSchedule:
    """Build the frozen C-FIX7 or D-EMED7 causal candidate stream."""

    values = np.ascontiguousarray(np.asarray(neural), dtype=np.float32)
    require(values.ndim == 2 and values.shape[1] == UNITS and np.isfinite(values).all(),
            "H1 chunk stream must be finite [T,176]")
    require(type(origin) is int and 0 <= origin <= values.shape[0], "chunk origin drift")
    require(type(chunk_length) is int and chunk_length >= 4, "chunk length drift")
    require(type(initial_members) is int and 1 <= initial_members < MAX_MEMBERS,
            "initial member count drift")
    required = MAX_MEMBERS - initial_members
    members: list[np.ndarray] = []
    completions: list[int] = []
    candidate_ends: list[int] = []
    energies: list[float] = []
    accepted: list[bool] = []
    start = origin
    while start + chunk_length <= values.shape[0] and len(members) < required:
        end = start + chunk_length
        chunk = values[start:end]
        energy = float(np.asarray(chunk, dtype=np.float64).mean(dtype=np.float64))
        require(np.isfinite(energy), "candidate energy is nonfinite")
        take = not energy_gated or not energies or energy >= float(np.median(np.asarray(energies, dtype=np.float64)))
        candidate_ends.append(end)
        energies.append(energy)
        accepted.append(bool(take))
        if take:
            members.append(resample_activity_member(chunk))
            completions.append(end)
        start = end
    require(len(candidate_ends) == len(energies) == len(accepted), "candidate trace length drift")
    require(all(left < right for left, right in zip(completions, completions[1:])),
            "accepted completion order drift")
    return CandidateSchedule(
        members=tuple(members),
        completion_end_exclusive=tuple(completions),
        candidate_end_exclusive=tuple(candidate_ends),
        candidate_energies=tuple(energies),
        accepted=tuple(accepted),
    )


def pool_cardinality(schedule: CandidateSchedule, *, initial_members: int, endpoint_inclusive: int) -> int:
    value = initial_members + schedule.accepted_before(endpoint_inclusive)
    require(initial_members <= value <= MAX_MEMBERS, "activity cardinality drift")
    return value


__all__ = (
    "CandidateSchedule", "H1CACError", "array_sha256", "build_chunk_schedule",
    "pool_cardinality", "require", "resample_activity_member",
)

