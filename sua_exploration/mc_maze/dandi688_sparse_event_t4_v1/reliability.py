"""Source-only Stage-0 reliability calculations, isolated from candidate tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import plan
from .core import require


@dataclass(frozen=True)
class SourceReliabilityAudit:
    """References must live in this audit-only container, never model handoff."""
    split_half: Mapping[str, np.ndarray]
    deployment_reference: Mapping[str, np.ndarray]


def assert_audit_is_disjoint(candidate_arrays: Sequence[np.ndarray], audit: SourceReliabilityAudit) -> None:
    for candidate in candidate_arrays:
        for table in (audit.split_half, audit.deployment_reference):
            for reference in table.values():
                require(not np.shares_memory(np.asarray(candidate), np.asarray(reference)), "candidate and reliability namespaces share memory")


def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape and left.ndim == 2, "reliability alignment")
    values = np.full(left.shape[1], np.nan, dtype=np.float64)
    for column in range(left.shape[1]):
        x, y = left[:, column], right[:, column]
        if np.isfinite(x).all() and np.isfinite(y).all() and x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
            values[column] = np.corrcoef(x, y)[0, 1]
    return values


def aggregate_column_reliability(session_correlations: Mapping[str, Sequence[float]], *, gate: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for column, correlations in session_correlations.items():
        values = np.asarray(correlations, dtype=np.float64)
        finite = values[np.isfinite(values)]
        clipped = np.clip(finite, -0.999999, 0.999999)
        fisher = float(np.tanh(np.arctanh(clipped).mean())) if clipped.size else float("nan")
        median = float(np.median(finite)) if finite.size else float("nan")
        positive = int(np.count_nonzero(finite > 0.0))
        if gate == "split_half":
            passed = bool(finite.size >= plan.SPLIT_HALF_MIN_FINITE_SESSIONS and fisher >= plan.SPLIT_HALF_FISHER_Z_MIN and median >= plan.SPLIT_HALF_MEDIAN_MIN and positive >= plan.SPLIT_HALF_POSITIVE_MIN)
        elif gate == "reference":
            passed = bool(fisher >= plan.REFERENCE_FISHER_Z_MIN and median >= plan.REFERENCE_MEDIAN_MIN and positive >= plan.REFERENCE_POSITIVE_MIN)
        else:
            raise ValueError(f"unknown gate {gate!r}")
        result[column] = {"finite_session_count": int(finite.size), "fisher_z_aggregate_r": fisher, "median_raw_r": median, "positive_session_count": positive, "passed": passed}
    return result


def retention_mask(split_half: Mapping[str, Mapping[str, object]], reference: Mapping[str, Mapping[str, object]]) -> tuple[bool, bool, bool, bool]:
    require(tuple(split_half) == tuple(plan.PROFILE_COLUMN_NAMES) == tuple(reference), "reliability column order drift")
    return tuple(bool(split_half[name]["passed"]) and bool(reference[name]["passed"]) for name in plan.PROFILE_COLUMN_NAMES)  # type: ignore[return-value]


def stage0_decision(*, estimator_a_pass: bool, estimator_c_pass: bool, mask: Sequence[bool]) -> dict[str, object]:
    require(len(mask) == plan.PROFILE_DIM, "reliability mask width")
    return {"estimator_route": "OPEN" if estimator_a_pass and estimator_c_pass else "CLOSED", "film_route": "OPEN" if any(mask) else "CLOSED", "reliability_mask": [bool(value) for value in mask], "decoder_numbers_opened": False, "gpu_opened": False}
