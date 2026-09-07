"""Pure aggregation contracts for M2 three-seed replication."""

from __future__ import annotations

from typing import Mapping

import numpy as np


class ReplicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplicationError(message)


def summarize(values: Mapping[str, float]) -> dict[str, object]:
    require(bool(values), "empty score map")
    ordered = {key: float(values[key]) for key in sorted(values)}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    require(np.isfinite(array).all(), "nonfinite score")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "values": ordered,
    }


def paired(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, object]:
    require(set(candidate) == set(reference) and bool(candidate), "paired keys disagree")
    deltas = {key: float(candidate[key] - reference[key]) for key in sorted(candidate)}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "positive": int(np.count_nonzero(values > 0)),
        "count": int(values.size),
        "deltas": deltas,
    }
