"""Pure primitives for the M2 T4 activity-budget screen."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import plan


class ScreenError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


@dataclasses.dataclass(frozen=True)
class CellSpec:
    name: str
    carrier_budget: int
    activity_budget: int

    def __post_init__(self) -> None:
        require(self.carrier_budget in plan.BUDGETS, "unsupported carrier budget")
        require(self.activity_budget in plan.BUDGETS, "unsupported activity budget")
        require(self.activity_budget >= self.carrier_budget, "activity cannot precede carrier support")
        require(
            self.name
            == (
                f"ridge_static_m{self.carrier_budget}"
                if self.activity_budget == self.carrier_budget
                else f"ridge_activity30_m{self.carrier_budget}"
            ),
            "cell name/semantics drift",
        )


CELL_SPECS = (
    CellSpec("ridge_static_m30", 30, 30),
    CellSpec("ridge_static_m10", 10, 10),
    CellSpec("ridge_activity30_m10", 10, 30),
    CellSpec("ridge_static_m4", 4, 4),
    CellSpec("ridge_activity30_m4", 4, 30),
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def select_activity_rows(
    calibration: np.ndarray,
    *,
    selected_indices: Sequence[int],
    activity_budget: int,
) -> np.ndarray:
    values = np.asarray(calibration, dtype=np.float32)
    require(values.ndim == 3, "calibration must be [trials,time,channels]")
    require(values.shape[0] >= plan.ACTIVITY_HORIZON, "fewer than 30 calibration trials")
    require(values.shape[1:] == (plan.TRIAL_LENGTH, plan.CHANNELS), "M2 calibration shape drift")
    require(activity_budget in plan.BUDGETS, "unknown activity budget")
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    require(selected.size in plan.BUDGETS, "selected support cardinality drift")
    require(np.unique(selected).size == selected.size, "selected support contains duplicates")
    require(int(selected.min()) >= 0 and int(selected.max()) < plan.ACTIVITY_HORIZON,
            "selected support leaves first-30 causal pool")
    require(activity_budget >= selected.size, "activity budget precedes carrier support")
    if activity_budget == selected.size:
        return np.ascontiguousarray(values[selected])
    require(activity_budget == plan.ACTIVITY_HORIZON, "only selected-M or first-30 activity is legal")
    return np.ascontiguousarray(values[:plan.ACTIVITY_HORIZON])


def select_common_post30_window_starts(
    session_window_starts: Sequence[int],
    trial_starts: Sequence[int],
) -> np.ndarray:
    windows = np.asarray(session_window_starts, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_starts, dtype=np.int64).reshape(-1)
    require(trials.size > plan.ACTIVITY_HORIZON, "within session lacks trial-30 boundary")
    require(np.all(np.diff(windows) > 0), "window starts must be unique and increasing")
    require(np.all(np.diff(trials) > 0), "trial starts must be unique and increasing")
    selected = np.ascontiguousarray(windows[windows >= trials[plan.ACTIVITY_HORIZON]])
    require(selected.size > 0, "within session has no post-30 query windows")
    return selected


def variance_weighted_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    require(y.ndim == 2 and y.shape[1] == 2 and p.shape == y.shape, "R2 shape drift")
    require(y.shape[0] >= 2, "R2 requires at least two rows")
    require(np.isfinite(y).all() and np.isfinite(p).all(), "R2 nonfinite")
    residual = np.square(y - p).sum(axis=0, dtype=np.float64)
    centered = y - y.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(axis=0, dtype=np.float64)
    denominator = float(total.sum(dtype=np.float64))
    require(denominator > 0.0, "R2 target variance is zero")
    return float(1.0 - residual.sum(dtype=np.float64) / denominator)


def summarize_sessions(values: Mapping[str, float]) -> dict[str, object]:
    require(bool(values), "empty session map")
    ordered = {key: float(values[key]) for key in sorted(values)}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    require(np.isfinite(array).all(), "session R2 nonfinite")
    return {
        "session_count": int(array.size),
        "equal_session_mean": float(array.mean()),
        "equal_session_median": float(np.median(array)),
        "per_session_r2": ordered,
    }


def paired_contrast(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, object]:
    require(set(candidate) == set(reference) and bool(candidate), "paired session set mismatch")
    differences = {
        key: float(candidate[key] - reference[key]) for key in sorted(candidate)
    }
    values = np.asarray(list(differences.values()), dtype=np.float64)
    return {
        "candidate_minus_reference_mean": float(values.mean()),
        "candidate_minus_reference_median": float(np.median(values)),
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "session_count": int(values.size),
        "per_session_delta": differences,
    }
