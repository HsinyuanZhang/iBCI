"""Pure contracts and numerical helpers for the M2 comparator matrix."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from . import plan


class ComparatorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparatorError(message)


@dataclasses.dataclass(frozen=True)
class CellSpec:
    name: str
    family: str
    budget: int
    support_selection: str
    supervision: str
    source: str = "new"

    def __post_init__(self) -> None:
        require(self.budget in plan.BUDGETS, "unsupported budget")
        require(self.family in {"spint", "t4_mean", "t4_ols", "t4_ridge", "direct_ridge", "pv"},
                "unsupported comparator family")
        require(self.support_selection in {"chronological", "matched_doptimal", "doptimal_or_chronological"},
                "unsupported support selection")
        require(self.source in {"new", "reused_parent"}, "unsupported cell source")


CELL_SPECS = (
    CellSpec("spint_chronological_m30", "spint", 30, "chronological", "activity_only"),
    CellSpec("spint_chronological_m10", "spint", 10, "chronological", "activity_only"),
    CellSpec("spint_chronological_m4", "spint", 4, "chronological", "activity_only"),
    CellSpec("spint_matched_doptimal_m4", "spint", 4, "matched_doptimal", "activity_plus_selection_labels"),
    CellSpec("t4_mean_side_m30", "t4_mean", 30, "doptimal_or_chronological", "activity_only_mean_side_intervention"),
    CellSpec("t4_mean_side_m10", "t4_mean", 10, "doptimal_or_chronological", "activity_only_mean_side_intervention"),
    CellSpec("t4_mean_side_m4", "t4_mean", 4, "doptimal_or_chronological", "activity_plus_selection_labels_mean_side_intervention"),
    CellSpec("t4_ols_static_m30", "t4_ols", 30, "doptimal_or_chronological", "trial_direction_labels"),
    CellSpec("t4_ols_static_m10", "t4_ols", 10, "doptimal_or_chronological", "trial_direction_labels"),
    CellSpec("t4_ols_static_m4", "t4_ols", 4, "doptimal_or_chronological", "trial_direction_labels"),
    CellSpec("t4_ridge_static_m30", "t4_ridge", 30, "doptimal_or_chronological", "trial_direction_labels", "reused_parent"),
    CellSpec("t4_ridge_static_m10", "t4_ridge", 10, "doptimal_or_chronological", "trial_direction_labels", "reused_parent"),
    CellSpec("t4_ridge_activity30_m10", "t4_ridge", 10, "doptimal_or_chronological", "trial_direction_labels_plus_unlabelled_activity", "reused_parent"),
    CellSpec("t4_ridge_static_m4", "t4_ridge", 4, "doptimal_or_chronological", "trial_direction_labels", "reused_parent"),
    CellSpec("t4_ridge_activity30_m4", "t4_ridge", 4, "doptimal_or_chronological", "trial_direction_labels_plus_unlabelled_activity", "reused_parent"),
    CellSpec("direct_ridge_m30", "direct_ridge", 30, "doptimal_or_chronological", "dense_velocity_labels"),
    CellSpec("direct_ridge_m10", "direct_ridge", 10, "doptimal_or_chronological", "dense_velocity_labels"),
    CellSpec("direct_ridge_m4", "direct_ridge", 4, "doptimal_or_chronological", "dense_velocity_labels"),
    CellSpec("population_vector_m30", "pv", 30, "doptimal_or_chronological", "trial_direction_plus_dense_velocity_labels"),
    CellSpec("population_vector_m10", "pv", 10, "doptimal_or_chronological", "trial_direction_plus_dense_velocity_labels"),
    CellSpec("population_vector_m4", "pv", 4, "doptimal_or_chronological", "trial_direction_plus_dense_velocity_labels"),
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


def file_sha256(path: object) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chronological_indices(budget: int) -> np.ndarray:
    require(budget in plan.BUDGETS, "unknown chronological budget")
    return np.arange(budget, dtype=np.int64)


def dense_support_target_bins(
    trial_starts: Sequence[int], total_bins: int, selected_indices: Sequence[int]
) -> np.ndarray:
    """Return W50 endpoints whose complete histories stay inside selected trials.

    A selected trial shorter than W50 contributes no dense-label row.  This is
    different from the B3S identity path, where the production data module pads
    variable-length trials: padding has no honest velocity label and therefore
    must never be promoted to dense Ridge/PV supervision.
    """
    starts = np.asarray(trial_starts, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    require(starts.size >= plan.ACTIVITY_HORIZON, "fewer than 30 support trials")
    require(np.all(np.diff(starts) > 0), "support trial starts are not increasing")
    require(total_bins > int(starts[-1]), "invalid support time axis")
    require(selected.size in plan.BUDGETS and np.unique(selected).size == selected.size,
            "selected support cardinality/uniqueness drift")
    require(int(selected.min()) >= 0 and int(selected.max()) < plan.ACTIVITY_HORIZON,
            "selected support leaves first-30 pool")
    rows: list[np.ndarray] = []
    for index in selected:
        start = int(starts[index])
        stop = int(starts[index + 1]) if int(index) + 1 < starts.size else int(total_bins)
        if stop - start < plan.WINDOW_SIZE:
            continue
        rows.append(np.arange(start + plan.WINDOW_SIZE - 1, stop, dtype=np.int64))
    require(bool(rows), "selected support has no within-trial W50 endpoint")
    result = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
    require(result.size >= 3 and np.unique(result).size == result.size, "dense support rows invalid")
    return result


def materialize_windows(neural: np.ndarray, target_bins: Sequence[int]) -> np.ndarray:
    values = np.asarray(neural, dtype=np.float32)
    targets = np.asarray(target_bins, dtype=np.int64).reshape(-1)
    require(values.ndim == 2 and values.shape[1] == plan.CHANNELS, "neural shape drift")
    require(targets.size > 0 and int(targets.min()) >= plan.WINDOW_SIZE - 1,
            "window endpoint lacks W50 history")
    require(int(targets.max()) < values.shape[0], "window endpoint outside neural data")
    offsets = np.arange(-(plan.WINDOW_SIZE - 1), 1, dtype=np.int64)
    windows = values[targets[:, None] + offsets[None, :]].reshape(targets.size, -1)
    require(np.isfinite(windows).all(), "window features are nonfinite")
    return np.ascontiguousarray(windows, dtype=np.float32)


def variance_weighted_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    require(y.ndim == 2 and y.shape[1] == plan.OUTPUT_DIM and p.shape == y.shape,
            "R2 shape drift")
    require(y.shape[0] >= 2 and np.isfinite(y).all() and np.isfinite(p).all(), "R2 input invalid")
    residual = np.square(y - p).sum(axis=0, dtype=np.float64)
    centered = y - y.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(axis=0, dtype=np.float64)
    require(float(total.sum()) > 0.0, "R2 target variance is zero")
    return float(1.0 - residual.sum(dtype=np.float64) / total.sum(dtype=np.float64))


def summarize_sessions(values: Mapping[str, float]) -> dict[str, object]:
    require(bool(values), "empty session summary")
    ordered = {name: float(values[name]) for name in sorted(values)}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    require(np.isfinite(array).all(), "session score is nonfinite")
    return {
        "session_count": int(array.size),
        "equal_session_mean": float(array.mean()),
        "equal_session_median": float(np.median(array)),
        "per_session_r2": ordered,
    }


def paired_contrast(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, object]:
    require(set(candidate) == set(reference) and bool(candidate), "paired session set mismatch")
    deltas = {name: float(candidate[name] - reference[name]) for name in sorted(candidate)}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    return {
        "candidate_minus_reference_mean": float(values.mean()),
        "candidate_minus_reference_median": float(np.median(values)),
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "session_count": int(values.size),
        "per_session_delta": deltas,
    }
