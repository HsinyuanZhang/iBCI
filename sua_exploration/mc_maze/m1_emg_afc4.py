"""Source-frozen EMG low-rank analytic functional carriers for native M1.

This module is intentionally independent of the streaming data module and of
every formal/EvalAI endpoint.  It specifies the small, closed-form carrier
used by the M1 AFC4 feasibility audit:

1. source sessions determine EMG mean, scale, PCA order, and PCA signs;
2. a target session only projects its *calibration* EMG into that frozen basis;
3. per-unit neural rates are fit with a fixed-ridge affine encoding model; and
4. the result is exposed through exactly four coordinates.

The q=2 interface is ``[w1, w2, ||W||, b]``.  The q=3 interface is
``[w1, w2, w3, b]``.  Neither constructor receives future trials, a decoder,
or any behavioral test target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SEMANTICS_VERSION = "m1_source_frozen_emg_afc4_v1"
SUPPORT_TRIALS = 10
RIDGE_ALPHA = 1.0
EPS = 1.0e-12


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty 2-D matrix, got {array.shape}")
    return array


@dataclass(frozen=True)
class SourceFrozenEMGBasis:
    """Source-only EMG standardizer and deterministically oriented PCA basis."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray  # [q, emg_dims], decreasing source singular-value order.
    singular_values: np.ndarray
    source_energy_explained: np.ndarray
    sign_anchor_indices: np.ndarray

    @property
    def q(self) -> int:
        return int(self.components.shape[0])

    def transform(self, emg: np.ndarray) -> np.ndarray:
        values = _matrix(emg, name="emg")
        if values.shape[1] != self.mean.size:
            raise ValueError("EMG dimension differs from source-frozen basis")
        return ((values - self.mean) / self.scale) @ self.components.T

    def reconstruction_energy_explained(self, emg: np.ndarray) -> float:
        values = _matrix(emg, name="emg")
        standardized = (values - self.mean) / self.scale
        scores = standardized @ self.components.T
        residual = standardized - scores @ self.components
        denominator = float(np.square(standardized).sum())
        if denominator <= EPS:
            raise ValueError("EMG has zero source-standardized energy")
        return float(1.0 - np.square(residual).sum() / denominator)


def fit_source_frozen_emg_basis(source_emg: Iterable[np.ndarray], *, q: int) -> SourceFrozenEMGBasis:
    """Fit mean/scale/PCA exclusively from a nonempty iterable of source sessions.

    Component order is decreasing SVD singular value.  Each component is then
    multiplied by a deterministic sign: its largest-magnitude loading is made
    nonnegative, ties resolving to the lowest loading index via ``argmax``.
    This makes the basis reproducible without orienting it against target data.
    """
    matrices = tuple(_matrix(value, name=f"source_emg[{index}]") for index, value in enumerate(source_emg))
    if not matrices:
        raise ValueError("source_emg cannot be empty")
    dimensions = {matrix.shape[1] for matrix in matrices}
    if len(dimensions) != 1:
        raise ValueError("source EMG sessions have inconsistent dimensions")
    stacked = np.concatenate(matrices, axis=0)
    if not (1 <= int(q) <= min(stacked.shape)):
        raise ValueError(f"q={q} is invalid for source EMG shape {stacked.shape}")
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0, ddof=0)
    scale[scale <= EPS] = 1.0
    standardized = (stacked - mean) / scale
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[: int(q)].copy()
    anchors = np.argmax(np.abs(components), axis=1).astype(np.int64)
    for row, anchor in enumerate(anchors):
        if components[row, anchor] < 0.0:
            components[row] *= -1.0
    total_energy = float(np.square(singular_values).sum())
    explained = np.square(singular_values[: int(q)]) / total_energy
    return SourceFrozenEMGBasis(
        mean=mean,
        scale=scale,
        components=components,
        singular_values=singular_values[: int(q)].copy(),
        source_energy_explained=explained,
        sign_anchor_indices=anchors,
    )


def affine_design_report(scores: np.ndarray) -> dict[str, object]:
    """Return rank/conditioning for the exact affine M-trial calibration design."""
    z = _matrix(scores, name="scores")
    design = np.column_stack((np.ones(len(z), dtype=np.float64), z))
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = float(max(design.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int(np.sum(singular > tolerance))
    condition = float(np.inf) if rank < design.shape[1] else float(singular[0] / singular[-1])
    return {
        "n_trials": int(design.shape[0]),
        "n_columns": int(design.shape[1]),
        "rank": rank,
        "full_rank": bool(rank == design.shape[1]),
        "condition_number": condition,
        "singular_values": singular.tolist(),
        "rank_tolerance": tolerance,
    }


def fit_unit_encoding(scores: np.ndarray, neural_rates: np.ndarray, *, alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Fit raw-rate ``r_i = b_i + w_i^T z`` with fixed ridge on w, not b."""
    z = _matrix(scores, name="scores")
    rates = _matrix(neural_rates, name="neural_rates")
    if z.shape[0] != rates.shape[0]:
        raise ValueError("scores and neural_rates must have equal trial counts")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    design = np.column_stack((np.ones(len(z), dtype=np.float64), z))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ rates)
    return beta[1:].T.copy(), beta[0].copy()


def afc4_from_encoding(weights: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Compress a q=2/q=3 encoding fit into the frozen four-coordinate API."""
    w = _matrix(weights, name="weights")
    b = np.asarray(intercept, dtype=np.float64)
    if b.ndim != 1 or b.size != w.shape[0] or not np.isfinite(b).all():
        raise ValueError("intercept must be finite [units] and match weights")
    if w.shape[1] == 2:
        return np.column_stack((w[:, 0], w[:, 1], np.linalg.norm(w, axis=1), b))
    if w.shape[1] == 3:
        return np.column_stack((w[:, 0], w[:, 1], w[:, 2], b))
    raise ValueError("AFC4 supports exactly q=2 or q=3")


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= EPS else float(x.dot(y) / denominator)


def deterministic_split_half_reliability(scores: np.ndarray, neural_rates: np.ndarray, *, alpha: float = RIDGE_ALPHA) -> dict[str, object]:
    """Odd/even support-trial coefficient reliability; no future trial is read."""
    z = _matrix(scores, name="scores")
    rates = _matrix(neural_rates, name="neural_rates")
    if len(z) != SUPPORT_TRIALS or len(rates) != SUPPORT_TRIALS:
        raise ValueError(f"split-half reliability requires exactly {SUPPORT_TRIALS} support trials")
    even = np.arange(0, SUPPORT_TRIALS, 2, dtype=np.int64)
    odd = np.arange(1, SUPPORT_TRIALS, 2, dtype=np.int64)
    w_even, b_even = fit_unit_encoding(z[even], rates[even], alpha=alpha)
    w_odd, b_odd = fit_unit_encoding(z[odd], rates[odd], alpha=alpha)
    return {
        "split": {"even_trial_indices": even.tolist(), "odd_trial_indices": odd.tolist()},
        "even_design": affine_design_report(z[even]),
        "odd_design": affine_design_report(z[odd]),
        "weight_flattened_pearson": _pearson(w_even, w_odd),
        "intercept_pearson": _pearson(b_even, b_odd),
        "afc4_flattened_pearson": _pearson(
            afc4_from_encoding(w_even, b_even), afc4_from_encoding(w_odd, b_odd)
        ),
    }
