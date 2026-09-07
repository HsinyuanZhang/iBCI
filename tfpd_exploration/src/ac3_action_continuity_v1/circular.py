"""Circular statistics for the AC3-0 screen.

Everything here is pure numpy on angles; no data access, no torch.  The
review-critical objects are:

* :func:`circular_mean` / :func:`resultant_length` -- the R-GE group ensemble
  of row R0.5 (section 23 amendment 1), including the exact behaviour on
  opposite-direction cancellation (resultant length -> 0, mean undefined);
* :func:`calibration_curve` -- the REQUIRED rho_GE credibility calibration
  output (section 23 amendment 4);
* :func:`nearest_canonical` -- snap mismatch, delegated to the frozen
  ``nearest_canonical_direction`` of the CDM-D core so the snap grid is the
  sealed 8-direction grid, never a local re-implementation.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core

from . import plan


class AC3CircularError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3CircularError(message)


def wrap_rad(angles: np.ndarray) -> np.ndarray:
    """Wrap radians into ``[-pi, pi)``."""
    values = np.asarray(angles, dtype=np.float64)
    _require(bool(np.isfinite(values).all()), "circular wrap received nonfinite angles")
    return np.mod(values + math.pi, 2.0 * math.pi) - math.pi


def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Absolute wrapped angular distance ``|wrap(a - b)|`` in ``[0, pi]``."""
    return np.abs(wrap_rad(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def circular_mean(angles: Sequence[float] | np.ndarray) -> float:
    """Mean angle of a finite nonempty sample (``atan2`` of the resultant)."""
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    _require(values.size >= 1 and bool(np.isfinite(values).all()), "circular mean needs finite nonempty angles")
    mean = math.atan2(float(np.sin(values).sum()), float(np.cos(values).sum()))
    return float(wrap_rad(mean))


def resultant_length(angles: Sequence[float] | np.ndarray) -> float:
    """Resultant length ``R`` of the unit-vector mean, in ``[0, 1]``."""
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    _require(values.size >= 1 and bool(np.isfinite(values).all()), "resultant length needs finite nonempty angles")
    size = values.size
    return float(np.hypot(np.cos(values).sum() / size, np.sin(values).sum() / size))


def group_ensemble(angles: Sequence[float]) -> dict[str, object]:
    """The R-GE construction: circular mean of the view directions + rho_GE.

    Opposite-direction cancellation is legal and *must* be visible: two views
    at ``+x`` and two at ``-x`` give ``rho_GE = 0`` and an undefined mean
    (reported as ``None`` with ``degenerate=True``), which is exactly the
    low-credibility case the calibration curve has to expose.
    """
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    _require(values.size >= 1, "group ensemble needs at least one view angle")
    rho = resultant_length(values)
    degenerate = rho <= 1.0e-12
    theta = None if degenerate else circular_mean(values)
    return {
        "theta_rad": theta,
        "rho": rho,
        "views": int(values.size),
        "degenerate": bool(degenerate),
    }


def nearest_canonical(theta_rad: float) -> tuple[int, float]:
    """Snap to the sealed 8-direction canonical grid (delegated to the core)."""
    _require(theta_rad is not None and math.isfinite(float(theta_rad)), "canonical snap needs a finite angle")
    index, distance = cdm_core.nearest_canonical_direction(float(theta_rad))
    return int(index), float(distance)


def snap_mismatch_rate(thetas: Sequence[Optional[float]], true_indices: Sequence[int]) -> dict[str, object]:
    """Fraction of estimates whose canonical snap differs from the true snap."""
    estimates = list(thetas)
    truths = list(true_indices)
    _require(len(estimates) == len(truths) and estimates, "snap mismatch needs paired nonempty sequences")
    compared = 0
    mismatch = 0
    for theta, true_index in zip(estimates, truths):
        if theta is None or not math.isfinite(float(theta)):
            continue
        index, _distance = nearest_canonical(float(theta))
        compared += 1
        if index != int(true_index):
            mismatch += 1
    return {
        "compared": compared,
        "mismatch": mismatch,
        "rate": (float(mismatch) / float(compared)) if compared else None,
    }


def mean_circular_error(
    thetas: Sequence[Optional[float]], true_thetas: Sequence[Optional[float]],
) -> dict[str, object]:
    """Mean circular error over the trials where both estimates are defined."""
    _require(len(thetas) == len(true_thetas) and thetas, "circular error needs paired nonempty sequences")
    errors: list[float] = []
    undefined = 0
    for theta, truth in zip(thetas, true_thetas):
        if theta is None or truth is None:
            undefined += 1
            continue
        errors.append(float(circular_distance(float(theta), float(truth))))
    return {
        "compared": len(errors),
        "undefined_pairs": undefined,
        "mean_rad": (float(sum(errors) / len(errors)) if errors else None),
        "median_rad": (float(np.median(errors)) if errors else None),
        "max_rad": (float(max(errors)) if errors else None),
    }


def calibration_curve(
    credibility: Sequence[float], errors_rad: Sequence[float],
    mismatch_flags: Optional[Sequence[int]] = None,
    bins: Sequence[float] = plan.METRICS["calibration_bins"],
) -> dict[str, object]:
    """Credibility vs true direction error, the REQUIRED R0.5 output.

    Bins the credibility (``rho_GE`` for R0.5, the learned resultant length for
    the learned rows) into the predeclared bins and reports, per bin, the mean
    true circular error, the mean snap-mismatch flag and the occupancy.  A
    monotone non-increasing error across bins is the calibration property the
    zero-learning credibility candidate for AC3-1 needs.
    """
    cred = np.asarray(credibility, dtype=np.float64)
    err = np.asarray(errors_rad, dtype=np.float64)
    _require(
        cred.ndim == err.ndim == 1 and cred.size == err.size and cred.size >= 1
        and bool(np.isfinite(cred).all()) and bool(np.isfinite(err).all()) and bool((cred >= 0).all()),
        "calibration curve needs paired finite credibility/error arrays",
    )
    flags: Optional[np.ndarray]
    if mismatch_flags is None:
        flags = None
    else:
        flags = np.asarray(mismatch_flags, dtype=np.float64)
        _require(flags.shape == cred.shape and bool(np.isfinite(flags).all()),
                 "calibration mismatch flags must align with the credibility array")
    edges = [float(item) for item in bins]
    _require(len(edges) >= 2 and all(left < right for left, right in zip(edges, edges[1:])),
             "calibration bins must be increasing")
    rows: list[dict[str, object]] = []
    for left, right in zip(edges, edges[1:]):
        selected = (cred >= left) & (cred < right)
        count = int(selected.sum())
        if count == 0:
            rows.append({"bin": [left, right], "count": 0, "mean_error_rad": None,
                         "mean_snap_mismatch_rate": None})
            continue
        rows.append({
            "bin": [left, right],
            "count": count,
            "mean_error_rad": float(err[selected].mean()),
            "mean_snap_mismatch_rate": (
                None if flags is None else float(flags[selected].mean())
            ),
        })
    finite_rows = [row["mean_error_rad"] for row in rows if row["mean_error_rad"] is not None]
    # Monotonicity check on consecutive nonempty bins (ties allowed).
    monotone = True
    previous: Optional[float] = None
    for row in rows:
        value = row["mean_error_rad"]
        if value is None:
            continue
        if previous is not None and value > previous + 1.0e-9:
            monotone = False
        previous = value
    return {
        "bins": rows,
        "n": int(cred.size),
        "mean_error_overall_rad": float(err.mean()) if err.size else None,
        "error_monotone_nonincreasing": bool(monotone),
        "n_nonempty_bins": len(finite_rows),
    }


def soft_prototype_readout(
    embedding: np.ndarray, prototypes: Mapping[int, np.ndarray], temperature: float = 0.1,
) -> tuple[float, np.ndarray]:
    """Label-free direction readout: similarity-weighted circular mean.

    ``prototypes`` maps a canonical index to its mean training embedding.  The
    returned angle is the circular mean of the canonical angles weighted by
    ``softmax(<z, p_k> / temperature)``.  This readout never sees a label for
    the evaluated trial; it is recorded as a diagnostic for the contrastive
    rows so that a direction result cannot be attributed to the linear probe.
    """
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    _require(bool(np.isfinite(vector).all()), "prototype readout needs a finite embedding")
    _require(bool(prototypes) and temperature > 0.0, "prototype readout needs prototypes and a positive temperature")
    indices = sorted(int(key) for key in prototypes)
    matrix = np.stack([np.asarray(prototypes[index], dtype=np.float64).reshape(-1) for index in indices])
    _require(matrix.shape[1] == vector.size, "prototype/embedding dimension drift")
    sims = matrix @ vector / float(temperature)
    sims = sims - sims.max()
    weights = np.exp(sims)
    weights = weights / weights.sum()
    angles = np.asarray([cdm_core.CANONICAL_DIRECTIONS_RAD[index] for index in indices], dtype=np.float64)
    theta = float(wrap_rad(math.atan2(float((weights * np.sin(angles)).sum()), float((weights * np.cos(angles)).sum()))))
    return theta, weights
