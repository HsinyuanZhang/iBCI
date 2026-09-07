"""SLOT-AUDIT V1 scoring — the house R2 law plus numpy residual statistics.

House law (work order section 4.1): per-session variance-weighted R2 across
the 2 output dims, replicating ``src/tfpd_lane/matched_scorer.session_r2``
(torchmetrics ``R2Score(multioutput="variance_weighted")``, float32) exactly.
``house_session_r2`` BRIDGES to that function (never reimplements it); the
float64 numpy replica below is the analytically identical form

    R2 = 1 - sum_d SSres_d / sum_d SStot_d,   SStot_d = sum (y - mean_d)^2

used for cross-checks and for statistics computed in float64 (the measured
float32-vs-float64 gap in this lane is ~7e-8, recorded in the receipts).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SlotAuditScoringError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditScoringError(message)


def house_session_r2(prediction_f32: Any, target_f32: Any) -> float:
    """The governing per-session R2 through the house torch scorer (CPU)."""
    import torch

    from src.tfpd_lane.matched_scorer import session_r2

    pred = np.ascontiguousarray(prediction_f32, dtype=np.float32)
    target = np.ascontiguousarray(target_f32, dtype=np.float32)
    _require(
        pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == 2,
        "house R2 shape drift",
    )
    _require(
        bool(np.isfinite(pred).all()) and bool(np.isfinite(target).all()),
        "house R2 nonfinite input",
    )
    return float(session_r2(
        torch.from_numpy(pred.copy()), torch.from_numpy(target.copy()),
    ))


def variance_weighted_r2_numpy(prediction: Any, target: Any) -> float:
    """The float64 analytical twin of the house variance-weighted R2."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _require(
        pred.shape == truth.shape and pred.ndim == 2,
        "numpy R2 shape drift",
    )
    residual = np.sum((pred - truth) ** 2, axis=0)
    centered = truth - truth.mean(axis=0, keepdims=True)
    total = np.sum(centered ** 2, axis=0)
    _require(bool(np.all(total > 0.0)), "numpy R2 zero target variance")
    return float(1.0 - float(residual.sum()) / float(total.sum()))


def residual_statistics(prediction: Any, target: Any) -> dict[str, Any]:
    """MSE, per-dim bias (mean pred - target) and per-dim residual variance."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _require(pred.shape == truth.shape and pred.ndim == 2, "residual stats shape drift")
    residual = pred - truth
    return {
        "n": int(residual.shape[0]),
        "mse": float(np.mean(residual ** 2)),
        "bias": [float(value) for value in residual.mean(axis=0)],
        "residual_variance": [float(value) for value in residual.var(axis=0)],
        "residual_std": [float(value) for value in residual.std(axis=0)],
    }


def pooled_correlation_from_sums(sums: Mapping[str, float]) -> dict[str, float]:
    """Pearson correlation/covariance/stds from pooled sums (N, sum x, sum y,
    sum xx, sum yy, sum xy)."""
    n = float(sums["n"])
    _require(n > 1.0, "pooled correlation needs > 1 pair")
    mean_x = float(sums["sx"]) / n
    mean_y = float(sums["sy"]) / n
    var_x = float(sums["sxx"]) / n - mean_x * mean_x
    var_y = float(sums["syy"]) / n - mean_y * mean_y
    cov = float(sums["sxy"]) / n - mean_x * mean_y
    var_x = max(var_x, 0.0)
    var_y = max(var_y, 0.0)
    std_x = float(np.sqrt(var_x))
    std_y = float(np.sqrt(var_y))
    corr = float(cov / (std_x * std_y)) if std_x > 0.0 and std_y > 0.0 else 0.0
    return {"n": n, "covariance": cov, "std_x": std_x, "std_y": std_y,
            "correlation": corr}
