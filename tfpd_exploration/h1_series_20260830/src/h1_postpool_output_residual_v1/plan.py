"""Frozen laws for H1 post-pool output residual V1."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "h1_postpool_output_residual_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
MAX_ABS_BETA = 4.0
MIN_DENOMINATOR = 1.0e-12
MIN_OOF_MEAN_GAIN = 0.005
MIN_OOF_NONNEGATIVE_DATES = 4
MIN_OOF_WORST_DATE = -0.010
EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
PREDECESSOR_SCORE_SHA256 = "62ebd18db6ce504c01414bd392a84a617c2b2aa2049af1e68b90e34bfc238d83"
PREDECESSOR_TERMINAL_SHA256 = "ec86e26c513c3ccb9ab4ca5f8d0f0ce244e6b162b507c635f581078559a1804a"


class H1OutputResidualError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1OutputResidualError(message)


def fit_beta(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Fit the unique equal-date/equal-recording/equal-R2 scalar residual."""

    _need(bool(rows), "source beta rows are empty")
    dates = tuple(sorted({str(row["date"]) for row in rows}))
    _need(len(dates) >= 2, "source beta needs at least two dates")
    counts = Counter(str(row["date"]) for row in rows)
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        target = np.asarray(row["target"], dtype=np.float64)
        static = np.asarray(row["static"], dtype=np.float64)
        post = np.asarray(row["post"], dtype=np.float64)
        _need(target.ndim == 2 and target.shape == static.shape == post.shape, "beta row geometry drift")
        _need(np.isfinite(target).all() and np.isfinite(static).all() and np.isfinite(post).all(), "beta row nonfinite")
        centered = target - target.mean(axis=0, keepdims=True, dtype=np.float64)
        tss = float(np.square(centered).sum(dtype=np.float64))
        _need(np.isfinite(tss) and tss > 0.0, "beta row TSS is invalid")
        date = str(row["date"])
        weight = 1.0 / (len(dates) * counts[date] * tss)
        delta = post - static
        error = target - static
        numerator += weight * float(np.sum(delta * error, dtype=np.float64))
        denominator += weight * float(np.sum(delta * delta, dtype=np.float64))
    _need(np.isfinite(numerator) and np.isfinite(denominator) and denominator > MIN_DENOMINATOR,
          "beta fit is singular or nonfinite")
    beta = numerator / denominator
    _need(np.isfinite(beta) and abs(beta) <= MAX_ABS_BETA, "beta exceeds frozen stability bound")
    return {
        "beta": float(beta),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "max_abs_beta": MAX_ABS_BETA,
        "source_date_count": float(len(dates)),
        "source_recording_count": float(len(rows)),
    }


def apply_beta(static: Any, post: Any, beta: float) -> np.ndarray:
    left = np.asarray(static, dtype=np.float32)
    right = np.asarray(post, dtype=np.float32)
    value = float(beta)
    _need(left.shape == right.shape and left.ndim == 2, "output residual geometry drift")
    _need(np.isfinite(left).all() and np.isfinite(right).all() and np.isfinite(value), "output residual nonfinite")
    if value == 0.0:
        return left
    result = np.ascontiguousarray(left + np.float32(value) * (right - left), dtype=np.float32)
    _need(np.isfinite(result).all(), "output residual result is nonfinite")
    return result


def decide_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _need(tuple(str(row["outer_date"]) for row in rows) == DATE_ORDER, "OOF date order drift")
    deltas = [float(row["delta_vs_static"]) for row in rows]
    mean = sum(deltas) / len(deltas)
    nonnegative = sum(value >= 0.0 for value in deltas)
    worst = min(deltas)
    passed = mean >= MIN_OOF_MEAN_GAIN and nonnegative >= MIN_OOF_NONNEGATIVE_DATES and worst >= MIN_OOF_WORST_DATE
    return {
        "mean_gain": mean,
        "nonnegative_dates": nonnegative,
        "worst_date_gain": worst,
        "required_mean_gain": MIN_OOF_MEAN_GAIN,
        "required_nonnegative_dates": MIN_OOF_NONNEGATIVE_DATES,
        "required_worst_date_gain": MIN_OOF_WORST_DATE,
        "pass": passed,
        "verdict": "PASS_FOR_ALL_SOURCE_BETA_AND_PACKAGE" if passed else "STOP_H1_POSTPOOL_OUTPUT_RESIDUAL",
    }


__all__ = ("DATE_ORDER", "EXPECTED_GPU0_UUID", "PREDECESSOR_SCORE_SHA256",
           "PREDECESSOR_TERMINAL_SHA256", "SCHEMA", "apply_beta", "decide_oof", "fit_beta")

