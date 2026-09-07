"""Shared H1 learning metrics. Train units are raw model output vs scale*y."""

from __future__ import annotations

from typing import Any

import numpy as np

from tfpd_exploration.src.m1_h1_activity_headroom_v1.core import variance_weighted_r2


def _arr(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def std_l2(array: np.ndarray) -> float:
    std = _arr(array).std(axis=0)
    return float(np.linalg.norm(std))


def report_pair(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.ascontiguousarray(pred, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    zero = np.zeros_like(target)
    mean = np.broadcast_to(target.mean(axis=0, keepdims=True), target.shape).copy()
    pred_std = std_l2(pred)
    tgt_std = std_l2(target)
    return {
        "n": int(target.shape[0]),
        "r2": variance_weighted_r2(pred, target),
        "mse": float(np.mean(np.square(pred - target))),
        "bias_l2": float(np.linalg.norm(pred.mean(axis=0) - target.mean(axis=0))),
        "pred_std_l2": pred_std,
        "target_std_l2": tgt_std,
        "pred_std_over_target_std": pred_std / tgt_std if tgt_std > 0 else None,
        "zero_r2": variance_weighted_r2(zero, target),
        "zero_mse": float(np.mean(np.square(zero - target))),
        "mean_r2": variance_weighted_r2(mean, target),
        "mean_mse": float(np.mean(np.square(mean - target))),
        "beats_zero_r2": bool(variance_weighted_r2(pred, target) > variance_weighted_r2(zero, target)),
        "beats_mean_r2": bool(variance_weighted_r2(pred, target) > variance_weighted_r2(mean, target)),
    }
