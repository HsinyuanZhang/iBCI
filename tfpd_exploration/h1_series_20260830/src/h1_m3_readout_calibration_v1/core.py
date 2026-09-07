"""CPU float64 closed-form H1 readout maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .plan import FAMILIES, SCALE_FLOOR


@dataclass(frozen=True)
class ReadoutMap:
    family: str
    ridge: float
    p_mean: np.ndarray
    p_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    weight: np.ndarray
    intercept: np.ndarray


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7 or result.shape[0] < 8 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [N,7]")
    return result


def fit(prediction: Any, target: Any, *, family: str, ridge: float) -> ReadoutMap:
    p = _matrix(prediction, "prediction")
    y = _matrix(target, "target")
    if p.shape != y.shape or family not in FAMILIES or float(ridge) < 0.0 or not np.isfinite(float(ridge)):
        raise ValueError("readout fit contract drift")
    pm, ym = p.mean(0), y.mean(0)
    ps = np.maximum(p.std(0), SCALE_FLOOR)
    ys = np.maximum(y.std(0), SCALE_FLOOR)
    zp, zy = (p - pm) / ps, (y - ym) / ys
    lam = float(ridge)
    if family == "MAT7":
        design = np.concatenate((zp, np.ones((zp.shape[0], 1), dtype=np.float64)), axis=1)
        regularizer = np.diag(np.asarray([lam ** 0.5] * 7 + [0.0], dtype=np.float64))
        solution = np.linalg.lstsq(
            np.concatenate((design, regularizer), axis=0),
            np.concatenate((zy, np.zeros((8, 7), dtype=np.float64)), axis=0),
            rcond=None,
        )[0]
        weight, intercept = solution[:7], solution[7]
    else:
        weight = np.zeros((7, 7), dtype=np.float64)
        intercept = np.zeros(7, dtype=np.float64)
        for index in range(7):
            design = np.stack((zp[:, index], np.ones(zp.shape[0], dtype=np.float64)), axis=1)
            solution = np.linalg.lstsq(
                np.concatenate((design, np.diag([lam ** 0.5, 0.0])), axis=0),
                np.concatenate((zy[:, index], np.zeros(2, dtype=np.float64))),
                rcond=None,
            )[0]
            weight[index, index], intercept[index] = solution
    if not np.isfinite(weight).all() or not np.isfinite(intercept).all():
        raise ValueError("readout solution nonfinite")
    return ReadoutMap(family, lam, pm, ps, ym, ys, weight, intercept)


def apply(model: ReadoutMap, prediction: Any) -> np.ndarray:
    p = _matrix(prediction, "prediction")
    normalized = (p - model.p_mean) / model.p_scale
    result = (normalized @ model.weight + model.intercept) * model.y_scale + model.y_mean
    if result.shape != p.shape or not np.isfinite(result).all():
        raise ValueError("readout output drift")
    return np.ascontiguousarray(result, dtype=np.float32)


def template(target: Any, count: int) -> np.ndarray:
    y = _matrix(target, "target")
    if type(count) is not int or count < 1:
        raise ValueError("template count drift")
    return np.ascontiguousarray(np.repeat(y.mean(0, keepdims=True), count, axis=0), dtype=np.float32)


__all__ = ("ReadoutMap", "apply", "fit", "template")

