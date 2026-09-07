"""Frozen ReLU nonnegative projection. No learnable parameters."""
from __future__ import annotations

import inspect

import numpy as np

from . import plan


class RectifyError(RuntimeError):
    """Fail closed for rectifier-law drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RectifyError(message)


THRESHOLD = 0.0
LEARNABLE = False
NAME = "relu_nonnegative_projection"


def relu_nonnegative_projection(x: np.ndarray) -> np.ndarray:
    """R(x)=max(x,0). Elementwise float64. Threshold is exactly 0.0."""
    raw = np.asarray(x, dtype=np.float64)
    _require(raw.size > 0 and np.isfinite(raw).all(), "rectifier input must be finite")
    return np.maximum(raw, THRESHOLD)


def assert_frozen_law() -> None:
    source = inspect.getsource(relu_nonnegative_projection)
    _require("np.maximum" in source, "rectifier must be np.maximum")
    _require("np.abs" not in source, "abs is a forbidden alternative")
    _require(THRESHOLD == 0.0, "threshold drift")
    _require(LEARNABLE is False, "learnable rectifier is forbidden")
    _require(plan.RECTIFIER_LAW["threshold"] == 0.0, "plan threshold drift")
    _require(plan.RECTIFIER_LAW["learnable_parameters"] == 0, "plan learnable drift")


def mass_report(x: np.ndarray) -> dict[str, object]:
    raw = np.asarray(x, dtype=np.float64)
    _require(raw.size > 0 and np.isfinite(raw).all(), "mass report input")
    negative = raw < 0.0
    rectified = relu_nonnegative_projection(raw)
    negative_mass = float(-np.sum(raw[negative])) if np.any(negative) else 0.0
    return {
        "name": NAME,
        "threshold": THRESHOLD,
        "clipped_count": int(np.sum(negative)),
        "clipped_fraction": float(np.mean(negative)),
        "negative_fraction": float(np.mean(negative)),
        "negative_mass": negative_mass,
        "maximum_undershoot": float(np.min(raw)),
        "rectified_negative_fraction": float(np.mean(rectified < 0.0)),
        "rectified_minimum": float(np.min(rectified)),
        "rectified_maximum": float(np.max(rectified)),
        "zero_fraction": float(np.mean(rectified == 0.0)),
    }


def rectified_signal_view(stored: dict[str, object], emg_rect: np.ndarray, mass: dict[str, object]) -> dict[str, object]:
    view = dict(stored)
    rect = np.asarray(emg_rect, dtype=np.float64)
    view["rectification"] = NAME
    view["minimum"] = float(np.min(rect))
    view["maximum"] = float(np.max(rect))
    view["negative_fraction"] = float(np.mean(rect < 0.0))
    view["zero_fraction"] = float(np.mean(rect == 0.0))
    view["finite_count"] = int(np.isfinite(rect).sum())
    view["rectifier_mass"] = mass
    return view
