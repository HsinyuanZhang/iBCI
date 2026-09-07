"""The AC3-U rotation law (work order §4), pure numpy on one trial.

For trial row ``i``, group ``g``, trajectory ``v_g`` with net displacement
``d_g = sum_t v_g[t]``:

    delta = wrap(theta_hat_i - atan2(d_g[1], d_g[0]))
    v'_g   = R(delta) @ v_g^T, transposed back

* the math runs in float64 and the result is cast back to the incoming dtype;
* a rigid rotation preserves every per-bin speed, every net displacement norm
  and the movement mask, so the frozen magnitude/movement gates are unchanged;
  only the direction content changes (the frozen canonical-snap gate sees the
  rotated direction by design);
* when ``theta_hat_i`` is undefined (NaN) the whole trial passes through
  unrotated -- the predeclared fallback of §4;
* the ``VelocityValidityEvidence`` object is NEVER rebuilt: the rotated view is
  ``core.CompletedVelocityPrediction(rotated_velocity, item.validity)`` with the
  SAME evidence instance, so the frozen ``observe`` law sees an identical
  capability type.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core

from . import plan


class AC3URotationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3URotationError(message)


def wrap_rad(value: float) -> float:
    """Wrap one angle into ``[-pi, pi)`` (rotation by ``a`` equals ``a - 2pi``)."""
    return float(np.mod(float(value) + math.pi, 2.0 * math.pi) - math.pi)


def net_displacement(velocity: Any) -> np.ndarray:
    """``d = sum_t v[t]`` in float64 (the rotation law's own definition)."""
    values = np.asarray(velocity, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 2, "rotation law needs a [P, 2] trajectory")
    return values.sum(axis=0)


def rotation_delta(theta_hat: float, displacement: np.ndarray) -> float:
    """``wrap(theta_hat - atan2(d[1], d[0]))`` for one group's displacement."""
    _require(math.isfinite(float(theta_hat)), "rotation delta needs a finite theta_hat")
    vector = np.asarray(displacement, dtype=np.float64)
    _require(vector.shape == (2,), "rotation delta needs one [2] displacement")
    return wrap_rad(float(theta_hat) - math.atan2(float(vector[1]), float(vector[0])))


def rotate_trajectory(velocity: Any, delta: float) -> np.ndarray:
    """``R(delta) @ v^T`` transposed back, float64 math, original dtype out."""
    values = np.asarray(velocity)
    _require(values.ndim == 2 and values.shape[1] == 2, "rotation needs a [P, 2] trajectory")
    _require(
        np.issubdtype(values.dtype, np.floating) and bool(np.isfinite(values).all()),
        "rotation needs a finite floating trajectory",
    )
    _require(math.isfinite(float(delta)), "rotation angle must be finite")
    cosine = math.cos(float(delta))
    sine = math.sin(float(delta))
    # Row-vector form of R(delta): v' = v @ R(delta)^T with
    # R(delta)^T = [[cos, sin], [-sin, cos]].
    rotated = np.asarray(values, dtype=np.float64) @ np.asarray(
        [[cosine, sine], [-sine, cosine]], dtype=np.float64,
    )
    _require(bool(np.isfinite(rotated).all()), "rotation produced a nonfinite trajectory")
    return np.ascontiguousarray(rotated, dtype=values.dtype)


def norm_preservation_report(
    before: Sequence[Any], after: Sequence[Any],
) -> Mapping[str, object]:
    """Per-bin speed and net-displacement norm preservation across four groups."""
    speed_gaps: list[float] = []
    displacement_gaps: list[float] = []
    for left, right in zip(before, after):
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        _require(left_array.shape == right_array.shape, "rotation changed the trajectory shape")
        left_speed = np.linalg.norm(left_array, axis=1)
        right_speed = np.linalg.norm(right_array, axis=1)
        scale = np.maximum(left_speed, 1.0)
        speed_gaps.append(float(np.max(np.abs(left_speed - right_speed) / scale)))
        left_norm = float(np.linalg.norm(left_array.sum(axis=0)))
        right_norm = float(np.linalg.norm(right_array.sum(axis=0)))
        displacement_gaps.append(
            abs(left_norm - right_norm) / max(left_norm, 1.0) if left_norm > 0.0 else abs(right_norm),
        )
    return {
        "max_relative_speed_gap": max(speed_gaps) if speed_gaps else 0.0,
        "max_relative_displacement_norm_gap": max(displacement_gaps) if displacement_gaps else 0.0,
        "tolerance": plan.NORM_PRESERVATION_TOLERANCE,
    }


def assert_norm_preservation(before: Sequence[Any], after: Sequence[Any]) -> None:
    report = norm_preservation_report(before, after)
    _require(
        float(report["max_relative_speed_gap"]) <= plan.NORM_PRESERVATION_TOLERANCE
        and float(report["max_relative_displacement_norm_gap"]) <= plan.NORM_PRESERVATION_TOLERANCE,
        f"rotation law violated norm preservation: {dict(report)}",
    )


def rotate_group_predictions(
    group_predictions: Sequence[Any],
    theta_hat: Optional[float],
    *,
    row_id: str = "",
) -> Tuple[Tuple[Any, ...], Mapping[str, object]]:
    """Apply the §4 rotation law to the four complementary views of one trial.

    Returns the rebuilt views -- each carrying the ORIGINAL validity evidence
    object -- plus a typed record.  ``theta_hat`` non-finite means the
    predeclared fallback: the incoming tuple is returned unchanged.
    """
    rows = tuple(group_predictions)
    _require(len(rows) == core.GROUP_COUNT, "the rotation law needs exactly four group predictions")
    if theta_hat is None or not math.isfinite(float(theta_hat)):
        return rows, {
            "row": row_id,
            "applied": False,
            "reason": "undefined_theta_hat_predeclared_fallback",
            "deltas_rad": [],
            "norm_report": {"max_relative_speed_gap": 0.0, "max_relative_displacement_norm_gap": 0.0},
        }
    theta = float(theta_hat)
    deltas: list[float] = []
    rotated: list[Any] = []
    for item in rows:
        delta = rotation_delta(theta, net_displacement(item.velocity))
        deltas.append(delta)
        rotated.append(core.CompletedVelocityPrediction(
            rotate_trajectory(item.velocity, delta), item.validity,
        ))
    report = norm_preservation_report([item.velocity for item in rows],
                                      [item.velocity for item in rotated])
    _require(
        float(report["max_relative_speed_gap"]) <= plan.NORM_PRESERVATION_TOLERANCE
        and float(report["max_relative_displacement_norm_gap"]) <= plan.NORM_PRESERVATION_TOLERANCE,
        f"rotation law violated norm preservation: {dict(report)}",
    )
    return tuple(rotated), {
        "row": row_id,
        "applied": True,
        "reason": None,
        "deltas_rad": deltas,
        "norm_report": dict(report),
    }


def displacement_angles(group_predictions: Sequence[Any]) -> np.ndarray:
    """``atan2`` of each view's net displacement, NaN where the displacement is 0."""
    angles = np.full(core.GROUP_COUNT, np.nan, dtype=np.float64)
    for index, item in enumerate(group_predictions):
        vector = net_displacement(item.velocity)
        if float(np.linalg.norm(vector)) > 0.0:
            angles[index] = math.atan2(float(vector[1]), float(vector[0]))
    return angles


def displacement_norms(group_predictions: Sequence[Any]) -> np.ndarray:
    norms = np.zeros(core.GROUP_COUNT, dtype=np.float64)
    for index, item in enumerate(group_predictions):
        norms[index] = float(np.linalg.norm(net_displacement(item.velocity)))
    return norms
