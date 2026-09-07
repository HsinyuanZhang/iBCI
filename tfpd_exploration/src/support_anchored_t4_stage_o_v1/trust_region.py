"""Support design-precision trust region (design §4.3).

    delta = T4_candidate - T4_support          (per-unit coefficient space)
    P_support = Xs^T Xs + production ridge      (the anchor's A0, per group)
    D2 = delta^T P_support delta                (aggregate over groups/units)
    T4_active = T4_support + alpha_M * project(delta, D2 <= c_M)

The projection is the exact radial scaling of the single quadratic constraint:
``delta * sqrt(c_M / D2)`` iff ``D2 > c_M``.  ``P_support`` is derived only
from the frozen labeled-support design.  ``c_M`` is a source-calibrated
GEOMETRIC trust threshold (within-6 only); it is not a credible region and
must never be described as one.  With ``alpha_M = 0`` the rebuild reproduces
``T4_support`` bit-exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core

from . import plan
from .anchor import SupportAnchor


class TrustRegionError(ValueError):
    """Fail-closed error for malformed trust-region inputs or a broken bound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrustRegionError(message)


@dataclass(frozen=True)
class TrustRegionOutcome:
    """The projected movement of one commit and the resulting active carrier."""

    active_t4: np.ndarray = field(repr=False, compare=False)
    alpha_M: float
    c_M: Optional[float]
    d2_unprojected: float
    d2_projected_scaled: float
    projected: bool
    projection_scale: float
    delta_unprojected_frobenius: float
    delta_projected_frobenius: float
    movement_frobenius: float
    per_group_d2: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(np.asarray(self.active_t4).shape[1] == 4 and np.isfinite(self.active_t4).all(),
                 "trust-region active carrier drift")
        _require(math.isfinite(float(self.d2_unprojected)) and float(self.d2_unprojected) >= 0.0,
                 "D2 must be finite and nonnegative")
        _require(0.0 <= float(self.alpha_M) <= 1.0 + 1.0e-12, "alpha_M must be inside [0, 1]")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_o_trust_region_v1",
            "alpha_M": float(self.alpha_M),
            "c_M": None if self.c_M is None else float(self.c_M),
            "d2_unprojected": float(self.d2_unprojected),
            "d2_projected_scaled": float(self.d2_projected_scaled),
            "projected": bool(self.projected),
            "projection_scale": float(self.projection_scale),
            "delta_unprojected_frobenius": float(self.delta_unprojected_frobenius),
            "delta_projected_frobenius": float(self.delta_projected_frobenius),
            "movement_frobenius": float(self.movement_frobenius),
            "per_group_d2": [float(item) for item in self.per_group_d2],
            "active_t4_sha256": cdm_core.array_digest(self.active_t4),
            "geometry": plan.TRUST_REGION_LAW["geometry"],
            "coverage_claim_forbidden": plan.TRUST_REGION_LAW["coverage_claim_forbidden"],
        }


def aggregate_d2(
    anchor: SupportAnchor, delta_blocks: Sequence[np.ndarray],
) -> tuple[float, tuple[float, ...]]:
    """``D2 = sum over groups and valid units of delta_u^T A0_g delta_u``."""
    _require(len(delta_blocks) == anchor.groups.group_count, "D2 needs one delta block per group")
    per_group: list[float] = []
    total = 0.0
    for group in range(anchor.groups.group_count):
        delta = np.asarray(delta_blocks[group], dtype=np.float64)
        _require(delta.shape == (3, anchor.per_group[group].units.size), "delta block shape drift")
        _require(np.isfinite(delta).all(), "delta block must be finite")
        quadratic = float(np.sum((anchor.per_group[group].A0 @ delta) * delta))
        _require(quadratic >= 0.0, "negative quadratic form under a positive-definite A0")
        per_group.append(quadratic)
        total += quadratic
    return total, tuple(per_group)


def project_delta(
    anchor: SupportAnchor, delta_blocks: Sequence[np.ndarray], *, c_M: Optional[float],
) -> tuple[list[np.ndarray], float, float, bool]:
    """Radial projection onto ``D2 <= c_M``; ``c_M=None`` disables projection.

    Returns ``(projected_blocks, d2_before, d2_after, projected_flag)``.
    """
    d2_before, _per_group = aggregate_d2(anchor, delta_blocks)
    if c_M is None:
        return [np.asarray(item, dtype=np.float64) for item in delta_blocks], d2_before, d2_before, False
    _require(math.isfinite(float(c_M)) and float(c_M) > 0.0, "c_M must be a positive finite threshold")
    if d2_before <= float(c_M):
        return [np.asarray(item, dtype=np.float64) for item in delta_blocks], d2_before, d2_before, False
    scale = math.sqrt(float(c_M) / d2_before)
    projected = [np.asarray(item, dtype=np.float64) * scale for item in delta_blocks]
    d2_after, _per_group = aggregate_d2(anchor, projected)
    _require(
        d2_after <= float(c_M) + plan.GATE_BOUNDARY_EPSILON * max(1.0, float(c_M)),
        "projection failed to restore the trust-region bound",
    )
    return projected, d2_before, d2_after, True


def _frobenius(blocks: Sequence[np.ndarray]) -> float:
    return float(math.sqrt(sum(float(np.sum(np.asarray(item) ** 2)) for item in blocks)))


def active_carrier(
    anchor: SupportAnchor,
    candidate_blocks: Sequence[np.ndarray],
    *,
    alpha_M: float,
    c_M: Optional[float],
) -> TrustRegionOutcome:
    """``T4_active = T4_support + alpha_M * project(candidate - support)``.

    The delta lives in per-unit coefficient space; the active carrier is
    rebuilt through the production ``[a, c, hypot(a,c), b]`` mirror in the
    sealed support carrier's own dtype, and ``alpha_M = 0`` returns the sealed
    support carrier itself bit-exactly.
    """
    _require(math.isfinite(float(alpha_M)) and 0.0 <= float(alpha_M) <= 1.0,
             "alpha_M must lie in [0, 1]")
    support_blocks = anchor.coefficients(anchor.support_t4)
    delta_blocks = [
        np.asarray(candidate_blocks[group], dtype=np.float64) - np.asarray(support_blocks[group], dtype=np.float64)
        for group in range(anchor.groups.group_count)
    ]
    projected_blocks, d2_before, d2_after, projected = project_delta(anchor, delta_blocks, c_M=c_M)
    if float(alpha_M) == 0.0:
        # The exact no-op law: alpha_M = 0 returns the sealed support carrier
        # itself (the very array), byte-identical, with the projected movement
        # still fully recorded for the receipt.
        active = np.asarray(anchor.support_t4)
    else:
        moved_blocks = [
            float(alpha_M) * np.asarray(projected_blocks[group]) + np.asarray(support_blocks[group])
            for group in range(anchor.groups.group_count)
        ]
        active = anchor.rebuild_t4(moved_blocks)
    movement = float(np.linalg.norm(
        np.asarray(active, dtype=np.float64) - np.asarray(anchor.support_t4, dtype=np.float64),
    ))
    _require(math.isfinite(movement), "carrier movement must be finite")
    _per_group_d2 = aggregate_d2(anchor, delta_blocks)[1]
    return TrustRegionOutcome(
        active_t4=active,
        alpha_M=float(alpha_M),
        c_M=None if c_M is None else float(c_M),
        d2_unprojected=d2_before,
        d2_projected_scaled=d2_after,
        projected=projected,
        projection_scale=1.0 if not projected else math.sqrt(
            (d2_after / d2_before) if d2_before > 0.0 else 1.0,
        ),
        delta_unprojected_frobenius=_frobenius(delta_blocks),
        delta_projected_frobenius=_frobenius(projected_blocks),
        movement_frobenius=movement,
        per_group_d2=_per_group_d2,
    )


def calibrate_c_M(d2_values: Sequence[float]) -> float:
    """The pre-registered within-6 calibration statistic: the MEDIAN of D2."""
    values = np.asarray([float(item) for item in d2_values], dtype=np.float64)
    _require(values.size >= 1 and np.isfinite(values).all() and np.all(values >= 0.0),
             "c_M calibration needs nonnegative finite D2 values")
    median = float(np.median(values))
    _require(math.isfinite(median) and median > 0.0,
             "c_M calibration produced a degenerate threshold")
    return median
