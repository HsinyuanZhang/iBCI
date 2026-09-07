"""The SUA support anchor: the stage-O A0/b0 law on the sealed M4 ridge carrier.

The frozen Stage-O/Stage-P machinery (``support_anchored_t4_stage_p_v1.gate``,
``support_anchored_t4_stage_o_v1.trust_region``) is orchestrated verbatim, so
this anchor provides exactly the interface those modules consume:

* ``groups``               -- a ``cdm_core.ComplementaryGroups``;
* ``per_group[g]``         -- ``.group/.units/.counts/.A0/.b0/.design_row_count``;
* ``coefficients(t4)``     -- per-group ``(3, U)`` (a, c, b) blocks;
* ``rebuild_t4(blocks)``   -- the production ``[a, c, m, b]`` mirror;
* ``support_t4``/``digest``/``a0_b0_digest``.

Two SUA-specific laws are the point of this module (both disclosed in the
receipt):

* the support design uses the sealed carrier's CONTINUOUS calibration angles
  with the ``fit_ridge_t4`` column order ``[cos, sin, 1]`` and penalty
  ``diag(n*lambda, n*lambda, 0)``, ``lambda = 0.1`` -- algebraically the same
  Stage-O law, arithmetically the sealed fit;
* the modulation column is rebuilt with ``sqrt(a*a + c*c)`` (the sealed
  ``fit_ridge_t4`` arithmetic), not ``hypot``.

``from_labeled_support`` proves the zero-evidence fallback bitwise: solving
``(A0, b0)`` per group and rebuilding reproduces the sealed carrier
bit-for-bit, because ``b0`` is sliced from ONE full-unit design product and a
per-column LU solve is column-wise identical to the sealed whole-matrix
solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core


class SUAAnchorError(ValueError):
    """Fail-closed error for a malformed or drifting SUA anchor."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SUAAnchorError(message)


def _immutable(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _frozen_array_payload(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "sha256": cdm_core.array_digest(value),
    }


@dataclass(frozen=True)
class SUAGroupAnchor:
    """One complementary group's immutable support sufficient statistics."""

    group: int
    units: np.ndarray = field(repr=False, compare=False)
    counts: np.ndarray = field(repr=False, compare=False)
    A0: np.ndarray = field(repr=False, compare=False)
    b0: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(0 <= int(self.group) < int(cdm_core.GROUP_COUNT), "anchor group index drift")
        units = np.asarray(self.units)
        counts = np.asarray(self.counts)
        A0 = np.asarray(self.A0)
        b0 = np.asarray(self.b0)
        _require(units.ndim == 1 and units.size >= 1, "anchor units must be [U]")
        _require(counts.ndim == 1, "anchor counts must be one-dimensional")
        _require(A0.shape == (3, 3) and b0.shape == (3, units.size), "anchor A0/b0 shape drift")
        _require(np.all(np.linalg.eigvalsh(A0) > 0.0), "anchor A0 must be positive definite")
        _require(np.isfinite(A0).all() and np.isfinite(b0).all() and np.all(counts >= 0),
                 "anchor nonfinite statistics")
        object.__setattr__(self, "units", _immutable(units, dtype=np.int64))
        object.__setattr__(self, "counts", _immutable(counts, dtype=np.int64))
        object.__setattr__(self, "A0", _immutable(A0, dtype=np.float64))
        object.__setattr__(self, "b0", _immutable(b0, dtype=np.float64))

    @property
    def design_row_count(self) -> int:
        return int(np.asarray(self.counts).sum())

    def payload(self) -> dict[str, Any]:
        return {
            "group": int(self.group),
            "units": [int(item) for item in np.asarray(self.units).tolist()],
            "counts": [int(item) for item in np.asarray(self.counts).tolist()],
            "A0": _frozen_array_payload(np.asarray(self.A0)),
            "b0": _frozen_array_payload(np.asarray(self.b0)),
            "support_trial_count": self.design_row_count,
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))


def support_design(angles_rad: Sequence[float]) -> np.ndarray:
    """``[cos, sin, 1]`` -- the sealed ``fit_ridge_t4`` column order."""
    theta = np.asarray([float(item) for item in angles_rad], dtype=np.float64)
    _require(theta.ndim == 1 and theta.size >= 1 and np.isfinite(theta).all(),
             "support angles must be finite [n]")
    return np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size)))


def support_penalty(support_rows: int, *, normalized_lambda: float) -> np.ndarray:
    """The exact sealed fixed-ridge penalty of the SUA trial-table fit."""
    weight = float(support_rows) * float(normalized_lambda)
    return np.diag((weight, weight, 0.0))


def format_t4_rows(coefficients: np.ndarray) -> np.ndarray:
    """``[3, U]`` (a, c, b) -> ``[U, 4]`` (a, c, sqrt(a*a+c*c), b).

    The modulation column uses the sealed carrier's ``np.sqrt(a*a + c*c)``
    arithmetic, NOT ``np.hypot``: the zero-evidence fallback must reproduce
    the sealed M4 ridge carrier bit-for-bit.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == 3, "coefficient block must be [3, U]")
    a, c, b = values[0], values[1], values[2]
    rows = np.column_stack((a, c, np.sqrt(a * a + c * c), b))
    _require(np.isfinite(rows).all(), "formatted carrier rows are nonfinite")
    return rows


class SUASupportAnchor:
    """The immutable per-group A0/b0 bank plus the sealed support carrier."""

    def __init__(
        self,
        *,
        groups: cdm_core.ComplementaryGroups,
        support_t4: np.ndarray,
        support_rates_sha256: str,
        support_angles_rad: Sequence[float],
        per_group: Sequence[SUAGroupAnchor],
        normalized_lambda: float,
        parity: Mapping[str, Any],
    ) -> None:
        support = np.asarray(support_t4)
        _require(support.shape == (groups.units, 4), "support T4 shape drift")
        _require(np.isfinite(support).all(), "support T4 must be finite")
        _require(len(per_group) == groups.group_count, "anchor group count drift")
        for group_anchor in per_group:
            _require(group_anchor.group < groups.group_count, "anchor group binding drift")
            _require(
                np.array_equal(np.asarray(group_anchor.units), groups.unit_indices(group_anchor.group)),
                "anchor unit binding drift",
            )
            _require(
                group_anchor.design_row_count == len(support_angles_rad),
                "anchor design row count must equal the support trial count",
            )
        _require(float(normalized_lambda) == cdm_core.RIDGE_NORMALIZED_LAMBDA,
                 "anchor must use the production normalized lambda 0.1")
        self.groups = groups
        self.support_t4 = _immutable(support)
        self.support_rates_sha256 = str(support_rates_sha256)
        self.support_angles_rad = tuple(float(item) for item in support_angles_rad)
        self.per_group = tuple(per_group)
        self.normalized_lambda = float(normalized_lambda)
        self.support_coefficient_parity = dict(parity)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_labeled_support(
        cls,
        *,
        groups: cdm_core.ComplementaryGroups,
        support_trial_rates_hz: np.ndarray,
        support_angles_rad: Sequence[float],
        support_t4: np.ndarray,
        normalized_lambda: float = cdm_core.RIDGE_NORMALIZED_LAMBDA,
    ) -> "SUASupportAnchor":
        rates = np.ascontiguousarray(np.asarray(support_trial_rates_hz), dtype=np.float64)
        angles = np.asarray([float(item) for item in support_angles_rad], dtype=np.float64)
        support = np.asarray(support_t4)
        n_support = int(rates.shape[0])
        _require(rates.ndim == 2 and rates.shape[1] == groups.units and n_support >= 3,
                 "support rates must be [trials >= 3, units]")
        _require(np.isfinite(rates).all(), "support rates must be finite")
        _require(angles.shape == (n_support,), "support angle count drift")
        _require(support.shape == (groups.units, 4), "support T4 shape drift")
        design = support_design(angles.tolist())
        penalty = support_penalty(n_support, normalized_lambda=normalized_lambda)
        a0 = np.ascontiguousarray(design.T @ design + penalty, dtype=np.float64)
        _require(np.all(np.linalg.eigvalsh(a0) > 0.0), "support A0 is not positive definite")
        # ONE full-unit product, then per-group column slices: per-column LU
        # solves are bitwise identical to the sealed whole-matrix solve.
        b0_all = np.ascontiguousarray(design.T @ rates, dtype=np.float64)
        canonical = np.asarray(cdm_core.CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
        counts = np.zeros(canonical.size, dtype=np.int64)
        for angle in angles:
            distance = np.abs(np.arctan2(np.sin(angle - canonical), np.cos(angle - canonical)))
            counts[int(np.argmin(distance))] += 1
        per_group: list[SUAGroupAnchor] = []
        parity_max_abs = 0.0
        parity_bitwise = True
        for group in range(groups.group_count):
            units = groups.unit_indices(group)
            b0 = np.ascontiguousarray(b0_all[:, units], dtype=np.float64)
            solved = np.linalg.solve(a0, b0)
            per_group.append(SUAGroupAnchor(
                group=group, units=units, counts=counts, A0=a0, b0=b0,
            ))
            formatted = np.ascontiguousarray(format_t4_rows(solved), dtype=np.float32)
            sealed_rows = np.ascontiguousarray(support[units], dtype=np.float32)
            parity_max_abs = max(parity_max_abs, float(np.max(np.abs(
                formatted.astype(np.float64) - sealed_rows.astype(np.float64)
            ))))
            parity_bitwise = parity_bitwise and bool(np.array_equal(formatted, sealed_rows))
        return cls(
            groups=groups,
            support_t4=support,
            support_rates_sha256=cdm_core.array_digest(rates),
            support_angles_rad=angles.tolist(),
            per_group=per_group,
            normalized_lambda=float(normalized_lambda),
            parity={
                "law": (
                    "per-group solve(A0, b0) -> [a, c, sqrt(a*a+c*c), b] float32 vs the "
                    "sealed fit_ridge_t4 carrier"
                ),
                "max_abs_coefficient_difference": parity_max_abs,
                "rebuilt_rows_bitwise_equal": parity_bitwise,
            },
        )

    # -- coefficient space --------------------------------------------------

    def coefficients(self, t4: np.ndarray) -> tuple[np.ndarray, ...]:
        values = np.asarray(t4)
        _require(values.shape == (self.groups.units, 4), "T4 coefficient extraction shape drift")
        blocks = np.asarray(values, dtype=np.float64)
        return tuple(
            np.ascontiguousarray(blocks[self.groups.unit_indices(group)][:, [0, 1, 3]].T)
            for group in range(self.groups.group_count)
        )

    def rebuild_t4(self, coefficient_blocks: Sequence[np.ndarray]) -> np.ndarray:
        """Rebuild the production ``[a, c, m, b]`` carrier, float32.

        Valid units take their group's refit rows; invalid units keep the
        sealed support rows (the Stage-O ``_reconstruct`` law).  The result is
        presented in the sealed carrier's own float32 dtype, so a no-movement
        rebuild is byte-identical to the sealed carrier.
        """
        _require(len(coefficient_blocks) == self.groups.group_count,
                 "rebuild needs one block per group")
        support = np.asarray(self.support_t4)
        seed = np.array(support, dtype=np.float32, copy=True)
        for group in range(self.groups.group_count):
            units = self.groups.unit_indices(group)
            formatted = format_t4_rows(np.asarray(coefficient_blocks[group], dtype=np.float64))
            seed[units] = np.ascontiguousarray(formatted, dtype=np.float32)
        result = np.ascontiguousarray(seed, dtype=np.float32)
        result.setflags(write=False)
        _require(np.isfinite(result).all(), "rebuilt carrier is nonfinite")
        return result

    # -- digests ------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "cdm_p1_sua_v1_support_anchor_v1",
            "groups_sha256": self.groups.digest,
            "support_t4": _frozen_array_payload(np.asarray(self.support_t4)),
            "support_angles_rad": [float(item) for item in self.support_angles_rad],
            "support_rates_sha256": self.support_rates_sha256,
            "normalized_lambda": float(self.normalized_lambda),
            "per_group": [item.payload() for item in self.per_group],
            "support_coefficient_parity": dict(self.support_coefficient_parity),
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))

    @property
    def a0_b0_digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
            "per_group": [item.digest for item in self.per_group],
            "support_t4_sha256": cdm_core.array_digest(np.asarray(self.support_t4)),
            "support_rates_sha256": self.support_rates_sha256,
        }))


def build_groups(
    initial_raw_t4: np.ndarray, channel_ids: np.ndarray,
) -> cdm_core.ComplementaryGroups:
    """Complementary groups over the sealed carrier under the frozen law.

    Valid units are finite rows with positive modulation (``m > 0``); the
    assignment itself is the frozen ``cdm_core.build_complementary_groups``
    round-robin over ``(atan2(c, a), magnitude, channel_id)``.
    """
    raw = np.asarray(initial_raw_t4, dtype=np.float64)
    channels = np.asarray(channel_ids, dtype=np.int64)
    _require(raw.ndim == 2 and raw.shape[1] == 4 and raw.shape[0] == channels.size,
             "carrier/channel topology drift")
    valid = np.isfinite(raw).all(axis=1) & (raw[:, 2] > 0.0)
    _require(int(valid.sum()) >= cdm_core.GROUP_COUNT,
             "fewer than four valid carrier rows for complementary grouping")
    return cdm_core.build_complementary_groups(raw, channels, valid_mask=valid)
