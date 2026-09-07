"""The immutable labeled-support anchor of the Stage-O carrier memory.

Design §4.1.  For each complementary group this module preserves the EXACT
production T4 regression convention of
``src/causal_dual_memory_cell_d_v1/core.py``:

* ``fit_carriers_from_trial_table(..., mode=FIXED_RIDGE_BY_TRIAL)`` builds the
  per-trial design ``[cos(theta), sin(theta), 1]``, the penalty
  ``diag(n * lambda, n * lambda, 0)`` with ``lambda = 0.1``, and solves
  ``solve(design.T @ design + penalty, design.T @ rates)``;
* ``CarrierSufficientStatistics.from_labeled_support`` accumulates the same
  law as per-group counts/sums.

The anchor therefore carries, per group:

    A0 = Xs^T Xs + diag(n_s * lambda, n_s * lambda, 0)
    b0 = Xs^T rs   (response restricted to the group's units)
    T4_support = solve(A0, b0)

with ``T4_support`` additionally parity-checked against the SEALED
``fit_carriers_from_trial_table`` carrier the frozen runtime installs (that
sealed float32 carrier stays the exact no-update fallback and the O0 arm).
The support carrier, its sufficient statistics and their digests never change;
all Stage-O movement lives in the separate evidence bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core


class SupportAnchorError(ValueError):
    """Fail-closed error for a malformed or drifting support anchor."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SupportAnchorError(message)


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
class GroupSupportAnchor:
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
        _require(counts.shape == (len(cdm_core.CANONICAL_DIRECTIONS_RAD),), "anchor counts shape drift")
        _require(A0.shape == (3, 3) and b0.shape == (3, units.size), "anchor A0/b0 shape drift")
        _require(np.all(np.linalg.eigvalsh(A0) > 0.0), "anchor A0 must be positive definite")
        _require(np.isfinite(A0).all() and np.isfinite(b0).all() and np.all(counts >= 0), "anchor nonfinite statistics")
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


def _support_design(direction_indices: Sequence[int]) -> np.ndarray:
    theta = np.asarray(
        [cdm_core.CANONICAL_DIRECTIONS_RAD[int(index)] for index in direction_indices], dtype=np.float64,
    )
    return np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size, dtype=np.float64)))


def support_penalty(support_rows: int, *, normalized_lambda: float) -> np.ndarray:
    """The exact production fixed-ridge penalty of the trial-table fit."""
    weight = float(support_rows) * float(normalized_lambda)
    return np.diag((weight, weight, 0.0))


@dataclass(frozen=True)
class SupportAnchor:
    """The immutable per-group A0/b0 bank plus the sealed support carrier."""

    groups: cdm_core.ComplementaryGroups = field(repr=False, compare=False)
    support_t4: np.ndarray = field(repr=False, compare=False)
    support_direction_indices: tuple[int, ...]
    rates_sha256: str
    per_group: tuple[GroupSupportAnchor, ...] = field(repr=False, compare=False)
    statistics: cdm_core.CarrierSufficientStatistics = field(repr=False, compare=False)
    support_coefficient_parity: Mapping[str, Any] = field(repr=False, compare=False)
    normalized_lambda: float = cdm_core.RIDGE_NORMALIZED_LAMBDA

    def __post_init__(self) -> None:
        support = np.asarray(self.support_t4)
        _require(support.shape == (self.groups.units, 4), "support T4 shape drift")
        _require(np.isfinite(support).all(), "support T4 must be finite")
        _require(len(self.per_group) == self.groups.group_count, "anchor group count drift")
        for group_anchor in self.per_group:
            _require(group_anchor.group < self.groups.group_count, "anchor group binding drift")
            expected = self.groups.unit_indices(group_anchor.group)
            _require(np.array_equal(np.asarray(group_anchor.units), expected), "anchor unit binding drift")
        _require(
            float(self.normalized_lambda) == cdm_core.RIDGE_NORMALIZED_LAMBDA,
            "anchor must use the production normalized lambda 0.1",
        )
        _require(isinstance(self.statistics, cdm_core.CarrierSufficientStatistics), "anchor statistics type drift")
        object.__setattr__(self, "support_t4", _immutable(support))
        object.__setattr__(self, "support_direction_indices", tuple(int(item) for item in self.support_direction_indices))

    # -- construction -------------------------------------------------------

    @classmethod
    def from_labeled_support(
        cls,
        *,
        groups: cdm_core.ComplementaryGroups,
        support_trial_rates: Any,
        support_direction_indices: Sequence[int],
        support_t4: Any,
        normalized_lambda: float = cdm_core.RIDGE_NORMALIZED_LAMBDA,
    ) -> "SupportAnchor":
        """Build the anchor from the same inputs the frozen runtime feeds ``CarrierMemory``."""
        rates = np.ascontiguousarray(np.asarray(support_trial_rates), dtype=np.float64)
        directions = np.asarray([int(item) for item in support_direction_indices], dtype=np.int64)
        support = np.asarray(support_t4)
        n_support = int(rates.shape[0])
        _require(rates.ndim == 2 and rates.shape[1] == groups.units and n_support >= 1,
                 "support rates must be [trials, units]")
        _require(np.isfinite(rates).all(), "support rates must be finite")
        _require(directions.shape == (n_support,), "support direction count drift")
        _require(np.all((directions >= 0) & (directions < len(cdm_core.CANONICAL_DIRECTIONS_RAD))),
                 "support direction index out of range")
        design = _support_design(directions.tolist())
        penalty = support_penalty(n_support, normalized_lambda=normalized_lambda)
        a0_common = design.T @ design + penalty
        per_group: list[GroupSupportAnchor] = []
        statistics = cdm_core.CarrierSufficientStatistics.from_labeled_support(groups, rates, directions)
        parity_max_abs = 0.0
        parity_max_relative = 0.0
        parity_rebuilt_bitwise = True
        for group in range(groups.group_count):
            units = groups.unit_indices(group)
            b0 = np.ascontiguousarray(design.T @ rates[:, units], dtype=np.float64)
            per_group.append(GroupSupportAnchor(
                group=group, units=units, counts=np.asarray(statistics.counts[group]), A0=a0_common, b0=b0,
            ))
            solved = np.linalg.solve(a0_common, b0)
            sealed = np.asarray(support, dtype=np.float64)[units][:, [0, 1, 3]].T
            parity_max_abs = max(parity_max_abs, float(np.max(np.abs(solved - sealed))))
            parity_max_relative = max(parity_max_relative, float(np.max(
                np.abs(solved - sealed) / np.maximum(np.abs(sealed), 1.0e-12)
            )))
            # The float64 solve rounds to the SAME carrier rows the sealed table
            # fit produced (after the production float32 quantization and the
            # sealed dtype mirror): the empty-bank refit is the sealed carrier
            # bitwise.
            formatted = _format_t4_rows(solved)
            quantized = np.ascontiguousarray(formatted, dtype=np.float32).astype(np.float64)
            rebuilt = np.ascontiguousarray(quantized, dtype=np.asarray(support).dtype)
            parity_rebuilt_bitwise = parity_rebuilt_bitwise and np.array_equal(
                rebuilt, np.asarray(support)[units],
            )
        return cls(
            groups=groups,
            support_t4=support,
            support_direction_indices=tuple(int(item) for item in directions.tolist()),
            rates_sha256=cdm_core.array_digest(rates),
            per_group=tuple(per_group),
            statistics=statistics,
            normalized_lambda=float(normalized_lambda),
            support_coefficient_parity={
                "law": "solve(A0, b0) vs the sealed fit_carriers_from_trial_table carrier",
                "max_abs_coefficient_difference": parity_max_abs,
                "max_relative_coefficient_difference": parity_max_relative,
                "rebuilt_rows_bitwise_equal": bool(parity_rebuilt_bitwise),
            },
        )

    # -- coefficient space --------------------------------------------------

    def coefficients(self, t4: Any) -> tuple[np.ndarray, ...]:
        """Per-group (3, U) float64 coefficient blocks (a, c, b) of a [N,4] T4."""
        values = np.asarray(t4)
        _require(values.shape == (self.groups.units, 4), "T4 coefficient extraction shape drift")
        blocks = np.asarray(values, dtype=np.float64)
        return tuple(
            np.ascontiguousarray(blocks[self.groups.unit_indices(group)][:, [0, 1, 3]].T)
            for group in range(self.groups.group_count)
        )

    def rebuild_t4(self, coefficient_blocks: Sequence[np.ndarray]) -> np.ndarray:
        """Rebuild the production [a, c, hypot(a,c), b] carrier.

        Valid units take their group's refit rows; invalid units keep the sealed
        support rows, mirroring ``CarrierMemory._reconstruct``.  The production
        fixed-ridge estimand is quantized to float32 (``_reconstruct``'s law;
        the sealed initializer is the float32 table fit), and the result is
        presented in the SEALED support carrier's own dtype -- the frozen
        runtime installs a float64 ``active_t4`` holding the float32 table-fit
        values.  A no-movement rebuild is therefore byte-identical to the
        support carrier and the sealed normalizer path sees the dtype it always
        saw.
        """
        _require(len(coefficient_blocks) == self.groups.group_count, "rebuild needs one block per group")
        support = np.asarray(self.support_t4)
        seed = np.array(support, dtype=np.float64, copy=True)
        for group in range(self.groups.group_count):
            units = self.groups.unit_indices(group)
            formatted = _format_t4_rows(np.asarray(coefficient_blocks[group], dtype=np.float64))
            seed[units] = np.ascontiguousarray(formatted, dtype=np.float32).astype(np.float64)
        result = np.ascontiguousarray(seed, dtype=support.dtype)
        result.setflags(write=False)
        _require(np.isfinite(result).all(), "rebuilt carrier is nonfinite")
        return result

    # -- digests ------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_o_support_anchor_v1",
            "groups_sha256": self.groups.digest,
            "statistics_sha256": self.statistics.digest,
            "support_t4": _frozen_array_payload(np.asarray(self.support_t4)),
            "support_direction_indices": [int(item) for item in self.support_direction_indices],
            "support_rates_sha256": self.rates_sha256,
            "normalized_lambda": float(self.normalized_lambda),
            "per_group": [item.payload() for item in self.per_group],
            "support_coefficient_parity": dict(self.support_coefficient_parity),
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))

    @property
    def a0_b0_digest(self) -> str:
        """Digest of ONLY the sufficient statistics (the immutability proof)."""
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
            "per_group": [item.digest for item in self.per_group],
            "statistics_sha256": self.statistics.digest,
            "support_t4_sha256": cdm_core.array_digest(np.asarray(self.support_t4)),
        }))


def _format_t4_rows(coefficients: np.ndarray) -> np.ndarray:
    """[3, U] (a, c, b) -> [U, 4] (a, c, hypot(a, c), b), the production layout."""
    values = np.asarray(coefficients, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == 3, "coefficient block must be [3, U]")
    a, c, b = values[0], values[1], values[2]
    rows = np.column_stack((a, c, np.hypot(a, c), b))
    _require(np.isfinite(rows).all(), "formatted carrier rows are nonfinite")
    return rows
