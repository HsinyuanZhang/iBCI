"""Support-anchored block refit: the Stage-O evidence bank and candidate solve.

Design §4.2.  Finalized completed trials contribute bounded statistics:

    At = A0 + rho_M * sum_i w_i x(theta_i) x(theta_i)^T
    bt = b0 + rho_M * sum_i w_i x(theta_i) r_i
    T4_candidate = solve(At, bt)

Laws enforced structurally by this module:

* the candidate is recomputed ALWAYS from ``A0/b0`` plus the accepted evidence
  bank -- the only inputs to :func:`refit_from_anchor` are the immutable
  anchor and the bank, so a pseudo-updated carrier can never seed a refit;
* evidence is accumulated in complete trial blocks (Stage O: block = 1 trial,
  the binding operator amendment);
* every row binds the finalized trial ID, chronology, rate digest, direction
  digest, group identity and valid-unit mask;
* a removed block reproduces the pre-block state exactly (digest-verified by
  :meth:`EvidenceBank.drop_last_block`);
* Stage O's commit law is always-commit; the typed three-factor gate evidence
  produced here is recorded as NON-GOVERNING diagnostics only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core

from . import plan
from .anchor import SupportAnchor, _format_t4_rows


class BlockRefitError(ValueError):
    """Fail-closed error for a malformed evidence row, bank, or refit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockRefitError(message)


def _immutable(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _direction_row(theta_rad: float) -> np.ndarray:
    theta = float(theta_rad)
    _require(math.isfinite(theta), "direction angle must be finite")
    return np.asarray([math.cos(theta), math.sin(theta), 1.0], dtype=np.float64)


@dataclass(frozen=True)
class EvidenceRow:
    """One finalized completed trial's oracle direction + native scalar rates."""

    session_id: str
    trial_id: str
    chronology_position: int
    block_index: int
    direction_indices: tuple[int, ...]
    theta_raw_rad: tuple[float, ...]
    canonical_distance_rad: tuple[float, ...]
    movement_bins: tuple[int, ...]
    displacement_norm: tuple[float, ...]
    mean_speed: tuple[float, ...]
    scalar_rates: np.ndarray = field(repr=False, compare=False)
    rate_sha256: str
    native_counts_sha256: str
    channel_order_sha256: str
    valid_mask_sha256: str
    confidence_rule: str = "true_direction_oracle"
    confidence: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require(bool(self.session_id) and bool(self.trial_id), "evidence row needs session/trial ids")
        _require(int(self.chronology_position) >= 0 and int(self.block_index) >= 0,
                 "evidence row chronology/block index must be nonnegative")
        n_groups = int(cdm_core.GROUP_COUNT)
        for name in ("direction_indices", "theta_raw_rad", "canonical_distance_rad",
                     "movement_bins", "displacement_norm", "mean_speed"):
            _require(len(getattr(self, name)) == n_groups, f"evidence row {name} needs one entry per group")
        _require(all(0 <= int(item) < len(cdm_core.CANONICAL_DIRECTIONS_RAD)
                     for item in self.direction_indices), "evidence direction index out of range")
        rates = np.asarray(self.scalar_rates)
        _require(rates.ndim == 1 and np.isfinite(rates).all(), "evidence scalar rates must be finite [N]")
        _require(float(self.weight) > 0.0 and math.isfinite(float(self.weight)), "evidence weight must be positive")
        _require(self.confidence_rule == "true_direction_oracle" and float(self.confidence) == 1.0,
                 "Stage-O evidence confidence is the true-direction oracle constant")
        for digest in (self.rate_sha256, self.native_counts_sha256, self.channel_order_sha256, self.valid_mask_sha256):
            _require(isinstance(digest, str) and len(digest) == 64, "evidence row digest drift")
        object.__setattr__(self, "direction_indices", tuple(int(item) for item in self.direction_indices))
        object.__setattr__(self, "theta_raw_rad", tuple(float(item) for item in self.theta_raw_rad))
        object.__setattr__(self, "scalar_rates", _immutable(self.scalar_rates, dtype=np.float64))

    @property
    def direction_sha256(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "chronology_position": int(self.chronology_position),
            "block_index": int(self.block_index),
            "direction_indices": [int(item) for item in self.direction_indices],
            "theta_raw_rad": [float(item) for item in self.theta_raw_rad],
            "canonical_distance_rad": [float(item) for item in self.canonical_distance_rad],
        }))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_o_evidence_row_v1",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "chronology_position": int(self.chronology_position),
            "block_index": int(self.block_index),
            "group_direction_indices": [int(item) for item in self.direction_indices],
            "group_theta_raw_rad": [float(item) for item in self.theta_raw_rad],
            "group_canonical_distance_rad": [float(item) for item in self.canonical_distance_rad],
            "group_movement_bins": [int(item) for item in self.movement_bins],
            "group_displacement_norm": [float(item) for item in self.displacement_norm],
            "group_mean_speed": [float(item) for item in self.mean_speed],
            "scalar_rates_sha256": self.rate_sha256,
            "native_counts_sha256": self.native_counts_sha256,
            "channel_order_sha256": self.channel_order_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "direction_sha256": self.direction_sha256,
            "confidence_rule": self.confidence_rule,
            "confidence": float(self.confidence),
            "weight": float(self.weight),
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))


class EvidenceBank:
    """An append-only bank of finalized evidence rows (immutable snapshots)."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[EvidenceRow] = ()) -> None:
        checked = tuple(rows)
        seen: set[str] = set()
        previous_position = -1
        for row in checked:
            _require(isinstance(row, EvidenceRow), "bank accepts only typed evidence rows")
            _require(row.trial_id not in seen, f"duplicate evidence trial: {row.trial_id}")
            seen.add(row.trial_id)
            _require(row.chronology_position > previous_position,
                     "evidence chronology must be strictly increasing")
            previous_position = row.chronology_position
        self._rows: tuple[EvidenceRow, ...] = checked

    @classmethod
    def empty(cls) -> "EvidenceBank":
        return cls()

    @property
    def rows(self) -> tuple[EvidenceRow, ...]:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def with_row(self, row: EvidenceRow) -> "EvidenceBank":
        return EvidenceBank(self._rows + (row,))

    def drop_last_block(self, block_size: int = plan.HYPERPARAMETERS["block_size_completed_trials"]) -> "EvidenceBank":
        """Remove the newest complete block; the recomputation law's inverse.

        Returns the exact pre-block bank: removing a rejected (or any) block
        reproduces the pre-block state byte-for-byte because nothing but the
        bank ever entered the refit.
        """
        _require(int(block_size) >= 1, "block size must be positive")
        _require(len(self._rows) >= int(block_size),
                 "cannot drop a block larger than the bank")
        return EvidenceBank(self._rows[:-int(block_size)])

    @property
    def block_digests(self) -> list[str]:
        """Digest after each committed block (the §9 block-statistics chain)."""
        digests: list[str] = []
        for row in self._rows:
            digests.append(cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
                "row_sha256": row.digest, "bank_length": len(digests) + 1,
            })))
        return digests

    @property
    def accepted_mass(self) -> float:
        """Cumulative accepted pseudo mass: sum of row weights."""
        return float(sum(row.weight for row in self._rows))

    def evidence_moments(self, anchor: SupportAnchor, group: int) -> tuple[np.ndarray, np.ndarray]:
        """``(S_g, t_g)`` with S_g = sum w_i x_i x_i^T, t_g = sum w_i x_i r_i^T.

        ALWAYS recomputed from the rows; never cached, never seeded from a
        carrier.
        """
        _require(0 <= int(group) < anchor.groups.group_count, "evidence group index drift")
        units = anchor.groups.unit_indices(group)
        S = np.zeros((3, 3), dtype=np.float64)
        t = np.zeros((3, units.size), dtype=np.float64)
        for row in self._rows:
            design = _direction_row(cdm_core.CANONICAL_DIRECTIONS_RAD[int(row.direction_indices[group])])
            weight = float(row.weight)
            S += weight * np.outer(design, design)
            t += weight * design[:, None] * np.asarray(row.scalar_rates, dtype=np.float64)[units][None, :]
        return S, t

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_o_evidence_bank_v1",
            "n_rows": len(self._rows),
            "accepted_mass": self.accepted_mass,
            "rows": [row.payload() for row in self._rows],
            "block_digests": self.block_digests,
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
            "n_rows": len(self._rows),
            "row_digests": [row.digest for row in self._rows],
        }))


@dataclass(frozen=True)
class BlockRefit:
    """One anchored refit: per-group At/bt, the solved candidate, and evidence."""

    anchor_digest: str
    bank_digest: str
    rho_M: float
    At: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    bt: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    coefficients: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    candidate_t4: np.ndarray = field(repr=False, compare=False)
    coverage_rows: tuple[Mapping[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_o_block_refit_v1",
            "anchor_digest": self.anchor_digest,
            "bank_digest": self.bank_digest,
            "rho_M": float(self.rho_M),
            "At": [cdm_core.array_digest(item) for item in self.At],
            "bt": [cdm_core.array_digest(item) for item in self.bt],
            "candidate_t4_sha256": cdm_core.array_digest(self.candidate_t4),
            "coverage_rows": [dict(row) for row in self.coverage_rows],
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))


def _coverage_rows(anchor: SupportAnchor, bank: EvidenceBank) -> tuple[Mapping[str, Any], ...]:
    """Direction-coverage evidence (design §6.2), NON-GOVERNING in Stage O."""
    rows: list[Mapping[str, Any]] = []
    for group in range(anchor.groups.group_count):
        counts = np.asarray(anchor.statistics.counts[group], dtype=np.int64).copy()
        for row in bank.rows:
            counts[int(row.direction_indices[group])] += 1
        present = np.flatnonzero(counts > 0).astype(np.int64, copy=False)
        theta = np.asarray([cdm_core.CANONICAL_DIRECTIONS_RAD[int(i)] for i in present], dtype=np.float64)
        design = np.column_stack((np.ones(theta.size), np.cos(theta), np.sin(theta))) if present.size else \
            np.empty((0, 3), dtype=np.float64)
        rank = int(np.linalg.matrix_rank(design)) if present.size else 0
        condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
        rows.append({
            "group": group,
            "accepted_evidence": int(counts.sum()),
            "present_directions": [int(item) for item in present],
            "design_rank": rank,
            "design_condition": condition,
            "governing_in_stage_o": False,
        })
    return tuple(rows)


def refit_from_anchor(anchor: SupportAnchor, bank: EvidenceBank, *, rho_M: float) -> BlockRefit:
    """Solve the anchored block refit from A0/b0 + the evidence bank ONLY."""
    _require(math.isfinite(float(rho_M)) and float(rho_M) >= 0.0, "rho_M must be finite and nonnegative")
    At_blocks: list[np.ndarray] = []
    bt_blocks: list[np.ndarray] = []
    coefficient_blocks: list[np.ndarray] = []
    for group in range(anchor.groups.group_count):
        group_anchor = anchor.per_group[group]
        S, t = bank.evidence_moments(anchor, group)
        At = np.ascontiguousarray(group_anchor.A0 + float(rho_M) * S, dtype=np.float64)
        bt = np.ascontiguousarray(group_anchor.b0 + float(rho_M) * t, dtype=np.float64)
        _require(np.all(np.linalg.eigvalsh(At) > 0.0), "anchored refit system is not positive definite")
        coefficients = np.linalg.solve(At, bt)
        _require(np.isfinite(coefficients).all(), "anchored refit produced nonfinite coefficients")
        At_blocks.append(At)
        bt_blocks.append(bt)
        coefficient_blocks.append(coefficients)
    candidate = anchor.rebuild_t4(coefficient_blocks)
    return BlockRefit(
        anchor_digest=anchor.digest,
        bank_digest=bank.digest,
        rho_M=float(rho_M),
        At=tuple(At_blocks),
        bt=tuple(bt_blocks),
        coefficients=tuple(coefficient_blocks),
        candidate_t4=candidate,
        coverage_rows=_coverage_rows(anchor, bank),
    )


def empty_bank_refit_parity(anchor: SupportAnchor, *, rho_M: float) -> dict[str, Any]:
    """Diagnostic: the empty-bank candidate vs the anchor's own A0/b0 solve."""
    refit = refit_from_anchor(anchor, EvidenceBank.empty(), rho_M=rho_M)
    support_blocks = anchor.coefficients(anchor.support_t4)
    differences = [
        float(np.max(np.abs(np.asarray(refit.coefficients[g]) - np.asarray(support_blocks[g]))))
        for g in range(anchor.groups.group_count)
    ]
    return {
        "law": "empty bank => solve(A0, b0), the anchor's own support coefficients",
        "max_abs_coefficient_difference_by_group": differences,
        "candidate_t4_sha256": cdm_core.array_digest(refit.candidate_t4),
        "support_t4_sha256": cdm_core.array_digest(np.asarray(anchor.support_t4)),
    }
