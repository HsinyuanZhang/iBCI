"""The Stage-P evidence bank, anchored refit and three-factor commit gate.

Design §4.2 (bounded pseudo evidence), §6 (the gate) and the work order's
commit law:

* block = 1 completed trial;
* every row binds the finalized trial ID, chronology, rate digest, direction
  digest, group identity, per-group confidence and valid-unit mask;
* a row carries an OPTIONAL label per complementary group -- group ``g``'s
  moments only ever include rows where group ``g`` has a label, so a group
  whose measurement or coverage factors failed is untouched by that trial
  while the other groups may still commit (design §5.1 is per group);
* the candidate is ALWAYS recomputed from ``A0/b0`` plus the accepted bank
  (the Stage-O recomputation law), so a rejected block reproduces the
  pre-block state byte-for-byte: nothing but the bank ever entered the refit;
* a block may commit only if ALL THREE factors pass (design §6):
  measurement confidence (evaluated in :mod:`.direction_estimator` under the
  source-selected ``tau_d``), direction coverage (logdet increase,
  nondecreasing minimum eigenvalue, repetition cap, distinct-direction
  warmup, maximum pseudo mass) and the support trust region (the Stage-O
  ``D2 <= c_M`` design-precision law, evaluated on the proposal).

The immutable anchor, the trust-region projection and the movement law are
reused VERBATIM from the proven Stage-O package
(:class:`src.support_anchored_t4_stage_o_v1.anchor.SupportAnchor` and
:mod:`src.support_anchored_t4_stage_o_v1.trust_region`); nothing there is
edited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from src.support_anchored_t4_stage_o_v1 import trust_region
    from src.support_anchored_t4_stage_o_v1.anchor import SupportAnchor
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1 import trust_region
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1.anchor import SupportAnchor

from . import direction_estimator


class StagePGateError(ValueError):
    """Fail-closed error for a malformed bank, refit or gate decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StagePGateError(message)


def _immutable(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _direction_row(theta_rad: float) -> np.ndarray:
    theta = float(theta_rad)
    _require(math.isfinite(theta), "direction angle must be finite")
    return np.asarray([math.cos(theta), math.sin(theta), 1.0], dtype=np.float64)


#: The frozen log-library epsilon for the coverage inequalities that hold
#: exactly in real arithmetic (a rank-1 positive update strictly increases
#: logdet; Weyl's inequality keeps the minimum eigenvalue nondecreasing).
COVERAGE_EPSILON = 1.0e-12


# ---------------------------------------------------------------------------
# The evidence row and bank.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockLabel:
    """One group's committed label inside one evidence row."""

    group: int
    estimator: str
    included_groups: tuple[int, ...]
    theta_index: int
    theta_raw_rad: float
    theta_canonical_rad: float
    canonical_distance_rad: float
    weight: float
    confidence_rule: str
    dispersion_rad: Optional[float] = None
    resultant_length: Optional[float] = None
    agreement: Optional[float] = None
    early_late_distance_rad: Optional[float] = None
    displacement_norm: Optional[float] = None

    def __post_init__(self) -> None:
        _require(0 <= int(self.group) < cdm_core.GROUP_COUNT, "label group index drift")
        _require(0 <= int(self.theta_index) < len(cdm_core.CANONICAL_DIRECTIONS_RAD),
                 "label direction index out of range")
        _require(math.isfinite(float(self.theta_raw_rad)), "label raw angle must be finite")
        _require(math.isfinite(float(self.weight)) and 0.0 < float(self.weight) <= 1.0,
                 "label weight must lie in (0, 1]")

    @classmethod
    def from_measurement(cls, record: direction_estimator.GroupMeasurement) -> "BlockLabel":
        return cls(
            group=record.group,
            estimator=record.estimator,
            included_groups=tuple(int(item) for item in record.included_groups),
            theta_index=int(record.theta_index),
            theta_raw_rad=float(record.theta_raw_rad),
            theta_canonical_rad=float(record.theta_canonical_rad),
            canonical_distance_rad=float(record.canonical_distance_rad),
            weight=float(record.weight),
            confidence_rule=record.confidence_rule,
            dispersion_rad=record.dispersion_rad,
            resultant_length=record.resultant_length,
            agreement=record.agreement,
            early_late_distance_rad=record.early_late_distance_rad,
            displacement_norm=record.displacement_norm,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "group": int(self.group),
            "estimator": self.estimator,
            "included_groups": [int(item) for item in self.included_groups],
            "theta_index": int(self.theta_index),
            "theta_raw_rad": float(self.theta_raw_rad),
            "theta_canonical_rad": float(self.theta_canonical_rad),
            "canonical_distance_rad": float(self.canonical_distance_rad),
            "weight": float(self.weight),
            "confidence_rule": self.confidence_rule,
            "dispersion_rad": None if self.dispersion_rad is None else float(self.dispersion_rad),
            "resultant_length": None if self.resultant_length is None else float(self.resultant_length),
            "agreement": None if self.agreement is None else float(self.agreement),
            "early_late_distance_rad": (
                None if self.early_late_distance_rad is None else float(self.early_late_distance_rad)
            ),
            "displacement_norm": None if self.displacement_norm is None else float(self.displacement_norm),
        }


@dataclass(frozen=True)
class TrialFacts:
    """The immutable finalized-trial binding every evidence row carries."""

    session_id: str
    trial_id: str
    chronology_position: int
    scalar_rates: np.ndarray = field(repr=False, compare=False)
    rate_sha256: str
    native_counts_sha256: str
    channel_order_sha256: str
    valid_mask_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.session_id) and bool(self.trial_id), "trial facts need session/trial ids")
        _require(int(self.chronology_position) >= 0, "chronology position must be nonnegative")
        rates = np.asarray(self.scalar_rates)
        _require(rates.ndim == 1 and np.isfinite(rates).all(), "scalar rates must be finite [N]")
        for digest in (self.rate_sha256, self.native_counts_sha256,
                       self.channel_order_sha256, self.valid_mask_sha256):
            _require(isinstance(digest, str) and len(digest) == 64, "trial facts digest drift")
        object.__setattr__(self, "scalar_rates", _immutable(rates, dtype=np.float64))


@dataclass(frozen=True)
class EvidenceRowP:
    """One finalized completed trial's per-group labels plus the gate record."""

    facts: TrialFacts
    block_index: int
    labels: tuple[Optional[BlockLabel], ...]
    gate_evidence: Mapping[str, Any]
    measurement_sha256: str

    def __post_init__(self) -> None:
        _require(int(self.block_index) >= 0, "evidence block index must be nonnegative")
        _require(len(self.labels) == cdm_core.GROUP_COUNT, "evidence row needs one slot per group")
        _require(any(item is not None for item in self.labels),
                 "a committed evidence row needs at least one label")
        for group, label in enumerate(self.labels):
            _require(label is None or int(label.group) == group,
                     "evidence label is bound to the wrong group slot")
        _require(isinstance(self.measurement_sha256, str) and len(self.measurement_sha256) == 64,
                 "evidence measurement digest drift")

    @property
    def trial_id(self) -> str:
        return self.facts.trial_id

    @property
    def chronology_position(self) -> int:
        return int(self.facts.chronology_position)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_p_evidence_row_v1",
            "session_id": self.facts.session_id,
            "trial_id": self.facts.trial_id,
            "chronology_position": int(self.facts.chronology_position),
            "block_index": int(self.block_index),
            "labels": [None if item is None else item.payload() for item in self.labels],
            "scalar_rates_sha256": self.facts.rate_sha256,
            "native_counts_sha256": self.facts.native_counts_sha256,
            "channel_order_sha256": self.facts.channel_order_sha256,
            "valid_mask_sha256": self.facts.valid_mask_sha256,
            "measurement_sha256": self.measurement_sha256,
            "gate_evidence": dict(self.gate_evidence),
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))


class EvidenceBankP:
    """An append-only bank of finalized Stage-P evidence rows."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[EvidenceRowP] = ()) -> None:
        checked = tuple(rows)
        seen: set[str] = set()
        previous_position = -1
        for position, row in enumerate(checked):
            _require(isinstance(row, EvidenceRowP), "bank accepts only typed Stage-P rows")
            _require(row.trial_id not in seen, f"duplicate evidence trial: {row.trial_id}")
            seen.add(row.trial_id)
            _require(row.chronology_position > previous_position,
                     "evidence chronology must be strictly increasing")
            _require(row.block_index == position,
                     "evidence block index must equal the bank position")
            previous_position = row.chronology_position
        self._rows: tuple[EvidenceRowP, ...] = checked

    @classmethod
    def empty(cls) -> "EvidenceBankP":
        return cls()

    @property
    def rows(self) -> tuple[EvidenceRowP, ...]:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def with_row(self, row: EvidenceRowP) -> "EvidenceBankP":
        return EvidenceBankP(self._rows + (row,))

    def drop_last_block(self, block_size: int = 1) -> "EvidenceBankP":
        """Remove the newest complete block; the recomputation law's inverse."""
        _require(int(block_size) == 1, "the Stage-P commit law binds block = 1 completed trial")
        _require(len(self._rows) >= 1, "cannot drop a block from an empty bank")
        return EvidenceBankP(self._rows[:-1])

    # -- per-group sufficient statistics ------------------------------------

    def direction_counts(self, group: int) -> np.ndarray:
        _require(0 <= int(group) < cdm_core.GROUP_COUNT, "bank group index drift")
        counts = np.zeros(len(cdm_core.CANONICAL_DIRECTIONS_RAD), dtype=np.int64)
        for row in self._rows:
            label = row.labels[int(group)]
            if label is not None:
                counts[int(label.theta_index)] += 1
        return counts

    def accepted_mass(self, group: int) -> float:
        return float(sum(
            float(row.labels[int(group)].weight)
            for row in self._rows if row.labels[int(group)] is not None
        ))

    def evidence_moments(self, anchor: SupportAnchor, group: int) -> tuple[np.ndarray, np.ndarray]:
        """``(S_g, t_g)`` over ONLY the rows where group ``g`` carries a label."""
        _require(0 <= int(group) < anchor.groups.group_count, "evidence group index drift")
        units = anchor.groups.unit_indices(group)
        S = np.zeros((3, 3), dtype=np.float64)
        t = np.zeros((3, units.size), dtype=np.float64)
        for row in self._rows:
            label = row.labels[int(group)]
            if label is None:
                continue
            design = _direction_row(cdm_core.CANONICAL_DIRECTIONS_RAD[int(label.theta_index)])
            weight = float(label.weight)
            S += weight * np.outer(design, design)
            t += weight * design[:, None] * np.asarray(row.facts.scalar_rates, dtype=np.float64)[units][None, :]
        return S, t

    @property
    def block_digests(self) -> list[str]:
        digests: list[str] = []
        for row in self._rows:
            digests.append(cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
                "row_sha256": row.digest, "bank_length": len(digests) + 1,
            })))
        return digests

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "support_anchored_t4_stage_p_evidence_bank_v1",
            "n_rows": len(self._rows),
            "accepted_mass_by_group": [self.accepted_mass(group) for group in range(cdm_core.GROUP_COUNT)],
            "rows": [row.payload() for row in self._rows],
            "block_digests": self.block_digests,
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes({
            "n_rows": len(self._rows),
            "row_digests": [row.digest for row in self._rows],
        }))


# ---------------------------------------------------------------------------
# The anchored refit (Stage-O law, per-group optional labels).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockRefitP:
    """One anchored refit: per-group At/bt, the solved candidate and coverage."""

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
            "schema": "support_anchored_t4_stage_p_block_refit_v1",
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


def _coverage_row(anchor: SupportAnchor, bank: EvidenceBankP, group: int) -> dict[str, Any]:
    counts = bank.direction_counts(group)
    present = np.flatnonzero(counts > 0).astype(np.int64, copy=False)
    return {
        "group": int(group),
        "accepted_evidence_rows": int(counts.sum()),
        "accepted_mass": bank.accepted_mass(group),
        "present_directions": [int(item) for item in present],
        "distinct_directions": int(present.size),
        "direction_histogram": [int(item) for item in counts],
    }


def refit_from_anchor_p(
    anchor: SupportAnchor, bank: EvidenceBankP, *, rho_M: float,
) -> BlockRefitP:
    """Solve the anchored block refit from A0/b0 + the Stage-P bank ONLY."""
    _require(math.isfinite(float(rho_M)) and float(rho_M) >= 0.0,
             "rho_M must be finite and nonnegative")
    At_blocks: list[np.ndarray] = []
    bt_blocks: list[np.ndarray] = []
    coefficient_blocks: list[np.ndarray] = []
    for group in range(anchor.groups.group_count):
        group_anchor = anchor.per_group[group]
        S, t = bank.evidence_moments(anchor, group)
        At = np.ascontiguousarray(group_anchor.A0 + float(rho_M) * S, dtype=np.float64)
        bt = np.ascontiguousarray(group_anchor.b0 + float(rho_M) * t, dtype=np.float64)
        _require(np.all(np.linalg.eigvalsh(At) > 0.0),
                 "anchored refit system is not positive definite")
        coefficients = np.linalg.solve(At, bt)
        _require(np.isfinite(coefficients).all(), "anchored refit produced nonfinite coefficients")
        At_blocks.append(At)
        bt_blocks.append(bt)
        coefficient_blocks.append(coefficients)
    candidate = anchor.rebuild_t4(coefficient_blocks)
    return BlockRefitP(
        anchor_digest=anchor.digest,
        bank_digest=bank.digest,
        rho_M=float(rho_M),
        At=tuple(At_blocks),
        bt=tuple(bt_blocks),
        coefficients=tuple(coefficient_blocks),
        candidate_t4=candidate,
        coverage_rows=tuple(_coverage_row(anchor, bank, group) for group in range(anchor.groups.group_count)),
    )


# ---------------------------------------------------------------------------
# The three-factor commit gate (design §6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateThresholds:
    """The source-selected (within-6) gate thresholds of one cell."""

    tau_d: float
    r_max: int
    d_min: int
    max_mass_relative: float

    def __post_init__(self) -> None:
        _require(math.isfinite(float(self.tau_d)) and 0.0 < float(self.tau_d) <= math.pi,
                 "tau_d must be a finite angle in (0, pi]")
        _require(int(self.r_max) >= 1, "r_max must be positive")
        _require(int(self.d_min) >= 1, "d_min must be positive")
        _require(math.isfinite(float(self.max_mass_relative)) and float(self.max_mass_relative) > 0.0,
                 "max pseudo mass multiple must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "tau_d_rad": float(self.tau_d),
            "r_max_repetition_per_direction": int(self.r_max),
            "d_min_distinct_directions": int(self.d_min),
            "max_pseudo_mass_relative_to_support_rows": float(self.max_mass_relative),
        }


@dataclass(frozen=True)
class BlockDecision:
    """The typed three-factor decision for one completed-trial block."""

    committed: bool
    labels: tuple[Optional[BlockLabel], ...]
    per_group_reasons: tuple[Optional[str], ...]
    block_reason: Optional[str]
    d2_unprojected: Optional[float]
    factor_evidence: Mapping[str, Any]
    coefficients: Optional[tuple[np.ndarray, ...]] = field(default=None, repr=False, compare=False)
    row: Optional[EvidenceRowP] = field(default=None, repr=False, compare=False)


def _coverage_factors(
    anchor: SupportAnchor,
    bank: EvidenceBankP,
    group: int,
    *,
    rho_M: float,
    thresholds: GateThresholds,
    theta_index: int,
    weight: float,
) -> dict[str, Any]:
    """The §6.2 direction-coverage factors for one (group, candidate label)."""
    counts_before = bank.direction_counts(group)
    counts = counts_before.copy()
    counts[int(theta_index)] += 1
    novel = int(counts_before[int(theta_index)]) == 0
    distinct = int(np.flatnonzero(counts > 0).size)
    repetition_ok = bool(int(counts[int(theta_index)]) <= int(thresholds.r_max))
    # The no-deadlock form of §6.2's "required number of distinct directions":
    # a block may commit if it is direction-NOVEL for the group, or once the
    # bank already spans >= d_min distinct canonical directions.  Without the
    # novelty clause a warmup could never complete (rejected rows never enter
    # the bank); with it, a session that only ever repeats one direction
    # contributes at most one row per group -- a repeated high-confidence
    # direction is not enough to update a full cosine tuning carrier.
    distinct_ok = bool(distinct >= int(thresholds.d_min) or novel)
    support_rows = int(anchor.per_group[group].design_row_count)
    mass_with = float(rho_M) * (bank.accepted_mass(group) + float(weight))
    mass_limit = float(thresholds.max_mass_relative) * support_rows
    mass_ok = bool(mass_with <= mass_limit + COVERAGE_EPSILON * max(1.0, mass_limit))
    S_current, _t = bank.evidence_moments(anchor, group)
    A_current = np.ascontiguousarray(anchor.per_group[group].A0 + float(rho_M) * S_current)
    design = _direction_row(cdm_core.CANONICAL_DIRECTIONS_RAD[int(theta_index)])
    A_with = np.ascontiguousarray(A_current + float(rho_M) * float(weight) * np.outer(design, design))
    logdet_current = float(np.linalg.slogdet(A_current)[1])
    logdet_with = float(np.linalg.slogdet(A_with)[1])
    min_eig_current = float(np.linalg.eigvalsh(A_current).min())
    min_eig_with = float(np.linalg.eigvalsh(A_with).min())
    logdet_ok = bool(logdet_with > logdet_current)
    min_eig_ok = bool(min_eig_with >= min_eig_current - COVERAGE_EPSILON * max(1.0, abs(min_eig_current)))
    return {
        "group": int(group),
        "repetition_count": int(counts[int(theta_index)]),
        "repetition_ok": repetition_ok,
        "direction_novel": bool(novel),
        "distinct_directions_with_row": distinct,
        "distinct_ok": distinct_ok,
        "distinct_directions_required": int(thresholds.d_min),
        "accepted_mass_with_row": mass_with,
        "max_pseudo_mass_limit": mass_limit,
        "mass_ok": mass_ok,
        "logdet_increase": logdet_with - logdet_current,
        "logdet_increase_ok": logdet_ok,
        "min_eigenvalue_before": min_eig_current,
        "min_eigenvalue_with_row": min_eig_with,
        "min_eigenvalue_nondecreasing": min_eig_ok,
        "coverage_ok": bool(repetition_ok and distinct_ok and mass_ok and logdet_ok and min_eig_ok),
    }


def evaluate_block(
    anchor: SupportAnchor,
    bank: EvidenceBankP,
    records: Sequence[Optional[direction_estimator.GroupMeasurement]],
    *,
    facts: TrialFacts,
    measurement_sha256: str,
    rho_M: float,
    c_M: Optional[float],
    thresholds: GateThresholds,
    block_index: int,
) -> BlockDecision:
    """Evaluate the three factors for one finalized trial's candidate labels.

    Factors 1 (measurement confidence) already ran in
    :func:`direction_estimator.measure_trial`; this function owns factor 2
    (direction coverage, per group) and factor 3 (the Stage-O support trust
    region, per block).  A block whose aggregate unprojected design-precision
    distance exceeds ``c_M`` is rejected WHOLE (no group commits), which is the
    §6.3 answer to "is the proposed displacement plausible under labeled
    support?" -- the proposal, not its projection, must be plausible.  The
    hypothetical refit binds the trial's REAL finalized rates and binding, so
    the decision's D2 is exactly the committed block's D2.
    """
    _require(len(records) == cdm_core.GROUP_COUNT, "gate needs one record slot per group")
    _require(int(block_index) == len(bank), "gate block index must extend the bank")
    candidates = [record for record in records if record is not None]
    if not candidates:
        return BlockDecision(
            committed=False,
            labels=(None,) * cdm_core.GROUP_COUNT,
            per_group_reasons=("measurement_factor_rejected",) * cdm_core.GROUP_COUNT,
            block_reason="no_group_passed_measurement",
            d2_unprojected=None,
            factor_evidence={"block_index": int(block_index), "factors": {
                "measurement_confidence": False, "direction_coverage": False,
                "support_trust_region": False,
            }},
        )
    coverage_rows: list[dict[str, Any]] = []
    labels: list[Optional[BlockLabel]] = [None] * cdm_core.GROUP_COUNT
    reasons: list[Optional[str]] = [None] * cdm_core.GROUP_COUNT
    for group, record in enumerate(records):
        if record is None:
            reasons[group] = "measurement_factor_rejected"
            continue
        factors = _coverage_factors(
            anchor, bank, group, rho_M=rho_M, thresholds=thresholds,
            theta_index=int(record.theta_index), weight=float(record.weight),
        )
        coverage_rows.append(factors)
        if factors["coverage_ok"]:
            labels[group] = BlockLabel.from_measurement(record)
        else:
            reasons[group] = "direction_coverage_rejected"
    if not any(item is not None for item in labels):
        return BlockDecision(
            committed=False,
            labels=(None,) * cdm_core.GROUP_COUNT,
            per_group_reasons=tuple(reasons),
            block_reason="direction_coverage_rejected_all_groups",
            d2_unprojected=None,
            factor_evidence={"block_index": int(block_index), "coverage_rows": coverage_rows,
                             "factors": {"measurement_confidence": True,
                                         "direction_coverage": False,
                                         "support_trust_region": False}},
        )
    # Factor 3: the Stage-O design-precision law on the hypothetical block.
    evidence: dict[str, Any] = {
        "block_index": int(block_index),
        "coverage_rows": coverage_rows,
    }
    row = EvidenceRowP(
        facts=facts, block_index=int(block_index), labels=tuple(labels),
        gate_evidence=evidence, measurement_sha256=measurement_sha256,
    )
    hypothetical = EvidenceBankP(bank.rows + (row,))
    refit = refit_from_anchor_p(anchor, hypothetical, rho_M=rho_M)
    support_blocks = anchor.coefficients(anchor.support_t4)
    delta_blocks = [
        np.asarray(refit.coefficients[group], dtype=np.float64) - np.asarray(support_blocks[group], dtype=np.float64)
        for group in range(anchor.groups.group_count)
    ]
    d2, per_group_d2 = trust_region.aggregate_d2(anchor, delta_blocks)
    evidence["d2_unprojected"] = float(d2)
    evidence["c_M"] = None if c_M is None else float(c_M)
    evidence["per_group_d2"] = [float(item) for item in per_group_d2]
    trust_ok = True if c_M is None else bool(d2 <= float(c_M))
    evidence["factors"] = {
        "measurement_confidence": True, "direction_coverage": True,
        "support_trust_region": bool(trust_ok),
    }
    if not trust_ok:
        return BlockDecision(
            committed=False,
            labels=(None,) * cdm_core.GROUP_COUNT,
            per_group_reasons=tuple(reasons),
            block_reason="support_trust_region_exceeded",
            d2_unprojected=float(d2),
            factor_evidence=evidence,
        )
    return BlockDecision(
        committed=True,
        labels=tuple(labels),
        per_group_reasons=tuple(reasons),
        block_reason="block_committed",
        d2_unprojected=float(d2),
        factor_evidence=evidence,
        coefficients=refit.coefficients,
        row=row,
    )


def empty_bank_refit_parity_p(anchor: SupportAnchor, *, rho_M: float) -> dict[str, Any]:
    """Diagnostic: the empty-bank Stage-P candidate vs the anchor's own solve."""
    refit = refit_from_anchor_p(anchor, EvidenceBankP.empty(), rho_M=rho_M)
    support_blocks = anchor.coefficients(anchor.support_t4)
    differences = [
        float(np.max(np.abs(np.asarray(refit.coefficients[g]) - np.asarray(support_blocks[g]))))
        for g in range(anchor.groups.group_count)
    ]
    return {
        "law": "empty Stage-P bank => solve(A0, b0), the anchor's own support coefficients",
        "max_abs_coefficient_difference_by_group": differences,
        "candidate_t4_sha256": cdm_core.array_digest(refit.candidate_t4),
        "support_t4_sha256": cdm_core.array_digest(np.asarray(anchor.support_t4)),
    }
