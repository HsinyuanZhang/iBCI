"""The deployable completed-trial pseudo-direction measurement (design §5).

Everything here is a predeclared function of the four frozen complementary
group forwards of one FINALIZED completed trial.  No query behavior label, no
target R2, no future trial and no carrier outcome enters the measurement.

Laws (frozen in :mod:`.plan`, one per Stage-P row):

* ``cross_group_circular`` (P1)  -- per group ``g`` the label is the circular
  mean of the OTHER groups' completed-trial directions,
  ``theta_hat_minus_g = atan2(mean sin, mean cos)`` over ``h != g``; group
  ``g``'s own prediction never labels group ``g`` (design §5.1);
* ``cross_group_trajectory`` (P2/P3/P4) -- trajectory-level aggregation: the
  per-group integrated displacements ``d_hat_h = sum_t v_hat_h[t] * dt`` (the
  exact integration domain of the frozen ``pseudo_direction_from_velocity``
  law) are SUMMED over ``h != g`` and the single ``atan2`` is taken AFTER the
  sum, plus the early/late displacement agreement statistic (design §5.2);
* ``same_group`` (P5, diagnostic only, self-referential) -- group ``g``'s own
  frozen per-group direction labels group ``g``.

Per-group direction facts always come from the frozen production law
``core.pseudo_direction_from_velocity`` under the session carrier's own
``CDMDConfig`` (the same law the Stage-O replay used), applied to the frozen
held-group forward views.  Weights are predeclared functions of §5.3 evidence
only: the complementary-group resultant length (P1), resultant length times
the early/late agreement factor (P2/P4), or the constant 1.0 (P3/P5).
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


class DirectionEstimatorError(ValueError):
    """Fail-closed error for a malformed measurement request or record."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectionEstimatorError(message)


#: The predeclared scale of the early/late agreement factor (design §5.3):
#: agreement = max(0, 1 - circular_distance(theta_early, theta_late) / (pi/2)),
#: and a measurement whose early/late ensemble directions disagree by >= pi/2
#: fails the trajectory-coherence pass (design §6.1) and yields no label.
EARLY_LATE_DISAGREEMENT_SCALE_RAD = math.pi / 2.0

DIRECTION_LAWS = ("cross_group_circular", "cross_group_trajectory", "same_group")
WEIGHT_LAWS = (
    "resultant_length",
    "resultant_times_early_late_agreement",
    "constant_one",
)
BINDINGS = ("complementary_exclusion", "rotated_group_and_confidence")


# ---------------------------------------------------------------------------
# Per-group frozen-law trajectory evidence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupTrajectoryEvidence:
    """One held-group forward's frozen direction facts and displacement rows."""

    group: int
    accepted: bool
    reason: Optional[str]
    theta_raw_rad: Optional[float]
    theta_index: Optional[int]
    theta_canonical_rad: Optional[float]
    canonical_distance_rad: Optional[float]
    movement_bins: int
    displacement_norm: Optional[float]
    mean_speed: Optional[float]
    displacement: np.ndarray = field(repr=False, compare=False)
    early_displacement: np.ndarray = field(repr=False, compare=False)
    late_displacement: np.ndarray = field(repr=False, compare=False)

    def payload(self) -> dict[str, Any]:
        return {
            "group": int(self.group),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "theta_raw_rad": self.theta_raw_rad,
            "theta_index": None if self.theta_index is None else int(self.theta_index),
            "theta_canonical_rad": self.theta_canonical_rad,
            "canonical_distance_rad": self.canonical_distance_rad,
            "movement_bins": int(self.movement_bins),
            "displacement_norm": self.displacement_norm,
            "mean_speed": self.mean_speed,
        }


def _immutable_vector(value: Sequence[float]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    _require(array.shape == (2,), "trajectory displacement must be [2]")
    array.setflags(write=False)
    return array


def group_trajectory_evidence(view: Any, *, group: int, config: Any) -> GroupTrajectoryEvidence:
    """Run the frozen per-group direction law and retain its displacement rows.

    ``pseudo_direction_from_velocity`` is called EXACTLY as the frozen CDM
    state machine and the Stage-O replay call it; the displacement vectors are
    the law's own integration (``sum_t v[t] * dt`` over the movement mask),
    retained once so the trajectory-level aggregation never re-integrates with
    a different arithmetic.  The early/late split is the predecladed first/second
    half of the SAME movement-mask rows (split at ``movement_bins // 2``).
    """
    _require(0 <= int(group) < cdm_core.GROUP_COUNT, "trajectory evidence group index drift")
    pseudo = cdm_core.pseudo_direction_from_velocity(
        view.velocity, view.validity.valid_mask, config=config,
    )
    mask = np.asarray(view.validity.valid_mask, dtype=np.bool_)
    selected = np.asarray(view.velocity, dtype=np.float64)[mask]
    dt = float(config.dt)
    displacement = selected.sum(axis=0) * dt
    split = int(selected.shape[0]) // 2
    early = selected[:split].sum(axis=0) * dt
    late = selected[split:].sum(axis=0) * dt
    return GroupTrajectoryEvidence(
        group=int(group),
        accepted=bool(pseudo.accepted),
        reason=None if pseudo.reason is None else pseudo.reason.value,
        theta_raw_rad=pseudo.theta_raw_rad,
        theta_index=pseudo.theta_index,
        theta_canonical_rad=pseudo.theta_canonical_rad,
        canonical_distance_rad=pseudo.canonical_distance_rad,
        movement_bins=int(pseudo.movement_bins),
        displacement_norm=pseudo.displacement_norm,
        mean_speed=pseudo.mean_speed,
        displacement=_immutable_vector(displacement),
        early_displacement=_immutable_vector(early),
        late_displacement=_immutable_vector(late),
    )


# ---------------------------------------------------------------------------
# The cross-group measurement records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupMeasurement:
    """One group's deployable pseudo-direction label and confidence evidence."""

    group: int
    estimator: str
    included_groups: tuple[int, ...]
    theta_raw_rad: float
    theta_index: int
    theta_canonical_rad: float
    canonical_distance_rad: float
    dispersion_rad: Optional[float]
    resultant_length: Optional[float]
    agreement: Optional[float]
    early_theta_rad: Optional[float]
    late_theta_rad: Optional[float]
    early_late_distance_rad: Optional[float]
    displacement_norm: float
    weight: float
    confidence_rule: str

    def payload(self) -> dict[str, Any]:
        return {
            "group": int(self.group),
            "estimator": self.estimator,
            "included_groups": [int(item) for item in self.included_groups],
            "theta_raw_rad": float(self.theta_raw_rad),
            "theta_index": int(self.theta_index),
            "theta_canonical_rad": float(self.theta_canonical_rad),
            "canonical_distance_rad": float(self.canonical_distance_rad),
            "dispersion_rad": None if self.dispersion_rad is None else float(self.dispersion_rad),
            "resultant_length": None if self.resultant_length is None else float(self.resultant_length),
            "agreement": None if self.agreement is None else float(self.agreement),
            "early_theta_rad": None if self.early_theta_rad is None else float(self.early_theta_rad),
            "late_theta_rad": None if self.late_theta_rad is None else float(self.late_theta_rad),
            "early_late_distance_rad": (
                None if self.early_late_distance_rad is None else float(self.early_late_distance_rad)
            ),
            "displacement_norm": float(self.displacement_norm),
            "weight": float(self.weight),
            "confidence_rule": self.confidence_rule,
        }


@dataclass(frozen=True)
class TrialMeasurement:
    """The full measurement of one finalized trial under one row's law."""

    law: str
    weight_law: str
    binding: str
    accepted: bool
    reason: Optional[str]
    per_group: tuple[Optional[GroupMeasurement], ...]
    rejection_reasons: tuple[Optional[str], ...]
    group_evidence: tuple[Mapping[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "law": self.law,
            "weight_law": self.weight_law,
            "binding": self.binding,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "per_group": [
                None if item is None else item.payload() for item in self.per_group
            ],
            "rejection_reasons": list(self.rejection_reasons),
            "group_evidence": [dict(item) for item in self.group_evidence],
        }

    @property
    def digest(self) -> str:
        return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(self.payload()))


def _circular_summary(thetas: Sequence[float]) -> tuple[float, float]:
    """``(circular mean, mean resultant length)`` of unit vectors."""
    values = np.asarray([float(item) for item in thetas], dtype=np.float64)
    _require(values.ndim == 1 and values.size >= 1, "circular summary needs angles")
    sine = float(np.mean(np.sin(values)))
    cosine = float(np.mean(np.cos(values)))
    return float(math.atan2(sine, cosine)), float(math.hypot(sine, cosine))


def _max_pairwise_distance(thetas: Sequence[float]) -> float:
    values = [float(item) for item in thetas]
    return float(max(
        cdm_core.circular_distance(left, right)
        for position, left in enumerate(values) for right in values[position + 1:]
    ))


def _snap(theta_rad: float, *, config: Any) -> tuple[int, float, float]:
    index, distance = cdm_core.nearest_canonical_direction(
        theta_rad, canonical_directions_rad=config.canonical_directions_rad,
    )
    return int(index), float(distance), float(config.canonical_directions_rad[index])


def _agreement(early_theta: float, late_theta: float) -> tuple[float, float]:
    """``(agreement factor in [0, 1], early/late circular distance)``."""
    distance = float(cdm_core.circular_distance(early_theta, late_theta))
    return max(0.0, 1.0 - distance / EARLY_LATE_DISAGREEMENT_SCALE_RAD), distance


def _measurement_records(
    evidence: Sequence[GroupTrajectoryEvidence],
    *,
    law: str,
    weight_law: str,
    tau_d: float,
    config: Any,
) -> tuple[Optional[GroupMeasurement], ...]:
    """Build one (possibly rejected) record per group under one cross-group law."""
    records: list[Optional[GroupMeasurement]] = []
    reasons: list[Optional[str]] = []
    for group in range(cdm_core.GROUP_COUNT):
        included = tuple(h for h in range(cdm_core.GROUP_COUNT) if h != group)
        thetas = [float(evidence[h].theta_raw_rad) for h in included]  # type: ignore[arg-type]
        dispersion = _max_pairwise_distance(thetas)
        if law == "cross_group_circular":
            theta, resultant = _circular_summary(thetas)
            aggregate_norm = float(min(float(evidence[h].displacement_norm) for h in included))  # type: ignore[arg-type]
            agreement_value: Optional[float] = None
            early_theta: Optional[float] = None
            late_theta: Optional[float] = None
            early_late_distance: Optional[float] = None
        elif law == "cross_group_trajectory":
            displacement = np.zeros(2, dtype=np.float64)
            early = np.zeros(2, dtype=np.float64)
            late = np.zeros(2, dtype=np.float64)
            for h in included:
                displacement = displacement + np.asarray(evidence[h].displacement, dtype=np.float64)
                early = early + np.asarray(evidence[h].early_displacement, dtype=np.float64)
                late = late + np.asarray(evidence[h].late_displacement, dtype=np.float64)
            theta = float(math.atan2(float(displacement[1]), float(displacement[0])))
            aggregate_norm = float(np.linalg.norm(displacement))
            _theta_only, resultant = _circular_summary(thetas)
            early_theta = float(math.atan2(float(early[1]), float(early[0])))
            late_theta = float(math.atan2(float(late[1]), float(late[0])))
            agreement_value, early_late_distance = _agreement(early_theta, late_theta)
        else:
            raise DirectionEstimatorError(f"unknown cross-group law: {law}")
        # -- the measurement-confidence factor (design §6.1) -------------------
        reason: Optional[str] = None
        if dispersion > float(tau_d):
            reason = cdm_core.UpdateRejectionReason.COMPLEMENTARY_DISAGREEMENT.value
        elif not math.isfinite(aggregate_norm) or aggregate_norm < float(config.minimum_displacement):
            reason = cdm_core.UpdateRejectionReason.LOW_DISPLACEMENT.value
        elif law == "cross_group_trajectory" and (
            agreement_value is None or not float(agreement_value) > 0.0
        ):
            reason = cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE.value
        if reason is not None:
            records.append(None)
            reasons.append(reason)
            continue
        index, canonical_distance, canonical = _snap(theta, config=config)
        if canonical_distance > float(config.max_canonical_distance_rad):
            records.append(None)
            reasons.append(cdm_core.UpdateRejectionReason.CANONICAL_DIRECTION_TOO_FAR.value)
            continue
        weight = _weight(
            weight_law=weight_law,
            resultant_length=resultant,
            agreement=agreement_value,
        )
        records.append(GroupMeasurement(
            group=group,
            estimator=law,
            included_groups=tuple(int(item) for item in included),
            theta_raw_rad=float(theta),
            theta_index=index,
            theta_canonical_rad=canonical,
            canonical_distance_rad=canonical_distance,
            dispersion_rad=float(dispersion),
            resultant_length=float(resultant),
            agreement=None if agreement_value is None else float(agreement_value),
            early_theta_rad=None if early_theta is None else float(early_theta),
            late_theta_rad=None if late_theta is None else float(late_theta),
            early_late_distance_rad=None if early_late_distance is None else float(early_late_distance),
            displacement_norm=float(aggregate_norm),
            weight=float(weight),
            confidence_rule=_confidence_rule(law=law, weight_law=weight_law),
        ))
        reasons.append(None)
    return tuple(records)


def _weight(*, weight_law: str, resultant_length: float, agreement: Optional[float]) -> float:
    if weight_law == "resultant_length":
        value = float(resultant_length)
    elif weight_law == "resultant_times_early_late_agreement":
        _require(agreement is not None, "agreement weight needs the trajectory law")
        value = float(resultant_length) * float(agreement)
    elif weight_law == "constant_one":
        value = 1.0
    else:
        raise DirectionEstimatorError(f"unknown weight law: {weight_law}")
    _require(math.isfinite(value) and 0.0 < value <= 1.0,
             f"confidence weight outside (0, 1]: {value}")
    return value


def _confidence_rule(*, law: str, weight_law: str) -> str:
    return f"{law}+{weight_law}"


def _rotate_binding(
    records: Sequence[Optional[GroupMeasurement]],
) -> tuple[Optional[GroupMeasurement], ...]:
    """The deterministic P4 group/confidence row shuffle (design §7.2/§8.3).

    Group ``g`` receives the trajectory-law record of group ``(g+1) mod K``
    and the CONFIDENCE of group ``(g+2) mod K`` of the same finalized trial.
    The rotation is within-trial, so it needs no future knowledge (a cross-trial
    reversal would break causality), it is fully deterministic with no tunable
    randomness, and it preserves every within-trial marginal while destroying
    both the complementary-group exclusion and the direction/confidence
    binding.  Because group ``g`` now receives a label derived from a
    prediction that INCLUDED group ``g``'s own units, the control is exactly
    the self-labeling binding the deployable law forbids.
    """
    rotated: list[Optional[GroupMeasurement]] = []
    count = cdm_core.GROUP_COUNT
    for group in range(count):
        source = records[(group + 1) % count]
        weight_source = records[(group + 2) % count]
        if source is None or weight_source is None:
            rotated.append(None)
            continue
        rotated.append(GroupMeasurement(
            group=group,
            estimator=source.estimator,
            included_groups=tuple(int(item) for item in source.included_groups),
            theta_raw_rad=source.theta_raw_rad,
            theta_index=source.theta_index,
            theta_canonical_rad=source.theta_canonical_rad,
            canonical_distance_rad=source.canonical_distance_rad,
            dispersion_rad=source.dispersion_rad,
            resultant_length=source.resultant_length,
            agreement=source.agreement,
            early_theta_rad=source.early_theta_rad,
            late_theta_rad=source.late_theta_rad,
            early_late_distance_rad=source.early_late_distance_rad,
            displacement_norm=source.displacement_norm,
            weight=float(weight_source.weight),
            confidence_rule=source.confidence_rule + "+rotated_group_and_confidence",
        ))
    return tuple(rotated)


def measure_trial(
    views: Sequence[Any],
    *,
    config: Any,
    law: str,
    weight_law: str,
    binding: str,
    tau_d: float,
) -> TrialMeasurement:
    """Measure one finalized trial under one Stage-P row's law.

    The four frozen per-group direction facts must ALL be acceptable under the
    frozen production gates (the frozen CDM law's own all-four requirement,
    reused verbatim by Stage O); a trial whose any held-group forward fails the
    frozen displacement/speed/snap gates yields no label for ANY group and the
    frozen typed reason is recorded.
    """
    _require(law in DIRECTION_LAWS, f"unknown direction law: {law}")
    _require(weight_law in WEIGHT_LAWS, f"unknown weight law: {weight_law}")
    _require(binding in BINDINGS, f"unknown record binding: {binding}")
    rows = tuple(views)
    _require(len(rows) == cdm_core.GROUP_COUNT, "measurement needs exactly four group views")
    if law == "same_group":
        _require(weight_law == "constant_one",
                 "the same-group diagnostic uses the constant confidence law")
    _require(math.isfinite(float(tau_d)) and 0.0 < float(tau_d) <= math.pi,
             "tau_d must be a finite angle in (0, pi]")
    evidence = tuple(
        group_trajectory_evidence(view, group=group, config=config)
        for group, view in enumerate(rows)
    )
    payloads = tuple(item.payload() for item in evidence)
    failed = next((item for item in evidence if not item.accepted), None)
    if failed is not None:
        return TrialMeasurement(
            law=law, weight_law=weight_law, binding=binding,
            accepted=False, reason=failed.reason, per_group=(None,) * cdm_core.GROUP_COUNT,
            rejection_reasons=tuple(failed.reason for _ in range(cdm_core.GROUP_COUNT)),
            group_evidence=payloads,
        )
    if law == "same_group":
        records: tuple[Optional[GroupMeasurement], ...] = tuple(
            GroupMeasurement(
                group=group,
                estimator="same_group_self_labeling",
                included_groups=(int(group),),
                theta_raw_rad=float(item.theta_raw_rad),  # type: ignore[arg-type]
                theta_index=int(item.theta_index),  # type: ignore[arg-type]
                theta_canonical_rad=float(item.theta_canonical_rad),  # type: ignore[arg-type]
                canonical_distance_rad=float(item.canonical_distance_rad),  # type: ignore[arg-type]
                dispersion_rad=None,
                resultant_length=None,
                agreement=None,
                early_theta_rad=None,
                late_theta_rad=None,
                early_late_distance_rad=None,
                displacement_norm=float(item.displacement_norm),  # type: ignore[arg-type]
                weight=_weight(weight_law=weight_law, resultant_length=1.0, agreement=None),
                confidence_rule=_confidence_rule(law=law, weight_law=weight_law)
                + "+self_referential",
            )
            for group, item in enumerate(evidence)
        )
        reasons: tuple[Optional[str], ...] = (None,) * cdm_core.GROUP_COUNT
    else:
        records = _measurement_records(
            evidence, law=law, weight_law=weight_law, tau_d=float(tau_d), config=config,
        )
        reasons = tuple(
            None if item is not None else "measurement_factor_rejected"
            for item in records
        )
    if binding == "rotated_group_and_confidence":
        records = _rotate_binding(records)
    return TrialMeasurement(
        law=law, weight_law=weight_law, binding=binding,
        accepted=any(item is not None for item in records),
        reason=None if any(item is not None for item in records) else "no_group_passed_measurement",
        per_group=records,
        rejection_reasons=reasons,
        group_evidence=payloads,
    )
