"""Conditional-posterior, frozen-support precision gate for CDM-D V2.

This is deliberately a thin transition wrapper.  The shared independent
activity state machine still owns B8, pseudo-direction, design, departure,
and commit semantics.  V2 only vetoes an already accepted carrier proposal
against a support-only conditional Gaussian-ridge posterior region.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# Use a package-relative dependency so the exact gate can be embedded in both
# the historical top-level ``src`` route and a qualified
# ``tfpd_exploration.src`` route.  The latter is required when the evaluated
# decoder itself owns the unrelated top-level ``src`` namespace.
from ..causal_dual_memory_cell_d_v1 import core

from . import plan


class PrecisionAwareV2TransitionError(RuntimeError):
    """Fail closed for V2 posterior precision or transition drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionAwareV2TransitionError(message)


def _immutable(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _array_payload(value: np.ndarray) -> dict[str, object]:
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": core.array_digest(value)}


def _canonical_design(direction_indices: np.ndarray) -> np.ndarray:
    _require(direction_indices.ndim == 1 and int(direction_indices.size) in plan.TRANSITION_BUDGETS,
             "precision V2 support directions must be exact M4 or M10")
    _require(direction_indices.dtype == np.int64, "precision V2 support directions must be int64")
    _require(np.all((direction_indices >= 0) & (direction_indices < len(core.CANONICAL_DIRECTIONS_RAD))),
             "precision V2 support direction out of canonical range")
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[int(item)] for item in direction_indices], dtype=np.float64)
    return np.ascontiguousarray(np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size))), dtype=np.float64)


def _posterior_terms(
    support_rates: np.ndarray, support_direction_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return exact float64 conditional Gaussian-ridge posterior terms."""
    design = _canonical_design(support_direction_indices)
    budget = int(support_rates.shape[0])
    penalty = np.diag((budget * plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
                       budget * plan.FIXED_RIDGE_NORMALIZED_LAMBDA, 0.0)).astype(np.float64)
    normal = np.ascontiguousarray(design.T @ design + penalty, dtype=np.float64)
    try:
        normal_inverse = np.linalg.inv(normal)
        beta = np.linalg.solve(normal, design.T @ support_rates)
    except np.linalg.LinAlgError as error:
        raise PrecisionAwareV2TransitionError("precision V2 fixed-ridge normal matrix is singular") from error
    residual = np.ascontiguousarray(support_rates - design @ beta, dtype=np.float64)
    hat = np.ascontiguousarray(design @ normal_inverse @ design.T, dtype=np.float64)
    degrees_of_freedom = float(budget - float(np.trace(hat)))
    _require(math.isfinite(degrees_of_freedom) and degrees_of_freedom > 0.0,
             "precision V2 conditional posterior degrees-of-freedom drift")
    variance = np.maximum(
        np.sum(residual * residual, axis=0) / degrees_of_freedom,
        plan.POSTERIOR_VARIANCE_FLOOR,
    )
    posterior = np.einsum("u,ij->uij", variance, normal_inverse, optimize=True)
    return design, normal, normal_inverse, beta, degrees_of_freedom, variance, posterior[:, :2, :2]


@dataclass(frozen=True)
class SupportConditionalPosterior:
    """Exact M4/M10 support-only conditional Gaussian-ridge posterior."""

    budget: int
    support_rates: np.ndarray = field(repr=False, compare=False)
    support_direction_indices: np.ndarray = field(repr=False, compare=False)
    valid_mask: np.ndarray = field(repr=False, compare=False)
    fixed_ridge_t4: np.ndarray = field(repr=False, compare=False)
    posterior_covariance_ac: np.ndarray = field(repr=False, compare=False)
    residual_variance: np.ndarray = field(repr=False, compare=False)
    design: np.ndarray = field(repr=False, compare=False)
    normal_matrix: np.ndarray = field(repr=False, compare=False)
    normal_inverse: np.ndarray = field(repr=False, compare=False)
    residual_degrees_of_freedom: float
    groups_sha256: str | None = None

    def __post_init__(self) -> None:
        budget = int(self.budget)
        rates = np.asarray(self.support_rates)
        directions = np.asarray(self.support_direction_indices)
        valid = np.asarray(self.valid_mask)
        fitted = np.asarray(self.fixed_ridge_t4)
        covariance = np.asarray(self.posterior_covariance_ac)
        variance = np.asarray(self.residual_variance)
        design = np.asarray(self.design)
        normal = np.asarray(self.normal_matrix)
        inverse = np.asarray(self.normal_inverse)
        _require(budget in plan.TRANSITION_BUDGETS, "precision V2 posterior budget must be M4 or M10")
        _require(rates.shape == (budget, valid.size) and rates.dtype == np.float64 and np.isfinite(rates).all(),
                 "precision V2 support rates must be finite float64 [M,N]")
        _require(directions.shape == (budget,) and directions.dtype == np.int64,
                 "precision V2 support-direction topology drift")
        _require(valid.shape == (rates.shape[1],) and valid.dtype == np.bool_ and int(valid.sum()) >= core.GROUP_COUNT,
                 "precision V2 valid-unit topology drift")
        _require(fitted.shape == (rates.shape[1], 4) and fitted.dtype == np.float32 and np.isfinite(fitted).all(),
                 "precision V2 fixed-ridge carrier topology drift")
        _require(covariance.shape == (rates.shape[1], 2, 2) and covariance.dtype == np.float64,
                 "precision V2 posterior covariance topology drift")
        _require(variance.shape == (rates.shape[1],) and variance.dtype == np.float64,
                 "precision V2 residual-variance topology drift")
        _require(design.shape == (budget, 3) and design.dtype == np.float64,
                 "precision V2 design topology drift")
        _require(normal.shape == inverse.shape == (3, 3) and normal.dtype == inverse.dtype == np.float64,
                 "precision V2 normal-matrix topology drift")
        _require(math.isfinite(float(self.residual_degrees_of_freedom))
                 and float(self.residual_degrees_of_freedom) > 0.0,
                 "precision V2 residual degrees-of-freedom must be positive finite")
        _require(np.isfinite(covariance[valid]).all() and np.isfinite(variance[valid]).all()
                 and np.all(variance[valid] >= plan.POSTERIOR_VARIANCE_FLOOR),
                 "precision V2 valid posterior topology drift")
        _require(np.isnan(covariance[~valid]).all() and np.isnan(variance[~valid]).all(),
                 "precision V2 invalid units must remain explicit NaN rows")
        expected_design, expected_normal, expected_inverse, _beta, expected_dof, expected_variance, expected_covariance = (
            _posterior_terms(rates, directions)
        )
        expected_covariance = np.ascontiguousarray(expected_covariance, dtype=np.float64)
        expected_variance = np.ascontiguousarray(expected_variance, dtype=np.float64)
        expected_covariance[~valid] = np.nan
        expected_variance[~valid] = np.nan
        _require(np.array_equal(design, expected_design), "precision V2 canonical support design drift")
        _require(np.allclose(normal, expected_normal, rtol=0.0, atol=1.0e-14),
                 "precision V2 normal matrix drift")
        _require(np.allclose(inverse, expected_inverse, rtol=0.0, atol=1.0e-14),
                 "precision V2 normal inverse drift")
        _require(abs(float(self.residual_degrees_of_freedom) - expected_dof) <= 1.0e-12,
                 "precision V2 residual degrees-of-freedom drift")
        _require(np.allclose(variance, expected_variance, rtol=1.0e-12, atol=1.0e-14, equal_nan=True),
                 "precision V2 residual variance is not conditional posterior law")
        _require(np.allclose(covariance, expected_covariance, rtol=1.0e-12, atol=1.0e-14, equal_nan=True),
                 "precision V2 covariance is not conditional Gaussian-ridge posterior")
        fixed = core.fit_carriers_from_trial_table(
            rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
        )
        _require(np.array_equal(fitted, fixed), "precision V2 fixed-ridge coefficient parity drift")
        for index in np.flatnonzero(valid):
            matrix = covariance[int(index)]
            _require(np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12),
                     "precision V2 posterior covariance symmetry drift")
            eigenvalues = np.linalg.eigvalsh(matrix)
            _require(np.isfinite(eigenvalues).all() and float(eigenvalues.min()) > 0.0,
                     "precision V2 posterior covariance must be positive definite")
        if self.groups_sha256 is not None:
            plan.require_sha256(self.groups_sha256, "precision V2 groups SHA")
        object.__setattr__(self, "support_rates", _immutable(rates, dtype=np.float64))
        object.__setattr__(self, "support_direction_indices", _immutable(directions, dtype=np.int64))
        object.__setattr__(self, "valid_mask", _immutable(valid, dtype=np.bool_))
        object.__setattr__(self, "fixed_ridge_t4", _immutable(fitted, dtype=np.float32))
        object.__setattr__(self, "posterior_covariance_ac", _immutable(covariance, dtype=np.float64))
        object.__setattr__(self, "residual_variance", _immutable(variance, dtype=np.float64))
        object.__setattr__(self, "design", _immutable(design, dtype=np.float64))
        object.__setattr__(self, "normal_matrix", _immutable(normal, dtype=np.float64))
        object.__setattr__(self, "normal_inverse", _immutable(inverse, dtype=np.float64))

    @classmethod
    def from_support_only_fixed_ridge(
        cls,
        *,
        support_rates: Any,
        support_direction_indices: Any,
        valid_mask: Any,
        groups_sha256: str | None = None,
    ) -> "SupportConditionalPosterior":
        rates = np.ascontiguousarray(np.asarray(support_rates, dtype=np.float64))
        directions = np.ascontiguousarray(np.asarray(support_direction_indices, dtype=np.int64))
        valid = np.ascontiguousarray(np.asarray(valid_mask, dtype=np.bool_))
        _require(rates.ndim == 2 and rates.shape[0] in plan.TRANSITION_BUDGETS and rates.shape[1] >= core.GROUP_COUNT,
                 "precision V2 support-rate topology must be M4/M10 by N")
        _require(directions.shape == (rates.shape[0],) and valid.shape == (rates.shape[1],),
                 "precision V2 support input topology drift")
        _require(np.isfinite(rates).all(), "precision V2 support rates must be finite")
        design, normal, inverse, _beta, dof, variance, covariance = _posterior_terms(rates, directions)
        covariance = np.ascontiguousarray(covariance, dtype=np.float64)
        variance = np.ascontiguousarray(variance, dtype=np.float64)
        covariance[~valid] = np.nan
        variance[~valid] = np.nan
        fitted = core.fit_carriers_from_trial_table(
            rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
        )
        return cls(
            budget=int(rates.shape[0]), support_rates=rates, support_direction_indices=directions,
            valid_mask=valid, fixed_ridge_t4=fitted, posterior_covariance_ac=covariance,
            residual_variance=variance, design=design, normal_matrix=normal, normal_inverse=inverse,
            residual_degrees_of_freedom=dof, groups_sha256=groups_sha256,
        )

    @property
    def frozen_initial_active_t4(self) -> np.ndarray:
        """The one frozen float32 support carrier promoted only for V2 math."""
        return _immutable(self.fixed_ridge_t4, dtype=np.float64)

    @property
    def digest(self) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_v2_support_conditional_posterior_v1",
            "budget": int(self.budget),
            "fit_mode": core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL.value,
            "normalized_lambda": plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
            "covariance_estimand": "conditional_gaussian_ridge_posterior_sigma2_A_inverse",
            "sampling_sandwich_covariance_used": False,
            "support_rates": _array_payload(self.support_rates),
            "support_direction_indices": _array_payload(self.support_direction_indices),
            "valid_mask": _array_payload(self.valid_mask),
            "fixed_ridge_t4": _array_payload(self.fixed_ridge_t4),
            "posterior_covariance_ac": _array_payload(self.posterior_covariance_ac),
            "residual_variance": _array_payload(self.residual_variance),
            "design": _array_payload(self.design),
            "normal_matrix": _array_payload(self.normal_matrix),
            "normal_inverse": _array_payload(self.normal_inverse),
            "residual_degrees_of_freedom": float(self.residual_degrees_of_freedom),
            "posterior_variance_floor": plan.POSTERIOR_VARIANCE_FLOOR,
            "groups_sha256_or_null": self.groups_sha256,
            "pseudo_labels_used": False,
            "decoder_token_used": False,
        }

    def validate_initial_carrier(self, carrier: core.CarrierMemory) -> None:
        """Bind posterior math to the exact support-only consumer carrier."""
        _require(isinstance(carrier, core.CarrierMemory), "precision V2 needs typed carrier memory")
        _require(carrier.config.support_budget_m == self.budget
                 and carrier.config.active_fit_mode is core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
                 "precision V2 carrier budget/fit-mode drift")
        _require(np.array_equal(carrier.groups.valid_mask, self.valid_mask), "precision V2 carrier valid-mask drift")
        if self.groups_sha256 is not None:
            _require(carrier.groups.digest == self.groups_sha256, "precision V2 complementary-group drift")
        active = np.ascontiguousarray(np.asarray(carrier.active_t4, dtype=np.float32))
        initial = np.ascontiguousarray(np.asarray(carrier.initial_raw_t4, dtype=np.float32))
        expected_digest = core.array_digest(self.fixed_ridge_t4)
        _require(core.array_digest(active) == expected_digest and core.array_digest(initial) == expected_digest,
                 "precision V2 carrier is not the frozen support-only fixed-ridge initializer")


@dataclass(frozen=True)
class FamilywiseCredibleDecision:
    """Immutable evidence for a proposal measured against frozen support."""

    accepted: bool
    threshold: float
    valid_unit_count: int
    invalid_unit_count: int
    max_mahalanobis_squared: float
    min_mahalanobis_squared: float
    reference_initial_sha256: str
    proposed_sha256: str
    delta_ac: np.ndarray = field(repr=False, compare=False)
    mahalanobis_squared: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta_ac)
        statistic = np.asarray(self.mahalanobis_squared)
        _require(type(self.accepted) is bool, "precision V2 decision accepted flag must be exact bool")
        _require(delta.ndim == 2 and delta.shape[1] == 2 and delta.dtype == np.float64,
                 "precision V2 delta topology drift")
        _require(statistic.shape == (delta.shape[0],) and statistic.dtype == np.float64,
                 "precision V2 statistic topology drift")
        _require(type(self.valid_unit_count) is int and type(self.invalid_unit_count) is int
                 and self.valid_unit_count + self.invalid_unit_count == delta.shape[0],
                 "precision V2 decision valid/invalid cardinality drift")
        expected_threshold = plan.bonferroni_chi2_df2_threshold(self.valid_unit_count)
        _require(float(self.threshold) == expected_threshold, "precision V2 Bonferroni threshold drift")
        _require(math.isfinite(float(self.max_mahalanobis_squared)) and math.isfinite(float(self.min_mahalanobis_squared)),
                 "precision V2 decision summary must be finite")
        finite_statistics = statistic[np.isfinite(statistic)]
        _require(finite_statistics.size == self.valid_unit_count and int(np.isnan(statistic).sum()) == self.invalid_unit_count,
                 "precision V2 statistic valid/invalid topology drift")
        _require(np.all(finite_statistics >= -1.0e-12)
                 and abs(float(finite_statistics.max()) - float(self.max_mahalanobis_squared)) <= 1.0e-12
                 and abs(float(finite_statistics.min()) - float(self.min_mahalanobis_squared)) <= 1.0e-12,
                 "precision V2 decision statistic summary drift")
        _require(self.accepted is (float(self.max_mahalanobis_squared) <= expected_threshold),
                 "precision V2 accepted flag/statistic relation drift")
        plan.require_sha256(self.reference_initial_sha256, "precision V2 frozen reference SHA")
        plan.require_sha256(self.proposed_sha256, "precision V2 proposal SHA")
        object.__setattr__(self, "delta_ac", _immutable(delta, dtype=np.float64))
        object.__setattr__(self, "mahalanobis_squared", _immutable(statistic, dtype=np.float64))

    @property
    def digest(self) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_v2_familywise_credible_decision_v1",
            "accepted": self.accepted,
            "rejection_reason_or_null": None if self.accepted else core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION.value,
            "familywise_alpha": plan.FAMILYWISE_ALPHA,
            "threshold_law": "-2*log(0.05/N_valid)",
            "threshold": float(self.threshold),
            "valid_unit_count": self.valid_unit_count,
            "invalid_unit_count": self.invalid_unit_count,
            "max_mahalanobis_squared": float(self.max_mahalanobis_squared),
            "min_mahalanobis_squared": float(self.min_mahalanobis_squared),
            "reference": "frozen_support_only_initial_fixed_ridge",
            "reference_initial_sha256": self.reference_initial_sha256,
            "proposed_sha256": self.proposed_sha256,
            "delta_ac": _array_payload(self.delta_ac),
            "mahalanobis_squared": _array_payload(self.mahalanobis_squared),
            "decoder_token_used": False,
        }


def decide_against_frozen_support(
    posterior: SupportConditionalPosterior,
    *,
    frozen_initial_active_t4: Any,
    proposed_active_t4: Any,
) -> FamilywiseCredibleDecision:
    """Evaluate a candidate only against the immutable support initializer."""
    _require(isinstance(posterior, SupportConditionalPosterior), "precision V2 requires typed posterior")
    reference = np.ascontiguousarray(np.asarray(frozen_initial_active_t4, dtype=np.float64))
    proposed = np.ascontiguousarray(np.asarray(proposed_active_t4, dtype=np.float64))
    _require(reference.shape == proposed.shape == posterior.fixed_ridge_t4.shape,
             "precision V2 frozen reference/proposal carrier topology drift")
    _require(np.isfinite(reference).all() and np.isfinite(proposed).all(),
             "precision V2 frozen reference/proposal carrier must be finite")
    _require(core.array_digest(np.ascontiguousarray(reference, dtype=np.float32))
             == core.array_digest(posterior.fixed_ridge_t4),
             "precision V2 decision reference is not exact frozen support carrier")
    delta = np.ascontiguousarray(proposed[:, :2] - reference[:, :2], dtype=np.float64)
    statistics = np.full((delta.shape[0],), np.nan, dtype=np.float64)
    for unit in np.flatnonzero(posterior.valid_mask):
        covariance = posterior.posterior_covariance_ac[int(unit)]
        try:
            statistics[int(unit)] = float(delta[int(unit)] @ np.linalg.solve(covariance, delta[int(unit)]))
        except np.linalg.LinAlgError as error:
            raise PrecisionAwareV2TransitionError("precision V2 posterior covariance solve failed") from error
    valid_statistics = statistics[posterior.valid_mask]
    _require(valid_statistics.size == int(posterior.valid_mask.sum())
             and np.isfinite(valid_statistics).all(),
             "precision V2 valid posterior statistic drift")
    threshold = plan.bonferroni_chi2_df2_threshold(int(valid_statistics.size))
    return FamilywiseCredibleDecision(
        accepted=bool(float(valid_statistics.max()) <= threshold), threshold=threshold,
        valid_unit_count=int(valid_statistics.size), invalid_unit_count=int((~posterior.valid_mask).sum()),
        max_mahalanobis_squared=float(valid_statistics.max()), min_mahalanobis_squared=float(valid_statistics.min()),
        reference_initial_sha256=core.array_digest(np.ascontiguousarray(reference, dtype=np.float64)),
        proposed_sha256=core.array_digest(np.ascontiguousarray(proposed, dtype=np.float64)),
        delta_ac=delta, mahalanobis_squared=statistics,
    )


@dataclass(frozen=True)
class PrecisionTransitionOutcomeV2:
    """V2 receipt evidence retaining the shared independent-activity result."""

    budget: int
    m30_deployment_noop: bool
    core_outcome: core.IndependentActivityUpdateOutcome | None = field(repr=False, compare=False)
    posterior_decision: FamilywiseCredibleDecision | None = field(repr=False, compare=False)
    state_before_sha256: str = ""
    state_after_sha256: str = ""
    prediction_before_sha256: str = ""
    prediction_after_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.budget in plan.BUDGETS, "precision V2 outcome budget drift")
        for label, value in (
            ("state before", self.state_before_sha256), ("state after", self.state_after_sha256),
            ("prediction before", self.prediction_before_sha256), ("prediction after", self.prediction_after_sha256),
        ):
            plan.require_sha256(value, f"precision V2 {label} SHA")
        if self.budget == 30:
            _require(self.m30_deployment_noop and self.core_outcome is None and self.posterior_decision is None,
                     "precision V2 M30 must be literal no-op")
            _require(self.state_before_sha256 == self.state_after_sha256
                     and self.prediction_before_sha256 == self.prediction_after_sha256,
                     "precision V2 M30 state/prediction drift")
        else:
            _require(not self.m30_deployment_noop and isinstance(self.core_outcome, core.IndependentActivityUpdateOutcome),
                     "precision V2 M4/M10 core outcome drift")
            if self.core_outcome.carrier_transition_committed:
                _require(isinstance(self.posterior_decision, FamilywiseCredibleDecision)
                         and self.posterior_decision.accepted,
                         "precision V2 committed carrier lacks accepted posterior decision")
            elif self.core_outcome.carrier_rejection_reason is core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION:
                _require(isinstance(self.posterior_decision, FamilywiseCredibleDecision)
                         and not self.posterior_decision.accepted,
                         "precision V2 precision rejection lacks rejected posterior decision")
            else:
                _require(self.posterior_decision is None,
                         "precision V2 pre-existing core rejection may not be relabelled")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_v2_transition_outcome_v1",
            "budget": self.budget,
            "m30_deployment_noop": self.m30_deployment_noop,
            "core_independent_activity_outcome": None if self.core_outcome is None else self.core_outcome.payload(),
            "conditional_posterior_decision": None if self.posterior_decision is None else self.posterior_decision.payload(),
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "prediction_before_sha256": self.prediction_before_sha256,
            "prediction_after_sha256": self.prediction_after_sha256,
            "target_accessed": False,
            "decoder_token_used": False,
            "normalizer_refit": False,
            "model_parameter_update": False,
        }


class PrecisionAwareIndependentActivityV2:
    """One V2 posterior veto layered after the frozen core proposal gates."""

    def __init__(
        self, *, memory: core.IndependentActivityCausalDualMemory,
        posterior: SupportConditionalPosterior | None,
    ) -> None:
        _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
                 "precision V2 requires IndependentActivityCausalDualMemory")
        budget = memory.state.carrier.config.support_budget_m
        if budget == 30:
            _require(posterior is None, "precision V2 M30 must not construct posterior statistic")
            frozen = None
        else:
            _require(isinstance(posterior, SupportConditionalPosterior),
                     "precision V2 M4/M10 needs typed posterior")
            posterior.validate_initial_carrier(memory.state.carrier)
            frozen = _immutable(memory.state.carrier.active_t4, dtype=np.float64)
        self._memory = memory
        self._posterior = posterior
        self._frozen_initial_active_t4 = frozen

    @property
    def memory(self) -> core.IndependentActivityCausalDualMemory:
        return self._memory

    @property
    def posterior(self) -> SupportConditionalPosterior | None:
        return self._posterior

    @staticmethod
    def _prediction_digest(value: core.PredictionInputs) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(value.payload()))

    def _precision_rejected_pending(
        self, pending: core.IndependentActivityPendingTrialUpdate,
    ) -> core.IndependentActivityPendingTrialUpdate:
        _require(pending.activity_transition_ready and pending.carrier_transition_accepted,
                 "precision V2 may veto only an accepted core carrier proposal")
        _require(isinstance(pending.candidate_state, core.DualMemoryState),
                 "precision V2 accepted core proposal lacks candidate state")
        candidate = core.DualMemoryState(
            activity=pending.candidate_state.activity,
            carrier=self._memory.state.carrier,
            committed_query_trials=pending.candidate_state.committed_query_trials,
        )
        return core.IndependentActivityPendingTrialUpdate(
            base_state_digest=pending.base_state_digest,
            activity_transition_ready=True,
            activity_fifo_changed=pending.activity_fifo_changed,
            carrier_transition_accepted=False,
            activity_rejection_reason=None,
            carrier_rejection_reason=core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION,
            pseudo_directions=pending.pseudo_directions,
            scalar_rates=pending.scalar_rates,
            carrier_proposal=pending.carrier_proposal,
            candidate_state=candidate,
            fallback=pending.fallback,
            completed_trial_evidence=pending.completed_trial_evidence,
        )

    def observe_and_commit(
        self,
        *,
        budget: int,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        complementary_predictions: Sequence[Any],
    ) -> PrecisionTransitionOutcomeV2:
        """Read pre-query state, delegate core gates, then apply V2 if eligible."""
        _require(budget in plan.BUDGETS and budget == self._memory.state.carrier.config.support_budget_m,
                 "precision V2 requested budget/memory state drift")
        before_state = self._memory.state.digest
        before_prediction = self._prediction_digest(self._memory.read_prediction_inputs())
        if budget == 30:
            return PrecisionTransitionOutcomeV2(
                budget=30, m30_deployment_noop=True, core_outcome=None, posterior_decision=None,
                state_before_sha256=before_state, state_after_sha256=before_state,
                prediction_before_sha256=before_prediction, prediction_after_sha256=before_prediction,
            )
        _require(isinstance(self._posterior, SupportConditionalPosterior)
                 and isinstance(self._frozen_initial_active_t4, np.ndarray),
                 "precision V2 M4/M10 posterior/frozen support reference is absent")
        pending = self._memory.observe_completed_trial(
            b3s_trial_activity=b3s_trial_activity,
            carrier_trial_counts=carrier_trial_counts,
            complementary_predictions=complementary_predictions,
        )
        decision: FamilywiseCredibleDecision | None = None
        if pending.carrier_transition_accepted:
            _require(isinstance(pending.carrier_proposal, core.CarrierProposal)
                     and isinstance(pending.carrier_proposal.candidate, core.CarrierMemory),
                     "precision V2 accepted core proposal lacks typed candidate carrier")
            decision = decide_against_frozen_support(
                self._posterior,
                frozen_initial_active_t4=self._frozen_initial_active_t4,
                proposed_active_t4=pending.carrier_proposal.candidate.active_t4,
            )
            if not decision.accepted:
                pending = self._precision_rejected_pending(pending)
        outcome = self._memory.commit_independent(pending)
        after_state = self._memory.state.digest
        after_prediction = self._prediction_digest(self._memory.read_prediction_inputs())
        return PrecisionTransitionOutcomeV2(
            budget=budget, m30_deployment_noop=False, core_outcome=outcome, posterior_decision=decision,
            state_before_sha256=before_state, state_after_sha256=after_state,
            prediction_before_sha256=before_prediction, prediction_after_sha256=after_prediction,
        )
