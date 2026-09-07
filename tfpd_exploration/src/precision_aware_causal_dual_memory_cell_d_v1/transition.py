"""Support-precision gate layered over the accepted independent CDM-D state.

This module owns no model, decoder, normalizer, parser, or source reader.  It
operates exclusively on a typed ``IndependentActivityCausalDualMemory`` and
the exact support-only trial table that created its initial fixed-ridge
carrier.  Consequently the precision statistic cannot become a live decoder
feature or a pseudo-label refit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core

from . import plan


class PrecisionTransitionError(RuntimeError):
    """Fail closed for a precision statistic or state-transition drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionTransitionError(message)


def _immutable(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    array.setflags(write=False)
    return array


def _frozen_payload(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": core.array_digest(value),
    }


def _canonical_direction_design(direction_indices: np.ndarray) -> np.ndarray:
    _require(direction_indices.ndim == 1 and direction_indices.size in plan.TRANSITION_BUDGETS,
             "precision support directions must be exact M4 or M10")
    _require(np.issubdtype(direction_indices.dtype, np.integer), "precision support directions must be integers")
    _require(np.all((direction_indices >= 0) & (direction_indices < len(core.CANONICAL_DIRECTIONS_RAD))),
             "precision support direction out of canonical range")
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[int(item)] for item in direction_indices], dtype=np.float64)
    return np.ascontiguousarray(np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size))), dtype=np.float64)


@dataclass(frozen=True)
class SupportPrecision:
    """Fixed M4/M10 [a,c] coefficient uncertainty from source support only."""

    budget: int
    support_rates: np.ndarray = field(repr=False, compare=False)
    support_direction_indices: np.ndarray = field(repr=False, compare=False)
    valid_mask: np.ndarray = field(repr=False, compare=False)
    fixed_ridge_t4: np.ndarray = field(repr=False, compare=False)
    covariance_ac: np.ndarray = field(repr=False, compare=False)
    residual_variance: np.ndarray = field(repr=False, compare=False)
    design: np.ndarray = field(repr=False, compare=False)
    normal_matrix: np.ndarray = field(repr=False, compare=False)
    effective_residual_degrees_of_freedom: float
    groups_sha256: str | None = None

    def __post_init__(self) -> None:
        budget = int(self.budget)
        rates = np.asarray(self.support_rates)
        directions = np.asarray(self.support_direction_indices)
        valid = np.asarray(self.valid_mask)
        fitted = np.asarray(self.fixed_ridge_t4)
        covariance = np.asarray(self.covariance_ac)
        residual = np.asarray(self.residual_variance)
        design = np.asarray(self.design)
        normal = np.asarray(self.normal_matrix)
        _require(budget in plan.TRANSITION_BUDGETS, "precision support budget must be M4 or M10")
        _require(rates.shape == (budget, valid.size) and rates.dtype == np.float64 and np.isfinite(rates).all(),
                 "precision support rates must be finite float64 [M,N]")
        _require(directions.shape == (budget,) and directions.dtype == np.int64,
                 "precision support directions must be int64 [M]")
        _require(valid.shape == (rates.shape[1],) and valid.dtype == np.bool_,
                 "precision valid mask must be bool [N]")
        _require(int(valid.sum()) >= core.GROUP_COUNT, "precision needs at least four valid units")
        _require(fitted.shape == (rates.shape[1], 4) and fitted.dtype == np.float32 and np.isfinite(fitted).all(),
                 "precision fixed-ridge carrier topology drift")
        _require(covariance.shape == (rates.shape[1], 2, 2) and covariance.dtype == np.float64,
                 "precision covariance topology drift")
        _require(residual.shape == (rates.shape[1],) and residual.dtype == np.float64,
                 "precision residual-variance topology drift")
        _require(design.shape == (budget, 3) and design.dtype == np.float64,
                 "precision design topology drift")
        _require(normal.shape == (3, 3) and normal.dtype == np.float64 and np.isfinite(normal).all(),
                 "precision normal matrix drift")
        _require(math.isfinite(float(self.effective_residual_degrees_of_freedom))
                 and float(self.effective_residual_degrees_of_freedom) > 0.0,
                 "precision residual degrees of freedom must be positive finite")
        _require(np.isfinite(covariance[valid]).all() and np.isfinite(residual[valid]).all(),
                 "precision valid covariance/residual rows must be finite")
        _require(np.isnan(covariance[~valid]).all() and np.isnan(residual[~valid]).all(),
                 "precision invalid units must be explicit NaN/excluded rows")
        _require(np.all(residual[valid] >= plan.RESIDUAL_VARIANCE_FLOOR),
                 "precision residual variance floor drift")
        # A caller may not synthesize a friendlier covariance after fitting.
        # Rebuild the complete closed-form support-only law here, so the
        # receipt object is an authority rather than merely a container for
        # arbitrary matrices.
        expected_design = _canonical_direction_design(directions)
        _require(np.array_equal(design, expected_design), "precision canonical support design drift")
        expected_penalty = np.diag((budget * plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
                                    budget * plan.FIXED_RIDGE_NORMALIZED_LAMBDA, 0.0)).astype(np.float64)
        expected_normal = np.ascontiguousarray(expected_design.T @ expected_design + expected_penalty, dtype=np.float64)
        _require(np.allclose(normal, expected_normal, rtol=0.0, atol=1.0e-14),
                 "precision fixed-ridge normal matrix drift")
        try:
            expected_inverse = np.linalg.inv(expected_normal)
            expected_beta = np.linalg.solve(expected_normal, expected_design.T @ rates)
        except np.linalg.LinAlgError as error:
            raise PrecisionTransitionError("precision fixed-ridge normal matrix is singular") from error
        expected_residual = rates - expected_design @ expected_beta
        expected_dof = float(budget - float(np.trace(expected_design @ expected_inverse @ expected_design.T)))
        _require(abs(float(self.effective_residual_degrees_of_freedom) - expected_dof) <= 1.0e-12,
                 "precision effective residual degrees-of-freedom drift")
        expected_variance = np.maximum(
            np.sum(expected_residual * expected_residual, axis=0) / expected_dof,
            plan.RESIDUAL_VARIANCE_FLOOR,
        )
        expected_covariance = np.einsum(
            "u,ij->uij", expected_variance,
            expected_inverse @ (expected_design.T @ expected_design) @ expected_inverse,
            optimize=True,
        )[:, :2, :2]
        expected_covariance[~valid] = np.nan
        expected_variance[~valid] = np.nan
        _require(np.allclose(covariance, expected_covariance, rtol=1.0e-12, atol=1.0e-14, equal_nan=True),
                 "precision covariance is not the frozen support-only closed form")
        _require(np.allclose(residual, expected_variance, rtol=1.0e-12, atol=1.0e-14, equal_nan=True),
                 "precision residual variance is not the frozen support-only closed form")
        expected_fit = core.fit_carriers_from_trial_table(
            rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
        )
        _require(np.array_equal(fitted, expected_fit), "precision fixed-ridge coefficient parity drift")
        for index in np.flatnonzero(valid):
            matrix = covariance[int(index)]
            _require(np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12),
                     "precision covariance must be symmetric")
            eigenvalues = np.linalg.eigvalsh(matrix)
            _require(np.isfinite(eigenvalues).all() and float(eigenvalues.min()) > 0.0,
                     "precision valid covariance must be positive definite")
        if self.groups_sha256 is not None:
            plan.require_sha256(self.groups_sha256, "precision groups SHA")
        object.__setattr__(self, "support_rates", _immutable(rates, dtype=np.float64))
        object.__setattr__(self, "support_direction_indices", _immutable(directions, dtype=np.int64))
        object.__setattr__(self, "valid_mask", _immutable(valid, dtype=np.bool_))
        object.__setattr__(self, "fixed_ridge_t4", _immutable(fitted, dtype=np.float32))
        object.__setattr__(self, "covariance_ac", _immutable(covariance, dtype=np.float64))
        object.__setattr__(self, "residual_variance", _immutable(residual, dtype=np.float64))
        object.__setattr__(self, "design", _immutable(design, dtype=np.float64))
        object.__setattr__(self, "normal_matrix", _immutable(normal, dtype=np.float64))

    @classmethod
    def from_support_only_fixed_ridge(
        cls,
        *,
        support_rates: Any,
        support_direction_indices: Any,
        valid_mask: Any,
        groups_sha256: str | None = None,
    ) -> "SupportPrecision":
        """Construct the frozen float64 covariance without pseudo-label data."""
        rates = np.ascontiguousarray(np.asarray(support_rates, dtype=np.float64))
        directions = np.ascontiguousarray(np.asarray(support_direction_indices, dtype=np.int64))
        valid = np.ascontiguousarray(np.asarray(valid_mask, dtype=np.bool_))
        _require(rates.ndim == 2 and rates.shape[0] in plan.TRANSITION_BUDGETS and rates.shape[1] >= core.GROUP_COUNT,
                 "precision support rate topology must be M4/M10 by N")
        _require(directions.shape == (rates.shape[0],), "precision support direction length drift")
        _require(valid.shape == (rates.shape[1],), "precision support valid-mask drift")
        _require(np.isfinite(rates).all(), "precision support rates must be finite")
        design = _canonical_direction_design(directions)
        penalty = np.diag((rates.shape[0] * plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
                           rates.shape[0] * plan.FIXED_RIDGE_NORMALIZED_LAMBDA, 0.0)).astype(np.float64)
        normal = np.ascontiguousarray(design.T @ design + penalty, dtype=np.float64)
        try:
            normal_inverse = np.linalg.inv(normal)
            beta = np.linalg.solve(normal, design.T @ rates)
        except np.linalg.LinAlgError as error:
            raise PrecisionTransitionError("precision fixed-ridge normal matrix is singular") from error
        residual = rates - design @ beta
        hat = design @ normal_inverse @ design.T
        dof = float(rates.shape[0] - float(np.trace(hat)))
        _require(math.isfinite(dof) and dof > 0.0, "precision effective residual degrees of freedom drift")
        residual_variance_all = np.maximum(np.sum(residual * residual, axis=0) / dof, plan.RESIDUAL_VARIANCE_FLOOR)
        covariance_basis = normal_inverse @ (design.T @ design) @ normal_inverse
        covariance_all = np.einsum("u,ij->uij", residual_variance_all, covariance_basis, optimize=True)
        # Only valid rows are ever eligible for the transition statistic.  We
        # retain invalid topology in place rather than dropping/reordering it.
        covariance = np.asarray(covariance_all[:, :2, :2], dtype=np.float64)
        residual_variance = np.asarray(residual_variance_all, dtype=np.float64)
        covariance[~valid] = np.nan
        residual_variance[~valid] = np.nan
        reference = core.fit_carriers_from_trial_table(
            rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
        )
        core.assert_production_parity(
            reference, rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        return cls(
            budget=int(rates.shape[0]), support_rates=rates, support_direction_indices=directions,
            valid_mask=valid, fixed_ridge_t4=reference, covariance_ac=covariance,
            residual_variance=residual_variance, design=design, normal_matrix=normal,
            effective_residual_degrees_of_freedom=dof, groups_sha256=groups_sha256,
        )

    @property
    def digest(self) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_support_precision_v1",
            "budget": int(self.budget),
            "fit_mode": core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL.value,
            "normalized_lambda": plan.FIXED_RIDGE_NORMALIZED_LAMBDA,
            "support_rates": _frozen_payload(self.support_rates),
            "support_direction_indices": _frozen_payload(self.support_direction_indices),
            "valid_mask": _frozen_payload(self.valid_mask),
            "fixed_ridge_t4": _frozen_payload(self.fixed_ridge_t4),
            "covariance_ac": _frozen_payload(self.covariance_ac),
            "residual_variance": _frozen_payload(self.residual_variance),
            "design": _frozen_payload(self.design),
            "normal_matrix": _frozen_payload(self.normal_matrix),
            "effective_residual_degrees_of_freedom": float(self.effective_residual_degrees_of_freedom),
            "residual_variance_floor": plan.RESIDUAL_VARIANCE_FLOOR,
            "groups_sha256_or_null": self.groups_sha256,
            "pseudo_labels_used": False,
            "decoder_token_used": False,
        }

    def validate_carrier(self, carrier: core.CarrierMemory) -> None:
        """Bind the statistic to the exact support-only consumer carrier."""
        _require(isinstance(carrier, core.CarrierMemory), "precision route needs typed carrier memory")
        _require(carrier.config.support_budget_m == self.budget, "precision/carrier support budget drift")
        _require(carrier.config.active_fit_mode is core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
                 "precision route requires fixed-ridge active carrier")
        _require(np.array_equal(carrier.groups.valid_mask, self.valid_mask), "precision/carrier valid-mask drift")
        if self.groups_sha256 is not None:
            _require(carrier.groups.digest == self.groups_sha256, "precision/carrier complementary-group drift")
        core.assert_production_parity(
            carrier.active_t4, self.support_rates, self.support_direction_indices,
            mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        # ``CarrierMemory`` retains the initial carrier in its historical
        # float64 snapshot even though the fixed-ridge reference is emitted
        # as float32.  Bind values after the one frozen float32 boundary,
        # rather than falsely treating that intentional storage dtype as a
        # different estimand.
        active_float32 = np.ascontiguousarray(np.asarray(carrier.active_t4, dtype=np.float32))
        _require(core.array_digest(active_float32) == core.array_digest(self.fixed_ridge_t4),
                 "precision/carrier initial fixed-ridge value/dtype boundary drift")


@dataclass(frozen=True)
class CredibleRegionDecision:
    """Receipt-safe result of the scalar [a,c] state-transition test."""

    accepted: bool
    max_mahalanobis_squared: float
    min_mahalanobis_squared: float
    threshold: float
    valid_unit_count: int
    invalid_unit_count: int
    delta_ac: np.ndarray = field(repr=False, compare=False)
    mahalanobis_squared: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta_ac)
        statistic = np.asarray(self.mahalanobis_squared)
        _require(type(self.accepted) is bool, "precision decision accepted flag must be exact bool")
        _require(delta.ndim == 2 and delta.shape[1] == 2 and delta.dtype == np.float64,
                 "precision delta must be float64 [N,2]")
        _require(statistic.shape == (delta.shape[0],) and statistic.dtype == np.float64,
                 "precision statistic must be float64 [N]")
        _require(int(self.valid_unit_count) + int(self.invalid_unit_count) == delta.shape[0],
                 "precision decision valid/invalid cardinality drift")
        _require(math.isfinite(float(self.threshold)) and float(self.threshold) == plan.CREDIBLE_REGION_CHI2_DF2_95,
                 "precision decision threshold drift")
        _require(math.isfinite(float(self.max_mahalanobis_squared))
                 and math.isfinite(float(self.min_mahalanobis_squared)),
                 "precision decision summary must be finite")
        object.__setattr__(self, "delta_ac", _immutable(delta, dtype=np.float64))
        object.__setattr__(self, "mahalanobis_squared", _immutable(statistic, dtype=np.float64))

    @property
    def digest(self) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_credible_region_decision_v1",
            "accepted": self.accepted,
            "rejection_reason_or_null": None if self.accepted else core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION.value,
            "max_mahalanobis_squared": float(self.max_mahalanobis_squared),
            "min_mahalanobis_squared": float(self.min_mahalanobis_squared),
            "threshold": float(self.threshold),
            "valid_unit_count": int(self.valid_unit_count),
            "invalid_unit_count": int(self.invalid_unit_count),
            "delta_ac": _frozen_payload(self.delta_ac),
            "mahalanobis_squared": _frozen_payload(self.mahalanobis_squared),
            "decoder_token_used": False,
        }


def decide_credible_region(
    precision: SupportPrecision, *, current_active_t4: Any, proposed_active_t4: Any,
) -> CredibleRegionDecision:
    """Evaluate only the candidate state delta against support-only covariance."""
    current = np.ascontiguousarray(np.asarray(current_active_t4, dtype=np.float64))
    proposed = np.ascontiguousarray(np.asarray(proposed_active_t4, dtype=np.float64))
    _require(current.shape == proposed.shape == precision.fixed_ridge_t4.shape,
             "precision current/proposed carrier topology drift")
    _require(np.isfinite(current).all() and np.isfinite(proposed).all(),
             "precision current/proposed carrier must be finite")
    delta = np.ascontiguousarray(proposed[:, :2] - current[:, :2], dtype=np.float64)
    statistic = np.full((delta.shape[0],), np.nan, dtype=np.float64)
    for unit in np.flatnonzero(precision.valid_mask):
        covariance = precision.covariance_ac[int(unit)]
        try:
            statistic[int(unit)] = float(delta[int(unit)] @ np.linalg.solve(covariance, delta[int(unit)]))
        except np.linalg.LinAlgError as error:
            raise PrecisionTransitionError("precision covariance solve failed") from error
    valid_statistics = statistic[precision.valid_mask]
    _require(valid_statistics.size >= core.GROUP_COUNT and np.isfinite(valid_statistics).all(),
             "precision valid statistic must be finite for every valid unit")
    accepted = bool(float(valid_statistics.max()) <= plan.CREDIBLE_REGION_CHI2_DF2_95)
    return CredibleRegionDecision(
        accepted=accepted,
        max_mahalanobis_squared=float(valid_statistics.max()),
        min_mahalanobis_squared=float(valid_statistics.min()),
        threshold=plan.CREDIBLE_REGION_CHI2_DF2_95,
        valid_unit_count=int(precision.valid_mask.sum()),
        invalid_unit_count=int((~precision.valid_mask).sum()),
        delta_ac=delta,
        mahalanobis_squared=statistic,
    )


@dataclass(frozen=True)
class PrecisionTransitionOutcome:
    """Route-owned outcome that retains the core independent activity proof."""

    budget: int
    m30_deployment_noop: bool
    core_outcome: core.IndependentActivityUpdateOutcome | None = field(repr=False, compare=False)
    decision: CredibleRegionDecision | None = field(repr=False, compare=False)
    state_before_sha256: str = ""
    state_after_sha256: str = ""
    prediction_before_sha256: str = ""
    prediction_after_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.budget in plan.BUDGETS, "precision outcome budget drift")
        for label, value in (
            ("state before", self.state_before_sha256), ("state after", self.state_after_sha256),
            ("prediction before", self.prediction_before_sha256), ("prediction after", self.prediction_after_sha256),
        ):
            plan.require_sha256(value, label)
        if self.budget == 30:
            _require(self.m30_deployment_noop is True and self.core_outcome is None and self.decision is None,
                     "M30 precision outcome must be a literal deployment no-op")
            _require(self.state_before_sha256 == self.state_after_sha256
                     and self.prediction_before_sha256 == self.prediction_after_sha256,
                     "M30 precision output/state must remain exact")
        else:
            _require(self.m30_deployment_noop is False and isinstance(self.core_outcome, core.IndependentActivityUpdateOutcome)
                     and isinstance(self.decision, CredibleRegionDecision),
                     "M4/M10 precision outcome must retain core outcome and decision")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_transition_outcome_v1",
            "budget": self.budget,
            "m30_deployment_noop": self.m30_deployment_noop,
            "core_outcome": None if self.core_outcome is None else self.core_outcome.payload(),
            "credible_region": None if self.decision is None else self.decision.payload(),
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "prediction_before_sha256": self.prediction_before_sha256,
            "prediction_after_sha256": self.prediction_after_sha256,
            "target_accessed": False,
            "decoder_token_used": False,
            "normalizer_refit": False,
            "model_parameter_update": False,
        }


class PrecisionAwareIndependentActivity:
    """Narrow wrapper that adds one post-proposal precision gate to core V3.

    It delegates B3S validation, one-shot pseudo directions, B8/design and
    departure validation, and the actual independent commit to the shared
    state machine.  It never modifies those policies.
    """

    def __init__(
        self, *, memory: core.IndependentActivityCausalDualMemory, precision: SupportPrecision | None,
    ) -> None:
        _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
                 "precision route requires IndependentActivityCausalDualMemory")
        budget = memory.state.carrier.config.support_budget_m
        if budget == 30:
            _require(precision is None, "M30 deployment no-op must not construct a precision statistic")
        else:
            _require(isinstance(precision, SupportPrecision), "M4/M10 requires typed support precision")
            precision.validate_carrier(memory.state.carrier)
        self._memory = memory
        self._precision = precision

    @property
    def memory(self) -> core.IndependentActivityCausalDualMemory:
        return self._memory

    @property
    def precision(self) -> SupportPrecision | None:
        return self._precision

    @staticmethod
    def _prediction_digest(value: core.PredictionInputs) -> str:
        return core.sha256_bytes(core.canonical_json_bytes(value.payload()))

    def _precision_rejected_pending(
        self, pending: core.IndependentActivityPendingTrialUpdate,
    ) -> core.IndependentActivityPendingTrialUpdate:
        _require(pending.activity_transition_ready and pending.carrier_transition_accepted,
                 "only accepted core carrier proposal may be precision-rejected")
        _require(isinstance(pending.candidate_state, core.DualMemoryState), "accepted core proposal lacks candidate state")
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
    ) -> PrecisionTransitionOutcome:
        """Read one causal query state, then commit at most one typed transition."""
        _require(budget in plan.BUDGETS, "precision transition budget drift")
        _require(budget == self._memory.state.carrier.config.support_budget_m,
                 "precision requested budget/memory state drift")
        before_state = self._memory.state.digest
        before_prediction = self._prediction_digest(self._memory.read_prediction_inputs())
        if budget == 30:
            # M30 must exactly replay sealed deployment state.  Not invoking
            # observe avoids a carrier proposal, an activity count increment,
            # and any pseudo-label dependence.
            return PrecisionTransitionOutcome(
                budget=30, m30_deployment_noop=True, core_outcome=None, decision=None,
                state_before_sha256=before_state, state_after_sha256=before_state,
                prediction_before_sha256=before_prediction, prediction_after_sha256=before_prediction,
            )
        _require(isinstance(self._precision, SupportPrecision), "M4/M10 precision statistic is absent")
        _require(budget == self._precision.budget, "precision support-budget/memory route drift")
        pending = self._memory.observe_completed_trial(
            b3s_trial_activity=b3s_trial_activity,
            carrier_trial_counts=carrier_trial_counts,
            complementary_predictions=complementary_predictions,
        )
        if pending.carrier_transition_accepted:
            _require(isinstance(pending.carrier_proposal, core.CarrierProposal)
                     and isinstance(pending.carrier_proposal.candidate, core.CarrierMemory),
                     "accepted core proposal lacks typed candidate carrier")
            decision = decide_credible_region(
                self._precision,
                current_active_t4=self._memory.state.carrier.active_t4,
                proposed_active_t4=pending.carrier_proposal.candidate.active_t4,
            )
            if not decision.accepted:
                pending = self._precision_rejected_pending(pending)
        else:
            # A pre-existing B8/design/departure rejection is not relabelled
            # as a precision failure.  Its zero delta is recorded only so
            # every M4/M10 outcome has a typed numeric gate receipt.
            decision = decide_credible_region(
                self._precision,
                current_active_t4=self._memory.state.carrier.active_t4,
                proposed_active_t4=self._memory.state.carrier.active_t4,
            )
        outcome = self._memory.commit_independent(pending)
        after_state = self._memory.state.digest
        after_prediction = self._prediction_digest(self._memory.read_prediction_inputs())
        return PrecisionTransitionOutcome(
            budget=budget, m30_deployment_noop=False, core_outcome=outcome, decision=decision,
            state_before_sha256=before_state, state_after_sha256=after_state,
            prediction_before_sha256=before_prediction, prediction_after_sha256=after_prediction,
        )
