"""Exact NumPy implementation of the E02-derived posterior estimator.

This module deliberately imports NumPy only.  It has no filesystem, Torch,
CUDA, model, checkpoint, data-loader, or result-root surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import plan


class PosteriorContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PosteriorContractError(message)


def _array(value: object, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == ndim, f"{name} must have ndim={ndim}, got {result.shape}")
    _require(result.size > 0 and np.isfinite(result).all(), f"{name} must be nonempty and finite")
    return np.ascontiguousarray(result)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def directional_design(theta_radians: object) -> np.ndarray:
    theta = _array(theta_radians, name="theta_radians", ndim=1)
    return np.ascontiguousarray(
        np.stack((np.ones_like(theta), np.cos(theta), np.sin(theta)), axis=1),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PosteriorFit:
    raw_t4: np.ndarray
    beta_bac: np.ndarray
    ols_beta_bac: np.ndarray
    residual_variance: np.ndarray
    posterior_covariance_ac: np.ndarray
    design_rank: int
    residual_dof: int
    prior_variance: float
    trial_count: int
    raw_t4_sha256: str

    def __post_init__(self) -> None:
        units = int(self.raw_t4.shape[0])
        _require(self.raw_t4.shape == (units, 4), "raw_t4 must be [units,4]")
        _require(self.beta_bac.shape == (units, 3), "beta_bac must be [units,3]")
        _require(self.ols_beta_bac.shape == (units, 3), "ols_beta_bac must be [units,3]")
        _require(self.residual_variance.shape == (units,), "residual variance must be [units]")
        _require(
            self.posterior_covariance_ac.shape == (units, 2, 2),
            "posterior covariance must be [units,2,2]",
        )
        for value in (
            self.raw_t4,
            self.beta_bac,
            self.ols_beta_bac,
            self.residual_variance,
            self.posterior_covariance_ac,
        ):
            _require(value.dtype == np.float64 and np.isfinite(value).all(), "fit arrays must be finite float64")
        _require(self.design_rank == 3 and self.residual_dof > 0, "fit must bind rank 3 and positive residual DOF")
        _require(math.isfinite(self.prior_variance) and self.prior_variance > 0.0, "invalid prior variance")
        _require(self.raw_t4_sha256 == array_sha256(self.raw_t4), "raw T4 digest drift")


@dataclass(frozen=True)
class SourcePrior:
    variance: float
    raw_second_moment: float
    expected_ols_noise: float
    variance_floor: float
    source_session_count: int
    unit_fit_count: int
    source_budget: int
    body_sha256: str

    def payload_without_sha(self) -> dict[str, object]:
        return {
            "schema": "budget_matched_posterior_source_prior_v1",
            "variance": self.variance,
            "raw_second_moment": self.raw_second_moment,
            "expected_ols_noise": self.expected_ols_noise,
            "variance_floor": self.variance_floor,
            "source_session_count": self.source_session_count,
            "unit_fit_count": self.unit_fit_count,
            "source_budget": self.source_budget,
            "source_only": True,
            "shared_unchanged_across_budgets": [30, 10, 4],
            "e02_reference_prior_not_reused": plan.E02_REFERENCE_PRIOR_VARIANCE_ONLY,
        }

    def __post_init__(self) -> None:
        _require(math.isfinite(self.variance) and self.variance > 0.0, "prior variance must be positive")
        _require(self.source_budget == 30, "source prior must be fit at M30")
        _require(self.source_session_count > 0 and self.unit_fit_count > 0, "empty source prior")
        expected = _json_sha(self.payload_without_sha())
        _require(self.body_sha256 == expected, "source-prior body digest drift")


@dataclass(frozen=True)
class PosteriorNormalizer:
    mean: np.ndarray
    std: np.ndarray
    per_budget_row_count: Mapping[int, int]
    per_budget_rows_sha256: Mapping[int, str]
    rows_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require(self.mean.shape == (4,) and self.std.shape == (4,), "normalizer moments must be [4]")
        _require(self.mean.dtype == np.float64 and self.std.dtype == np.float64, "normalizer must be float64")
        _require(np.isfinite(self.mean).all() and np.isfinite(self.std).all(), "nonfinite normalizer")
        _require(bool(np.all(self.std > 0.0)), "normalizer standard deviations must be positive")
        _require(set(self.per_budget_row_count) == set(plan.NORMALIZER_BUDGET_ORDER), "budget count topology drift")
        _require(set(self.per_budget_rows_sha256) == set(plan.NORMALIZER_BUDGET_ORDER), "budget digest topology drift")
        counts = [self.per_budget_row_count[m] for m in plan.NORMALIZER_BUDGET_ORDER]
        _require(all(type(n) is int and n > 0 for n in counts) and len(set(counts)) == 1, "budgets must have equal row counts")
        _require(self.body_sha256 == _json_sha(self.payload_without_sha()), "normalizer body digest drift")

    def payload_without_sha(self) -> dict[str, object]:
        return {
            "schema": "budget_matched_posterior_normalizer_v1",
            "row_order": "budget_M4_M10_M30_then_caller_source_roster_then_unit",
            "mean_float64": [float(x) for x in self.mean],
            "std_float64": [float(x) for x in self.std],
            "ddof": 0,
            "per_budget_row_count": {str(m): self.per_budget_row_count[m] for m in plan.NORMALIZER_BUDGET_ORDER},
            "per_budget_rows_sha256": {str(m): self.per_budget_rows_sha256[m] for m in plan.NORMALIZER_BUDGET_ORDER},
            "rows_sha256": self.rows_sha256,
            "source_only": True,
            "equal_budget_weight": True,
            "ordinary_ols_normalizer_reused": False,
        }

    def normalize(self, raw_t4: object) -> np.ndarray:
        raw = _array(raw_t4, name="raw_t4", ndim=2)
        _require(raw.shape[1] == 4, "raw_t4 must be [units,4]")
        return np.ascontiguousarray((raw - self.mean) / self.std, dtype=np.float64)


def _json_sha(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _ols_evidence(trial_rates: object, theta_radians: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    rates = _array(trial_rates, name="trial_rates", ndim=2)
    theta = _array(theta_radians, name="theta_radians", ndim=1)
    _require(rates.shape[1] == theta.shape[0], "rates/theta trial-count mismatch")
    design = directional_design(theta)
    rank = int(np.linalg.matrix_rank(design))
    dof = int(design.shape[0] - rank)
    _require(rank == 3 and dof > 0, plan.M4_FAILURE)
    gram = design.T @ design
    try:
        inverse = np.linalg.inv(gram)
        ols = np.linalg.solve(gram, design.T @ rates.T).T
    except np.linalg.LinAlgError as error:
        raise PosteriorContractError("rank-3 OLS solve failed") from error
    residual = rates - (design @ ols.T).T
    sigma2 = np.einsum("ij,ij->i", residual, residual) / float(dof)
    _require(np.isfinite(sigma2).all() and np.all(sigma2 >= 0.0), "invalid residual variance")
    return rates, design, ols, rank, dof, np.ascontiguousarray(sigma2)


def fit_posterior_mean(trial_rates: object, theta_radians: object, *, prior_variance: float) -> PosteriorFit:
    _require(math.isfinite(prior_variance) and prior_variance > 0.0, "prior_variance must be finite and positive")
    rates, design, ols, rank, dof, sigma2 = _ols_evidence(trial_rates, theta_radians)
    gram = design.T @ design
    rhs = design.T @ rates.T
    beta = np.empty_like(ols)
    covariance_ac = np.empty((rates.shape[0], 2, 2), dtype=np.float64)
    for unit in range(rates.shape[0]):
        penalty = float(sigma2[unit] / prior_variance)
        system = gram + np.diag((0.0, penalty, penalty))
        try:
            beta[unit] = np.linalg.solve(system, rhs[:, unit])
            covariance = float(sigma2[unit]) * np.linalg.inv(system)
        except np.linalg.LinAlgError as error:
            raise PosteriorContractError("posterior solve failed") from error
        covariance_ac[unit] = covariance[1:3, 1:3]
    raw = np.stack((beta[:, 1], beta[:, 2], np.hypot(beta[:, 1], beta[:, 2]), beta[:, 0]), axis=1)
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    return PosteriorFit(
        raw_t4=raw,
        beta_bac=np.ascontiguousarray(beta),
        ols_beta_bac=np.ascontiguousarray(ols),
        residual_variance=sigma2,
        posterior_covariance_ac=np.ascontiguousarray(covariance_ac),
        design_rank=rank,
        residual_dof=dof,
        prior_variance=float(prior_variance),
        trial_count=int(rates.shape[1]),
        raw_t4_sha256=array_sha256(raw),
    )


def fit_source_prior(
    source_sessions: Sequence[tuple[object, object]],
    *,
    variance_floor: float = plan.PRIOR_VARIANCE_FLOOR,
) -> SourcePrior:
    _require(isinstance(source_sessions, Sequence) and len(source_sessions) > 0, "source sessions must be nonempty")
    _require(math.isfinite(variance_floor) and variance_floor > 0.0, "variance floor must be positive")
    squared: list[np.ndarray] = []
    noise: list[np.ndarray] = []
    units = 0
    for rates_value, theta_value in source_sessions:
        rates, design, ols, _rank, _dof, sigma2 = _ols_evidence(rates_value, theta_value)
        inverse_ac_trace = float(np.trace(np.linalg.inv(design.T @ design)[1:3, 1:3]))
        squared.append(np.square(ols[:, 1]) + np.square(ols[:, 2]))
        noise.append(sigma2 * inverse_ac_trace)
        units += int(rates.shape[0])
    raw_second = float(np.concatenate(squared).mean() / 2.0)
    expected_noise = float(np.concatenate(noise).mean() / 2.0)
    variance = float(max(variance_floor, raw_second - expected_noise))
    provisional = {
        "schema": "budget_matched_posterior_source_prior_v1",
        "variance": variance,
        "raw_second_moment": raw_second,
        "expected_ols_noise": expected_noise,
        "variance_floor": float(variance_floor),
        "source_session_count": len(source_sessions),
        "unit_fit_count": units,
        "source_budget": 30,
        "source_only": True,
        "shared_unchanged_across_budgets": [30, 10, 4],
        "e02_reference_prior_not_reused": plan.E02_REFERENCE_PRIOR_VARIANCE_ONLY,
    }
    return SourcePrior(
        variance=variance,
        raw_second_moment=raw_second,
        expected_ols_noise=expected_noise,
        variance_floor=float(variance_floor),
        source_session_count=len(source_sessions),
        unit_fit_count=units,
        source_budget=30,
        body_sha256=_json_sha(provisional),
    )


def fit_equal_budget_normalizer(raw_rows_by_budget: Mapping[int, object]) -> PosteriorNormalizer:
    _require(set(raw_rows_by_budget) == set(plan.NORMALIZER_BUDGET_ORDER), "normalizer requires exact M4/M10/M30 mappings")
    arrays: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    digests: dict[int, str] = {}
    for budget in plan.NORMALIZER_BUDGET_ORDER:
        rows = _array(raw_rows_by_budget[budget], name=f"M{budget} posterior rows", ndim=2)
        _require(rows.shape[1] == 4, "posterior normalizer rows must be [units,4]")
        arrays[budget] = rows
        counts[budget] = int(rows.shape[0])
        digests[budget] = array_sha256(rows)
    _require(len(set(counts.values())) == 1, "posterior normalizer budgets must have equal source-row counts")
    combined = np.ascontiguousarray(np.concatenate([arrays[m] for m in plan.NORMALIZER_BUDGET_ORDER], axis=0))
    mean = np.ascontiguousarray(combined.mean(axis=0), dtype=np.float64)
    std = np.ascontiguousarray(np.sqrt(np.square(combined - mean).mean(axis=0)), dtype=np.float64)
    _require(bool(np.all(std > 0.0)), "posterior normalizer has a zero-variance column")
    provisional = {
        "schema": "budget_matched_posterior_normalizer_v1",
        "row_order": "budget_M4_M10_M30_then_caller_source_roster_then_unit",
        "mean_float64": [float(x) for x in mean],
        "std_float64": [float(x) for x in std],
        "ddof": 0,
        "per_budget_row_count": {str(m): counts[m] for m in plan.NORMALIZER_BUDGET_ORDER},
        "per_budget_rows_sha256": {str(m): digests[m] for m in plan.NORMALIZER_BUDGET_ORDER},
        "rows_sha256": array_sha256(combined),
        "source_only": True,
        "equal_budget_weight": True,
        "ordinary_ols_normalizer_reused": False,
    }
    return PosteriorNormalizer(
        mean=mean,
        std=std,
        per_budget_row_count=counts,
        per_budget_rows_sha256=digests,
        rows_sha256=array_sha256(combined),
        body_sha256=_json_sha(provisional),
    )


def angular_reliability(
    fit: PosteriorFit,
    *,
    eps: float = 1.0e-8,
    flat_value: float = -20.0,
) -> np.ndarray:
    _require(math.isfinite(eps) and eps > 0.0, "eps must be positive")
    _require(math.isfinite(flat_value), "flat value must be finite")
    mu = fit.beta_bac[:, 1:3]
    norm2 = np.einsum("ij,ij->i", mu, mu)
    result = np.full(mu.shape[0], float(flat_value), dtype=np.float64)
    valid = norm2 > eps
    if np.any(valid):
        unit = mu[valid] / np.sqrt(norm2[valid])[:, None]
        perpendicular = np.stack((-unit[:, 1], unit[:, 0]), axis=1)
        covariance = fit.posterior_covariance_ac[valid]
        directional_variance = np.einsum("ij,ijk,ik->i", perpendicular, covariance, perpendicular)
        ratio = directional_variance / (norm2[valid] + eps)
        result[valid] = -np.log(ratio + eps)
    _require(np.isfinite(result).all(), "angular reliability is nonfinite")
    return np.ascontiguousarray(result)

