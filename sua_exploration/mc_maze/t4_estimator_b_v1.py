"""Experiment B v1: source-only CPU selection of analytic T4 estimators.

This module has no decoder, no development/formal-session resolver, and no GPU dependency.
It receives exactly fifty chronological rewarded trials already selected by the existing
datamodule path.  Trials ``[0,30)`` are the only calibration support and ``[30,50)`` are
the only prospective score window.  It deliberately never writes raw counts to disk.

The three pre-registered candidates are:

* ``eb_ridge``: a non-zero-mean, full-covariance empirical-Bayes prior estimated only from
  the current outer fold's 26 source sessions (not W3 / not q_unit+M);
* ``second_harmonic``: a five-column rate fit whose *export and score* use only its derived
  first-harmonic four-vector;
* ``poisson_irls``: fixed-iteration log-link IRLS, deterministically projected back to a
  rate-space first-harmonic four-vector before prospective scoring.

The implementation is intentionally CPU-only.  The real runner is guarded in the companion
script; pure functions here are covered by synthetic no-NWB tests.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from mc_maze.t4_cross_budget_audit import REPEATABILITY_SEEDS, disjoint_trial_count_partitions
from mc_maze.t4_cross_budget_protocol import direction_design_metadata, fit_t4_prefix
from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD


N_SOURCE_SESSIONS = 27
N_CALIBRATION_TRIALS = 30
SCORE_TRIAL_START = 30
SCORE_TRIAL_STOP = 50
WITHIN_TRIAL_THINNING_SEEDS: tuple[int, ...] = tuple(REPEATABILITY_SEEDS)
MIN_DIRECTION_NORM = 1.0e-8
RATE_FLOOR_HZ = 1.0e-6
EB_LAMBDA_GRID: tuple[float, ...] = (0.25, 1.0, 4.0)
EB_PRIOR_VERSION = "outer26_full_covariance_nonzero_mean_empirical_bayes_v1_not_w3"
SECOND_HARMONIC_VERSION = "trial_rate_ols_1h_plus_2h_export_and_score_1h_only_v1"
POISSON_IRLS_MAX_ITERATIONS = 32
POISSON_IRLS_TOLERANCE = 1.0e-7
POISSON_ETA_CLIP = (-12.0, 12.0)
POISSON_VERSION = "fixed_irls32_tol1e-7_eta[-12,12]_quadrature_to_rate_1h_v1"
CANDIDATE_NAMES: tuple[str, ...] = ("eb_ridge", "second_harmonic", "poisson_irls")


@dataclass(frozen=True)
class SessionCounts:
    """One source-only session's in-RAM count receipt for Experiment B."""

    name: str
    counts: np.ndarray  # integer [units, 50]
    durations_s: np.ndarray  # float [50]
    direction_indices: np.ndarray  # int [50], canonical 0..7 or -1

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        durations = np.asarray(self.durations_s, dtype=np.float64)
        directions = np.asarray(self.direction_indices, dtype=np.int64)
        if counts.ndim != 2 or counts.shape[1] != SCORE_TRIAL_STOP:
            raise ValueError("Experiment B requires integer counts [units,50]")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("counts must be nonnegative integers")
        if durations.shape != (SCORE_TRIAL_STOP,) or np.any(~np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("durations must be finite positive [50]")
        if directions.shape != (SCORE_TRIAL_STOP,) or np.any((directions < -1) | (directions > 7)):
            raise ValueError("directions must be canonical -1..7 [50]")
        if not self.name:
            raise ValueError("session name is required")


@dataclass(frozen=True)
class DescriptorFit:
    """A fitted exported four-vector and explicit fit-health counters."""

    t4: np.ndarray  # [units,4], [a,c,m,b], NaN row means invalid
    design_rank: int
    rank_deficient_units: int
    nonconverged_units: int
    invalid_units: int
    calibration_ops_proxy: int
    persistent_state_bytes: int
    temporary_workspace_bytes: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EmpiricalBayesPrior:
    mean_bac: np.ndarray  # [b,a,c]
    covariance_bac: np.ndarray  # [3,3]
    source_session_names: tuple[str, ...]


def _support_arrays(session: SessionCounts, counts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(session.counts if counts is None else counts, dtype=np.float64)
    if matrix.shape != session.counts.shape:
        raise ValueError("alternate count matrix must preserve full [units,50] shape")
    return (
        matrix[:, :N_CALIBRATION_TRIALS],
        np.asarray(session.durations_s[:N_CALIBRATION_TRIALS], dtype=np.float64),
        np.asarray(session.direction_indices[:N_CALIBRATION_TRIALS], dtype=np.int64),
    )


def _rate_design(direction_indices: np.ndarray, *, second_harmonic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(direction_indices, dtype=np.int64)
    valid = directions >= 0
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(item)] for item in directions[valid]], dtype=np.float64)
    columns = [np.ones(theta.size), np.cos(theta), np.sin(theta)]
    if second_harmonic:
        columns.extend([np.cos(2.0 * theta), np.sin(2.0 * theta)])
    return np.stack(columns, axis=1), valid


def _to_t4_from_bac(bac: np.ndarray) -> np.ndarray:
    values = np.asarray(bac, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("bac must be [units,3]")
    b, a, c = values[:, 0], values[:, 1], values[:, 2]
    return np.column_stack([a, c, np.hypot(a, c), b]).astype(np.float64, copy=False)


def _ordinary_fit(session: SessionCounts, *, counts: np.ndarray | None = None) -> DescriptorFit:
    support_counts, durations, directions = _support_arrays(session, counts)
    rates = support_counts / durations[None, :]
    base = fit_t4_prefix(rates, durations, directions, budget=N_CALIBRATION_TRIALS)
    invalid = int((~np.isfinite(base.t4).all(axis=1)).sum())
    return DescriptorFit(
        t4=np.asarray(base.t4, dtype=np.float64), design_rank=int(base.design_rank),
        rank_deficient_units=(int(base.t4.shape[0]) if not base.fit_defined else 0),
        nonconverged_units=0, invalid_units=invalid,
        calibration_ops_proxy=int(base.t4.shape[0] * N_CALIBRATION_TRIALS * 3),
        persistent_state_bytes=int(base.t4.shape[0] * 4 * 8), temporary_workspace_bytes=0,
        metadata={"fit": "ordinary_equal_direction_mean_first_harmonic"},
    )


def estimate_eb_prior(training_sessions: Sequence[SessionCounts]) -> EmpiricalBayesPrior:
    """Estimate a full-covariance population prior from outer-train sessions only.

    This is purposely unlike W3: it estimates a nonzero [b,a,c] mean and full covariance
    from unit-level source fits.  The caller owns the outer-fold exclusion invariant.
    """
    if not training_sessions:
        raise ValueError("EB prior needs at least one outer-training session")
    rows: list[np.ndarray] = []
    for session in training_sessions:
        fit = _ordinary_fit(session)
        finite = np.isfinite(fit.t4).all(axis=1)
        if finite.any():
            rows.append(np.column_stack([fit.t4[finite, 3], fit.t4[finite, 0], fit.t4[finite, 1]]))
    if not rows:
        raise ValueError("EB prior has no rank-valid source unit")
    all_rows = np.concatenate(rows, axis=0)
    mean = np.mean(all_rows, axis=0)
    if all_rows.shape[0] < 2:
        covariance = np.eye(3, dtype=np.float64)
    else:
        covariance = np.cov(all_rows, rowvar=False, ddof=1)
    scale = float(np.trace(covariance) / 3.0) if np.isfinite(covariance).all() else 1.0
    covariance = np.asarray(covariance, dtype=np.float64) + np.eye(3) * max(scale, 1.0e-6) * 1.0e-5
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("EB prior is non-finite")
    return EmpiricalBayesPrior(mean, covariance, tuple(item.name for item in training_sessions))


def fit_eb_ridge(session: SessionCounts, prior: EmpiricalBayesPrior, *, shrinkage_lambda: float,
                 counts: np.ndarray | None = None) -> DescriptorFit:
    if shrinkage_lambda not in EB_LAMBDA_GRID:
        raise ValueError("EB lambda not pre-registered")
    support_counts, durations, directions = _support_arrays(session, counts)
    design, valid = _rate_design(directions)
    unit_count = support_counts.shape[0]
    if np.linalg.matrix_rank(design) != 3:
        return DescriptorFit(np.full((unit_count, 4), np.nan), int(np.linalg.matrix_rank(design)), unit_count, 0, unit_count,
                             0, unit_count * 4 * 8, 0, {"reason": "rank_deficient_shared_design"})
    rates = support_counts[:, valid] / durations[None, valid]
    xtx = design.T @ design
    prior_precision = np.linalg.pinv(prior.covariance_bac) / float(shrinkage_lambda)
    out = np.full((unit_count, 3), np.nan, dtype=np.float64)
    for unit in range(unit_count):
        y = rates[unit]
        ols = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ ols
        sigma2 = float(np.dot(residual, residual) / max(1, design.shape[0] - 3))
        likelihood_precision = xtx / max(sigma2, 1.0e-6)
        precision = likelihood_precision + prior_precision
        try:
            out[unit] = np.linalg.solve(precision, likelihood_precision @ ols + prior_precision @ prior.mean_bac)
        except np.linalg.LinAlgError:
            continue
    t4 = _to_t4_from_bac(out)
    invalid = int((~np.isfinite(t4).all(axis=1)).sum())
    return DescriptorFit(t4, 3, 0, 0, invalid, unit_count * (N_CALIBRATION_TRIALS * 9 + 54),
                         unit_count * 4 * 8, int(unit_count * (3 * 3 + 3) * 8),
                         {"candidate": "eb_ridge", "prior_version": EB_PRIOR_VERSION,
                          "prior_source_session_names": prior.source_session_names,
                          "shrinkage_lambda": float(shrinkage_lambda), "output": "[a,c,m,b]"})


def fit_second_harmonic(session: SessionCounts, *, counts: np.ndarray | None = None) -> DescriptorFit:
    support_counts, durations, directions = _support_arrays(session, counts)
    design, valid = _rate_design(directions, second_harmonic=True)
    unit_count = support_counts.shape[0]
    rank = int(np.linalg.matrix_rank(design))
    if rank != 5:
        return DescriptorFit(np.full((unit_count, 4), np.nan), rank, unit_count, 0, unit_count, 0,
                             unit_count * 4 * 8, 0, {"reason": "rank_deficient_second_harmonic_design"})
    beta = np.linalg.lstsq(design, support_counts[:, valid].T / durations[valid, None], rcond=None)[0].T
    # beta columns are [b,a,c,a2,c2]; a2/c2 are estimation-only nuisance terms.
    t4 = _to_t4_from_bac(beta[:, :3])
    invalid = int((~np.isfinite(t4).all(axis=1)).sum())
    return DescriptorFit(t4, rank, 0, 0, invalid, unit_count * N_CALIBRATION_TRIALS * 5,
                         unit_count * 4 * 8, int(unit_count * 5 * 8),
                         {"candidate": "second_harmonic", "version": SECOND_HARMONIC_VERSION,
                          "scoring_uses": "derived_first_harmonic_[a,c,m,b]_only"})


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64); mu = np.maximum(np.asarray(mu, dtype=np.float64), RATE_FLOOR_HZ)
    positive = y > 0.0
    term = mu - y
    term[positive] += y[positive] * np.log(y[positive] / mu[positive])
    return float(2.0 * np.sum(term))


def _poisson_rate_to_first_harmonic(beta: np.ndarray) -> np.ndarray:
    """Project a log-link rate curve to exported rate-space [b,a,c] deterministically."""
    theta = np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
    design = np.column_stack([np.ones(8), np.cos(theta), np.sin(theta)])
    log_rates = np.clip(design @ beta, POISSON_ETA_CLIP[0], POISSON_ETA_CLIP[1])
    rates = np.exp(log_rates)
    return np.linalg.lstsq(design, rates, rcond=None)[0]


def fit_poisson_irls(session: SessionCounts, *, counts: np.ndarray | None = None) -> DescriptorFit:
    support_counts, durations, directions = _support_arrays(session, counts)
    design, valid = _rate_design(directions)
    unit_count = support_counts.shape[0]
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        return DescriptorFit(np.full((unit_count, 4), np.nan), rank, unit_count, 0, unit_count, 0,
                             unit_count * 4 * 8, 0, {"reason": "rank_deficient_poisson_design"})
    x = design; exposure = durations[valid]; y_all = support_counts[:, valid]
    out = np.full((unit_count, 3), np.nan, dtype=np.float64); nonconverged = 0
    for unit, y in enumerate(y_all):
        beta = np.linalg.lstsq(x, np.log((y + 0.5) / exposure), rcond=None)[0]
        converged = False
        for _ in range(POISSON_IRLS_MAX_ITERATIONS):
            eta = np.clip(x @ beta + np.log(exposure), POISSON_ETA_CLIP[0], POISSON_ETA_CLIP[1])
            mu = np.exp(eta)
            hessian = x.T @ (mu[:, None] * x) + np.eye(3) * 1.0e-8
            gradient = x.T @ (y - mu)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            beta_next = beta + step
            if float(np.max(np.abs(step))) <= POISSON_IRLS_TOLERANCE:
                beta = beta_next; converged = True; break
            beta = beta_next
        if not converged:
            nonconverged += 1; continue
        out[unit] = _poisson_rate_to_first_harmonic(beta)
    t4 = _to_t4_from_bac(out)
    invalid = int((~np.isfinite(t4).all(axis=1)).sum())
    return DescriptorFit(t4, rank, 0, nonconverged, invalid,
                         unit_count * POISSON_IRLS_MAX_ITERATIONS * (N_CALIBRATION_TRIALS * 9 + 27),
                         unit_count * 4 * 8, int(unit_count * (N_CALIBRATION_TRIALS * 4 + 18) * 8),
                         {"candidate": "poisson_irls", "version": POISSON_VERSION,
                          "max_iterations": POISSON_IRLS_MAX_ITERATIONS, "tolerance": POISSON_IRLS_TOLERANCE,
                          "eta_clip": list(POISSON_ETA_CLIP), "output": "quadrature_projected_rate_[a,c,m,b]"})


def descriptor_rate_hz(t4: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return [units,trials] rate using only the exported first-harmonic four-vector."""
    values = np.asarray(t4, dtype=np.float64); dirs = np.asarray(directions, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("t4 must be [units,4]")
    rates = np.full((values.shape[0], dirs.size), np.nan, dtype=np.float64)
    valid = dirs >= 0
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(item)] for item in dirs[valid]], dtype=np.float64)
    rates[:, valid] = values[:, 3, None] + values[:, 0, None] * np.cos(theta) + values[:, 1, None] * np.sin(theta)
    return np.maximum(rates, RATE_FLOOR_HZ)


def prospective_deviance(session: SessionCounts, fit: DescriptorFit) -> float | None:
    score_dirs = session.direction_indices[SCORE_TRIAL_START:SCORE_TRIAL_STOP]
    valid_trials = score_dirs >= 0
    if not valid_trials.any():
        return None
    finite_units = np.isfinite(fit.t4).all(axis=1)
    if not finite_units.any():
        return None
    rates = descriptor_rate_hz(fit.t4[finite_units], score_dirs[valid_trials])
    counts = session.counts[finite_units, SCORE_TRIAL_START:SCORE_TRIAL_STOP][:, valid_trials]
    exposure = session.durations_s[SCORE_TRIAL_START:SCORE_TRIAL_STOP][valid_trials]
    return _poisson_deviance(counts, rates * exposure[None, :])


def paired_cosine_reliability(session: SessionCounts, candidate_fit: Callable[[np.ndarray], DescriptorFit]) -> dict:
    """Eight-seed common-valid, disjoint-within-trial directional reliability delta."""
    records: list[dict] = []
    for seed in WITHIN_TRIAL_THINNING_SEEDS:
        parts = disjoint_trial_count_partitions(session.counts[:, :N_CALIBRATION_TRIALS], seed=int(seed))
        full_left = np.zeros_like(session.counts); full_right = np.zeros_like(session.counts)
        full_left[:, :N_CALIBRATION_TRIALS] = parts[0] * 2
        full_right[:, :N_CALIBRATION_TRIALS] = parts[1] * 2
        ordinary_left, ordinary_right = _ordinary_fit(session, counts=full_left), _ordinary_fit(session, counts=full_right)
        candidate_left, candidate_right = candidate_fit(full_left), candidate_fit(full_right)
        vectors = [item.t4[:, :2] for item in (ordinary_left, ordinary_right, candidate_left, candidate_right)]
        common = np.logical_and.reduce([np.isfinite(item).all(axis=1) & (np.linalg.norm(item, axis=1) > MIN_DIRECTION_NORM) for item in vectors])
        if not common.any():
            records.append({"seed": int(seed), "status": "undefined_no_common_valid_unit"}); continue
        def median_cosine(left: np.ndarray, right: np.ndarray) -> float:
            numerator = np.sum(left[common] * right[common], axis=1)
            denom = np.linalg.norm(left[common], axis=1) * np.linalg.norm(right[common], axis=1)
            return float(np.median(numerator / denom))
        ordinary = median_cosine(ordinary_left.t4[:, :2], ordinary_right.t4[:, :2])
        candidate = median_cosine(candidate_left.t4[:, :2], candidate_right.t4[:, :2])
        records.append({"seed": int(seed), "status": "defined", "common_valid_unit_count": int(common.sum()),
                        "ordinary_median_cosine": ordinary, "candidate_median_cosine": candidate,
                        "delta": candidate - ordinary})
    deltas = [float(item["delta"]) for item in records if item["status"] == "defined"]
    return {"seeds": list(WITHIN_TRIAL_THINNING_SEEDS), "records": records,
            "defined_seed_count": len(deltas), "median_delta": (float(np.median(deltas)) if deltas else None)}


def _candidate_fit_fn(name: str, prior: EmpiricalBayesPrior | None, config: Mapping[str, object]) -> Callable[[SessionCounts, np.ndarray | None], DescriptorFit]:
    if name == "eb_ridge":
        if prior is None: raise ValueError("EB requires an outer-training prior")
        value = float(config["shrinkage_lambda"])
        return lambda session, counts=None: fit_eb_ridge(session, prior, shrinkage_lambda=value, counts=counts)
    if name == "second_harmonic":
        return lambda session, counts=None: fit_second_harmonic(session, counts=counts)
    if name == "poisson_irls":
        return lambda session, counts=None: fit_poisson_irls(session, counts=counts)
    raise ValueError(f"unknown Experiment B candidate: {name}")


def evaluate_session(name: str, session: SessionCounts, training_sessions: Sequence[SessionCounts], config: Mapping[str, object]) -> dict:
    """Evaluate one prospective outer/inner fold.  ``training_sessions`` excludes ``session``."""
    candidate = str(config["candidate"])
    prior = estimate_eb_prior(training_sessions) if candidate == "eb_ridge" else None
    fit_fn = _candidate_fit_fn(candidate, prior, config)
    ordinary = _ordinary_fit(session); upgraded = fit_fn(session)
    ordinary_deviance, candidate_deviance = prospective_deviance(session, ordinary), prospective_deviance(session, upgraded)
    ratio = None if ordinary_deviance is None or candidate_deviance is None or ordinary_deviance <= 0 else candidate_deviance / ordinary_deviance
    reliability = paired_cosine_reliability(session, lambda counts: fit_fn(session, counts))
    rank_increase = upgraded.rank_deficient_units > ordinary.rank_deficient_units
    nonconvergence_increase = upgraded.nonconverged_units > ordinary.nonconverged_units
    invalid_increase = upgraded.invalid_units > ordinary.invalid_units
    direction_counts, direction_rank, direction_condition, direction_balance = direction_design_metadata(
        session.direction_indices[:N_CALIBRATION_TRIALS]
    )
    defined = ratio is not None and reliability["median_delta"] is not None
    return {
        "session": name, "candidate": candidate, "config": dict(config), "defined": defined,
        "ordinary_prospective_deviance": ordinary_deviance, "candidate_prospective_deviance": candidate_deviance,
        "prospective_deviance_ratio": ratio, "reliability": reliability,
        "rank_deficient_units": {"ordinary": ordinary.rank_deficient_units, "candidate": upgraded.rank_deficient_units},
        "nonconverged_units": {"ordinary": ordinary.nonconverged_units, "candidate": upgraded.nonconverged_units},
        "invalid_units": {"ordinary": ordinary.invalid_units, "candidate": upgraded.invalid_units},
        "rank_increase": rank_increase, "nonconvergence_increase": nonconvergence_increase, "invalid_increase": invalid_increase,
        "trial_receipt": {"support": [0, N_CALIBRATION_TRIALS], "score": [SCORE_TRIAL_START, SCORE_TRIAL_STOP],
                          "support_direction_counts": direction_counts.tolist(), "support_design_rank": direction_rank,
                          "support_design_condition": direction_condition, "support_direction_balance": direction_balance,
                          "score_labelled_trial_count": int((session.direction_indices[SCORE_TRIAL_START:SCORE_TRIAL_STOP] >= 0).sum())},
        "cost": {"calibration_ops_proxy": upgraded.calibration_ops_proxy,
                 "persistent_state_bytes": upgraded.persistent_state_bytes,
                 "temporary_workspace_bytes": upgraded.temporary_workspace_bytes},
        "candidate_metadata": dict(upgraded.metadata),
    }


def _mean_defined(rows: Iterable[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def _inner_select_config(candidate: str, outer_training: Sequence[SessionCounts]) -> dict:
    """Nested deterministic selection; only EB has a pre-registered hyperparameter grid."""
    configs = ([{"candidate": candidate, "shrinkage_lambda": value} for value in EB_LAMBDA_GRID]
               if candidate == "eb_ridge" else [{"candidate": candidate}])
    summaries: list[tuple[float, float, dict]] = []
    for config in configs:
        ratios: list[float] = []
        for index, held in enumerate(outer_training):
            inner_train = list(outer_training[:index]) + list(outer_training[index + 1:])
            result = evaluate_session(held.name, held, inner_train, config)
            if result["prospective_deviance_ratio"] is not None:
                ratios.append(float(result["prospective_deviance_ratio"]))
        # Non-finite/no-score configs lose deterministically.  Lower deviance then lower lambda.
        summaries.append((float(np.mean(ratios)) if ratios else math.inf,
                          float(config.get("shrinkage_lambda", 0.0)), config))
    return dict(min(summaries, key=lambda item: (item[0], item[1]))[2])


def summarize_candidate(rows: Sequence[dict]) -> dict:
    ratios = [float(row["prospective_deviance_ratio"]) for row in rows if row["prospective_deviance_ratio"] is not None]
    deltas = [float(row["reliability"]["median_delta"]) for row in rows if row["reliability"]["median_delta"] is not None]
    joint = sum(bool(row["prospective_deviance_ratio"] is not None and row["prospective_deviance_ratio"] <= 1.0 and
                     row["reliability"]["median_delta"] is not None and row["reliability"]["median_delta"] >= 0.0)
                for row in rows)
    rank_increase = any(bool(row["rank_increase"]) for row in rows)
    nonconv_increase = any(bool(row["nonconvergence_increase"]) for row in rows)
    invalid_increase = any(bool(row["invalid_increase"]) for row in rows)
    mean_ratio = float(np.mean(ratios)) if ratios else None
    mean_delta = float(np.mean(deltas)) if deltas else None
    passed = (len(rows) == N_SOURCE_SESSIONS and mean_ratio is not None and mean_delta is not None and
              mean_ratio <= 0.98 and mean_delta >= 0.02 and joint >= 20 and not rank_increase and not nonconv_increase and not invalid_increase)
    return {"fold_count": len(rows), "mean_prospective_deviance_ratio": mean_ratio,
            "mean_reliability_delta": mean_delta, "joint_nonworse_fold_count": joint,
            "rank_increase_any": rank_increase, "nonconvergence_increase_any": nonconv_increase,
            "invalid_descriptor_increase_any": invalid_increase,
            "decision": "pass" if passed else "fail"}


def run_nested_candidate(candidate: str, sessions: Sequence[SessionCounts]) -> dict:
    """Run one candidate's strict source nested-LOSO selection with irreversible fail-fast.

    A candidate stops once more than seven folds cannot be joint non-worse, because reaching
    20/27 is then mathematically impossible.  This never promotes an early winner.
    """
    if len(sessions) != N_SOURCE_SESSIONS or len({item.name for item in sessions}) != N_SOURCE_SESSIONS:
        raise ValueError("real Experiment B requires exactly 27 unique source sessions")
    rows: list[dict] = []; joint_failures = 0
    for index, held in enumerate(sessions):
        outer_train = list(sessions[:index]) + list(sessions[index + 1:])
        config = _inner_select_config(candidate, outer_train)
        row = evaluate_session(held.name, held, outer_train, config); rows.append(row)
        row["outer_training_session_names"] = [item.name for item in outer_train]
        row["inner_selection"] = {"scope": "outer_train_only_nested_LOSO", "selected_config": dict(config)}
        is_joint = bool(row["prospective_deviance_ratio"] is not None and row["prospective_deviance_ratio"] <= 1.0 and
                        row["reliability"]["median_delta"] is not None and row["reliability"]["median_delta"] >= 0.0)
        joint_failures += int(not is_joint)
        if joint_failures > 7:
            summary = summarize_candidate(rows)
            summary.update({"decision": "fail_fast_joint_threshold_impossible", "joint_failures": joint_failures,
                            "remaining_folds_not_run": N_SOURCE_SESSIONS - len(rows)})
            return {"candidate": candidate, "outer_folds": rows, "summary": summary}
    return {"candidate": candidate, "outer_folds": rows, "summary": summarize_candidate(rows)}


def select_unique_winner(candidate_results: Mapping[str, Mapping[str, object]]) -> dict:
    """Apply frozen winner/tie semantics after all CPU candidate runs complete."""
    passed = []
    for name in CANDIDATE_NAMES:
        result = candidate_results[name]
        summary = result["summary"]
        if summary.get("decision") == "pass":
            rows = result["outer_folds"]
            cost = float(np.mean([row["cost"]["calibration_ops_proxy"] for row in rows]))
            state = float(np.mean([row["cost"]["persistent_state_bytes"] for row in rows]))
            passed.append((float(summary["mean_prospective_deviance_ratio"]), cost, state, name))
    if not passed:
        return {"decision": "no_cpu_winner_no_gpu", "winner": None}
    passed.sort()
    if len(passed) > 1 and passed[0][:3] == passed[1][:3]:
        return {"decision": "tie_after_frozen_breaks_no_gpu", "winner": None, "tied": [item[3] for item in passed if item[:3] == passed[0][:3]]}
    return {"decision": "unique_cpu_winner_requires_new_gpu_prelaunch", "winner": passed[0][3],
            "tie_break": {"prospective_deviance_ratio": passed[0][0], "calibration_ops_proxy": passed[0][1], "persistent_state_bytes": passed[0][2]}}


def run_all_candidates_parallel(sessions: Sequence[SessionCounts]) -> dict:
    """Run the three pre-registered candidate computations concurrently after one shared read."""
    with ThreadPoolExecutor(max_workers=len(CANDIDATE_NAMES), thread_name_prefix="t4_estimator_b_cpu") as executor:
        futures = {name: executor.submit(run_nested_candidate, name, sessions) for name in CANDIDATE_NAMES}
        results = {name: futures[name].result() for name in CANDIDATE_NAMES}
    return {"candidates": results, "winner": select_unique_winner(results)}
