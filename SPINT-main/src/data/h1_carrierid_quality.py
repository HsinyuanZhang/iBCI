"""Source-legal carrier-quality features for the H1 CarrierID follow-up.

This module deliberately does **not** change the sealed M=4 carrier estimator
or any existing H-C checkpoint.  It contains only diagnostic and candidate
feature primitives:

* ``confidence_from_m4_fit`` exposes the analytic EB posterior shrinkage
  weight already produced by the frozen four-trial fit;
* ``split_forward_transfer_quality`` fits the frozen ridge map on support
  trials 1--2 and scores its channel-wise carrier agreement against an
  independent support-trials-3--4 fit; and
* ``leave_one_date_out_channel_prior`` tests whether that channel-wise quality
  has a source-date-stable component.

All functions operate on an already loaded record and a frozen source plan.
They do not enumerate or load H1 target recordings, allocate CUDA, train a
model, or select a checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import EPS, PilotDataError, _project


QUALITY_SCHEMA = "h1_carrierid_quality_features_v1"
QUALITY_FEATURE_DIM = 1
MIN_SPLIT_TRIALS = 2


def _as_trials(values: Sequence[float], *, expected: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise PilotDataError(f"{name} requires exactly {expected} TrialNum values, got {len(result)}")
    if len(set(result)) != len(result):
        raise PilotDataError(f"{name} TrialNum values must be distinct")
    return result


def confidence_from_m4_fit(fit: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the per-channel analytic EB shrinkage weight in ``[0,1]``.

    ``fit_frozen_carrier`` already computes this posterior quantity for the
    legal four-trial M=4 carrier.  Treating it as an additional feature is a
    candidate design, not a claim that it improves decoding.
    """

    if "weight" not in fit:
        raise PilotDataError("M=4 fit lacks analytic EB weight")
    values = np.asarray(fit["weight"], dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise PilotDataError("analytic EB weight must be finite and nonempty")
    if np.any(values < -EPS) or np.any(values > 1.0 + EPS):
        raise PilotDataError("analytic EB weight escaped [0,1]")
    return np.clip(values, 0.0, 1.0)


def _fit_partial_raw_carrier(
    record: Any,
    plan: Any,
    trial_values: Sequence[float],
) -> dict[str, np.ndarray]:
    """Fit the frozen ridge map on an explicitly sized support sub-block.

    This is *not* a replacement deployment carrier.  It is used only to form
    the legal 1--2 versus 3--4 forward-transfer agreement statistic.  The
    source PCA, ridge coefficient, and output projection ``U`` all remain the
    immutable M=4 source-plan values; no target/date-specific transform is
    fitted here.
    """

    values = _as_trials(
        trial_values, expected=MIN_SPLIT_TRIALS, name="split carrier fit"
    )
    trials = [record.blocks_for(value) for value in values]
    if any(int(trial.rates.shape[0]) < 2 for trial in trials):
        raise PilotDataError("split carrier fit requires at least two 100-ms blocks per trial")
    rates = np.concatenate([np.asarray(trial.rates, dtype=np.float64) for trial in trials], axis=0)
    labels = np.concatenate([np.asarray(trial.velocity, dtype=np.float64) for trial in trials], axis=0)
    if rates.ndim != 2 or labels.ndim != 2 or rates.shape[0] != labels.shape[0]:
        raise PilotDataError("split carrier fit has malformed rate/velocity blocks")
    z = _project(rates, plan)
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(plan.ridge_lambda)
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    try:
        beta = np.linalg.solve(system, design.T @ labels)
    except np.linalg.LinAlgError as exc:
        raise PilotDataError("split carrier ridge system is singular") from exc
    raw_rows = (np.asarray(plan.pcs[: plan.q], dtype=np.float64).T @ beta[1:]) / np.asarray(
        plan.scale, dtype=np.float64
    )[:, None]
    raw_carrier = raw_rows @ np.asarray(plan.U, dtype=np.float64)
    expected_shape = (int(record.num_neurons), 4)
    if raw_carrier.shape != expected_shape or not np.isfinite(raw_carrier).all():
        raise PilotDataError(
            f"split carrier raw shape/finite contract failed: {raw_carrier.shape}, expected {expected_shape}"
        )
    return {
        "trial_values": np.asarray(values, dtype=np.float64),
        "beta": np.asarray(beta, dtype=np.float64),
        "raw_rows": np.asarray(raw_rows, dtype=np.float64),
        "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
        "block_count": np.asarray([sum(int(trial.rates.shape[0]) for trial in trials)], dtype=np.int64),
    }


def split_forward_transfer_quality(
    record: Any,
    plan: Any,
    support_trial_values: Sequence[float],
) -> dict[str, np.ndarray]:
    """Compute a per-channel 1--2 -> 3--4 carrier agreement feature.

    The carrier used by an eventual model remains the ordinary four-trial
    support carrier.  This quality scalar is legal at deployment because the
    first four support trials and their labels are already present; trials
    3--4 are *not* query data.  It is a forward-transfer **score**, not a
    second target fit and not an oracle.

    Agreement is ``(cosine + 1)/2`` of the two independently fitted raw
    four-dimensional carrier vectors after centering by the immutable source
    EB mean.  Invalid zero-norm channels are assigned the neutral value 0.5
    and explicitly marked in ``valid``.
    """

    values = _as_trials(
        support_trial_values, expected=4, name="split forward-transfer quality"
    )
    fit = _fit_partial_raw_carrier(record, plan, values[:2])
    score = _fit_partial_raw_carrier(record, plan, values[2:])
    center = np.asarray(plan.mu, dtype=np.float64).reshape(1, 4)
    first = np.asarray(fit["raw_carrier"], dtype=np.float64) - center
    second = np.asarray(score["raw_carrier"], dtype=np.float64) - center
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    valid = (first_norm > EPS) & (second_norm > EPS)
    cosine = np.zeros(first.shape[0], dtype=np.float64)
    cosine[valid] = np.einsum("ij,ij->i", first[valid], second[valid]) / (
        first_norm[valid] * second_norm[valid]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    agreement = (cosine + 1.0) / 2.0
    agreement[~valid] = 0.5
    if not np.isfinite(agreement).all():
        raise PilotDataError("split forward-transfer quality is nonfinite")
    return {
        "schema": np.asarray([QUALITY_SCHEMA]),
        "fit_trial_values": np.asarray(values[:2], dtype=np.float64),
        "score_trial_values": np.asarray(values[2:], dtype=np.float64),
        "quality": np.asarray(agreement, dtype=np.float64),
        "cosine": np.asarray(cosine, dtype=np.float64),
        "valid": np.asarray(valid, dtype=bool),
        "fit_raw_carrier": first,
        "score_raw_carrier": second,
        "fit_block_count": fit["block_count"],
        "score_block_count": score["block_count"],
    }


def source_scalar_feature_normalizer(values: np.ndarray) -> dict[str, float]:
    """Fit a scalar source-only z-normalizer for one candidate quality axis."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise PilotDataError("quality normalizer requires finite nonempty source values")
    mean = float(flat.mean())
    scale = float(flat.std(ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        raise PilotDataError("quality feature is degenerate under source-only normalization")
    return {"mean": mean, "scale": scale, "count": int(flat.size), "formula": "(x-source_mean)/source_std"}


def normalize_scalar_feature(values: np.ndarray, normalizer: Mapping[str, float]) -> np.ndarray:
    mean = float(normalizer["mean"])
    scale = float(normalizer["scale"])
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= EPS:
        raise PilotDataError("invalid scalar quality normalizer")
    normalized = (np.asarray(values, dtype=np.float64) - mean) / scale
    if not np.isfinite(normalized).all():
        raise PilotDataError("normalized quality feature is nonfinite")
    return normalized


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Finite-only Pearson correlation, returning NaN for a degenerate axis."""

    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise PilotDataError("Pearson inputs have different shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    left = left[valid] - left[valid].mean()
    right = right[valid] - right[valid].mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(left, right) / denom)


def leave_one_date_out_channel_prior(
    quality_by_session: Mapping[str, np.ndarray],
    date_by_session: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate a source-only channel prior with source-date LODO.

    For every held source date, the prior is the mean channel-quality vector
    from all *other* source dates.  Its correlation with each held session's
    observed split quality quantifies whether a static per-channel reliability
    prior has any reason to be carried into a new date.
    """

    if set(quality_by_session) != set(date_by_session):
        raise PilotDataError("quality/date session sets differ")
    if len(quality_by_session) < 2:
        raise PilotDataError("source-date prior needs at least two source sessions")
    names = tuple(sorted(quality_by_session))
    shape = np.asarray(quality_by_session[names[0]], dtype=np.float64).shape
    if len(shape) != 1 or shape[0] == 0:
        raise PilotDataError("per-channel quality must be a nonempty vector")
    values: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        value = np.asarray(quality_by_session[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise PilotDataError("quality vectors must have matching finite shape")
        values[name] = value
        groups[str(date_by_session[name])].append(name)
    if len(groups) < 2:
        raise PilotDataError("source-date prior needs at least two distinct dates")

    per_date: dict[str, Any] = {}
    pooled_prior: list[np.ndarray] = []
    pooled_observed: list[np.ndarray] = []
    for held_date in sorted(groups):
        held = tuple(sorted(groups[held_date]))
        train = tuple(name for name in names if date_by_session[name] != held_date)
        if not train:
            raise PilotDataError("LODO source prior left no training sessions")
        prior = np.stack([values[name] for name in train], axis=0).mean(axis=0)
        session_scores = {name: pearson_correlation(prior, values[name]) for name in held}
        per_date[held_date] = {
            "held_sessions": list(held),
            "training_sessions": list(train),
            "prior": prior,
            "prior_vs_session_pearson": session_scores,
            "mean_session_pearson": float(np.nanmean(np.asarray(list(session_scores.values()), dtype=np.float64))),
        }
        for name in held:
            pooled_prior.append(prior)
            pooled_observed.append(values[name])
    session_scores = [
        value
        for result in per_date.values()
        for value in result["prior_vs_session_pearson"].values()
    ]
    all_source_prior = np.stack([values[name] for name in names], axis=0).mean(axis=0)
    return {
        "per_channel_all_source_prior": all_source_prior,
        "per_date": per_date,
        "pooled_lodo_pearson": pearson_correlation(
            np.concatenate(pooled_prior), np.concatenate(pooled_observed)
        ),
        "median_session_lodo_pearson": float(np.nanmedian(np.asarray(session_scores, dtype=np.float64))),
        "mean_session_lodo_pearson": float(np.nanmean(np.asarray(session_scores, dtype=np.float64))),
        "source_date_count": len(groups),
        "source_session_count": len(names),
        "channels": int(shape[0]),
    }
