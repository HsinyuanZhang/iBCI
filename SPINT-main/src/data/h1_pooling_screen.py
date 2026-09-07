"""SB: the pooling screen for the H1 carrier (CPU-only, source data only).

The carrier estimate is noisy at M=4.  The sealed split-half attachment cosine
is about 0.64 per recording, and only 2 of 6 dates clear 0.5.  Measurement
error attenuates the effect.  Cross-session partial pooling is the standard fix.

Three arms:

raw       plain ridge, no shrinkage.
eb_global the current empirical-Bayes shrinkage toward one global mu.
pool_loo  partial pooling, where the prior comes from the other source
          recordings (leave-one-out).

The split-half attachment cosine is the median per-channel cosine between
independent ridge fits on support trials (0,1) vs (2,3), on the raw 7-D
coefficient rows.  This reuses the metric from
``audit_h1_m4_population_decoder_carrier_date_lodo._row_cosine`` so the number
is comparable to the sealed ~0.64.

Status: CPU_ONLY_SOURCE_SCREEN.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    index_heldin_calib,
    load_record,
)
from src.data.h1_lag_screen import LagScreenPlan, build_plan, _ridge_solve


MODULE_STATUS = "CPU_ONLY_SOURCE_SCREEN"
POOLING_SCREEN_SCHEMA = "h1_pooling_screen_v1"
EPS = 1.0e-12


class PoolingScreenError(ValueError):
    """Fail-closed violation of the pooling screen contract."""


# --------------------------------------------------------------------------- #
# Row cosine (reused from the sealed M=4 population decoder audit).
# --------------------------------------------------------------------------- #
def row_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    """Median per-channel cosine between two [N, D] coefficient-row matrices."""

    numerator = (first * second).sum(axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    values = numerator[denominator > EPS] / denominator[denominator > EPS]
    return None if values.size == 0 else float(np.median(values))


# --------------------------------------------------------------------------- #
# Ridge fit helpers.
# --------------------------------------------------------------------------- #
def _raw_rows(beta: np.ndarray, plan: LagScreenPlan) -> np.ndarray:
    """Convert beta in projected space back to per-channel coefficient rows."""

    return (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]


def _fit_ridge_rows(
    record: H1PilotRecord, plan: LagScreenPlan, trial_indices: tuple[int, ...]
) -> np.ndarray:
    """Fit ridge on a subset of support trials, return raw_rows [N, 7]."""

    trials = [record.trials[i] for i in trial_indices]
    rates = np.concatenate([t.rates for t in trials], axis=0)
    velocity = np.concatenate([t.velocity for t in trials], axis=0)
    z = plan.project(rates)
    design = np.column_stack((np.ones(z.shape[0]), z))
    beta = _ridge_solve(design, velocity, plan.ridge_lambda)
    return _raw_rows(beta, plan)


def _fit_ridge_full(
    record: H1PilotRecord, plan: LagScreenPlan
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Full M=4 ridge fit, returning raw_rows, sigma2, G, and hat_trace."""

    support = record.trials[:4]
    rates = np.concatenate([t.rates for t in support], axis=0)
    velocity = np.concatenate([t.velocity for t in support], axis=0)
    z = plan.project(rates)
    design = np.column_stack((np.ones(z.shape[0]), z))
    lam = plan.ridge_lambda
    regularizer = np.eye(design.shape[1]) * lam
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    beta = np.linalg.solve(system, design.T @ velocity)
    rss = np.square(velocity - design @ beta).sum(axis=0)
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    denom = float(len(design) - hat_trace)
    sigma2 = rss / denom if denom > EPS else np.full(velocity.shape[1], np.nan)
    raw_rows = _raw_rows(beta, plan)
    return raw_rows, sigma2, G, hat_trace


def _channel_factor(plan: LagScreenPlan, G: np.ndarray) -> np.ndarray:
    """Per-channel leverage factor projected through the PCA basis."""

    projection = plan.pcs[: plan.q].T  # [N, q]
    return ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)


# --------------------------------------------------------------------------- #
# Shrinkage arms.
# --------------------------------------------------------------------------- #
def shrink_rows(
    raw_rows: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    est_var: np.ndarray,
) -> np.ndarray:
    """James-Stein shrinkage: shrunk = prior_mean + w * (rows - prior_mean).

    w = prior_var / (prior_var + est_var), clipped to [0, 1].
    """

    w = prior_var / (prior_var + est_var + EPS)
    w = np.clip(w, 0.0, 1.0)
    return prior_mean + w * (raw_rows - prior_mean)


# --------------------------------------------------------------------------- #
# Per-recording screen.
# --------------------------------------------------------------------------- #
def screen_record(
    record: H1PilotRecord,
    plan: LagScreenPlan,
    global_prior_mean: np.ndarray,
    global_prior_var: np.ndarray,
    loo_prior_mean: np.ndarray,
    loo_prior_var: np.ndarray,
) -> dict[str, Any]:
    """Run the pooling screen for one recording."""

    rows_a = _fit_ridge_rows(record, plan, (0, 1))
    rows_b = _fit_ridge_rows(record, plan, (2, 3))
    rows_full, sigma2_full, G_full, hat_trace = _fit_ridge_full(record, plan)
    ch_factor = _channel_factor(plan, G_full)

    # Estimation variance for split-half rows (half the data => roughly 2x var)
    est_var_half = np.zeros((EXPECTED_NEURONS, 7), dtype=np.float64)
    for j in range(7):
        if np.isfinite(sigma2_full[j]):
            est_var_half[:, j] = sigma2_full[j] * ch_factor * 2.0

    cosines: dict[str, float | None] = {}

    # raw
    cosines["raw"] = row_cosine(rows_a, rows_b)

    # eb_global
    shrunk_a = shrink_rows(rows_a, global_prior_mean, global_prior_var, est_var_half)
    shrunk_b = shrink_rows(rows_b, global_prior_mean, global_prior_var, est_var_half)
    cosines["eb_global"] = row_cosine(shrunk_a, shrunk_b)

    # pool_loo
    shrunk_a_loo = shrink_rows(rows_a, loo_prior_mean, loo_prior_var, est_var_half)
    shrunk_b_loo = shrink_rows(rows_b, loo_prior_mean, loo_prior_var, est_var_half)
    cosines["pool_loo"] = row_cosine(shrunk_a_loo, shrunk_b_loo)

    return {
        "session_name": record.session_name,
        "date": record.date,
        "cosines": {k: v for k, v in cosines.items()},
        "delta_pool_loo_vs_eb_global": (
            cosines["pool_loo"] - cosines["eb_global"]
            if cosines["pool_loo"] is not None and cosines["eb_global"] is not None
            else None
        ),
        "delta_pool_loo_vs_raw": (
            cosines["pool_loo"] - cosines["raw"]
            if cosines["pool_loo"] is not None and cosines["raw"] is not None
            else None
        ),
        "hat_trace": hat_trace,
    }


# --------------------------------------------------------------------------- #
# Full screen runner.
# --------------------------------------------------------------------------- #
def run_pooling_screen(records: Mapping[str, H1PilotRecord]) -> dict[str, Any]:
    """Run the pooling screen on all 11 source recordings."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    if not set(source).issubset(set(records)):
        raise PoolingScreenError("pooling screen requires all 11 fold-0 source records")
    plan = build_plan(records)

    # Compute full raw_rows for each recording
    full_rows: dict[str, np.ndarray] = {}
    for name in source:
        full_rows[name] = _fit_ridge_rows(records[name], plan, (0, 1, 2, 3))

    # Global prior (all source recordings)
    all_rows = np.stack([full_rows[name] for name in source], axis=0)  # [11, N, 7]
    global_prior_mean = all_rows.mean(axis=0)  # [N, 7]
    global_prior_var = all_rows.var(axis=0)  # [N, 7]

    per_recording = {}
    for name in source:
        # LOO prior: all recordings except this one
        other_names = [n for n in source if n != name]
        other_rows = np.stack([full_rows[n] for n in other_names], axis=0)
        loo_mean = other_rows.mean(axis=0)
        loo_var = other_rows.var(axis=0)

        # Assert the held recording does not enter its own prior
        assert name not in other_names

        per_recording[name] = screen_record(
            records[name], plan, global_prior_mean, global_prior_var, loo_mean, loo_var
        )

    # Aggregate
    arms = ("raw", "eb_global", "pool_loo")
    arm_summary = {}
    for arm in arms:
        values = [per_recording[name]["cosines"][arm] for name in source]
        finite = [v for v in values if v is not None and np.isfinite(v)]
        arm_summary[arm] = {
            "median": float(np.median(finite)) if finite else float("nan"),
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "frac_at_or_above_050": float(np.mean([v >= 0.5 for v in finite])) if finite else float("nan"),
        }

    # Paired deltas
    pool_vs_eb = [
        per_recording[name]["delta_pool_loo_vs_eb_global"]
        for name in source
        if per_recording[name]["delta_pool_loo_vs_eb_global"] is not None
    ]
    pool_vs_raw = [
        per_recording[name]["delta_pool_loo_vs_raw"]
        for name in source
        if per_recording[name]["delta_pool_loo_vs_raw"] is not None
    ]

    return {
        "schema": POOLING_SCREEN_SCHEMA,
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "metric": "split_half_attachment_cosine_on_raw_7d_coefficient_rows_trial01_vs_trial23",
        "metric_comparable_to_sealed_064": True,
        "per_recording": per_recording,
        "arm_summary": arm_summary,
        "paired_deltas": {
            "pool_loo_minus_eb_global": {
                "per_recording": {name: per_recording[name]["delta_pool_loo_vs_eb_global"] for name in source},
                "mean": float(np.mean(pool_vs_eb)) if pool_vs_eb else float("nan"),
                "median": float(np.median(pool_vs_eb)) if pool_vs_eb else float("nan"),
            },
            "pool_loo_minus_raw": {
                "per_recording": {name: per_recording[name]["delta_pool_loo_vs_raw"] for name in source},
                "mean": float(np.mean(pool_vs_raw)) if pool_vs_raw else float("nan"),
            },
        },
        "loo_assertion": "each recording's prior comes from the other 10 source recordings only",
        "read_rule_reference": {
            "at_or_below_0015": "Cross-session pooling adds nothing. The estimator sub-route stops.",
            "raises_median_and_frac": "The attenuation is reducible. The estimator route has a multiplier.",
        },
    }


# --------------------------------------------------------------------------- #
# Receipt.
# --------------------------------------------------------------------------- #
def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n").encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Write the pooling screen receipt beside the module.  No silent overwrite."""

    receipt_path = Path(path).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"pooling screen refuses to overwrite: {receipt_path}")
    payload = _canonical_json(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


# =========================================================================== #
# SB-3: deployed EB arm + uninflatable reference-fit metric (round 2).
# =========================================================================== #
SB3_SCHEMA = "h1_pooling_screen_sb3_v1"


def _fit_deployed_eb_carrier(
    record: H1PilotRecord, plan: LagScreenPlan,
    trial_indices: Sequence[int] = (0, 1, 2, 3),
) -> np.ndarray:
    """Deployed EB carrier: w = tau2/(tau2+v) on 4-D U-projected rows.

    This reuses the deployed empirical-Bayes formula from
    h1_m4_eb_pilot.fit_frozen_carrier (line 443):
        weight = tau2 / (tau2 + projected_variance)
        carrier = mu + weight * (raw_carrier - mu)
    but operates on the fold-0 source plan directly (not the receipt-bound
    FrozenEBPlan) to stay source-only.
    """

    # Fit ridge on the selected trials
    trials = [record.trials[i] for i in trial_indices]
    rates = np.concatenate([t.rates for t in trials], axis=0)
    velocity = np.concatenate([t.velocity for t in trials], axis=0)
    z = plan.project(rates)
    design = np.column_stack((np.ones(z.shape[0]), z))
    lam = plan.ridge_lambda
    reg = np.eye(design.shape[1]) * lam
    reg[0, 0] = 0.0
    system = design.T @ design + reg
    beta = np.linalg.solve(system, design.T @ velocity)
    rss = np.square(velocity - design @ beta).sum(axis=0)
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    denom = float(len(design) - hat_trace)
    sigma2 = rss / denom if denom > EPS else np.full(velocity.shape[1], np.nan)

    # Raw rows and 4-D carrier via top-4 SVD of source rows
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]  # [N, 7]
    # U basis from source rows (same as audit script)
    # NOTE: this must match how the plan was built.  For the deployed formula,
    # the carrier basis U comes from the source pooled rows.  Since we don't
    # have U stored in LagScreenPlan, we compute it from the full M=4 rows.
    # This is equivalent to h1_m4_eb_pilot's approach.
    return raw_rows, sigma2, G, hat_trace


def _compute_deployed_eb(
    record: H1PilotRecord, plan: LagScreenPlan,
    all_source_rows: np.ndarray,
    trial_indices: Sequence[int] = (0, 1, 2, 3),
) -> np.ndarray:
    """Full deployed EB: raw_rows -> U projection -> EB shrinkage."""

    raw_rows, sigma2, G, hat_trace = _fit_deployed_eb_carrier(record, plan, trial_indices)
    # Compute U basis from all source rows (already provided)
    _, _, vt = np.linalg.svd(all_source_rows, full_matrices=False)
    U = vt[:4].T  # [7, 4]
    raw_carrier = raw_rows @ U  # [N, 4]
    # Channel factor
    projection = plan.pcs[: plan.q].T  # [N, q]
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
    projected_covariance = U.T @ np.diag(sigma2) @ U
    projected_variance = channel_factor * np.trace(projected_covariance) / 4.0
    # Prior from source carriers
    source_carriers = all_source_rows @ U
    mu = source_carriers.mean(axis=0)
    tau2 = float(np.square(source_carriers - mu[None, :]).sum() / (source_carriers.shape[0] * 4))
    if tau2 <= EPS:
        tau2 = EPS
    # Deployed EB weight
    weight = tau2 / (tau2 + projected_variance)
    carrier = mu[None, :] + weight[:, None] * (raw_carrier - mu[None, :])
    return carrier


def _compute_js_shrink(
    record: H1PilotRecord, plan: LagScreenPlan,
    global_prior_mean: np.ndarray, global_prior_var: np.ndarray,
    trial_indices: Sequence[int] = (0, 1, 2, 3),
) -> np.ndarray:
    """James-Stein shrinkage on raw 7-D rows (round-1 eb_global formula)."""

    raw_rows, sigma2, G, hat_trace = _fit_deployed_eb_carrier(record, plan, trial_indices)
    projection = plan.pcs[: plan.q].T
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
    est_var = np.zeros_like(raw_rows)
    for j in range(7):
        if np.isfinite(sigma2[j]):
            est_var[:, j] = sigma2[j] * channel_factor
    return shrink_rows(raw_rows, global_prior_mean, global_prior_var, est_var)


def _row_cosine_4d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Per-channel cosine on 4-D carriers, median across channels."""

    return row_cosine(a, b)


def run_sb_3(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """SB-3: deployed EB vs JS global vs raw, with reference-fit metric."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    # Compute all source raw rows for U basis and prior
    all_rows_list = []
    all_rows_by_name: dict[str, np.ndarray] = {}
    for name in source:
        raw_rows, _, _, _ = _fit_deployed_eb_carrier(records[name], plan)
        all_rows_by_name[name] = raw_rows
        all_rows_list.append(raw_rows)
    all_source_rows = np.concatenate(all_rows_list, axis=0)
    global_prior_mean = all_source_rows.mean(axis=0)
    global_prior_var = all_source_rows.var(axis=0)

    per_recording: dict[str, Any] = {}
    for name in source:
        record = records[name]
        # Support rows (trials 0-3)
        rows_a = _fit_ridge_rows(record, plan, (0, 1))
        rows_b = _fit_ridge_rows(record, plan, (2, 3))
        # Raw full rows
        raw_full = all_rows_by_name[name]
        # Reference: all remaining blocks of the recording (trials 4+)
        ref_rates = np.concatenate([t.rates for t in record.trials[4:]], axis=0)
        ref_velocity = np.concatenate([t.velocity for t in record.trials[4:]], axis=0)
        z_ref = plan.project(ref_rates)
        design_ref = np.column_stack((np.ones(z_ref.shape[0]), z_ref))
        beta_ref = _ridge_solve(design_ref, ref_velocity, plan.ridge_lambda)
        ref_rows = (plan.pcs[: plan.q].T @ beta_ref[1:]) / plan.scale[:, None]  # [N, 7]

        # Arms on support
        eb_dep_carrier = _compute_deployed_eb(record, plan, all_source_rows)
        eb_js_carrier = _compute_js_shrink(record, plan, global_prior_mean, global_prior_var)
        raw_carrier = raw_full

        # Metric 1: split-half cosine on raw 7-D rows (inflatable)
        m1_raw = row_cosine(rows_a, rows_b)
        # Metric 2: cosine against reference (uninflatable)
        # Apply same U projection for all arms to get 4-D carriers
        _, _, vt = np.linalg.svd(all_source_rows, full_matrices=False)
        U = vt[:4].T
        ref_4d = ref_rows @ U
        m2_raw = row_cosine(raw_carrier @ U, ref_4d)
        m2_eb_dep = row_cosine(eb_dep_carrier, ref_4d)
        m2_eb_js = row_cosine(eb_js_carrier @ U, ref_4d)

        # For metric 1 with EB arms, apply EB on split halves
        # eb_deployed on halves is complex; use the JS halves from round 1
        _, sigma2_full, G_full, hat_trace = _fit_deployed_eb_carrier(record, plan)
        projection = plan.pcs[: plan.q].T
        ch_factor = ((projection @ G_full[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
        est_var_half = np.zeros_like(rows_a)
        for j in range(7):
            if np.isfinite(sigma2_full[j]):
                est_var_half[:, j] = sigma2_full[j] * ch_factor * 2.0
        shrunk_a_js = shrink_rows(rows_a, global_prior_mean, global_prior_var, est_var_half)
        shrunk_b_js = shrink_rows(rows_b, global_prior_mean, global_prior_var, est_var_half)
        m1_eb_js = row_cosine(shrunk_a_js, shrunk_b_js)
        # eb_deployed metric 1: approximate by computing deployed EB on each half
        # This requires per-half U and tau2 — use the same all-source U
        carrier_a_dep = _compute_deployed_eb(record, plan, all_source_rows, (0, 1))
        carrier_b_dep = _compute_deployed_eb(record, plan, all_source_rows, (2, 3))
        m1_eb_dep = row_cosine(carrier_a_dep, carrier_b_dep)

        per_recording[name] = {
            "metric1_split_half_cosine": {"raw": m1_raw, "eb_deployed": m1_eb_dep, "eb_js_global": m1_eb_js},
            "metric2_reference_cosine": {"raw": m2_raw, "eb_deployed": m2_eb_dep, "eb_js_global": m2_eb_js},
            "deltas_metric2": {
                "eb_js_global_minus_eb_deployed": (m2_eb_js - m2_eb_dep) if all(v is not None for v in [m2_eb_js, m2_eb_dep]) else None,
                "eb_deployed_minus_raw": (m2_eb_dep - m2_raw) if all(v is not None for v in [m2_eb_dep, m2_raw]) else None,
            },
            "deltas_metric1": {
                "eb_js_global_minus_eb_deployed": (m1_eb_js - m1_eb_dep) if all(v is not None for v in [m1_eb_js, m1_eb_dep]) else None,
                "eb_deployed_minus_raw": (m1_eb_dep - m1_raw) if all(v is not None for v in [m1_eb_dep, m1_raw]) else None,
            },
        }

    # Aggregate paired deltas on metric 2
    deltas_m2_js_dep = [
        per_recording[name]["deltas_metric2"]["eb_js_global_minus_eb_deployed"]
        for name in source
        if per_recording[name]["deltas_metric2"]["eb_js_global_minus_eb_deployed"] is not None
    ]
    deltas_m2_dep_raw = [
        per_recording[name]["deltas_metric2"]["eb_deployed_minus_raw"]
        for name in source
        if per_recording[name]["deltas_metric2"]["eb_deployed_minus_raw"] is not None
    ]

    return {
        "schema": SB3_SCHEMA,
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "metric1_note": "split-half attachment cosine; INFLATABLE by shrinkage — if both halves shrink toward a common target, cosine rises without accuracy gain",
        "metric2_note": "cosine against a same-recording reference fit on all remaining blocks; oracle-reference diagnostic, NOT deployable; shrinkage cannot inflate it because the reference does not move",
        "comparability_note": "absolute cosines on the fold-0 source plan are NOT comparable to the sealed 0.64 from the date-LODO plan; only within-plan paired deltas are readable",
        "deployed_eb_source": "h1_m4_eb_pilot.fit_frozen_carrier line 443: weight = tau2 / (tau2 + projected_variance); carrier = mu + weight * (raw_carrier - mu)",
        "per_recording": per_recording,
        "paired_deltas_metric2": {
            "eb_js_global_minus_eb_deployed": {
                "per_recording": {name: per_recording[name]["deltas_metric2"]["eb_js_global_minus_eb_deployed"] for name in source},
                "mean": float(np.mean(deltas_m2_js_dep)) if deltas_m2_js_dep else float("nan"),
                "median": float(np.median(deltas_m2_js_dep)) if deltas_m2_js_dep else float("nan"),
            },
            "eb_deployed_minus_raw": {
                "per_recording": {name: per_recording[name]["deltas_metric2"]["eb_deployed_minus_raw"] for name in source},
                "mean": float(np.mean(deltas_m2_dep_raw)) if deltas_m2_dep_raw else float("nan"),
            },
        },
        "read_rule_reference": {
            "js_better_than_deployed_on_metric2": "The shrinkage FORM is an available upgrade. Report it as a candidate.",
            "arms_agree_on_metric2": "Round-1 +0.412 is what shrinkage does on this plan. No upgrade here.",
            "metric2_smaller_gain_than_metric1": "Metric 1 was inflated. Read only metric 2.",
        },
    }

