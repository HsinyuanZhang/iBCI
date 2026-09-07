"""SA: the lag screen for the H1 carrier (CPU-only, source data only).

The carrier holds W at lag 0.  A neuron with a lead or a lag has its lag-0
coefficient attenuated.  This screen measures whether real signal exists at a
non-zero lag, and whether the best lag differs between channels.

Two arms, reported separately:

Arm P (population): regresses velocity on projected rates, sweeping the
velocity lag.  Mirrors the current H1 estimator.  One lag applies to all
channels.

Arm E (per channel): regresses each channel's rate on velocity, sweeping the
velocity lag.  Mirrors the AFC4 encoding form r_i = b + w^T y.

The current H1 estimator is population-level, so a per-channel lag cannot be
expressed in it at all.  Arm E therefore measures the value of a change of
estimator form, not a tuning of the current one.

The null: the velocity inside each trial is circularly shifted by an offset of
at least 1 second (not in the tested lag set), and the identical argmax over
the same 21 lags is redone.  Eight pre-declared offsets give the null
distribution of R2(tau*)/R2(0) and of tau*.

Status: CPU_ONLY_SOURCE_SCREEN.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    BLOCK_BINS,
    BLOCK_SECONDS,
    EXPECTED_NEURONS,
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    VELOCITY_DIM,
    index_heldin_calib,
    load_record,
)


MODULE_STATUS = "CPU_ONLY_SOURCE_SCREEN"
LAG_SCREEN_SCHEMA = "h1_lag_screen_v1"

LAG_RANGE = tuple(range(-10, 11))            # -10..+10 bins, 20 ms steps
LAG_STEP_MS = 20
NULL_OFFSETS = (55, 65, 75, 85, 95, 105, 115, 125)  # >=50 bins = 1 s, not in lag set
NULL_OFFSETS_SECONDS = tuple(o * 0.02 for o in NULL_OFFSETS)
DEAD_CHANNEL = 66
Q_GRID = (2, 4, 8, 16)
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
EPS = 1.0e-12
BIN_MS = 20


class LagScreenError(ValueError):
    """Fail-closed violation of the lag screen contract."""


# --------------------------------------------------------------------------- #
# Plan construction (source-only, mirrors the M=4 population decoder audit).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LagScreenPlan:
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    source_grid_r2: float

    def project(self, rates: np.ndarray) -> np.ndarray:
        return ((rates - self.mean[None, :]) / self.scale[None, :]) @ self.pcs[: self.q].T


def _support_rates(record: H1PilotRecord) -> np.ndarray:
    support = record.trials[:4]
    return np.concatenate([t.rates for t in support], axis=0)


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge_lambda: float) -> np.ndarray:
    regularizer = np.eye(design.shape[1]) * ridge_lambda
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    return np.linalg.solve(system, design.T @ target)


def _score_query_r2(record: H1PilotRecord, plan: LagScreenPlan) -> float:
    support = record.trials[:4]
    query = record.trials[4:]
    rates_s = np.concatenate([t.rates for t in support], axis=0)
    vel_s = np.concatenate([t.velocity for t in support], axis=0)
    z_s = plan.project(rates_s)
    design_s = np.column_stack((np.ones(z_s.shape[0]), z_s))
    beta = _ridge_solve(design_s, vel_s, plan.ridge_lambda)
    fit_mean = vel_s.mean(axis=0)
    sse, tss = 0.0, 0.0
    for trial in query:
        z_q = plan.project(trial.rates)
        design_q = np.column_stack((np.ones(z_q.shape[0]), z_q))
        pred = design_q @ beta
        sse += float(np.square(trial.velocity - pred).sum())
        tss += float(np.square(trial.velocity - fit_mean[None, :]).sum())
    if tss <= EPS:
        raise LagScreenError(f"{record.session_name}: query TSS undefined")
    return float(1.0 - sse / tss)


def build_plan(records: Mapping[str, H1PilotRecord]) -> LagScreenPlan:
    """Build the source PCA plan with grid-selected q and ridge_lambda."""

    source = tuple(sorted(records.keys()))
    source_rates = np.concatenate([_support_rates(records[name]) for name in source], axis=0)
    mean = source_rates.mean(axis=0)
    scale = np.maximum(source_rates.std(axis=0), 1.0e-6)
    _, _, pcs = np.linalg.svd(
        (source_rates - mean[None, :]) / scale[None, :], full_matrices=False
    )
    best = (-np.inf, None, None)
    for q in Q_GRID:
        for lam in LAMBDA_GRID:
            provisional = LagScreenPlan(mean, scale, pcs, q, lam, 0.0)
            mean_r2 = float(np.mean([_score_query_r2(records[name], provisional) for name in source]))
            if mean_r2 > best[0]:
                best = (mean_r2, q, lam)
    return LagScreenPlan(mean, scale, pcs, best[1], best[2], best[0])


# --------------------------------------------------------------------------- #
# Lagged block construction.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaggedBlocks:
    tau: int
    rates: np.ndarray           # [n_valid, N]
    velocity: np.ndarray        # [n_valid, 7]
    n_total: int
    n_dropped: int


def build_lagged_blocks(
    record: H1PilotRecord, tau: int, *, velocity_override: np.ndarray | None = None
) -> LaggedBlocks:
    """Build 100 ms blocks with velocity shifted by tau bins.

    A block whose shifted bins leave its own TrialNum is dropped.  Channel 66
    rates are included in the output but flagged for exclusion in Arm E.
    """

    velocity = velocity_override if velocity_override is not None else record.velocity
    rates_list: list[np.ndarray] = []
    vel_list: list[np.ndarray] = []
    n_total = 0
    n_dropped = 0
    for trial in record.trials:
        n_total += trial.rates.shape[0]
        shifted = trial.block_indices + tau
        valid = np.ones(trial.rates.shape[0], dtype=bool)
        for col in range(BLOCK_BINS):
            idx = shifted[:, col]
            in_bounds = (idx >= 0) & (idx < record.trial_num.shape[0])
            valid &= in_bounds
        for i in np.flatnonzero(valid):
            idx = shifted[i]
            if not np.all(record.trial_num[idx] == trial.trial_number):
                valid[i] = False
                continue
            if not np.isfinite(velocity[idx]).all():
                valid[i] = False
        n_valid = int(valid.sum())
        n_dropped += int(trial.rates.shape[0]) - n_valid
        if n_valid > 0:
            rates_list.append(trial.rates[valid])
            vel_list.append(np.array([velocity[shifted[i]].mean(axis=0) for i in np.flatnonzero(valid)]))
    if not rates_list:
        return LaggedBlocks(tau, np.empty((0, EXPECTED_NEURONS)), np.empty((0, VELOCITY_DIM)), n_total, n_dropped)
    return LaggedBlocks(
        tau,
        np.concatenate(rates_list, axis=0).astype(np.float64),
        np.concatenate(vel_list, axis=0).astype(np.float64),
        n_total,
        n_dropped,
    )


# --------------------------------------------------------------------------- #
# R2 computation.
# --------------------------------------------------------------------------- #
def arm_p_r2(blocks: LaggedBlocks, plan: LagScreenPlan, ridge_lambda: float) -> float:
    """Population velocity reconstruction R2 at a given lag."""

    if blocks.rates.shape[0] < plan.q + 2:
        return float("nan")
    z = plan.project(blocks.rates)
    design = np.column_stack((np.ones(z.shape[0]), z))
    beta = _ridge_solve(design, blocks.velocity, ridge_lambda)
    pred = design @ beta
    rss = float(np.square(blocks.velocity - pred).sum())
    tss = float(np.square(blocks.velocity - blocks.velocity.mean(axis=0)[None, :]).sum())
    if tss <= EPS:
        return float("nan")
    return float(1.0 - rss / tss)


def arm_e_r2(blocks: LaggedBlocks, ridge_lambda: float, exclude_channels: frozenset[int] = frozenset({DEAD_CHANNEL})) -> np.ndarray:
    """Per-channel encoding R2: rate_i ~ [1, velocity], returns [N] array."""

    n_valid = blocks.rates.shape[0]
    n_channels = blocks.rates.shape[1]
    design = np.column_stack((np.ones(n_valid), blocks.velocity))  # [n_valid, 8]
    r2 = np.full(n_channels, np.nan, dtype=np.float64)
    reg = np.eye(design.shape[1]) * ridge_lambda
    reg[0, 0] = 0.0
    system = design.T @ design + reg
    for ch in range(n_channels):
        if ch in exclude_channels:
            continue
        target = blocks.rates[:, ch]
        if target.max() <= 0.0:
            continue
        beta_ch = np.linalg.solve(system, design.T @ target)
        pred = design @ beta_ch
        rss = float(np.square(target - pred).sum())
        tss = float(np.square(target - target.mean()).sum())
        if tss > EPS:
            r2[ch] = float(1.0 - rss / tss)
    return r2


# --------------------------------------------------------------------------- #
# Null: circular velocity shift within each trial.
# --------------------------------------------------------------------------- #
def circular_shift_velocity(record: H1PilotRecord, offset: int) -> np.ndarray:
    """Return a copy of record.velocity with each trial's velocity circularly shifted."""

    shifted = record.velocity.astype(np.float64).copy()
    for trial in record.trials:
        bins = trial.block_indices[:, 0]  # first bin of each block (representative)
    # Use all eval-valid bins of the trial for the shift
    for trial in record.trials:
        trial_bins = np.flatnonzero(
            (record.trial_num == trial.trial_number) & record.eval_mask
        )
        if trial_bins.size > 1:
            shifted[trial_bins] = np.roll(shifted[trial_bins], offset, axis=0)
    return shifted


# --------------------------------------------------------------------------- #
# Per-recording screen.
# --------------------------------------------------------------------------- #
def screen_record(record: H1PilotRecord, plan: LagScreenPlan) -> dict[str, Any]:
    """Run the full lag sweep for one recording: real + null."""

    # --- Real sweep ---
    arm_p_real = {}
    arm_e_real = {}
    dropped_counts = {}
    for tau in LAG_RANGE:
        blocks = build_lagged_blocks(record, tau)
        arm_p_real[tau] = arm_p_r2(blocks, plan, plan.ridge_lambda)
        arm_e_real[tau] = arm_e_r2(blocks, plan.ridge_lambda)
        dropped_counts[tau] = blocks.n_dropped
    arm_p_real_arr = np.array([arm_p_real[t] for t in LAG_RANGE])
    arm_p_tau_star = int(LAG_RANGE[int(np.nanargmax(arm_p_real_arr))])
    arm_p_ratio = float(arm_p_real[arm_p_tau_star] / arm_p_real[0]) if arm_p_real[0] > EPS else float("nan")

    arm_e_matrix = np.array([arm_e_real[t] for t in LAG_RANGE])  # [21, N]
    valid_mask = np.isfinite(arm_e_matrix).any(axis=0)
    tau_star_per_ch = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    ratio_per_ch = np.full(EXPECTED_NEURONS, np.nan, dtype=np.float64)
    for ch in range(EXPECTED_NEURONS):
        if not valid_mask[ch]:
            continue
        col = arm_e_matrix[:, ch]
        best_idx = int(np.argmax(col))
        tau_star_per_ch[ch] = LAG_RANGE[best_idx]
        if col[LAG_RANGE.index(0)] > EPS:
            ratio_per_ch[ch] = float(col[best_idx] / col[LAG_RANGE.index(0)])

    finite_tau = tau_star_per_ch[tau_star_per_ch != -999]
    finite_ratio = ratio_per_ch[np.isfinite(ratio_per_ch)]

    # --- Null sweep ---
    null_results: list[dict[str, Any]] = []
    for offset in NULL_OFFSETS:
        shifted_v = circular_shift_velocity(record, offset)
        arm_p_null = {}
        arm_e_null = {}
        for tau in LAG_RANGE:
            blocks = build_lagged_blocks(record, tau, velocity_override=shifted_v)
            arm_p_null[tau] = arm_p_r2(blocks, plan, plan.ridge_lambda)
            arm_e_null[tau] = arm_e_r2(blocks, plan.ridge_lambda)
        arm_p_null_arr = np.array([arm_p_null[t] for t in LAG_RANGE])
        null_p_tau_star = int(LAG_RANGE[int(np.nanargmax(arm_p_null_arr))])
        null_p_ratio = float(arm_p_null[null_p_tau_star] / arm_p_null[0]) if arm_p_null[0] > EPS else float("nan")
        arm_e_null_matrix = np.array([arm_e_null[t] for t in LAG_RANGE])
        null_tau_per_ch = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
        null_ratio_per_ch = np.full(EXPECTED_NEURONS, np.nan, dtype=np.float64)
        for ch in range(EXPECTED_NEURONS):
            if not valid_mask[ch]:
                continue
            col = arm_e_null_matrix[:, ch]
            best_idx = int(np.argmax(col))
            null_tau_per_ch[ch] = LAG_RANGE[best_idx]
            if col[LAG_RANGE.index(0)] > EPS:
                null_ratio_per_ch[ch] = float(col[best_idx] / col[LAG_RANGE.index(0)])
        null_finite_ratio = null_ratio_per_ch[np.isfinite(null_ratio_per_ch)]
        null_results.append({
            "offset_bins": offset,
            "offset_seconds": offset * 0.02,
            "arm_p_tau_star": null_p_tau_star,
            "arm_p_ratio": null_p_ratio,
            "arm_e_ratio_q50": float(np.nanmedian(null_finite_ratio)) if null_finite_ratio.size else float("nan"),
            "arm_e_ratio_q95": float(np.nanquantile(null_finite_ratio, 0.95)) if null_finite_ratio.size else float("nan"),
            "arm_e_tau_star_q50": float(np.nanmedian(null_tau_per_ch[null_tau_per_ch != -999].astype(float))) if (null_tau_per_ch != -999).any() else float("nan"),
            "arm_e_tau_star_iqr": float(np.nanpercentile(null_tau_per_ch[null_tau_per_ch != -999].astype(float), 75) - np.nanpercentile(null_tau_per_ch[null_tau_per_ch != -999].astype(float), 25)) if (null_tau_per_ch != -999).any() else float("nan"),
        })

    null_ratios = [r["arm_e_ratio_q50"] for r in null_results if np.isfinite(r["arm_e_ratio_q50"])]
    null_ratio_q95 = float(np.quantile(null_ratios, 0.95)) if len(null_ratios) >= 2 else float("nan")
    real_ratio_q50 = float(np.nanmedian(finite_ratio)) if finite_ratio.size else float("nan")
    frac_above_null_q95 = float(np.mean(ratio_per_ch[np.isfinite(ratio_per_ch)] > null_ratio_q95)) if (np.isfinite(ratio_per_ch).any() and np.isfinite(null_ratio_q95)) else float("nan")

    null_p_ratios = [r["arm_p_ratio"] for r in null_results if np.isfinite(r["arm_p_ratio"])]
    null_p_ratio_q95 = float(np.quantile(null_p_ratios, 0.95)) if len(null_p_ratios) >= 2 else float("nan")

    return {
        "session_name": record.session_name,
        "date": record.date,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "lag_range": list(LAG_RANGE),
        "null_offsets_bins": list(NULL_OFFSETS),
        "dead_channel_excluded": DEAD_CHANNEL,
        "arm_p": {
            "r2_by_lag": {str(t): arm_p_real[t] for t in LAG_RANGE},
            "tau_star": arm_p_tau_star,
            "ratio": arm_p_ratio,
            "null_ratio_q95": null_p_ratio_q95,
        },
        "arm_e": {
            "tau_star_median": float(np.nanmedian(finite_tau.astype(float))) if finite_tau.size else float("nan"),
            "tau_star_iqr": float(np.nanpercentile(finite_tau.astype(float), 75) - np.nanpercentile(finite_tau.astype(float), 25)) if finite_tau.size else float("nan"),
            "frac_abs_tau_le_1": float(np.mean(np.abs(finite_tau) <= 1)) if finite_tau.size else float("nan"),
            "ratio_median": real_ratio_q50,
            "ratio_iqr": float(np.nanpercentile(finite_ratio, 75) - np.nanpercentile(finite_ratio, 25)) if finite_ratio.size else float("nan"),
            "null_ratio_q95": null_ratio_q95,
            "frac_channels_above_null_q95": frac_above_null_q95,
        },
        "null_replicates": null_results,
        "dropped_blocks_by_lag": {str(t): dropped_counts[t] for t in LAG_RANGE},
        "_tau_star_per_ch": tau_star_per_ch.tolist(),
    }


# --------------------------------------------------------------------------- #
# Full screen runner.
# --------------------------------------------------------------------------- #
def run_lag_screen(records: Mapping[str, H1PilotRecord]) -> dict[str, Any]:
    """Run the lag screen on all 11 source recordings."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    if not set(source).issubset(set(records)):
        raise LagScreenError("lag screen requires all 11 fold-0 source records")
    plan = build_plan(records)
    per_recording = {}
    for name in source:
        per_recording[name] = screen_record(records[name], plan)

    # Cross-recording tau* stability for Arm E
    tau_star_arrays = []
    for name in source:
        rec_result = per_recording[name]
        raw = rec_result.pop("_tau_star_per_ch", None)
        tau_star_arrays.append(np.array(raw, dtype=np.int64) if raw is not None else np.full(EXPECTED_NEURONS, -999, dtype=np.int64))

    # Compute pairwise correlations of tau* between recordings
    stability = []
    for i in range(len(source)):
        for j in range(i + 1, len(source)):
            a, b = tau_star_arrays[i], tau_star_arrays[j]
            valid = (a != -999) & (b != -999)
            if valid.sum() > 5:
                a_std = np.std(a[valid].astype(float))
                b_std = np.std(b[valid].astype(float))
                if a_std > EPS and b_std > EPS:
                    corr = float(np.corrcoef(a[valid].astype(float), b[valid].astype(float))[0, 1])
                    stability.append({"pair": f"{source[i]}__{source[j]}", "corr": corr})

    stability_corrs = [s["corr"] for s in stability if np.isfinite(s["corr"])]
    return {
        "schema": LAG_SCREEN_SCHEMA,
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "lag_range": list(LAG_RANGE),
        "lag_step_ms": BIN_MS,
        "null_offsets_bins": list(NULL_OFFSETS),
        "null_offsets_seconds": list(NULL_OFFSETS_SECONDS),
        "dead_channel_excluded": DEAD_CHANNEL,
        "per_recording": per_recording,
        "tau_star_stability": {
            "pairwise_correlations": stability,
            "mean_corr": float(np.mean(stability_corrs)) if stability_corrs else float("nan"),
            "median_corr": float(np.median(stability_corrs)) if stability_corrs else float("nan"),
        },
        "read_rule_reference": {
            "no_separation": "No lag content. The lag route stops.",
            "arm_p_clear_lag": "A single global lag constant is a cheap fix. Report it as a candidate.",
            "arm_e_spread": "A per-channel lag basis has content. Justifies a change of estimator form.",
            "arm_e_concentrated": "One global lag is enough. Do not widen the carrier.",
        },
    }


# --------------------------------------------------------------------------- #
# Receipt.
# --------------------------------------------------------------------------- #
def _sanitize_nan(value: Any) -> Any:
    """Convert NaN/Inf floats to None so they are valid JSON."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nan(v) for v in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(_sanitize_nan(value), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n").encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Write the lag screen receipt beside the module.  No silent overwrite."""

    receipt_path = Path(path).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"lag screen refuses to overwrite: {receipt_path}")
    payload = _canonical_json(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


# =========================================================================== #
# SA-P2: uncensored population lag (round 2).
# =========================================================================== #
LAG_RANGE_WIDE = tuple(range(-10, 21))  # -10..+20 bins = -200..+400 ms


def arm_p_curve(record: H1PilotRecord, plan: LagScreenPlan, lag_range: Sequence[int]) -> dict[int, float]:
    """Compute R2(tau) over a lag range for one recording.  No argmax."""

    curve: dict[int, float] = {}
    for tau in lag_range:
        blocks = build_lagged_blocks(record, tau)
        curve[tau] = arm_p_r2(blocks, plan, plan.ridge_lambda)
    return curve


def loo_declared_lag_test(
    records: Mapping[str, H1PilotRecord], plan: LagScreenPlan,
    lag_range: Sequence[int] = LAG_RANGE_WIDE,
) -> dict[str, Any]:
    """Leave-one-recording-out declared-lag test.

    For each recording r: pool R2(tau) curves of the other 10 recordings,
    declare tau_r as the argmax of the pooled curve, then score R2(tau_r) and
    R2(0) on recording r WITHOUT searching on it.
    """

    source = tuple(sorted(records.keys()))
    curves = {name: arm_p_curve(records[name], plan, lag_range) for name in source}

    results: list[dict[str, Any]] = []
    for held in source:
        others = [n for n in source if n != held]
        # Pool: average R2(tau) across the other 10 recordings.
        pooled_curve = {}
        for tau in lag_range:
            vals = [curves[n][tau] for n in others if np.isfinite(curves[n][tau])]
            pooled_curve[tau] = float(np.mean(vals)) if vals else float("nan")
        finite_taus = [t for t in lag_range if np.isfinite(pooled_curve[t])]
        if not finite_taus:
            raise LagScreenError(f"pooled curve has no finite values when holding {held}")
        tau_r = max(finite_taus, key=lambda t: pooled_curve[t])
        # Score on held recording WITHOUT argmax.
        r2_at_tau_r = curves[held][tau_r]
        r2_at_zero = curves[held][0]
        delta = r2_at_tau_r - r2_at_zero if np.isfinite(r2_at_tau_r) and np.isfinite(r2_at_zero) else float("nan")
        ratio = r2_at_tau_r / r2_at_zero if abs(r2_at_zero) > EPS and np.isfinite(r2_at_zero) and np.isfinite(r2_at_tau_r) else float("nan")
        results.append({
            "held_recording": held,
            "declared_lag_tau_r": tau_r,
            "r2_at_tau_r": r2_at_tau_r,
            "r2_at_zero": r2_at_zero,
            "delta": delta,
            "ratio": ratio,
        })

    deltas = [r["delta"] for r in results if np.isfinite(r["delta"])]
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    # Exact two-sided sign test (binomial).
    n_nonzero = n_positive + n_negative
    if n_nonzero > 0:
        from scipy.stats import binomtest
        sign_test = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "two_sided_p_value": float(binomtest(n_positive, n_nonzero, 0.5, alternative="two-sided").pvalue),
        }
    else:
        sign_test = {"n_positive": 0, "n_negative": 0, "n_zero": n_zero, "two_sided_p_value": float("nan")}

    declared_lags = [r["declared_lag_tau_r"] for r in results]
    return {
        "per_recording": results,
        "deltas": deltas,
        "sign_test": sign_test,
        "declared_lags": declared_lags,
        "declared_lag_mean": float(np.mean(declared_lags)),
        "declared_lag_std": float(np.std(declared_lags)),
        "declared_lag_all_same": len(set(declared_lags)) == 1,
        "lag_range": list(lag_range),
    }


def run_sa_p2(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """SA-P2: widened sweep + leave-one-out declared-lag test."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    # Widened curves for each recording
    curves = {name: arm_p_curve(records[name], plan, LAG_RANGE_WIDE) for name in source}
    # Check if curve still rises at +20
    rising_at_edge = {}
    for name in source:
        c = curves[name]
        rising_at_edge[name] = c.get(20, float("nan")) > c.get(19, float("nan"))
    # LOO test
    loo = loo_declared_lag_test(records, plan, LAG_RANGE_WIDE)
    return {
        "schema": "h1_lag_screen_sa_p2_v1",
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "lag_range_wide": list(LAG_RANGE_WIDE),
        "r2_curves_by_recording": {name: {str(t): curves[name][t] for t in LAG_RANGE_WIDE} for name in source},
        "rising_at_edge_plus20": rising_at_edge,
        "loo_declared_lag_test": loo,
        "read_rule_reference": {
            "ten_of_eleven_positive_stable": "A single frozen global lag is a candidate. Report the value.",
            "mixed_signs": "Arm P stops.",
            "rising_at_plus20": "Open boundary. Do not widen again in this brief.",
        },
        "scope_note": "source-internal leave-one-out screen; proves nothing about a target session",
    }


# =========================================================================== #
# SA-E2: per-channel lag with usable null (round 2).
# =========================================================================== #
NULL_OFFSETS_R2 = tuple(range(55, 55 + 48 * 10, 10))  # 48 offsets, >=1s apart by 200ms
NULL_OFFSETS_R2_SECONDS = tuple(o * 0.02 for o in NULL_OFFSETS_R2)
SUPPORT_TRIAL_COUNT = 4


def build_lagged_blocks_m4_only(
    record: H1PilotRecord, tau: int, *, velocity_override: np.ndarray | None = None
) -> LaggedBlocks:
    """Build lagged blocks from the M=4 support trials only (trials 0-3)."""

    velocity = velocity_override if velocity_override is not None else record.velocity
    rates_list: list[np.ndarray] = []
    vel_list: list[np.ndarray] = []
    n_total = 0
    n_dropped = 0
    for trial in record.trials[:SUPPORT_TRIAL_COUNT]:
        n_total += trial.rates.shape[0]
        shifted = trial.block_indices + tau
        valid = np.ones(trial.rates.shape[0], dtype=bool)
        for col in range(BLOCK_BINS):
            idx = shifted[:, col]
            in_bounds = (idx >= 0) & (idx < record.trial_num.shape[0])
            valid &= in_bounds
        for i in np.flatnonzero(valid):
            idx = shifted[i]
            if not np.all(record.trial_num[idx] == trial.trial_number):
                valid[i] = False
                continue
            if not np.isfinite(velocity[idx]).all():
                valid[i] = False
        n_valid = int(valid.sum())
        n_dropped += int(trial.rates.shape[0]) - n_valid
        if n_valid > 0:
            rates_list.append(trial.rates[valid])
            vel_list.append(np.array([velocity[shifted[i]].mean(axis=0) for i in np.flatnonzero(valid)]))
    if not rates_list:
        return LaggedBlocks(tau, np.empty((0, EXPECTED_NEURONS)), np.empty((0, VELOCITY_DIM)), n_total, n_dropped)
    return LaggedBlocks(
        tau,
        np.concatenate(rates_list, axis=0).astype(np.float64),
        np.concatenate(vel_list, axis=0).astype(np.float64),
        n_total,
        n_dropped,
    )


def arm_e_tau_star_per_channel(
    blocks: LaggedBlocks, ridge_lambda: float, lag_range: Sequence[int] = LAG_RANGE,
    blocks_builder=build_lagged_blocks, record: H1PilotRecord | None = None,
    velocity_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tau_star_per_ch[N], ratio_per_ch[N]) for a lag sweep."""

    arm_e_matrix = np.full((len(lag_range), EXPECTED_NEURONS), np.nan)
    for i_tau, tau in enumerate(lag_range):
        if record is not None:
            b = blocks_builder(record, tau, velocity_override=velocity_override)
        else:
            b = blocks
        r2 = arm_e_r2(b, ridge_lambda)
        arm_e_matrix[i_tau] = r2
    tau_star = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    ratio = np.full(EXPECTED_NEURONS, np.nan, dtype=np.float64)
    valid = np.isfinite(arm_e_matrix).any(axis=0)
    idx_zero = list(lag_range).index(0)
    for ch in range(EXPECTED_NEURONS):
        if not valid[ch]:
            continue
        best = int(np.argmax(arm_e_matrix[:, ch]))
        tau_star[ch] = lag_range[best]
        if arm_e_matrix[idx_zero, ch] > EPS:
            ratio[ch] = float(arm_e_matrix[best, ch] / arm_e_matrix[idx_zero, ch])
    return tau_star, ratio


def pooled_rank_test(
    record: H1PilotRecord, ridge_lambda: float,
    lag_range: Sequence[int] = LAG_RANGE,
    null_offsets: Sequence[int] = NULL_OFFSETS_R2,
    blocks_builder=build_lagged_blocks,
) -> dict[str, Any]:
    """Pooled rank test: for each channel, rank the real ratio in its null draws."""

    # Real ratio per channel
    real_tau, real_ratio = arm_e_tau_star_per_channel(
        None, ridge_lambda, lag_range, blocks_builder, record=record
    )
    # Null ratios per channel per offset
    null_ratios = np.full((len(null_offsets), EXPECTED_NEURONS), np.nan)
    for i_off, offset in enumerate(null_offsets):
        shifted_v = circular_shift_velocity(record, offset)
        _, null_ratio = arm_e_tau_star_per_channel(
            None, ridge_lambda, lag_range, blocks_builder, record=record,
            velocity_override=shifted_v,
        )
        null_ratios[i_off] = null_ratio

    # For each channel, rank the real ratio in its null draws.
    # Under H0, real is a draw from the same distribution -> rank is uniform.
    ranks = np.full(EXPECTED_NEURONS, np.nan)
    valid = np.isfinite(real_ratio) & np.all(np.isfinite(null_ratios), axis=0)
    for ch in range(EXPECTED_NEURONS):
        if not valid[ch]:
            continue
        combined = np.concatenate([[real_ratio[ch]], null_ratios[:, ch]])
        rank = np.searchsorted(np.sort(combined), real_ratio[ch]) + 1
        ranks[ch] = rank / len(combined)  # normalised rank in (0, 1]

    finite_ranks = ranks[np.isfinite(ranks)]
    n_null = len(null_offsets)
    # Under null, ranks uniform on 1/(n_null+1), 2/(n_null+1), ..., 1.0
    # Pool the ranks across channels and compare to uniform.
    # Report fraction of channels in each quartile.
    quartile_counts = [
        int(np.sum((finite_ranks > 0.0) & (finite_ranks <= 0.25))),
        int(np.sum((finite_ranks > 0.25) & (finite_ranks <= 0.50))),
        int(np.sum((finite_ranks > 0.50) & (finite_ranks <= 0.75))),
        int(np.sum((finite_ranks > 0.75) & (finite_ranks <= 1.0))),
    ]
    expected_per_quartile = len(finite_ranks) / 4.0

    # frac > null q95 (secondary)
    frac_above_q95 = float(np.mean(finite_ranks > (1.0 - 1.0 / (n_null + 1)))) if finite_ranks.size else float("nan")

    return {
        "n_null_offsets": n_null,
        "n_channels_with_finite_ranks": int(finite_ranks.size),
        "rank_distribution_quartiles": quartile_counts,
        "expected_per_quartile": float(expected_per_quartile),
        "rank_mean": float(np.mean(finite_ranks)) if finite_ranks.size else float("nan"),
        "rank_std": float(np.std(finite_ranks)) if finite_ranks.size else float("nan"),
        "frac_channels_above_null_q95": frac_above_q95,
        "expected_frac_above_q95": 0.05,
        "rank_per_channel": ranks.tolist(),
    }


def split_half_stability(
    record: H1PilotRecord, ridge_lambda: float,
    lag_range: Sequence[int] = LAG_RANGE,
    scheme: str = "alternate",
    blocks_builder=build_lagged_blocks,
) -> dict[str, Any]:
    """Within-session split-half stability of tau*_i.

    scheme='alternate': odd-indexed blocks vs even-indexed blocks (primary).
    scheme='first_second': first half vs second half (sensitivity).
    """

    all_blocks = blocks_builder(record, 0)
    n = all_blocks.rates.shape[0]
    if scheme == "alternate":
        mask_a = np.arange(n) % 2 == 0
        mask_b = ~mask_a
    elif scheme == "first_second":
        mid = n // 2
        mask_a = np.arange(n) < mid
        mask_b = ~mask_a
    else:
        raise LagScreenError(f"unknown split scheme: {scheme}")

    # Compute tau* on each half by running the arm E sweep restricted to those blocks.
    # Since arm_e_r2 needs the full blocks object, we need to re-sweep with restricted blocks.
    tau_star_a = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    tau_star_b = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    arm_e_matrix_a = np.full((len(lag_range), EXPECTED_NEURONS), np.nan)
    arm_e_matrix_b = np.full((len(lag_range), EXPECTED_NEURONS), np.nan)
    for i_tau, tau in enumerate(lag_range):
        blocks_tau = blocks_builder(record, tau)
        n_tau = blocks_tau.rates.shape[0]
        # The mask may not align after dropping blocks at non-zero lags.  Use the
        # mask applied to the tau=0 block count; non-zero tau drops blocks at the
        #edges, so we apply the same fractional split.
        if scheme == "alternate":
            m_a = np.arange(n_tau) % 2 == 0
            m_b = ~m_a
        else:
            mid_tau = n_tau // 2
            m_a = np.arange(n_tau) < mid_tau
            m_b = ~m_a
        blocks_a = LaggedBlocks(tau, blocks_tau.rates[m_a], blocks_tau.velocity[m_a], n_tau, 0)
        blocks_b = LaggedBlocks(tau, blocks_tau.rates[m_b], blocks_tau.velocity[m_b], n_tau, 0)
        arm_e_matrix_a[i_tau] = arm_e_r2(blocks_a, ridge_lambda)
        arm_e_matrix_b[i_tau] = arm_e_r2(blocks_b, ridge_lambda)
    valid = np.isfinite(arm_e_matrix_a).any(axis=0) & np.isfinite(arm_e_matrix_b).any(axis=0)
    for ch in range(EXPECTED_NEURONS):
        if valid[ch]:
            tau_star_a[ch] = lag_range[int(np.argmax(arm_e_matrix_a[:, ch]))]
            tau_star_b[ch] = lag_range[int(np.argmax(arm_e_matrix_b[:, ch]))]

    # Correlation between halves
    both_valid = (tau_star_a != -999) & (tau_star_b != -999)
    if both_valid.sum() > 5:
        a_std = np.std(tau_star_a[both_valid].astype(float))
        b_std = np.std(tau_star_b[both_valid].astype(float))
        if a_std > EPS and b_std > EPS:
            corr = float(np.corrcoef(tau_star_a[both_valid].astype(float), tau_star_b[both_valid].astype(float))[0, 1])
        else:
            corr = float("nan")
    else:
        corr = float("nan")

    return {
        "scheme": scheme,
        "n_blocks_total": n,
        "n_channels_both_valid": int(both_valid.sum()),
        "split_half_corr": corr,
        "tau_star_a": tau_star_a.tolist(),
        "tau_star_b": tau_star_b.tolist(),
    }


def run_sa_e2(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """SA-E2: per-channel lag with usable null, both block settings."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    # Section 4.1: round 1 used ALL blocks (iterating record.trials = all 15 trials).
    round1_setting = "all_blocks_of_recording"
    settings = {"all_blocks": build_lagged_blocks, "m4_support_only": build_lagged_blocks_m4_only}

    per_setting: dict[str, Any] = {}
    for setting_name, blocks_builder in settings.items():
        per_recording: dict[str, Any] = {}
        for name in source:
            record = records[name]
            # Pooled rank test
            rank_result = pooled_rank_test(record, plan.ridge_lambda, LAG_RANGE, NULL_OFFSETS_R2, blocks_builder)
            # Split-half stability (alternate primary, first_second sensitivity)
            stability_alt = split_half_stability(record, plan.ridge_lambda, LAG_RANGE, "alternate", blocks_builder)
            stability_fs = split_half_stability(record, plan.ridge_lambda, LAG_RANGE, "first_second", blocks_builder)
            # Null split-half stability (use a circular offset)
            shifted_v = circular_shift_velocity(record, NULL_OFFSETS_R2[0])
            null_stab_alt = _null_split_half_corr(record, plan.ridge_lambda, LAG_RANGE, "alternate", shifted_v, blocks_builder)
            null_stab_fs = _null_split_half_corr(record, plan.ridge_lambda, LAG_RANGE, "first_second", shifted_v, blocks_builder)

            # Emit raw tau* per channel (all-blocks setting)
            real_tau, real_ratio = arm_e_tau_star_per_channel(
                None, plan.ridge_lambda, LAG_RANGE, blocks_builder, record=record
            )
            per_recording[name] = {
                "pooled_rank_test": rank_result,
                "split_half_stability": {
                    "alternate": {"corr": stability_alt["split_half_corr"], "n_channels": stability_alt["n_channels_both_valid"]},
                    "first_second": {"corr": stability_fs["split_half_corr"], "n_channels": stability_fs["n_channels_both_valid"]},
                },
                "split_half_stability_null": {
                    "alternate": {"corr": null_stab_alt},
                    "first_second": {"corr": null_stab_fs},
                },
                "tau_star_per_channel": real_tau.tolist(),
                "ratio_per_channel": real_ratio.tolist(),
            }
        per_setting[setting_name] = {
            "per_recording": per_recording,
            "rank_summary": {
                name: per_recording[name]["pooled_rank_test"]["rank_distribution_quartiles"]
                for name in source
            },
        }

    return {
        "schema": "h1_lag_screen_sa_e2_v1",
        "module_status": MODULE_STATUS,
        "round1_arm_e_setting": round1_setting,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "null_offsets": list(NULL_OFFSETS_R2),
        "n_null_offsets": len(NULL_OFFSETS_R2),
        "settings": per_setting,
        "read_rule_reference": {
            "ranks_uniform": "No per-channel lag content. Arm E stops.",
            "ranks_deviate_and_stability_exceeds_null": "Per-channel lag content exists and is estimable at deployment budget.",
            "ranks_deviate_but_stability_not_exceeds_null": "Structure exists but not estimable at M=4. Budget-limited, not absent.",
        },
    }


def _null_split_half_corr(
    record: H1PilotRecord, ridge_lambda: float,
    lag_range: Sequence[int], scheme: str, shifted_v: np.ndarray,
    blocks_builder=build_lagged_blocks,
) -> float:
    """Compute split-half stability correlation under a null velocity shift."""

    tau_star_a = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    tau_star_b = np.full(EXPECTED_NEURONS, -999, dtype=np.int64)
    arm_e_matrix_a = np.full((len(lag_range), EXPECTED_NEURONS), np.nan)
    arm_e_matrix_b = np.full((len(lag_range), EXPECTED_NEURONS), np.nan)
    for i_tau, tau in enumerate(lag_range):
        blocks_tau = blocks_builder(record, tau, velocity_override=shifted_v)
        n_tau = blocks_tau.rates.shape[0]
        if scheme == "alternate":
            m_a = np.arange(n_tau) % 2 == 0
            m_b = ~m_a
        else:
            mid_tau = n_tau // 2
            m_a = np.arange(n_tau) < mid_tau
            m_b = ~m_a
        blocks_a = LaggedBlocks(tau, blocks_tau.rates[m_a], blocks_tau.velocity[m_a], n_tau, 0)
        blocks_b = LaggedBlocks(tau, blocks_tau.rates[m_b], blocks_tau.velocity[m_b], n_tau, 0)
        arm_e_matrix_a[i_tau] = arm_e_r2(blocks_a, ridge_lambda)
        arm_e_matrix_b[i_tau] = arm_e_r2(blocks_b, ridge_lambda)
    valid = np.isfinite(arm_e_matrix_a).any(axis=0) & np.isfinite(arm_e_matrix_b).any(axis=0)
    for ch in range(EXPECTED_NEURONS):
        if valid[ch]:
            tau_star_a[ch] = lag_range[int(np.argmax(arm_e_matrix_a[:, ch]))]
            tau_star_b[ch] = lag_range[int(np.argmax(arm_e_matrix_b[:, ch]))]
    both_valid = (tau_star_a != -999) & (tau_star_b != -999)
    if both_valid.sum() > 5:
        a_std = np.std(tau_star_a[both_valid].astype(float))
        b_std = np.std(tau_star_b[both_valid].astype(float))
        if a_std > EPS and b_std > EPS:
            return float(np.corrcoef(tau_star_a[both_valid].astype(float), tau_star_b[both_valid].astype(float))[0, 1])
    return float("nan")


# =========================================================================== #
# R3: detrend control for the population lag (round 3).
# =========================================================================== #
R3_SCHEMA = "h1_lag_screen_r3_detrend_v1"
LAG_RANGE_R3 = tuple(range(-10, 41))  # -10..+40 bins = -200..+800 ms
DETREND_DEGREE_PRIMARY = 3


def detrend_trial(
    rates: np.ndarray, velocity: np.ndarray, degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-trial polynomial detrend of block rates and velocity.

    Regresses each channel and each velocity dimension on a polynomial basis
    of the block index (0..n-1, normalised to [0,1]).  Returns the residuals.
    Never crosses a trial boundary.
    """

    n = rates.shape[0]
    if n < degree + 2:
        # Not enough blocks to detrend at this degree; return as-is.
        return rates.copy(), velocity.copy()
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    basis = np.column_stack([x ** d for d in range(degree + 1)])  # [n, degree+1]
    # Solve least-squares for each channel/dim, subtract the fit.
    # rates: [n, N], velocity: [n, 7]
    coeff_r, _, _, _ = np.linalg.lstsq(basis, rates, rcond=None)
    coeff_v, _, _, _ = np.linalg.lstsq(basis, velocity, rcond=None)
    return rates - basis @ coeff_r, velocity - basis @ coeff_v


def detrend_record(
    record: H1PilotRecord, degree: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detrend every trial's block rates and velocity.  Returns per-trial lists."""

    detrended_rates: list[np.ndarray] = []
    detrended_velocity: list[np.ndarray] = []
    for trial in record.trials:
        r_resid, v_resid = detrend_trial(
            trial.rates.astype(np.float64), trial.velocity.astype(np.float64), degree
        )
        detrended_rates.append(r_resid)
        detrended_velocity.append(v_resid)
    return detrended_rates, detrended_velocity


def build_detrended_record(
    record: H1PilotRecord, degree: int,
) -> H1PilotRecord:
    """Return a synthetic H1PilotRecord whose trial rates/velocity are detrended residuals.

    The detrend is applied inside each trial only.  The global neural/velocity
    arrays are reconstructed so that build_lagged_blocks works unchanged.
    """

    from dataclasses import replace as dc_replace
    detrended_rates, detrended_velocity = detrend_record(record, degree)
    new_trials = tuple(
        dc_replace(trial, rates=r, velocity=v)
        for trial, r, v in zip(record.trials, detrended_rates, detrended_velocity)
    )
    # Rebuild global neural/velocity by replacing trial bins.
    new_neural = record.neural.astype(np.float64).copy()
    new_velocity = record.velocity.astype(np.float64).copy()
    for trial, r_resid, v_resid in zip(record.trials, detrended_rates, detrended_velocity):
        # The trial's block_indices give the original bin positions.
        for i_block in range(trial.rates.shape[0]):
            bins = trial.block_indices[i_block]
            # Each block is 5 bins.  The residual replaces the per-bin rate
            # uniformly (the block rate was a sum / 0.1; distribute the residual
            # rate evenly across the 5 bins).
            new_neural[bins] = r_resid[i_block] * BLOCK_SECONDS / BLOCK_BINS
            new_velocity[bins] = v_resid[i_block]
    return dc_replace(
        record,
        neural=np.asarray(new_neural, dtype=np.float32),
        velocity=np.asarray(new_velocity, dtype=np.float32),
        trials=new_trials,
    )


def run_r3_detrend(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """R3: detrend control for the population lag.

    Runs the arm-P sweep over -10..+40 on both the raw data and the detrended
    data (degree 3 primary), with the LOO declared-lag test on both.  Degrees 1
    and 5 are reported as sensitivity.
    """

    source = tuple(H1_M4_FOLD0_SOURCE)

    # --- Degree 3 (primary): raw vs detrended side-by-side ---
    # Build detrended records (degree 3)
    detrended_records_d3 = {name: build_detrended_record(records[name], 3) for name in source}

    # Raw curves
    raw_curves = {name: arm_p_curve(records[name], plan, LAG_RANGE_R3) for name in source}
    det_curves = {name: arm_p_curve(detrended_records_d3[name], plan, LAG_RANGE_R3) for name in source}

    # Raw LOO
    raw_loo = loo_declared_lag_test(records, plan, LAG_RANGE_R3)
    # Detrended LOO
    det_loo = loo_declared_lag_test(detrended_records_d3, plan, LAG_RANGE_R3)

    # Peak analysis
    def _peak_info(curves_by_rec: dict[str, dict[int, float]]) -> dict[str, Any]:
        info = {}
        for name in source:
            c = curves_by_rec[name]
            finite = {t: c[t] for t in LAG_RANGE_R3 if np.isfinite(c[t])}
            if not finite:
                info[name] = {"tau_star": None, "has_peak": False, "r2_at_zero": None, "rises_at_40": None}
                continue
            tau_star = max(finite, key=lambda t: finite[t])
            # A peak exists if R2 decreases after tau_star (at least one step).
            if tau_star < 40:
                has_peak = finite.get(tau_star + 1, -np.inf) < finite[tau_star]
            else:
                has_peak = False
            rises_at_40 = finite.get(40, -np.inf) > finite.get(39, -np.inf)
            info[name] = {
                "tau_star": int(tau_star),
                "has_peak": bool(has_peak),
                "r2_at_zero": float(finite.get(0, float("nan"))),
                "r2_at_tau_star": float(finite[tau_star]),
                "rises_at_40": bool(rises_at_40),
            }
        return info

    raw_peaks = _peak_info(raw_curves)
    det_peaks = _peak_info(det_curves)

    # --- Degree sensitivity (1 and 5): peak locations only ---
    sensitivity: dict[str, Any] = {}
    for deg in (1, 5):
        detrended_records_deg = {name: build_detrended_record(records[name], deg) for name in source}
        curves_deg = {name: arm_p_curve(detrended_records_deg[name], plan, LAG_RANGE_R3) for name in source}
        peaks_deg = _peak_info(curves_deg)
        loo_deg = loo_declared_lag_test(detrended_records_deg, plan, LAG_RANGE_R3)
        sensitivity[f"degree_{deg}"] = {
            "peaks": peaks_deg,
            "loo_sign_test": loo_deg["sign_test"],
            "loo_declared_lags": loo_deg["declared_lags"],
            "loo_declared_lag_mean": loo_deg["declared_lag_mean"],
            "r2_curves": {name: {str(t): curves_deg[name][t] for t in LAG_RANGE_R3} for name in source},
        }

    # --- Null on detrended data (circular shift + detrend) ---
    # The brief requires: apply identical detrend to real and null.  The circular
    # shift null shifts velocity within each trial; we detrend the shifted
    # velocity with the same polynomial degree before building lagged blocks.
    # Report the null curve for a few offsets to show the null tau* distribution.
    null_offsets_r3 = NULL_OFFSETS[:4]  # reuse 4 of the 8 offsets for tractable runtime
    det_null_curves_by_offset: dict[int, dict[str, dict[int, float]]] = {}
    for offset in null_offsets_r3:
        curves_for_offset = {}
        for name in source:
            shifted_v = circular_shift_velocity(records[name], offset)
            # Build a record with shifted velocity, then detrend it
            from dataclasses import replace as dc_replace
            shifted_record = dc_replace(records[name], velocity=shifted_v.astype(np.float32))
            det_shifted = build_detrended_record(shifted_record, DETREND_DEGREE_PRIMARY)
            curves_for_offset[name] = arm_p_curve(det_shifted, plan, LAG_RANGE_R3)
        det_null_curves_by_offset[offset] = curves_for_offset

    # Null peak info per offset (pooled across recordings)
    det_null_peaks: dict[str, Any] = {}
    for offset in null_offsets_r3:
        pooled_curve: dict[int, float] = {}
        for tau in LAG_RANGE_R3:
            vals = [det_null_curves_by_offset[offset][name][tau] for name in source if np.isfinite(det_null_curves_by_offset[offset][name][tau])]
            pooled_curve[tau] = float(np.mean(vals)) if vals else float("nan")
        finite = {t: pooled_curve[t] for t in LAG_RANGE_R3 if np.isfinite(pooled_curve[t])}
        tau_star = max(finite, key=lambda t: finite[t]) if finite else None
        det_null_peaks[str(offset)] = {"tau_star": int(tau_star) if tau_star is not None else None, "r2_at_tau_star": float(finite.get(tau_star, float("nan"))) if tau_star else None}

    return {
        "schema": R3_SCHEMA,
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "lag_range_r3": list(LAG_RANGE_R3),
        "detrend_degree_primary": DETREND_DEGREE_PRIMARY,
        "detrend_scope": "within-trial polynomial detrend; never crosses trial boundary",
        "identical_detrend_applied_to_real_and_null": True,
        "detrend_method_real": f"degree-{DETREND_DEGREE_PRIMARY} polynomial regression on block index, residual kept, per channel and per velocity dimension, within each trial",
        "detrend_method_null": f"identical degree-{DETREND_DEGREE_PRIMARY} detrend applied after the circular velocity shift, before the lag sweep",
        "raw_arm": {
            "r2_curves": {name: {str(t): raw_curves[name][t] for t in LAG_RANGE_R3} for name in source},
            "peaks": raw_peaks,
            "loo_declared_lag_test": raw_loo,
        },
        "detrended_arm": {
            "r2_curves": {name: {str(t): det_curves[name][t] for t in LAG_RANGE_R3} for name in source},
            "peaks": det_peaks,
            "loo_declared_lag_test": det_loo,
        },
        "sensitivity": sensitivity,
        "detrended_null": {
            "null_offsets": list(null_offsets_r3),
            "pooled_peaks": det_null_peaks,
        },
        "read_rule_reference": {
            "detrended_peak_near_260_360": "The lead is real at that latency. Report it as a candidate global lag.",
            "detrended_peak_moves_to_100_150": "The raw peak was drift-contaminated; the real lead is canonical. Report the detrended latency as the candidate.",
            "detrended_flattens_or_sign_breaks": "Arm P stops, and the lag route is closed.",
            "detrended_rises_at_40": "Open boundary. Do not widen again. Do not call it a candidate.",
        },
    }


# =========================================================================== #
# R4: artifact controls C2, C1, C3 (round 4).
# =========================================================================== #
R4_SCHEMA = "h1_lag_screen_r4_artifact_controls_v1"


def _trial_valid_mask(trial, tau: int, record: H1PilotRecord) -> np.ndarray:
    """Return a boolean mask of valid blocks for one trial at lag tau."""

    shifted = trial.block_indices + tau
    valid = np.ones(trial.rates.shape[0], dtype=bool)
    for col in range(BLOCK_BINS):
        idx = shifted[:, col]
        in_bounds = (idx >= 0) & (idx < record.trial_num.shape[0])
        valid &= in_bounds
    for i in np.flatnonzero(valid):
        idx = shifted[i]
        if not np.all(record.trial_num[idx] == trial.trial_number):
            valid[i] = False
    return valid


def compute_fixed_block_mask(record: H1PilotRecord, lag_range: Sequence[int]) -> list[np.ndarray]:
    """For each trial, return a boolean mask of blocks valid at ALL tested lags."""

    per_trial_masks: list[np.ndarray] = []
    for trial in record.trials:
        combined = np.ones(trial.rates.shape[0], dtype=bool)
        for tau in lag_range:
            combined &= _trial_valid_mask(trial, tau, record)
        per_trial_masks.append(combined)
    return per_trial_masks


def build_fixed_lagged_blocks(
    record: H1PilotRecord, tau: int,
    fixed_masks: list[np.ndarray] | None = None,
    *, velocity_override: np.ndarray | None = None,
) -> LaggedBlocks:
    """Build lagged blocks using a FIXED block set (valid at all lags).

    The fixed mask is computed once per recording from the lag range.  At each
    tau, the same blocks are used (the mask is from the intersection, not from
    the per-tau validity check).  This removes mechanism A.
    """

    if fixed_masks is None:
        fixed_masks = compute_fixed_block_mask(record, LAG_RANGE_R3)
    velocity = velocity_override if velocity_override is not None else record.velocity
    rates_list: list[np.ndarray] = []
    vel_list: list[np.ndarray] = []
    n_total = 0
    n_dropped = 0
    for trial, mask in zip(record.trials, fixed_masks):
        n_total += trial.rates.shape[0]
        n_valid = int(mask.sum())
        n_dropped += trial.rates.shape[0] - n_valid
        if n_valid > 0:
            shifted = trial.block_indices[mask] + tau
            rates_list.append(trial.rates[mask])
            vel_list.append(np.array([velocity[s].mean(axis=0) for s in shifted]))
    if not rates_list:
        return LaggedBlocks(tau, np.empty((0, EXPECTED_NEURONS)), np.empty((0, VELOCITY_DIM)), n_total, n_dropped)
    return LaggedBlocks(
        tau,
        np.concatenate(rates_list, axis=0).astype(np.float64),
        np.concatenate(vel_list, axis=0).astype(np.float64),
        n_total,
        n_dropped,
    )


def arm_p_r2_cv(
    blocks: LaggedBlocks, plan: LagScreenPlan, ridge_lambda: float, k: int = 5,
) -> tuple[float, float]:
    """Return (in_sample_r2, out_of_sample_r2) via k contiguous folds."""

    n = blocks.rates.shape[0]
    if n < plan.q + 2:
        return float("nan"), float("nan")
    z = plan.project(blocks.rates)
    design = np.column_stack((np.ones(n), z))

    # In-sample
    beta = _ridge_solve(design, blocks.velocity, ridge_lambda)
    pred = design @ beta
    rss_in = float(np.square(blocks.velocity - pred).sum())
    tss_in = float(np.square(blocks.velocity - blocks.velocity.mean(axis=0)[None, :]).sum())
    r2_in = float(1.0 - rss_in / tss_in) if tss_in > EPS else float("nan")

    # Out-of-sample: k contiguous folds
    fold_size = n // k
    if fold_size < plan.q + 2:
        return r2_in, float("nan")
    rss_out = 0.0
    tss_out = 0.0
    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < k - 1 else n
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
        if train_idx.size < plan.q + 2:
            continue
        design_train = design[train_idx]
        vel_train = blocks.velocity[train_idx]
        beta_fold = _ridge_solve(design_train, vel_train, ridge_lambda)
        pred_fold = design[test_idx] @ beta_fold
        vel_test = blocks.velocity[test_idx]
        vel_mean = vel_train.mean(axis=0)
        rss_out += float(np.square(vel_test - pred_fold).sum())
        tss_out += float(np.square(vel_test - vel_mean[None, :]).sum())
    r2_out = float(1.0 - rss_out / tss_out) if tss_out > EPS else float("nan")
    return r2_in, r2_out


def _zero_compute_diagnostic(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """Correlate R2(tau) with n_blocks(tau) from the variable-set curves."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    result = {}
    for name in source:
        r2_vals = []
        n_blocks_vals = []
        for tau in LAG_RANGE_R3:
            blocks = build_lagged_blocks(records[name], tau)
            r2 = arm_p_r2(blocks, plan, plan.ridge_lambda)
            r2_vals.append(r2)
            n_blocks_vals.append(blocks.rates.shape[0])
        r2_arr = np.array(r2_vals)
        n_arr = np.array(n_blocks_vals, dtype=np.float64)
        finite = np.isfinite(r2_arr)
        if finite.sum() > 3 and np.std(n_arr[finite]) > EPS and np.std(r2_arr[finite]) > EPS:
            corr = float(np.corrcoef(n_arr[finite], r2_arr[finite])[0, 1])
        else:
            corr = float("nan")
        result[name] = {
            "corr_n_blocks_vs_r2": corr,
            "n_blocks_range": [int(n_arr.min()), int(n_arr.max())],
            "r2_range": [float(np.nanmin(r2_arr)), float(np.nanmax(r2_arr))],
        }
    return result


def _arm_p_curve_fixed(
    record: H1PilotRecord, plan: LagScreenPlan, lag_range: Sequence[int],
    fixed_masks: list[np.ndarray], *,
    velocity_override: np.ndarray | None = None,
) -> dict[int, float]:
    """R2(tau) on the fixed block set."""

    curve: dict[int, float] = {}
    for tau in lag_range:
        blocks = build_fixed_lagged_blocks(
            record, tau, fixed_masks, velocity_override=velocity_override
        )
        curve[tau] = arm_p_r2(blocks, plan, plan.ridge_lambda)
    return curve


def _loo_declared_lag_test_fixed(
    records: Mapping[str, H1PilotRecord], plan: LagScreenPlan,
    lag_range: Sequence[int],
    fixed_masks_by_name: Mapping[str, list[np.ndarray]],
    velocity_overrides_by_name: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """LOO declared-lag test on the fixed block set."""

    source = tuple(sorted(records.keys()))
    curves = {}
    for name in source:
        v_override = velocity_overrides_by_name.get(name) if velocity_overrides_by_name else None
        curves[name] = _arm_p_curve_fixed(
            records[name], plan, lag_range, fixed_masks_by_name[name],
            velocity_override=v_override,
        )

    results: list[dict[str, Any]] = []
    for held in source:
        others = [n for n in source if n != held]
        pooled_curve = {}
        for tau in lag_range:
            vals = [curves[n][tau] for n in others if np.isfinite(curves[n][tau])]
            pooled_curve[tau] = float(np.mean(vals)) if vals else float("nan")
        finite_taus = [t for t in lag_range if np.isfinite(pooled_curve[t])]
        if not finite_taus:
            raise LagScreenError(f"pooled curve has no finite values when holding {held}")
        tau_r = max(finite_taus, key=lambda t: pooled_curve[t])
        r2_at_tau_r = curves[held][tau_r]
        r2_at_zero = curves[held][0]
        delta = r2_at_tau_r - r2_at_zero if np.isfinite(r2_at_tau_r) and np.isfinite(r2_at_zero) else float("nan")
        results.append({
            "held_recording": held,
            "declared_lag_tau_r": tau_r,
            "r2_at_tau_r": r2_at_tau_r,
            "r2_at_zero": r2_at_zero,
            "delta": delta,
        })

    deltas = [r["delta"] for r in results if np.isfinite(r["delta"])]
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    n_nonzero = n_positive + n_negative
    if n_nonzero > 0:
        from scipy.stats import binomtest
        p_val = float(binomtest(n_positive, n_nonzero, 0.5, alternative="two-sided").pvalue)
    else:
        p_val = float("nan")
    declared_lags = [r["declared_lag_tau_r"] for r in results]
    return {
        "per_recording": results,
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "sign_test_p_value": p_val,
        "declared_lags": declared_lags,
        "declared_lag_mean": float(np.mean(declared_lags)),
        "declared_lag_std": float(np.std(declared_lags)),
    }


def run_r4_controls(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> dict[str, Any]:
    """R4: three artifact controls for the population lag."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    # --- Step 0 answer ---
    step0 = {
        "r2_is_in_sample": True,
        "explanation": "arm_p_r2 fits beta on the same blocks it scores (design -> beta -> pred = design @ beta). No train/test split. C3 is mandatory.",
        "plan_value_out_of_sample": 0.0286,
        "observed_r2_approx": 0.080,
        "chance_in_sample_floor": float(plan.q + 1) / 627.0,
    }

    # --- C2.1: zero-compute diagnostic ---
    c2_diagnostic = _zero_compute_diagnostic(records, plan)

    # --- C2.2: fixed block set ---
    fixed_masks_by_name: dict[str, list[np.ndarray]] = {}
    fixed_block_info: dict[str, Any] = {}
    for name in source:
        masks = compute_fixed_block_mask(records[name], LAG_RANGE_R3)
        fixed_masks_by_name[name] = masks
        total = sum(t.rates.shape[0] for t in records[name].trials)
        kept = sum(int(m.sum()) for m in masks)
        fixed_block_info[name] = {"total_blocks": total, "fixed_blocks": kept, "fraction_kept": kept / total if total > 0 else 0.0}

    # Raw arm: variable set (from R3) vs fixed set
    raw_variable_curves = {name: arm_p_curve(records[name], plan, LAG_RANGE_R3) for name in source}
    raw_fixed_curves = {name: _arm_p_curve_fixed(records[name], plan, LAG_RANGE_R3, fixed_masks_by_name[name]) for name in source}

    # Detrended arm (degree 3): fixed set
    from dataclasses import replace as dc_replace
    detrended_records = {name: build_detrended_record(records[name], DETREND_DEGREE_PRIMARY) for name in source}
    det_fixed_masks_by_name: dict[str, list[np.ndarray]] = {}
    for name in source:
        det_fixed_masks_by_name[name] = compute_fixed_block_mask(detrended_records[name], LAG_RANGE_R3)
    det_fixed_curves = {name: _arm_p_curve_fixed(detrended_records[name], plan, LAG_RANGE_R3, det_fixed_masks_by_name[name]) for name in source}

    # Null on fixed set (4 offsets, real data detrended + circular shift)
    null_offsets_c2 = NULL_OFFSETS[:4]
    null_fixed_curves_by_offset: dict[int, dict[str, dict[int, float]]] = {}
    for offset in null_offsets_c2:
        curves_for_offset = {}
        for name in source:
            shifted_v = circular_shift_velocity(records[name], offset)
            shifted_rec = dc_replace(records[name], velocity=shifted_v.astype(np.float32))
            det_shifted = build_detrended_record(shifted_rec, DETREND_DEGREE_PRIMARY)
            det_masks = compute_fixed_block_mask(det_shifted, LAG_RANGE_R3)
            curves_for_offset[name] = _arm_p_curve_fixed(det_shifted, plan, LAG_RANGE_R3, det_masks)
        null_fixed_curves_by_offset[offset] = curves_for_offset

    # Peak analysis helper
    def _peak(curves_by_rec):
        info = {}
        for name in source:
            c = curves_by_rec[name]
            finite = {t: c[t] for t in LAG_RANGE_R3 if np.isfinite(c[t])}
            if not finite:
                info[name] = {"tau_star": None, "r2_at_zero": None}
                continue
            tau_star = max(finite, key=lambda t: finite[t])
            info[name] = {"tau_star": int(tau_star), "r2_at_zero": float(finite.get(0, float("nan"))), "r2_at_tau_star": float(finite[tau_star])}
        return info

    # Null pooled peaks per offset
    null_pooled_peaks: dict[str, Any] = {}
    for offset in null_offsets_c2:
        pooled = {}
        for tau in LAG_RANGE_R3:
            vals = [null_fixed_curves_by_offset[offset][name][tau] for name in source if np.isfinite(null_fixed_curves_by_offset[offset][name][tau])]
            pooled[tau] = float(np.mean(vals)) if vals else float("nan")
        finite = {t: pooled[t] for t in LAG_RANGE_R3 if np.isfinite(pooled[t])}
        tau_star = max(finite, key=lambda t: finite[t]) if finite else None
        null_pooled_peaks[str(offset)] = {"tau_star": int(tau_star) if tau_star is not None else None, "r2_at_tau_star": float(finite.get(tau_star, float("nan"))) if tau_star else None}

    # LOO on fixed set: raw and detrended
    raw_fixed_loo = _loo_declared_lag_test_fixed(records, plan, LAG_RANGE_R3, fixed_masks_by_name)
    det_fixed_loo = _loo_declared_lag_test_fixed(detrended_records, plan, LAG_RANGE_R3, det_fixed_masks_by_name)

    # --- C1: null LOO distribution ---
    # For each null offset, run the full LOO on the fixed-set detrended null curves
    c1_null_results: list[dict[str, Any]] = []
    for offset in null_offsets_c2:
        velocity_overrides = {}
        for name in source:
            shifted_v = circular_shift_velocity(records[name], offset)
            shifted_rec = dc_replace(records[name], velocity=shifted_v.astype(np.float32))
            det_shifted = build_detrended_record(shifted_rec, DETREND_DEGREE_PRIMARY)
            velocity_overrides[name] = det_shifted.velocity
        # For the null LOO, we need the detrended shifted records
        det_shifted_records = {}
        det_shifted_masks = {}
        for name in source:
            shifted_v = circular_shift_velocity(records[name], offset)
            shifted_rec = dc_replace(records[name], velocity=shifted_v.astype(np.float32))
            det_shifted_records[name] = build_detrended_record(shifted_rec, DETREND_DEGREE_PRIMARY)
            det_shifted_masks[name] = compute_fixed_block_mask(det_shifted_records[name], LAG_RANGE_R3)
        null_loo = _loo_declared_lag_test_fixed(
            det_shifted_records, plan, LAG_RANGE_R3, det_shifted_masks
        )
        c1_null_results.append({
            "offset": offset,
            "mean_delta": null_loo["mean_delta"],
            "n_positive": null_loo["n_positive"],
            "declared_lag_mean": null_loo["declared_lag_mean"],
        })
    # Where does the real detrended delta sit?
    real_mean_delta = det_fixed_loo["mean_delta"]
    real_n_positive = det_fixed_loo["n_positive"]
    null_mean_deltas = [r["mean_delta"] for r in c1_null_results if np.isfinite(r["mean_delta"])]
    null_n_positives = [r["n_positive"] for r in c1_null_results]
    c1_percentile_delta = float(np.mean(np.array(null_mean_deltas) < real_mean_delta)) if null_mean_deltas else float("nan")
    c1_percentile_npos = float(np.mean(np.array(null_n_positives) < real_n_positive)) if null_n_positives else float("nan")

    # --- C3: out-of-sample R2 via 5 contiguous folds ---
    c3_results: dict[str, Any] = {}
    for name in source:
        # Fixed-set, detrended, real
        for tau in [0, 8, 17, 18, 25, 34]:
            blocks = build_fixed_lagged_blocks(detrended_records[name], tau, det_fixed_masks_by_name[name])
            r2_in, r2_out = arm_p_r2_cv(blocks, plan, plan.ridge_lambda, k=5)
            c3_results.setdefault(name, {})[str(tau)] = {"r2_in": r2_in, "r2_out": r2_out}
        # Null at tau=0 and tau=17
        shifted_v = circular_shift_velocity(records[name], NULL_OFFSETS[0])
        shifted_rec = dc_replace(records[name], velocity=shifted_v.astype(np.float32))
        det_shifted = build_detrended_record(shifted_rec, DETREND_DEGREE_PRIMARY)
        det_shifted_masks = compute_fixed_block_mask(det_shifted, LAG_RANGE_R3)
        for tau in [0, 17]:
            blocks = build_fixed_lagged_blocks(det_shifted, tau, det_shifted_masks)
            r2_in, r2_out = arm_p_r2_cv(blocks, plan, plan.ridge_lambda, k=5)
            c3_results.setdefault(name, {}).setdefault("null", {})[str(tau)] = {"r2_in": r2_in, "r2_out": r2_out}

    return {
        "schema": R4_SCHEMA,
        "module_status": MODULE_STATUS,
        "step0": step0,
        "c2_zero_compute_diagnostic": c2_diagnostic,
        "c2_fixed_block_info": fixed_block_info,
        "c2_raw_variable_curves": {name: {str(t): raw_variable_curves[name][t] for t in LAG_RANGE_R3} for name in source},
        "c2_raw_fixed_curves": {name: {str(t): raw_fixed_curves[name][t] for t in LAG_RANGE_R3} for name in source},
        "c2_detrended_fixed_curves": {name: {str(t): det_fixed_curves[name][t] for t in LAG_RANGE_R3} for name in source},
        "c2_raw_fixed_peaks": _peak(raw_fixed_curves),
        "c2_detrended_fixed_peaks": _peak(det_fixed_curves),
        "c2_null_pooled_peaks": null_pooled_peaks,
        "c2_raw_fixed_loo": raw_fixed_loo,
        "c2_detrended_fixed_loo": det_fixed_loo,
        "c1_null_loo_distribution": {
            "per_offset": c1_null_results,
            "null_mean_deltas": null_mean_deltas,
            "null_n_positives": null_n_positives,
            "real_mean_delta": real_mean_delta,
            "real_n_positive": real_n_positive,
            "real_delta_percentile_in_null": c1_percentile_delta,
            "real_n_positive_percentile_in_null": c1_percentile_npos,
        },
        "c3_out_of_sample": c3_results,
        "read_rule_reference": {
            "c2_peak_disappears": "Mechanism A confirmed. Arm P stops and the lag route closes.",
            "c2_peak_survives_separates": "Continue to C1 and C3.",
            "c2_peak_survives_equal": "Continue but evidence still does not separate lead from null.",
            "c1_real_inside_null_bulk": "No lag-specific content. Arm P stops.",
            "c1_real_above_null": "The lead separates from the null. Report it as a candidate.",
        },
    }
