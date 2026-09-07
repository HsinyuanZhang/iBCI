#!/usr/bin/env python3
"""Held-in-only factorial stability audit for movement K4 and target T4.

This CPU program is diagnostic only. It neither trains nor evaluates a decoder,
refuses held-out files, and requires a hash-bound receipt before --run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
SCE = ROOT / "streaming_calibration_exp"
sys.path[:0] = [str(SUA), str(SCE)]

from falcon_challenge.config import FalconTask  # noqa: E402
from falcon_challenge.dataloaders import load_nwb  # noqa: E402
from mc_maze.general_carrier import CarrierFit, fit_encoding, mean_squared_error, predict_encoding  # noqa: E402
from src.data.falcon_t4_features import calibration_target_angles, t4_from_trial_sums  # noqa: E402

TASK = "m2"
M_GRID = (8, 10, 12, 16, 20, 24, 28, 30, 32, 33)
LEAD_GRID_BINS = tuple(range(-5, 11))  # raw 20-ms bins: -100 through +200 ms
RAW_BIN_MS, BLOCK_BINS, ACTIVE_EPSILON, T4_EXPOSURE_BINS = 20, 5, 1.0e-3, 100
N_RANDOM_BALANCE_REPETITIONS = 50
RANDOM_BALANCE_SEEDS = tuple(range(N_RANDOM_BALANCE_REPETITIONS))
BALANCE_DIRECTION_BINS, RIDGE_ALPHA_STANDARDIZED = 8, 1.0
PER_CHANNEL_MIN_LOTO_FOLDS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def heldin_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("**/*held-in-calib*.nwb"))
    if len(paths) != 7 or any("held-out" in str(path).lower() for path in paths):
        raise ValueError("requires exactly seven non-held-out M2 held-in-calib files")
    return paths


def session_name(path: Path) -> str:
    return path.name.split("_", 1)[1].split(".nwb", 1)[0]


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def k4_blocks(neural: np.ndarray, velocity: np.ndarray, trial_change: np.ndarray, *, lead: int, max_trials: int) -> dict[str, np.ndarray]:
    """K4 raw 100-ms blocks with lead and every span retained inside its trial."""
    neural, velocity = np.asarray(neural, float), np.asarray(velocity, float)
    starts = np.flatnonzero(np.asarray(trial_change, bool))
    if neural.ndim != 2 or velocity.ndim != 2 or velocity.shape[1] != 2 or neural.shape[0] != velocity.shape[0]:
        raise ValueError("expected matched [time,channel] neural and [time,2] M2 velocity")
    if len(starts) < max_trials:
        raise ValueError(f"requires {max_trials} calibration trials")
    ends, active = np.r_[starts[1:], len(trial_change)], ~np.all(np.abs(velocity) < ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []; ys: list[np.ndarray] = []; trials: list[int] = []; block_ids: list[int] = []
    block_id = 0
    for trial, (start, end) in enumerate(zip(starts[:max_trials], ends[:max_trials])):
        lo, hi = int(start) + max(0, -lead), int(end) - BLOCK_BINS - max(0, lead)
        for left in range(lo, hi + 1, BLOCK_BINS):
            right, yl, yr = left + BLOCK_BINS, left + lead, left + lead + BLOCK_BINS
            if not (active[left:right].all() and active[yl:yr].all()):
                continue
            rates.append(neural[left:right].sum(0) / (BLOCK_BINS * RAW_BIN_MS / 1000.0))
            ys.append(velocity[yl:yr].mean(0)); trials.append(trial); block_ids.append(block_id); block_id += 1
    if len(rates) < 10:
        raise ValueError("fewer than ten active K4 blocks")
    return {"rate": np.asarray(rates), "velocity": np.asarray(ys), "trial": np.asarray(trials), "block": np.asarray(block_ids)}


def t4_trial_arrays(neural: np.ndarray, trial_change: np.ndarray, angles: np.ndarray, *, max_trials: int) -> dict[str, np.ndarray]:
    """The frozen T4 first-100-bin raw-count exposure, no partial exposure."""
    starts = np.flatnonzero(np.asarray(trial_change, bool))
    if len(starts) < max_trials or len(angles) != len(starts):
        raise ValueError("T4 trial/angle alignment failure")
    ends = np.r_[starts[1:], len(trial_change)]
    sums, lengths = [], []
    for start, end in zip(starts[:max_trials], ends[:max_trials]):
        valid = min(int(end - start), T4_EXPOSURE_BINS)
        if valid < 1: raise ValueError("empty T4 trial")
        sums.append(neural[start:start + valid].sum(0)); lengths.append(valid)
    return {"sums": np.asarray(sums), "lengths": np.asarray(lengths), "angles": np.asarray(angles[:max_trials])}


def k4_split_masks(blocks: dict[str, np.ndarray], m: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    trial, block, prefix = blocks["trial"], blocks["block"], blocks["trial"] < m
    midpoint = m // 2
    return {
        "chronological_trial_half": (prefix & (trial < midpoint), prefix & (trial >= midpoint)),
        "odd_even_trial": (prefix & ((trial % 2) == 0), prefix & ((trial % 2) == 1)),
        "odd_even_block_optimistic": (prefix & ((block % 2) == 0), prefix & ((block % 2) == 1)),
    }


def t4_split_masks(m: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    idx, midpoint = np.arange(m), m // 2
    return {"chronological_trial_half": (idx < midpoint, idx >= midpoint), "odd_even_trial": ((idx % 2) == 0, (idx % 2) == 1)}


def design_audit(velocity: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(velocity)), velocity])
    rank = int(np.linalg.matrix_rank(design))
    eig = np.linalg.eigvalsh(np.cov(velocity, rowvar=False))[::-1] if len(velocity) > 1 else np.array([np.nan, np.nan])
    return {
        "active_blocks": int(len(velocity)), "design_rank": rank,
        "design_condition": float(np.linalg.cond(design)) if rank == 3 else float("inf"),
        "velocity_covariance_eigenvalues": [float(x) for x in eig],
        "velocity_eigen_ratio_min_over_max": float(eig[-1] / max(eig[0], 1e-12)),
    }


def fit_k4(blocks: dict[str, np.ndarray], mask: np.ndarray, *, ridge: bool = False) -> tuple[CarrierFit, dict[str, Any]]:
    if int(mask.sum()) < 10: raise ValueError("fewer than ten active K4 blocks")
    audit = design_audit(blocks["velocity"][mask])
    if audit["design_rank"] != 3: raise ValueError("rank-deficient K4 design")
    if not ridge:
        return fit_encoding(blocks["rate"], blocks["velocity"], mask, blocks["trial"], lag_bins=0, alpha=0.0), audit
    x, y = blocks["velocity"][mask], blocks["rate"][mask]
    mean, std = x.mean(0), x.std(0); std[std <= 1e-12] = 1.0
    z = (x - mean) / std
    coef = np.linalg.solve(z.T @ z + RIDGE_ALPHA_STANDARDIZED * np.eye(2), z.T @ (y - y.mean(0)))
    weights = (coef / std[:, None]).T
    return CarrierFit(weights=weights, intercept=y.mean(0) - mean @ weights.T, alpha=RIDGE_ALPHA_STANDARDIZED, lag_bins=0), audit


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first).ravel(), np.asarray(second).ravel()
    return float(np.corrcoef(first, second)[0, 1]) if np.std(first) > 1e-12 and np.std(second) > 1e-12 else float("nan")


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(np.ravel(first), np.ravel(second)) / denom) if denom > 1e-12 else float("nan")


def k4_component_metrics(first: CarrierFit, second: CarrierFit, normalizer: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Component reliability; normalizer is a fixed other-six full-prefix LOSO statistic."""
    w1, w2, b1, b2 = first.weights, second.weights, first.intercept, second.intercept
    n1, n2 = np.linalg.norm(w1, axis=1), np.linalg.norm(w2, axis=1)
    mean, std = normalizer
    z1 = np.column_stack([(n1 - mean[0]) / std[0], (b1 - mean[1]) / std[1]])
    z2 = np.column_stack([(n2 - mean[0]) / std[0], (b2 - mean[1]) / std[1]])
    row_denom = n1 * n2
    row_cos = np.sum(w1 * w2, axis=1) / np.maximum(row_denom, 1e-12)
    return {
        "wx_pearson": pearson(w1[:, 0], w2[:, 0]), "wy_pearson": pearson(w1[:, 1], w2[:, 1]),
        "w_flattened_pearson": pearson(w1, w2), "w_flattened_cosine": cosine(w1, w2),
        "w_per_channel_cosine_median": float(np.median(row_cos)),
        "w_modulation_weighted_cosine": float(np.sum(w1 * w2) / np.maximum(np.sum(row_denom), 1e-12)),
        "w_norm_pearson": pearson(n1, n2), "w_norm_cosine": cosine(n1, n2),
        "b_pearson": pearson(b1, b2), "b_cosine": cosine(b1, b2),
        "loso_standardized_norm_b_flattened_pearson": pearson(z1, z2),
        "loso_standardized_norm_b_flattened_cosine": cosine(z1, z2),
    }


def t4_fit(arrays: dict[str, np.ndarray], mask: np.ndarray, source: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    # The cache is frozen at max(M)=33, while every split mask is a first-M
    # mask. Slice the chronological prefix before boolean indexing; this is
    # exactly the original T4 estimator on that support, not a partial-exposure
    # or reindexed trial construction.
    m = len(mask)
    sums, lengths, angles = arrays["sums"][:m], arrays["lengths"][:m], arrays["angles"][:m]
    angles = angles[mask]
    directional = np.isfinite(angles)
    coverage = {"n_trials": int(mask.sum()), "n_directional_trials": int(directional.sum()), "direction_design_rank": None, "undefined_reason": None}
    if int(directional.sum()) < 3:
        coverage["undefined_reason"] = "fewer_than_three_directional_trials"; return None, coverage
    design = np.column_stack([np.ones(int(directional.sum())), np.cos(angles[directional]), np.sin(angles[directional])])
    coverage["direction_design_rank"] = int(np.linalg.matrix_rank(design))
    if coverage["direction_design_rank"] != 3:
        coverage["undefined_reason"] = "direction_design_rank_not_3"; return None, coverage
    try:
        return t4_from_trial_sums(sums[mask], lengths[mask], angles, source=source), coverage
    except ValueError as exc:
        coverage["undefined_reason"] = str(exc); return None, coverage


def t4_component_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    n1, n2 = first[:, 2], second[:, 2]
    row_denom = n1 * n2
    row_cos = np.sum(first[:, :2] * second[:, :2], axis=1) / np.maximum(row_denom, 1e-12)
    return {
        "a_pearson": pearson(first[:, 0], second[:, 0]), "c_pearson": pearson(first[:, 1], second[:, 1]),
        "ac_flattened_pearson": pearson(first[:, :2], second[:, :2]), "ac_flattened_cosine": cosine(first[:, :2], second[:, :2]),
        "ac_per_channel_cosine_median": float(np.median(row_cos)),
        "ac_modulation_weighted_cosine": float(np.sum(first[:, :2] * second[:, :2]) / np.maximum(np.sum(row_denom), 1e-12)),
        "m_pearson": pearson(first[:, 2], second[:, 2]), "m_cosine": cosine(first[:, 2], second[:, 2]),
        "b_pearson": pearson(first[:, 3], second[:, 3]), "b_cosine": cosine(first[:, 3], second[:, 3]),
    }


def chrono_future_ratio(blocks: dict[str, np.ndarray], m: int) -> float:
    first, second = k4_split_masks(blocks, m)["chronological_trial_half"]
    fit, _ = fit_k4(blocks, first)
    indices = np.flatnonzero(second)
    predicted = predict_encoding(fit, blocks["velocity"], indices, indices)
    mse = mean_squared_error(blocks["rate"][second], predicted)
    base = mean_squared_error(blocks["rate"][second], np.broadcast_to(blocks["rate"][first].mean(0), blocks["rate"][second].shape))
    return mse / base


def select_lag_nested(cache: dict[tuple[str, int], dict[str, np.ndarray]], target: str, m: int) -> tuple[int | None, dict[str, Any]]:
    sessions = sorted({session for session, _lead in cache})
    curve: dict[str, Any] = {}
    for lead in LEAD_GRID_BINS:
        values = []; records = []
        for session in sessions:
            if session == target: continue
            try:
                ratio = chrono_future_ratio(cache[(session, lead)], m)
                values.append(ratio); records.append({"session": session, "defined": True, "future_rate_mse_ratio": float(ratio)})
            except (ValueError, np.linalg.LinAlgError) as exc:
                records.append({"session": session, "defined": False, "failure_reason": str(exc)})
        curve[str(lead)] = {"valid_count": len(values), "eligible_all_six": len(values) == 6,
                            "mean_future_rate_mse_ratio": float(np.mean(values)) if values else None, "per_reference_session": records}
    eligible = [lead for lead in LEAD_GRID_BINS if curve[str(lead)]["eligible_all_six"]]
    if not eligible:
        return None, {"chosen_lead_bins": None, "selection_curve": curve, "eligible_leads": [],
                      "undefined_reason": "no_lead_has_all_six_reference_sessions_valid",
                      "scope": "other six held-in sessions; chronological trial-half future-rate MSE/rate-only MSE"}
    lead = min(eligible, key=lambda item: (curve[str(item)]["mean_future_rate_mse_ratio"], abs(item - 2), item))
    return lead, {"chosen_lead_bins": lead, "chosen_lead_ms": lead * RAW_BIN_MS, "selection_curve": curve, "eligible_leads": eligible,
                  "scope": "other six held-in sessions; chronological trial-half future-rate MSE/rate-only MSE"}


def loso_normalizer(cache: dict[tuple[str, int], dict[str, np.ndarray]], target: str, lead: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for session in sorted({session for session, _lead in cache}):
        if session == target: continue
        blocks = cache[(session, lead)]
        fit, _ = fit_k4(blocks, blocks["trial"] < m)
        values.append(np.column_stack([np.linalg.norm(fit.weights, axis=1), fit.intercept]))
    values = np.concatenate(values)
    std = values.std(0); std[std <= 1e-6] = 1.0
    return values.mean(0), std


def per_channel_loto_lag_half(block_cache: dict[int, dict[str, np.ndarray]], m: int, split_name: str, *, half: int) -> dict[str, Any]:
    """Within-half per-channel lag diagnostic: leave one *trial* out, never deploy."""
    channels = next(iter(block_cache.values()))["rate"].shape[1]
    scores = np.full((len(LEAD_GRID_BINS), channels), np.nan)
    fold_counts = np.zeros((len(LEAD_GRID_BINS), channels), dtype=int)
    for li, lead in enumerate(LEAD_GRID_BINS):
        blocks = block_cache[lead]
        split_mask = k4_split_masks(blocks, m)[split_name][half]
        trial_ids = np.unique(blocks["trial"][split_mask])
        for held_trial in trial_ids:
            test = split_mask & (blocks["trial"] == held_trial)
            train = split_mask & (blocks["trial"] != held_trial)
            if not test.any(): continue
            try: fit, _ = fit_k4(blocks, train)
            except (ValueError, np.linalg.LinAlgError): continue
            pred = blocks["velocity"][test] @ fit.weights.T + fit.intercept
            per_channel = np.mean((blocks["rate"][test] - pred) ** 2, axis=0)
            valid = np.isfinite(per_channel)
            old = scores[li].copy()
            n = fold_counts[li]
            scores[li, valid] = np.where(n[valid] == 0, per_channel[valid], (old[valid] * n[valid] + per_channel[valid]) / (n[valid] + 1))
            fold_counts[li, valid] += 1
    selected = np.full(channels, np.nan)
    for channel in range(channels):
        eligible = np.flatnonzero(fold_counts[:, channel] >= PER_CHANNEL_MIN_LOTO_FOLDS)
        if eligible.size:
            best = min(eligible, key=lambda i: (scores[i, channel], abs(LEAD_GRID_BINS[i] - 2), LEAD_GRID_BINS[i]))
            selected[channel] = LEAD_GRID_BINS[best]
    return {"per_channel_lag_bins": selected.tolist(), "n_defined_channels": int(np.isfinite(selected).sum()),
            "minimum_valid_loto_folds": PER_CHANNEL_MIN_LOTO_FOLDS,
            "median_best_lag_valid_folds": float(np.median(np.max(fold_counts, axis=0)[np.isfinite(selected)])) if np.isfinite(selected).any() else None}


def balanced_indices(blocks: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Equal 45-degree direction-bin blocks; an empty bin is an undefined arm."""
    available = np.flatnonzero(mask)
    theta = np.arctan2(blocks["velocity"][available, 1], blocks["velocity"][available, 0])
    labels = np.floor((theta + np.pi) / (2 * np.pi) * BALANCE_DIRECTION_BINS).astype(int) % BALANCE_DIRECTION_BINS
    bins = [available[labels == i] for i in range(BALANCE_DIRECTION_BINS)]
    counts = [int(len(x)) for x in bins]
    if min(counts, default=0) == 0:
        return None, {"defined": False, "reason": "one_or_more_empty_direction_bins", "bin_counts": counts}
    per_bin = min(counts)
    # Deterministic: retain chronological first blocks in each direction bin.
    return np.concatenate([x[:per_bin] for x in bins]), {"defined": True, "bin_counts": counts, "per_bin": per_bin, "total_blocks": per_bin * BALANCE_DIRECTION_BINS}


def fit_on_indices(blocks: dict[str, np.ndarray], indices: np.ndarray, *, ridge: bool) -> tuple[CarrierFit, dict[str, Any]]:
    mask = np.zeros(len(blocks["trial"]), dtype=bool); mask[indices] = True
    return fit_k4(blocks, mask, ridge=ridge)


def balanced_arm(blocks: dict[str, np.ndarray], first_mask: np.ndarray, second_mask: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Balanced OLS vs equal-size random OLS and all-block OLS/ridge; no threshold."""
    out: dict[str, Any] = {}
    first_idx, first_cov = balanced_indices(blocks, first_mask); second_idx, second_cov = balanced_indices(blocks, second_mask)
    out["balanced_coverage"] = {"first": first_cov, "second": second_cov}
    if first_idx is None or second_idx is None:
        out["balanced_ols"] = {"defined": False, "reason": "balanced coverage undefined"}
        out["random_equal_size_ols"] = {"defined": False, "reason": "balanced coverage undefined; no matched sample size"}
    else:
        try:
            first_fit, first_audit = fit_on_indices(blocks, first_idx, ridge=False)
            second_fit, second_audit = fit_on_indices(blocks, second_idx, ridge=False)
            out["balanced_ols"] = {"defined": True, "metrics": k4_component_metrics(first_fit, second_fit, normalizer),
                                   "first_audit": first_audit, "second_audit": second_audit}
        except (ValueError, np.linalg.LinAlgError) as exc:
            out["balanced_ols"] = {"defined": False, "reason": str(exc)}
        n_first, n_second = len(first_idx), len(second_idx)
        random_metrics = []
        for seed in RANDOM_BALANCE_SEEDS:
            rng1, rng2 = np.random.RandomState(seed), np.random.RandomState(10_000 + seed)
            candidate1, candidate2 = np.flatnonzero(first_mask), np.flatnonzero(second_mask)
            try:
                fit1, _ = fit_on_indices(blocks, rng1.choice(candidate1, size=n_first, replace=False), ridge=False)
                fit2, _ = fit_on_indices(blocks, rng2.choice(candidate2, size=n_second, replace=False), ridge=False)
                random_metrics.append(k4_component_metrics(fit1, fit2, normalizer))
            except (ValueError, np.linalg.LinAlgError):
                continue
        out["random_equal_size_ols"] = {"defined": bool(random_metrics), "n_requested": N_RANDOM_BALANCE_REPETITIONS, "n_valid": len(random_metrics),
            "median_metrics": {key: float(np.nanmedian([row[key] for row in random_metrics])) for key in random_metrics[0]} if random_metrics else None}
    for label, ridge in (("all_block_ols", False), ("all_block_ridge", True)):
        try:
            fit1, audit1 = fit_k4(blocks, first_mask, ridge=ridge); fit2, audit2 = fit_k4(blocks, second_mask, ridge=ridge)
            out[label] = {"defined": True, "metrics": k4_component_metrics(fit1, fit2, normalizer), "first_audit": audit1, "second_audit": audit2}
        except (ValueError, np.linalg.LinAlgError) as exc:
            out[label] = {"defined": False, "reason": str(exc)}
    return out


def lag_agreement(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a, b = np.asarray(first["per_channel_lag_bins"], float), np.asarray(second["per_channel_lag_bins"], float)
    keep = np.isfinite(a) & np.isfinite(b)
    if not keep.any(): return {"n_joint_defined_channels": 0, "exact_agreement_fraction": None, "absolute_delta_lag_bins_median": None, "spearman": None}
    from scipy.stats import spearmanr
    rho = spearmanr(a[keep], b[keep]).statistic if int(keep.sum()) >= 2 else np.nan
    return {"n_joint_defined_channels": int(keep.sum()), "exact_agreement_fraction": float(np.mean(a[keep] == b[keep])),
            "absolute_delta_lag_bins_median": float(np.median(np.abs(a[keep] - b[keep]))), "spearman": float(rho)}


def write_protocol(out: Path, data: Path) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    paths = heldin_paths(data)
    payload = {
        "schema_version": 1, "status": "frozen_before_data_compute", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "seven M2 held-in calibration NWBs only; CPU-only; no held-out/query/decoder",
        "inputs": [{"session": session_name(path), "path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
        "source_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "frozen_design": {
            "m_grid": list(M_GRID), "k4_lead_grid_bins": list(LEAD_GRID_BINS), "raw_bin_ms": RAW_BIN_MS,
            "k4_block_definition": "five 20-ms neural bins; velocity block at frozen lead; both spans inside same trial; every sample active",
            "global_lag": "each target session/M selects lead from other six sessions only by mean chronological A-to-B future-rate MSE/rate-only-MSE; tie: closest to +2, then smaller lag",
            "splits": {"k4": ["chronological_trial_half primary", "odd_even_trial drift diagnostic", "odd_even_block optimistic upper bound"],
                       "t4": ["chronological_trial_half", "odd_even_trial"], "t4_odd_even_block": "not applicable: forbidden partial-exposure T4"},
            "per_channel_lag_diagnostic": {"selection": "independent half-specific leave-one-trial-out encoding MSE across valid trials", "min_valid_folds": PER_CHANNEL_MIN_LOTO_FOLDS,
                "tie": "lower CV MSE; closest to +2 bins; then lower lag", "report": ["exact agreement", "absolute delta", "Spearman", "distribution"], "not_deployed": True},
            "k4_components": "Wx, Wy, norm, b; [norm,b] standardized with other-six full-prefix LOSO stats, never half reestimated",
            "balanced_exploratory": {"velocity_angle_bins": BALANCE_DIRECTION_BINS, "balanced_ols": "each half separately: chronological first per-bin min occupancy; undefined if any empty bin",
                "random_equal_size_ols": {"repetitions": N_RANDOM_BALANCE_REPETITIONS, "seeds": list(RANDOM_BALANCE_SEEDS), "same_half_without_replacement": True},
                "all_block_ols": True, "all_block_ridge": {"velocity_standardized_within_fit": True, "alpha": RIDGE_ALPHA_STANDARDIZED, "map_weights_back_to_native_units": True},
                "no_eigenratio_threshold": True},
        }, "git_status_before_compute": git_status(),
    }
    receipt = out / "protocol_receipt.json"; receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return receipt


def write_feasibility_addendum(out: Path, expected_receipt_sha: str) -> Path:
    """Bind the immutable original receipt to explicit fail-closed option A."""
    receipt = out / "protocol_receipt.json"
    if not receipt.exists() or sha256(receipt) != expected_receipt_sha:
        raise ValueError("prior protocol receipt is absent or SHA-mismatched")
    prior = json.loads(receipt.read_text())
    addendum = out / "protocol_feasibility_addendum_v1.json"
    if addendum.exists(): raise FileExistsError("refusing to overwrite feasibility addendum")
    payload = {
        "schema_version": 1, "status": "frozen_before_resumed_compute", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_receipt_path": str(receipt.resolve()), "prior_receipt_sha256": expected_receipt_sha,
        "prior_script_sha256": prior["source_script"]["sha256"], "updated_script_path": str(Path(__file__).resolve()), "updated_script_sha256": sha256(Path(__file__)),
        "first_fail_closed_exception": {"stage": "nested global-lag selection", "m": 8, "lead_bins_example": -5,
            "exception": "fewer than ten active K4 blocks in a chronological first half for a reference session"},
        "resolution": {
            "choice": "A", "rule": "For every target session/M, select only among leads with all six other reference sessions valid under unchanged >=10-block and rank-3 constraints.",
            "invalid_leads": "retain per-reference failure reasons and valid_count in raw selection curve",
            "empty_common_lead_set": "all K4 splits are nested_lag_undefined; T4 still runs",
            "no_changes": ["M grid", "lead grid", "Gate-A >=10-block floor", "rank-3 requirement", "held-in-only scope"],
        },
    }
    addendum.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return addendum


def write_implementation_fix_addendum(out: Path, receipt_sha: str, feasibility_sha: str) -> Path:
    """Chain a narrow T4 prefix-indexing repair without mutating prior freezes."""
    receipt, feasibility = out / "protocol_receipt.json", out / "protocol_feasibility_addendum_v1.json"
    if sha256(receipt) != receipt_sha or sha256(feasibility) != feasibility_sha: raise ValueError("prior receipt/addendum SHA mismatch")
    prior = json.loads(feasibility.read_text())
    addendum = out / "protocol_implementation_fix_addendum_v2.json"
    if addendum.exists(): raise FileExistsError("refusing to overwrite implementation-fix addendum")
    payload = {
        "schema_version": 1, "status": "frozen_before_resumed_compute", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_receipt_sha256": receipt_sha, "prior_feasibility_addendum_sha256": feasibility_sha,
        "prior_updated_script_sha256": prior["updated_script_sha256"], "updated_script_path": str(Path(__file__).resolve()), "updated_script_sha256": sha256(Path(__file__)),
        "first_fail_closed_exception": {"stage": "T4 held-in split evaluation", "exception": "boolean mask M length indexed max-M=33 T4 cache"},
        "repair": "t4_fit slices chronological first-M cache arrays before applying the M-length split mask; it preserves frozen first-100-bin T4 exposure and does not create partial-exposure T4.",
        "no_protocol_change": True,
    }
    addendum.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return addendum


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def strict_json(value: Any) -> Any:
    """Recursively map numpy scalars and non-finite values to strict JSON values."""
    if isinstance(value, dict): return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray): return strict_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def aggregate(k4_rows: list[dict[str, Any]], t4_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive coverage/median table only; intentionally contains no gate."""
    out: dict[str, Any] = {"k4": {}, "t4": {}, "no_posthoc_gate": True}
    for modality, rows in (("k4", k4_rows), ("t4", t4_rows)):
        for m in M_GRID:
            for split in sorted({row["split"] for row in rows}):
                group = [row for row in rows if row["m"] == m and row["split"] == split]
                defined = [row for row in group if row.get("defined")]
                numeric = sorted({key for row in defined for key, value in row.items() if isinstance(value, (int, float, np.number)) and key not in {"m", "chosen_global_lead_bins"}})
                summary = {"n_sessions": len(group), "n_defined": len(defined), "defined_fraction": len(defined) / len(group) if group else None,
                           "session_median": {key: float(np.nanmedian([row[key] for row in defined if row.get(key) is not None])) for key in numeric if any(row.get(key) is not None for row in defined)}}
                if modality == "k4":
                    leads = [row["chosen_global_lead_bins"] for row in defined if row.get("chosen_global_lead_bins") is not None]
                    summary["global_lag_bins_distribution"] = {str(lead): leads.count(lead) for lead in sorted(set(leads))}
                out[modality][f"M={m}|{split}"] = summary
    return strict_json(out)


def plot(out: Path, k4_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [row for row in k4_rows if row["split"] == "chronological_trial_half" and row.get("defined")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for session in sorted({row["session"] for row in primary}):
        rows = sorted((row for row in primary if row["session"] == session), key=lambda row: row["m"])
        ax.plot([row["m"] for row in rows], [row["w_flattened_pearson"] for row in rows], marker="o", label=session.replace("ses-", ""))
    ax.axvline(24, linestyle=":", color="tab:red", label="M=24 deployment boundary")
    ax.set(xlabel="chronological calibration trials M", ylabel="K4 flattened-W Pearson", title="Nested-LOSO-lag K4 reliability; held-in only")
    ax.grid(alpha=.25); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(out / "k4_factorial_primary.png", dpi=220); fig.savefig(out / "k4_factorial_primary.pdf"); plt.close(fig)


def run(out: Path, data: Path, expected_sha: str, expected_addendum_sha: str, expected_fix_sha: str) -> None:
    receipt = out / "protocol_receipt.json"
    if not receipt.exists() or sha256(receipt) != expected_sha: raise ValueError("missing or SHA-mismatched frozen protocol receipt")
    protocol = json.loads(receipt.read_text())
    if protocol.get("status") != "frozen_before_data_compute": raise ValueError("not a pre-compute receipt")
    addendum = out / "protocol_feasibility_addendum_v1.json"
    if not addendum.exists() or sha256(addendum) != expected_addendum_sha: raise ValueError("missing or SHA-mismatched feasibility addendum")
    addendum_data = json.loads(addendum.read_text())
    if addendum_data.get("prior_receipt_sha256") != expected_sha:
        raise ValueError("feasibility addendum does not bind the prior receipt")
    fix = out / "protocol_implementation_fix_addendum_v2.json"
    if not fix.exists() or sha256(fix) != expected_fix_sha: raise ValueError("missing or SHA-mismatched implementation-fix addendum")
    fix_data = json.loads(fix.read_text())
    if fix_data.get("prior_receipt_sha256") != expected_sha or fix_data.get("prior_feasibility_addendum_sha256") != expected_addendum_sha or fix_data.get("prior_updated_script_sha256") != addendum_data.get("updated_script_sha256") or fix_data.get("updated_script_sha256") != sha256(Path(__file__)):
        raise ValueError("implementation-fix addendum chain does not bind current script")
    paths = heldin_paths(data)
    frozen_inputs = {row["path"]: row["sha256"] for row in protocol["inputs"]}
    if any(frozen_inputs.get(str(path.resolve())) != sha256(path) for path in paths): raise ValueError("input provenance drift")
    raw: dict[str, dict[str, Any]] = {}
    for path in paths:
        neural, velocity, change, _ = load_nwb(path, FalconTask.m2)
        angles = calibration_target_angles(path, TASK)
        raw[session_name(path)] = {"neural": np.asarray(neural), "velocity": np.asarray(velocity), "change": np.asarray(change, bool), "angles": angles}
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    t4_cache: dict[str, dict[str, np.ndarray]] = {}
    for session, item in raw.items():
        t4_cache[session] = t4_trial_arrays(item["neural"], item["change"], item["angles"], max_trials=max(M_GRID))
        for lead in LEAD_GRID_BINS:
            cache[(session, lead)] = k4_blocks(item["neural"], item["velocity"], item["change"], lead=lead, max_trials=max(M_GRID))

    k4_rows: list[dict[str, Any]] = []; t4_rows: list[dict[str, Any]] = []; details: dict[str, Any] = {}
    for m in M_GRID:
        for session in sorted(raw):
            lead, selection = select_lag_nested(cache, session, m)
            detail_key = f"{session}|M={m}"
            details[detail_key] = {"nested_global_lag": selection, "splits": {}}
            if lead is None:
                for split in k4_split_masks(cache[(session, LEAD_GRID_BINS[0])], m):
                    k4_rows.append({"modality": "k4", "session": session, "m": m, "split": split, "defined": False,
                                    "undefined_reason": "nested_lag_undefined: no common six-reference valid lead", "chosen_global_lead_bins": None})
                # T4 must still be evaluated below even when K4 is undefined.
                for split, (first_mask, second_mask) in t4_split_masks(m).items():
                    first, cov_first = t4_fit(t4_cache[session], first_mask, f"{session}|M={m}|{split}|first")
                    second, cov_second = t4_fit(t4_cache[session], second_mask, f"{session}|M={m}|{split}|second")
                    row = {"modality": "t4", "session": session, "m": m, "split": split, "first_coverage": json.dumps(cov_first, sort_keys=True), "second_coverage": json.dumps(cov_second, sort_keys=True)}
                    if first is None or second is None: row.update({"defined": False, "undefined_reason": f"first={cov_first['undefined_reason']};second={cov_second['undefined_reason']}"})
                    else: row.update(t4_component_metrics(first, second)); row["defined"] = True
                    t4_rows.append(row)
                continue
            normalizer = loso_normalizer(cache, session, lead, m)
            blocks = cache[(session, lead)]
            details[detail_key]["loso_norm_b_statistics"] = {"mean": normalizer[0].tolist(), "std": normalizer[1].tolist()}
            for split, (first_mask, second_mask) in k4_split_masks(blocks, m).items():
                row = {"modality": "k4", "session": session, "m": m, "split": split, "chosen_global_lead_bins": lead, "chosen_global_lead_ms": lead * RAW_BIN_MS}
                try:
                    first_fit, first_audit = fit_k4(blocks, first_mask); second_fit, second_audit = fit_k4(blocks, second_mask)
                    row.update(k4_component_metrics(first_fit, second_fit, normalizer)); row["defined"] = True
                    detail = {"first_audit": first_audit, "second_audit": second_audit, "balanced": balanced_arm(blocks, first_mask, second_mask, normalizer),
                              "per_channel_loto_note": "computed independently in each half after primary fits"}
                    details[detail_key]["splits"][split] = detail
                except (ValueError, np.linalg.LinAlgError) as exc:
                    row.update({"defined": False, "undefined_reason": str(exc)})
                k4_rows.append(row)
            for split, (first_mask, second_mask) in t4_split_masks(m).items():
                first, cov_first = t4_fit(t4_cache[session], first_mask, f"{session}|M={m}|{split}|first")
                second, cov_second = t4_fit(t4_cache[session], second_mask, f"{session}|M={m}|{split}|second")
                row = {"modality": "t4", "session": session, "m": m, "split": split, "first_coverage": json.dumps(cov_first, sort_keys=True), "second_coverage": json.dumps(cov_second, sort_keys=True)}
                if first is None or second is None: row.update({"defined": False, "undefined_reason": f"first={cov_first['undefined_reason']};second={cov_second['undefined_reason']}"})
                else: row.update(t4_component_metrics(first, second)); row["defined"] = True
                t4_rows.append(row)

    # K4 LOTO must independently select a lag in each half.  Recompute using explicit
    # mask pairs to avoid using its sister half as an accidental validation set.
    for row in k4_rows:
        if not row.get("defined"): continue
        session, m, split = row["session"], row["m"], row["split"]
        blocks_by_lead = {lead: cache[(session, lead)] for lead in LEAD_GRID_BINS}
        lag_first = per_channel_loto_lag_half(blocks_by_lead, m, split, half=0)
        lag_second = per_channel_loto_lag_half(blocks_by_lead, m, split, half=1)
        agreement = lag_agreement(lag_first, lag_second)
        row.update({"per_channel_loto_joint_defined_channels": agreement["n_joint_defined_channels"],
                    "per_channel_loto_exact_agreement": agreement["exact_agreement_fraction"],
                    "per_channel_loto_abs_delta_bins_median": agreement["absolute_delta_lag_bins_median"],
                    "per_channel_loto_spearman": agreement["spearman"]})
        details[f"{session}|M={m}"]["splits"][split]["per_channel_loto_first"] = lag_first
        details[f"{session}|M={m}"]["splits"][split]["per_channel_loto_second"] = lag_second
        details[f"{session}|M={m}"]["splits"][split]["per_channel_loto_agreement"] = agreement

    payload = {"schema_version": 1, "status": "complete", "scope": protocol["scope"], "protocol_receipt": {"path": str(receipt.resolve()), "sha256": expected_sha},
               "protocol_feasibility_addendum": {"path": str(addendum.resolve()), "sha256": expected_addendum_sha},
               "protocol_implementation_fix_addendum": {"path": str(fix.resolve()), "sha256": expected_fix_sha},
               "k4_rows": k4_rows, "t4_rows": t4_rows, "details": details,
               "guardrail": "No held-out/query/decoder result was read; balanced and per-channel lag branches are exploratory diagnostics only."}
    payload = strict_json(payload)
    (out / "raw_factorial.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    csv_write(out / "k4_by_session.csv", k4_rows); csv_write(out / "t4_by_session.csv", t4_rows); plot(out, k4_rows)
    (out / "aggregate.json").write_text(json.dumps(aggregate(k4_rows, t4_rows), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"k4_records": len(k4_rows), "t4_records": len(t4_rows), "out": str(out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953")
    parser.add_argument("--out-dir", type=Path, default=SUA / "results/m2_carrier_stability_factorial_v1")
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--write-feasibility-addendum", action="store_true")
    parser.add_argument("--write-implementation-fix-addendum", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--protocol-sha")
    parser.add_argument("--feasibility-addendum-sha")
    parser.add_argument("--implementation-fix-addendum-sha")
    args = parser.parse_args()
    if sum((args.write_protocol, args.write_feasibility_addendum, args.write_implementation_fix_addendum, args.run)) != 1: raise ValueError("select exactly one operation")
    out, data = args.out_dir.resolve(), args.data_dir.resolve()
    if args.write_protocol:
        receipt = write_protocol(out, data); print(f"{receipt}\nsha256={sha256(receipt)}"); return
    if args.write_feasibility_addendum:
        if not args.protocol_sha: raise ValueError("--write-feasibility-addendum requires --protocol-sha")
        addendum = write_feasibility_addendum(out, args.protocol_sha); print(f"{addendum}\nsha256={sha256(addendum)}"); return
    if args.write_implementation_fix_addendum:
        if not args.protocol_sha or not args.feasibility_addendum_sha: raise ValueError("--write-implementation-fix-addendum requires receipt and v1 SHA")
        addendum = write_implementation_fix_addendum(out, args.protocol_sha, args.feasibility_addendum_sha); print(f"{addendum}\nsha256={sha256(addendum)}"); return
    if not args.protocol_sha or not args.feasibility_addendum_sha or not args.implementation_fix_addendum_sha: raise ValueError("--run requires all three protocol SHA values")
    run(out, data, args.protocol_sha, args.feasibility_addendum_sha, args.implementation_fix_addendum_sha)


if __name__ == "__main__":
    main()
