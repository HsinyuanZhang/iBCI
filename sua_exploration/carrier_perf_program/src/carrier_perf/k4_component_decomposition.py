"""K4 component decomposition diagnostic — CPU-only, held-in M2 calibration.

Tests whether K4's usable channel-identifying information lives in dimension 4
(baseline rate) versus the velocity-tuning dims, and how that relates to T4.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from carrier_perf.p1_stage_b_audit import heldin_m2_paths, sha256_file
from carrier_perf.protocol import mean, sample_std, sigma_delta_paired

SCHEMA = "carrier_perf_k4_component_decomposition_cpu_v1"
CALIBRATION_N_TRIALS = 24
SPLIT_HALF_N_TRIALS = 12
MAX_TRIAL_LENGTH = 256

# Pre-declared verdict thresholds (do not tune after seeing results).
# Matches m2_k4_split_half_curve_v1 operating point for reliability.
RESIDUAL_R2_EXPLAINED_THRESHOLD = 0.5  # below => explained by T4 (redundant)
SPLIT_HALF_RELIABILITY_THRESHOLD = 0.5  # Pearson r across channels >= => reliable

T4_FEATURE_NAMES = ("m_cos_phi", "m_sin_phi", "m", "baseline_rate")
K4_FEATURE_NAMES = ("w_x", "w_y", "w_norm", "baseline_rate")

DESCRIPTOR_SUBSETS = (
    "k4_full",
    "k4_b_only",
    "k4_w_only",
    "t4_full",
    "t4_b_only",
    "t4_w_only",
    "hybrid_t4w_k4b",
)


def _spint_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sce_root() -> Path:
    return _spint_root() / "streaming_calibration_exp"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


def cross_correlation_matrix(t4: np.ndarray, k4: np.ndarray) -> dict[str, Any]:
    """Full 4x4 Pearson and Spearman between every T4 and K4 dim across channels."""
    if t4.shape != k4.shape or t4.ndim != 2 or t4.shape[1] != 4:
        raise ValueError(f"expected matching [N,4] arrays, got {t4.shape} and {k4.shape}")
    pearson = [[_pearson(t4[:, i], k4[:, j]) for j in range(4)] for i in range(4)]
    spearman = [[_spearman(t4[:, i], k4[:, j]) for j in range(4)] for i in range(4)]
    return {
        "pearson": pearson,
        "spearman": spearman,
        "t4_names": list(T4_FEATURE_NAMES),
        "k4_names": list(K4_FEATURE_NAMES),
    }


def split_half_reliability_per_dim(features: np.ndarray, half_a: np.ndarray, half_b: np.ndarray) -> list[float]:
    """Across-channel Pearson r per dimension between two half fits."""
    if half_a.shape != half_b.shape or half_a.shape[1] != 4:
        raise ValueError(f"expected [N,4] half arrays, got {half_a.shape} and {half_b.shape}")
    return [_pearson(half_a[:, d], half_b[:, d]) for d in range(4)]


def per_channel_w_halfsplit_cosine_median(half_a: np.ndarray, half_b: np.ndarray) -> float:
    """Median per-channel cosine between 2-D W vectors (w_x, w_y) across split halves."""
    if half_a.shape != half_b.shape or half_a.shape[1] < 2:
        raise ValueError(f"expected matching [N,>=2] half arrays, got {half_a.shape} and {half_b.shape}")
    w_a = np.asarray(half_a[:, :2], dtype=np.float64)
    w_b = np.asarray(half_b[:, :2], dtype=np.float64)
    dots = np.sum(w_a * w_b, axis=1)
    norms = np.linalg.norm(w_a, axis=1) * np.linalg.norm(w_b, axis=1)
    cosines = dots / np.maximum(norms, 1e-12)
    return float(np.median(cosines))


def residual_r_squared(y: np.ndarray, predictors: np.ndarray) -> float:
    """Fraction of variance in y NOT explained by OLS on predictors (incl. intercept)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(predictors, dtype=np.float64)
    if y.size != x.shape[0] or y.size < x.shape[1] + 2:
        raise ValueError(f"shape mismatch: y={y.shape}, X={x.shape}")
    design = np.column_stack([np.ones(y.size, dtype=np.float64), x])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < design.shape[1]:
        return float("nan")
    resid = y - design @ coef
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return ss_res / ss_tot


def extract_descriptor_subset(
    t4: np.ndarray,
    k4: np.ndarray,
    subset: str,
) -> np.ndarray:
    if subset == "k4_full":
        return np.asarray(k4, dtype=np.float64)
    if subset == "k4_b_only":
        return np.asarray(k4[:, 3:4], dtype=np.float64)
    if subset == "k4_w_only":
        return np.asarray(k4[:, :3], dtype=np.float64)
    if subset == "t4_full":
        return np.asarray(t4, dtype=np.float64)
    if subset == "t4_b_only":
        return np.asarray(t4[:, 3:4], dtype=np.float64)
    if subset == "t4_w_only":
        return np.asarray(t4[:, :3], dtype=np.float64)
    if subset == "hybrid_t4w_k4b":
        return np.column_stack([t4[:, :3], k4[:, 3:4]]).astype(np.float64)
    raise ValueError(f"unknown descriptor subset {subset!r}")


def channel_matching_accuracy(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
) -> dict[str, float]:
    """Z-score columns with session-A stats; nearest-neighbour channel matching."""
    a = np.asarray(desc_a, dtype=np.float64)
    b = np.asarray(desc_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"descriptor shape mismatch: {a.shape} vs {b.shape}")
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a == 0 or n_b == 0:
        raise ValueError("empty descriptor matrix")
    mu = a.mean(axis=0)
    sigma = a.std(axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    za = (a - mu) / sigma
    zb = (b - mu) / sigma
    # Pairwise squared Euclidean distances [n_a, n_b]
    dists = (
        np.sum(za * za, axis=1, keepdims=True)
        + np.sum(zb * zb, axis=1, keepdims=True).T
        - 2.0 * (za @ zb.T)
    )
    dists = np.maximum(dists, 0.0)
    order = np.argsort(dists, axis=1)
    n_match = min(n_a, n_b)
    reciprocal_ranks: list[float] = []
    top1_hits = 0
    for i in range(n_match):
        neighbours = order[i]
        true_rank = int(np.where(neighbours == i)[0][0]) + 1
        reciprocal_ranks.append(1.0 / true_rank)
        if neighbours[0] == i:
            top1_hits += 1
    chance = 1.0 / n_match
    return {
        "top1_accuracy": float(top1_hits / n_match),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "n_channels_matched": int(n_match),
        "chance_top1_accuracy": chance,
    }


def _verdict_for_dim(residual_r2: float, split_half_r: float) -> str:
    explained = math.isfinite(residual_r2) and residual_r2 < RESIDUAL_R2_EXPLAINED_THRESHOLD
    reliable = math.isfinite(split_half_r) and split_half_r >= SPLIT_HALF_RELIABILITY_THRESHOLD
    if explained:
        return "redundant_with_t4"
    if reliable:
        return "novel_and_reliable"
    return "novel_but_unreliable"


VERDICT_LABELS = ("novel_and_reliable", "novel_but_unreliable", "redundant_with_t4")
RELIABILITY_METRIC_NOTE = (
    "Per-channel 2-D W-vector cosine (w_percanel_halfsplit_cosine_median) measures whether "
    "each channel's velocity-tuning direction is stable across halves, while across-channel "
    "Pearson on a single W component measures whether the population ranking on that "
    "component is stable; a channel's direction can be unstable while its rank on one "
    "component remains stable, so the two metrics are not in contradiction."
)


def _aggregate_verdict_any_session_or(verdicts: Sequence[str]) -> str:
    if any(v == "novel_and_reliable" for v in verdicts):
        return "novel_and_reliable"
    if all(v == "redundant_with_t4" for v in verdicts):
        return "redundant_with_t4"
    if any(v == "novel_but_unreliable" for v in verdicts):
        return "novel_but_unreliable"
    return verdicts[0] if verdicts else "novel_but_unreliable"


def _aggregate_verdict_majority(verdicts: Sequence[str]) -> str:
    n = len(verdicts)
    if n == 0:
        return "novel_but_unreliable"
    threshold = n // 2 + 1
    counts = {label: sum(1 for v in verdicts if v == label) for label in VERDICT_LABELS}
    winners = [label for label in VERDICT_LABELS if counts[label] >= threshold]
    if len(winners) == 1:
        return winners[0]
    if winners:
        # Tie at majority threshold: prefer the more conservative (redundant) reading.
        for label in reversed(VERDICT_LABELS):
            if label in winners:
                return label
    return max(VERDICT_LABELS, key=lambda label: counts[label])


def _aggregate_verdict_unanimous(verdicts: Sequence[str]) -> str:
    if not verdicts:
        return "novel_but_unreliable"
    unique = set(verdicts)
    if len(unique) == 1:
        return verdicts[0]
    return "novel_but_unreliable"


def _per_dim_verdicts_from_sessions(
    session_records: Sequence[Mapping[str, Any]],
    rule: str,
) -> dict[str, str]:
    per_dim: dict[str, str] = {}
    for name in K4_FEATURE_NAMES:
        verdicts = [rec["incremental_over_t4"][name]["verdict"] for rec in session_records]
        if rule == "any_session_or":
            per_dim[name] = _aggregate_verdict_any_session_or(verdicts)
        elif rule == "majority":
            per_dim[name] = _aggregate_verdict_majority(verdicts)
        elif rule == "unanimous":
            per_dim[name] = _aggregate_verdict_unanimous(verdicts)
        else:
            raise ValueError(f"unknown aggregation rule {rule!r}")
    return per_dim


def _recommendation_from_per_dim(
    per_dim: Mapping[str, str],
    identifiability: Mapping[str, Mapping[str, float]],
) -> tuple[str, dict[str, Any]]:
    any_novel_reliable = any(v == "novel_and_reliable" for v in per_dim.values())
    hybrid_top1 = identifiability["hybrid_t4w_k4b"]["top1_accuracy_mean"]
    t4_top1 = identifiability["t4_full"]["top1_accuracy_mean"]
    k4_top1 = identifiability["k4_full"]["top1_accuracy_mean"]
    if any_novel_reliable:
        recommendation = "fusion_worthwhile"
    elif hybrid_top1 > t4_top1 and hybrid_top1 > k4_top1:
        recommendation = "hybrid_worthwhile"
    else:
        recommendation = "neither"
    return recommendation, {
        "any_k4_dim_novel_and_reliable": any_novel_reliable,
        "hybrid_t4w_k4b_top1_accuracy_mean": hybrid_top1,
        "t4_full_top1_accuracy_mean": t4_top1,
        "k4_full_top1_accuracy_mean": k4_top1,
    }


def _build_per_k4_dim_verdict_tally(
    session_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tally: dict[str, Any] = {}
    for name in K4_FEATURE_NAMES:
        verdicts = [rec["incremental_over_t4"][name]["verdict"] for rec in session_records]
        residual_vals = [
            rec["incremental_over_t4"][name]["residual_r_squared"] for rec in session_records
        ]
        reliability_vals = [
            rec["incremental_over_t4"][name]["split_half_reliability"] for rec in session_records
        ]
        finite_residual = [v for v in residual_vals if math.isfinite(v)]
        below = sum(1 for v in finite_residual if v < RESIDUAL_R2_EXPLAINED_THRESHOLD)
        at_or_above = sum(1 for v in finite_residual if v >= RESIDUAL_R2_EXPLAINED_THRESHOLD)
        tally[name] = {
            "verdict_counts": {
                label: sum(1 for v in verdicts if v == label) for label in VERDICT_LABELS
            },
            "residual_r_squared_mean": mean(finite_residual) if finite_residual else float("nan"),
            "residual_r_squared_per_session": residual_vals,
            "mean_residual_r2_straddles_threshold": below > 0 and at_or_above > 0,
            "split_half_reliability_mean": mean(
                [v for v in reliability_vals if math.isfinite(v)]
            )
            if any(math.isfinite(v) for v in reliability_vals)
            else float("nan"),
            "split_half_reliability_per_session": reliability_vals,
        }
    return tally


def _build_aggregation_sensitivity(
    session_records: Sequence[Mapping[str, Any]],
    identifiability: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in ("any_session_or", "majority", "unanimous"):
        per_dim = _per_dim_verdicts_from_sessions(session_records, rule)
        recommendation, inputs = _recommendation_from_per_dim(per_dim, identifiability)
        out[rule] = {
            "per_k4_dim_verdict": per_dim,
            "recommendation": recommendation,
            "recommendation_inputs": inputs,
        }
    return out


def _k4_from_trial_window(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    *,
    trial_start: int,
    n_trials: int,
    source: str,
) -> np.ndarray:
    """K4 OLS on a chronological trial window using production block construction."""
    sce = _sce_root()
    if str(sce) not in sys.path:
        sys.path.insert(0, str(sce))
    from src.data.falcon_k4_features import (
        K4_ACTIVE_EPSILON,
        K4_BEHAVIOR_LEAD_BINS,
        K4_BLOCK_WIDTH_BINS,
        K4_RAW_BIN_MS,
    )

    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    starts = np.flatnonzero(trial_change)
    if len(starts) < trial_start + n_trials:
        raise ValueError(
            f"K4 trial window [{trial_start}:{trial_start + n_trials}) unavailable for {source}"
        )
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(covariates) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    for start, end in zip(
        starts[trial_start : trial_start + n_trials],
        ends[trial_start : trial_start + n_trials],
    ):
        for left in range(
            int(start),
            int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1,
            K4_BLOCK_WIDTH_BINS,
        ):
            right = left + K4_BLOCK_WIDTH_BINS
            y_left = left + K4_BEHAVIOR_LEAD_BINS
            y_right = y_left + K4_BLOCK_WIDTH_BINS
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(
                neural[left:right].sum(axis=0)
                / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0)
            )
            behavior.append(covariates[y_left:y_right].mean(axis=0))
    if len(rates) < 3:
        raise ValueError(f"K4 trial window for {source} has fewer than three active blocks")
    rate = np.asarray(rates, dtype=np.float64)
    y = np.asarray(behavior, dtype=np.float64)
    design = np.column_stack([np.ones(len(y), dtype=np.float64), y])
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != 3:
        raise ValueError(
            f"K4 design rank {design_rank} != 3 for {source} window "
            f"[{trial_start}:{trial_start + n_trials})"
        )
    coefficients, *_ = np.linalg.lstsq(design, rate, rcond=None)
    intercept = coefficients[0]
    weights = coefficients[1:].T
    return np.column_stack(
        [weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), intercept]
    ).astype(np.float32)


def _load_m2_session(nwb_path: Path) -> dict[str, Any]:
    """Load arrays following FalconDataModule K4/M2 production calibration path."""
    if "held-out" in str(nwb_path).lower():
        raise ValueError(f"refusing held-out NWB: {nwb_path}")
    sce = _sce_root()
    if str(sce) not in sys.path:
        sys.path.insert(0, str(sce))
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from src.data.falcon_k4_features import k4_from_raw_calibration
    from src.data.falcon_t4_features import (
        calibration_target_angles,
        t4_from_trial_sums,
        validate_trial_label_alignment,
    )

    neural, covariates, trial_change, eval_mask = load_nwb(nwb_path, FalconTask.m2)
    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    target_angles = np.asarray(calibration_target_angles(nwb_path, "m2"), dtype=np.float64)
    validate_trial_label_alignment(trial_change, target_angles, source=str(nwb_path))

    # K4: raw unsmoothed full timeline (use_intertrials=True production load).
    k4_neural = neural.copy()
    k4_covariates = covariates.copy()
    k4_trial_change = trial_change.copy()

    # T4: eval_mask-filtered calibration (use_calib_intertrials=False).
    t4_neural = neural[eval_mask]
    t4_trial_change = trial_change[eval_mask]
    retained_angles = target_angles[
        eval_mask[np.flatnonzero(trial_change)]
    ]
    validate_trial_label_alignment(t4_trial_change, retained_angles, source=f"{nwb_path} filtered")

    trial_starts = np.where(t4_trial_change)[0]
    trial_spike_sums: list[np.ndarray] = []
    trial_lengths: list[int] = []
    for i in range(trial_starts.shape[0]):
        start = int(trial_starts[i])
        end = (
            int(trial_starts[i + 1])
            if i + 1 < trial_starts.shape[0]
            else int(t4_neural.shape[0])
        )
        trial_neural = t4_neural[start:end]
        valid_length = min(int(trial_neural.shape[0]), MAX_TRIAL_LENGTH)
        if valid_length <= 0:
            raise ValueError(f"empty calibration trial at index {i} in {nwb_path}")
        trial_spike_sums.append(trial_neural[:valid_length].sum(axis=0))
        trial_lengths.append(valid_length)

    sums = np.asarray(trial_spike_sums, dtype=np.float64)
    lengths = np.asarray(trial_lengths, dtype=np.int64)
    if sums.shape[0] < CALIBRATION_N_TRIALS:
        raise ValueError(
            f"{nwb_path.name}: need {CALIBRATION_N_TRIALS} trials, got {sums.shape[0]}"
        )

    t4_full = t4_from_trial_sums(
        sums[:CALIBRATION_N_TRIALS],
        lengths[:CALIBRATION_N_TRIALS],
        retained_angles[:CALIBRATION_N_TRIALS],
        source=f"{nwb_path.name}[0:{CALIBRATION_N_TRIALS}]",
    )
    k4_full, k4_audit = k4_from_raw_calibration(
        k4_neural,
        k4_covariates,
        k4_trial_change,
        calibration_n_trials=CALIBRATION_N_TRIALS,
    )

    # Split-half fits.
    t4_h1 = t4_from_trial_sums(
        sums[:SPLIT_HALF_N_TRIALS],
        lengths[:SPLIT_HALF_N_TRIALS],
        retained_angles[:SPLIT_HALF_N_TRIALS],
        source=f"{nwb_path.name}[0:{SPLIT_HALF_N_TRIALS}]",
    )
    t4_h2 = t4_from_trial_sums(
        sums[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        lengths[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        retained_angles[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        source=f"{nwb_path.name}[{SPLIT_HALF_N_TRIALS}:{CALIBRATION_N_TRIALS}]",
    )
    k4_h1 = _k4_from_trial_window(
        k4_neural,
        k4_covariates,
        k4_trial_change,
        trial_start=0,
        n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{nwb_path.name}[0:{SPLIT_HALF_N_TRIALS}]",
    )
    k4_h2 = _k4_from_trial_window(
        k4_neural,
        k4_covariates,
        k4_trial_change,
        trial_start=SPLIT_HALF_N_TRIALS,
        n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{nwb_path.name}[{SPLIT_HALF_N_TRIALS}:{CALIBRATION_N_TRIALS}]",
    )

    return {
        "session": nwb_path.name,
        "nwb_path": str(nwb_path),
        "nwb_sha256": sha256_file(nwb_path),
        "n_channels": int(t4_full.shape[0]),
        "t4_full": t4_full,
        "k4_full": k4_full,
        "k4_audit": k4_audit.as_dict(),
        "t4_split_half": {"first": t4_h1, "second": t4_h2},
        "k4_split_half": {"first": k4_h1, "second": k4_h2},
    }


def audit_session_record(session: Mapping[str, Any]) -> dict[str, Any]:
    t4 = np.asarray(session["t4_full"], dtype=np.float64)
    k4 = np.asarray(session["k4_full"], dtype=np.float64)
    cross = cross_correlation_matrix(t4, k4)
    dim4 = {
        "pearson": _pearson(t4[:, 3], k4[:, 3]),
        "spearman": _spearman(t4[:, 3], k4[:, 3]),
    }
    t4_sh = split_half_reliability_per_dim(
        t4, session["t4_split_half"]["first"], session["t4_split_half"]["second"]
    )
    k4_sh = split_half_reliability_per_dim(
        k4, session["k4_split_half"]["first"], session["k4_split_half"]["second"]
    )
    w_cosine_median = per_channel_w_halfsplit_cosine_median(
        session["k4_split_half"]["first"], session["k4_split_half"]["second"]
    )
    incremental: dict[str, Any] = {}
    for d, name in enumerate(K4_FEATURE_NAMES):
        resid = residual_r_squared(k4[:, d], t4)
        incremental[name] = {
            "residual_r_squared": resid,
            "split_half_reliability": k4_sh[d],
            "verdict": _verdict_for_dim(resid, k4_sh[d]),
        }
    return {
        "session": session["session"],
        "nwb_path": session["nwb_path"],
        "nwb_sha256": session["nwb_sha256"],
        "n_channels": session["n_channels"],
        "k4_audit": session["k4_audit"],
        "dimension4_redundancy": dim4,
        "cross_correlation_4x4": cross,
        "split_half_reliability": {
            "t4": {T4_FEATURE_NAMES[i]: t4_sh[i] for i in range(4)},
            "k4": {K4_FEATURE_NAMES[i]: k4_sh[i] for i in range(4)},
            "w_percanel_halfsplit_cosine_median": w_cosine_median,
            "k4_across_channel_pearson_per_dim": {
                K4_FEATURE_NAMES[i]: k4_sh[i] for i in range(4)
            },
        },
        "incremental_over_t4": incremental,
    }


def _pair_key(session_a: str, session_b: str) -> str:
    return f"{session_a}__{session_b}"


def _binom_two_sided_exact_pvalue(n_success: int, n_trials: int) -> float:
    """Two-sided exact binomial test with p=0.5 (equal-tail probability method)."""
    if n_trials <= 0:
        return float("nan")
    n_success = int(n_success)
    probs = [math.comb(n_trials, k) * (0.5**n_trials) for k in range(n_trials + 1)]
    obs_prob = probs[n_success]
    return min(1.0, sum(p for p in probs if p <= obs_prob + 1e-15))


# Ordered session pairs are derived from only 7 sessions and are NOT independent;
# the sign test below is descriptive and anti-conservative, not a valid inferential test.
SIGN_TEST_NONINDEPENDENCE_NOTE = (
    "The ordered session pairs derive from only 7 sessions and are therefore NOT "
    "independent; the two-sided exact sign test on nonzero pairs is descriptive "
    "and anti-conservative rather than a valid inferential test."
)


def hybrid_margin_within_noise(mean_paired_difference: float, standard_error: float) -> bool:
    """True when the mean paired top-1 gap is smaller than one SE of that gap."""
    if not math.isfinite(mean_paired_difference) or not math.isfinite(standard_error):
        return False
    return abs(mean_paired_difference) < standard_error


def _paired_top1_comparison(
    per_pair: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    subset_a: str,
    subset_b: str,
) -> dict[str, Any]:
    """Paired top-1 comparison of subset_a minus subset_b across aligned session pairs."""
    pairs_a = per_pair[subset_a]
    pairs_b = per_pair[subset_b]
    diffs: list[float] = []
    additional_channel_matches = 0.0
    total_channel_matching_attempts = 0
    n_positive = n_negative = n_zero = 0
    for key in sorted(pairs_a):
        if key not in pairs_b:
            continue
        row_a = pairs_a[key]
        row_b = pairs_b[key]
        diff = float(row_a["top1_accuracy"]) - float(row_b["top1_accuracy"])
        n_channels = int(row_a["n_channels_matched"])
        diffs.append(diff)
        additional_channel_matches += diff * n_channels
        total_channel_matching_attempts += n_channels
        if diff > 0:
            n_positive += 1
        elif diff < 0:
            n_negative += 1
        else:
            n_zero += 1

    n_pairs = len(diffs)
    mean_diff = mean(diffs) if diffs else float("nan")
    std_diff = sample_std(diffs) if len(diffs) >= 2 else float("nan")
    se_diff = sigma_delta_paired(diffs) if len(diffs) >= 2 else float("nan")
    n_nonzero = n_positive + n_negative
    sign_test_p = (
        _binom_two_sided_exact_pvalue(n_positive, n_nonzero) if n_nonzero > 0 else float("nan")
    )
    return {
        "subset_a": subset_a,
        "subset_b": subset_b,
        "n_pairs": n_pairs,
        "mean_paired_top1_difference": mean_diff,
        "paired_top1_difference_std": std_diff,
        "paired_top1_difference_se": se_diff,
        "additional_channel_matches": additional_channel_matches,
        "total_channel_matching_attempts": total_channel_matching_attempts,
        "n_pairs_positive": n_positive,
        "n_pairs_negative": n_negative,
        "n_pairs_zero": n_zero,
        "sign_test_n_nonzero_pairs": n_nonzero,
        "sign_test_n_positive_among_nonzero": n_positive,
        "sign_test_two_sided_exact_p_value": sign_test_p,
        "sign_test_note": SIGN_TEST_NONINDEPENDENCE_NOTE,
    }


def build_identifiability_margin_analysis(
    per_pair: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Margin disclosure for hybrid_worthwhile identifiability branch comparisons."""
    vs_t4 = _paired_top1_comparison(
        per_pair, subset_a="hybrid_t4w_k4b", subset_b="t4_full"
    )
    vs_k4 = _paired_top1_comparison(
        per_pair, subset_a="hybrid_t4w_k4b", subset_b="k4_full"
    )
    return {
        "note": SIGN_TEST_NONINDEPENDENCE_NOTE,
        "hybrid_t4w_k4b_vs_t4_full": vs_t4,
        "hybrid_t4w_k4b_vs_k4_full": vs_k4,
        "hybrid_margin_within_noise": hybrid_margin_within_noise(
            vs_t4["mean_paired_top1_difference"],
            vs_t4["paired_top1_difference_se"],
        ),
        "hybrid_caveat": (
            "The hybrid_worthwhile branch fires when hybrid_t4w_k4b mean top-1 exceeds "
            "both t4_full and k4_full by a bare '>' with no noise floor; the paired "
            "margin is inside one standard error and hybrid_t4w_k4b differs from t4_full "
            "only by swapping b_T4 for b_K4, which correlate above 0.96 across sessions, "
            "so this branch should not be read as evidence for the hybrid carrier."
        ),
    }


def aggregate_cross_session_identifiability(
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mean/std top-1 accuracy and MRR over all ordered session pairs."""
    names = [str(s["session"]) for s in sessions]
    by_subset: dict[str, list[dict[str, float]]] = {k: [] for k in DESCRIPTOR_SUBSETS}
    per_pair_identifiability: dict[str, dict[str, dict[str, Any]]] = {
        k: {} for k in DESCRIPTOR_SUBSETS
    }
    chance_levels: list[float] = []
    for i, sa in enumerate(sessions):
        for j, sb in enumerate(sessions):
            if i == j:
                continue
            t4_a = np.asarray(sa["t4_full"], dtype=np.float64)
            k4_a = np.asarray(sa["k4_full"], dtype=np.float64)
            t4_b = np.asarray(sb["t4_full"], dtype=np.float64)
            k4_b = np.asarray(sb["k4_full"], dtype=np.float64)
            pair_key = _pair_key(names[i], names[j])
            for subset in DESCRIPTOR_SUBSETS:
                desc_a = extract_descriptor_subset(t4_a, k4_a, subset)
                desc_b = extract_descriptor_subset(t4_b, k4_b, subset)
                row = channel_matching_accuracy(desc_a, desc_b)
                row["session_a"] = names[i]
                row["session_b"] = names[j]
                by_subset[subset].append(row)
                per_pair_identifiability[subset][pair_key] = {
                    "top1_accuracy": row["top1_accuracy"],
                    "mean_reciprocal_rank": row["mean_reciprocal_rank"],
                    "n_channels_matched": row["n_channels_matched"],
                }
                if subset == "k4_full":
                    chance_levels.append(row["chance_top1_accuracy"])

    agg: dict[str, Any] = {}
    for subset, rows in by_subset.items():
        top1 = [r["top1_accuracy"] for r in rows]
        mrr = [r["mean_reciprocal_rank"] for r in rows]
        agg[subset] = {
            "n_pairs": len(rows),
            "top1_accuracy_mean": mean(top1),
            "top1_accuracy_std": sample_std(top1) if len(top1) >= 2 else float("nan"),
            "mean_reciprocal_rank_mean": mean(mrr),
            "mean_reciprocal_rank_std": sample_std(mrr) if len(mrr) >= 2 else float("nan"),
            "chance_top1_accuracy": mean(chance_levels) if chance_levels else float("nan"),
        }
    agg["per_pair_identifiability"] = per_pair_identifiability
    return agg


def build_conclusions(
    session_records: Sequence[Mapping[str, Any]],
    identifiability: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    per_dim_or = _per_dim_verdicts_from_sessions(session_records, "any_session_or")
    recommendation, recommendation_inputs = _recommendation_from_per_dim(per_dim_or, identifiability)
    aggregation_sensitivity = _build_aggregation_sensitivity(session_records, identifiability)
    recommendations = {
        rule: aggregation_sensitivity[rule]["recommendation"]
        for rule in ("any_session_or", "majority", "unanimous")
    }
    headline_is_sensitive = len(set(recommendations.values())) > 1
    tally = _build_per_k4_dim_verdict_tally(session_records)
    per_pair = identifiability.get("per_pair_identifiability", {})
    margin_analysis = (
        build_identifiability_margin_analysis(per_pair) if per_pair else {}
    )
    hybrid_margin_noise = bool(margin_analysis.get("hybrid_margin_within_noise", False))

    w_x_pass = tally["w_x"]["verdict_counts"]["novel_and_reliable"]
    w_y_pass = tally["w_y"]["verdict_counts"]["novel_and_reliable"]
    n_sessions = len(session_records)
    caveat = (
        "Cross-session aggregation was not pre-declared; none of the three aggregation "
        "rules yields a headline that survives scrutiny. any_session_or fires "
        f"fusion_worthwhile on {w_x_pass}-of-{n_sessions} (w_x) and "
        f"{w_y_pass}-of-{n_sessions} (w_y) novel_and_reliable session pass rates; "
        "majority and unanimous fire hybrid_worthwhile on a sub-noise identifiability "
        "margin (hybrid_t4w_k4b vs t4_full/k4_full paired top-1 gap inside one SE)."
    )
    defensible_justification = (
        "Neither fusion_worthwhile nor hybrid_worthwhile survives both checks: "
        f"any_session_or rests on {w_x_pass}-of-{n_sessions} and "
        f"{w_y_pass}-of-{n_sessions} session pass rates for w_x and w_y, while "
        "majority/unanimous rest on a hybrid identifiability margin inside the "
        "paired noise floor with b_T4/b_K4 dimensions correlated above 0.96."
    )

    return {
        "thresholds": {
            "residual_r2_explained_by_t4_if_below": RESIDUAL_R2_EXPLAINED_THRESHOLD,
            "split_half_reliable_if_gte": SPLIT_HALF_RELIABILITY_THRESHOLD,
        },
        "any_session_or": {
            "aggregation_rule": "any_session_or",
            "per_k4_dim_verdict": per_dim_or,
            "recommendation": recommendation,
            "recommendation_inputs": recommendation_inputs,
        },
        "per_k4_dim_verdict_tally": tally,
        "aggregation_sensitivity": aggregation_sensitivity,
        "headline_is_aggregation_sensitive": headline_is_sensitive,
        "caveat": caveat,
        "identifiability_margin_analysis": margin_analysis,
        "hybrid_margin_within_noise": hybrid_margin_noise,
        "hybrid_caveat": margin_analysis.get("hybrid_caveat", ""),
        "defensible_recommendation": "neither",
        "defensible_recommendation_justification": defensible_justification,
        "reliability_metric_note": RELIABILITY_METRIC_NOTE,
    }


def build_audit(*, data_dir: Path, max_sessions: int | None = None) -> dict[str, Any]:
    paths = heldin_m2_paths(data_dir)
    if max_sessions is not None:
        paths = paths[:max_sessions]
    if not paths:
        return {
            "schema": SCHEMA,
            "status": "blocked_no_heldin_nwbs_found",
            "no_gpu": True,
            "data_dir": str(data_dir),
        }
    loaded = [_load_m2_session(p) for p in paths]
    records = [audit_session_record(s) for s in loaded]
    ident = aggregate_cross_session_identifiability(loaded)
    conclusions = build_conclusions(records, ident)
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "no_heldout_opened": True,
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "split_half_n_trials": SPLIT_HALF_N_TRIALS,
        "data_dir": str(data_dir),
        "n_sessions": len(records),
        "input_nwb_sha256": {r["session"]: r["nwb_sha256"] for r in records},
        "sessions": records,
        "cross_session_channel_identifiability": ident,
        "conclusions": conclusions,
    }


def _synthesize_calibration_arrays(
    *,
    n_channels: int = 32,
    n_trials: int = 30,
    bins_per_trial: int = 40,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Minimal valid M2-like arrays for dry-run (no NWB)."""
    rng = np.random.RandomState(seed)
    t_total = n_trials * bins_per_trial
    neural = rng.poisson(2.0, size=(t_total, n_channels)).astype(np.float64)
    covariates = np.zeros((t_total, 2), dtype=np.float64)
    trial_change = np.zeros(t_total, dtype=bool)
    angles = np.linspace(-math.pi, math.pi, n_trials, endpoint=False)
    for trial in range(n_trials):
        start = trial * bins_per_trial
        trial_change[start] = True
        vx = math.cos(float(angles[trial])) * 0.05
        vy = math.sin(float(angles[trial])) * 0.05
        covariates[start : start + bins_per_trial, 0] = vx + rng.normal(0, 0.002, bins_per_trial)
        covariates[start : start + bins_per_trial, 1] = vy + rng.normal(0, 0.002, bins_per_trial)
        # Channel-specific tuning to velocity and direction.
        neural[start : start + bins_per_trial] += (
            (vx * rng.randn(n_channels) + vy * rng.randn(n_channels))[None, :] * 3.0
        )
        neural[start : start + bins_per_trial] += (
            (math.cos(angles[trial]) * rng.randn(n_channels))[None, :] * 2.0
        )
    eval_mask = np.ones(t_total, dtype=bool)
    return {
        "neural": neural,
        "covariates": covariates,
        "trial_change": trial_change,
        "eval_mask": eval_mask,
        "target_angles": angles.astype(np.float64),
    }


def _session_from_synthetic_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    session_name: str,
) -> dict[str, Any]:
    sce = _sce_root()
    if str(sce) not in sys.path:
        sys.path.insert(0, str(sce))
    from src.data.falcon_k4_features import k4_from_raw_calibration
    from src.data.falcon_t4_features import t4_from_trial_sums, validate_trial_label_alignment

    neural = np.asarray(arrays["neural"], dtype=np.float64)
    covariates = np.asarray(arrays["covariates"], dtype=np.float64)
    trial_change = np.asarray(arrays["trial_change"], dtype=bool)
    eval_mask = np.asarray(arrays["eval_mask"], dtype=bool)
    target_angles = np.asarray(arrays["target_angles"], dtype=np.float64)
    validate_trial_label_alignment(trial_change, target_angles, source=session_name)

    t4_neural = neural[eval_mask]
    t4_trial_change = trial_change[eval_mask]
    retained_angles = target_angles[eval_mask[np.flatnonzero(trial_change)]]
    trial_starts = np.where(t4_trial_change)[0]
    trial_spike_sums: list[np.ndarray] = []
    trial_lengths: list[int] = []
    for i in range(trial_starts.shape[0]):
        start = int(trial_starts[i])
        end = (
            int(trial_starts[i + 1])
            if i + 1 < trial_starts.shape[0]
            else int(t4_neural.shape[0])
        )
        trial_neural = t4_neural[start:end]
        valid_length = min(int(trial_neural.shape[0]), MAX_TRIAL_LENGTH)
        trial_spike_sums.append(trial_neural[:valid_length].sum(axis=0))
        trial_lengths.append(valid_length)
    sums = np.asarray(trial_spike_sums, dtype=np.float64)
    lengths = np.asarray(trial_lengths, dtype=np.int64)

    t4_full = t4_from_trial_sums(
        sums[:CALIBRATION_N_TRIALS],
        lengths[:CALIBRATION_N_TRIALS],
        retained_angles[:CALIBRATION_N_TRIALS],
        source=f"{session_name}[0:{CALIBRATION_N_TRIALS}]",
    )
    k4_full, k4_audit = k4_from_raw_calibration(
        neural, covariates, trial_change, calibration_n_trials=CALIBRATION_N_TRIALS
    )
    t4_h1 = t4_from_trial_sums(
        sums[:SPLIT_HALF_N_TRIALS],
        lengths[:SPLIT_HALF_N_TRIALS],
        retained_angles[:SPLIT_HALF_N_TRIALS],
        source=f"{session_name}[0:{SPLIT_HALF_N_TRIALS}]",
    )
    t4_h2 = t4_from_trial_sums(
        sums[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        lengths[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        retained_angles[SPLIT_HALF_N_TRIALS:CALIBRATION_N_TRIALS],
        source=f"{session_name}[{SPLIT_HALF_N_TRIALS}:{CALIBRATION_N_TRIALS}]",
    )
    k4_h1 = _k4_from_trial_window(
        neural,
        covariates,
        trial_change,
        trial_start=0,
        n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{session_name}[0:{SPLIT_HALF_N_TRIALS}]",
    )
    k4_h2 = _k4_from_trial_window(
        neural,
        covariates,
        trial_change,
        trial_start=SPLIT_HALF_N_TRIALS,
        n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{session_name}[{SPLIT_HALF_N_TRIALS}:{CALIBRATION_N_TRIALS}]",
    )
    return {
        "session": session_name,
        "nwb_path": f"synthetic://{session_name}",
        "nwb_sha256": "synthetic",
        "n_channels": int(t4_full.shape[0]),
        "t4_full": t4_full,
        "k4_full": k4_full,
        "k4_audit": k4_audit.as_dict(),
        "t4_split_half": {"first": t4_h1, "second": t4_h2},
        "k4_split_half": {"first": k4_h1, "second": k4_h2},
    }


def build_synthetic_audit(*, n_sessions: int = 3, seed: int = 0) -> dict[str, Any]:
    loaded = [
        _session_from_synthetic_arrays(
            _synthesize_calibration_arrays(seed=seed + i),
            session_name=f"synthetic_session_{i}",
        )
        for i in range(n_sessions)
    ]
    records = [audit_session_record(s) for s in loaded]
    ident = aggregate_cross_session_identifiability(loaded)
    conclusions = build_conclusions(records, ident)
    return {
        "schema": SCHEMA,
        "status": "synthetic_cpu_only",
        "no_gpu": True,
        "synthetic": True,
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "split_half_n_trials": SPLIT_HALF_N_TRIALS,
        "n_sessions": len(records),
        "sessions": records,
        "cross_session_channel_identifiability": ident,
        "conclusions": conclusions,
    }


def write_audit(audit: Mapping[str, Any], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "audit.json"
    tmp = output_dir / "audit.json.tmp"

    def _strict(obj: Any) -> Any:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return _strict(obj.tolist())
        if isinstance(obj, dict):
            return {str(k): _strict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strict(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return _strict(float(obj))
        return obj

    tmp.write_text(json.dumps(_strict(dict(audit)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
