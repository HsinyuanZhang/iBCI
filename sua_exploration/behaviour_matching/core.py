"""Pure-NumPy estimator for cross-session behavioural nearest-neighbour matching.

Frozen protocol: ``sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_PROTOCOL_20260814.md``.

Nothing here opens an NWB, imports Torch, or touches a GPU, so the test suite can
exercise every statistical guard on synthetic arrays. File discovery and NWB reading
live in ``sua_exploration.behaviour_matching.loaders``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "behaviour_matching_feasibility_v1"
PROTOCOL_DOCUMENT = "sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_PROTOCOL_20260814.md"

# Frozen section 4.
GLOBAL_SEED = 20260814
M_QUERY = 512
M_REF = 1024
MIN_POOL = 16

# Frozen section 5.
SD_FLOOR = 1.0e-8

# Frozen section 8.
RATIO_TIGHT_MEDIAN_MAX = 1.25
RATIO_TIGHT_P90_MAX = 1.50
RATIO_MARGINAL_MEDIAN_MAX = 2.00
ABS_TIGHT_MAX = 0.10
ABS_MARGINAL_MAX = 0.25

# Frozen section 10.
SWEEP_REF_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MAX_SWEEP_PAIRS = 40
MIN_SWEEP_POINTS = 4
SCALING_TOLERANCE_FACTOR = 1.5
COMMON_REF_SIZE = 16

# Centre-out canonical target directions (multiples of pi/4).
N_CANONICAL_DIRECTIONS = 8
PHASE_QUANTILES = 5


class BehaviourMatchingError(ValueError):
    """A protocol invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BehaviourMatchingError(message)


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def derived_seed(*parts: str) -> int:
    """Deterministic uint64 seed from the global seed and stable string parts."""
    payload = "|".join((str(GLOBAL_SEED), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# section 4: reference / query construction
# ---------------------------------------------------------------------------


def chronological_halves(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference pool = chronologically first half, query pool = second half."""
    require(n_samples >= 2, f"need at least two samples to split, got {n_samples}")
    cut = n_samples // 2
    return np.arange(0, cut, dtype=np.int64), np.arange(cut, n_samples, dtype=np.int64)


def subsample(pool: np.ndarray, size: int, *, cohort: str, session: str, role: str) -> np.ndarray:
    """Seeded draw without replacement, truncated to the pool when the pool is smaller.

    Implemented as a prefix of one seeded permutation of the pool, so that for a fixed
    ``(cohort, session, role)`` the draws at different sizes are nested. That makes the
    protocol section 10 size sweep monotone in the reference set rather than re-randomised
    at every point, without changing the marginal law of any individual draw.
    """
    pool = np.asarray(pool, dtype=np.int64)
    require(pool.ndim == 1 and pool.size > 0, "pool must be a non-empty 1-D index array")
    require(size > 0, "subsample size must be positive")
    if pool.size <= size:
        return np.sort(pool)
    rng = np.random.default_rng(derived_seed(cohort, session, role))
    return np.sort(rng.permutation(pool)[:size])


# ---------------------------------------------------------------------------
# section 5: normalization
# ---------------------------------------------------------------------------


def grouped_zscore_stats(
    matrices: Sequence[np.ndarray],
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cohort-pooled mean/sd per normalization group, broadcast back to every dimension.

    ``groups[j]`` names the normalization group of dimension ``j``; all dimensions in a
    group share one mean and one sd (for ``win100`` the group is the velocity channel, so
    the 50 lags of a channel are not individually rescaled).
    """
    groups = np.asarray(groups, dtype=np.int64)
    require(len(matrices) > 0, "need at least one matrix to pool normalization statistics")
    dim = int(groups.size)
    for matrix in matrices:
        require(matrix.ndim == 2 and matrix.shape[1] == dim, f"matrix shape {matrix.shape} does not match d={dim}")
    stacked = np.concatenate([np.asarray(m, dtype=np.float64) for m in matrices], axis=0)
    require(np.isfinite(stacked).all(), "pooled behaviour contains non-finite values")
    mean = np.empty(dim, dtype=np.float64)
    sd = np.empty(dim, dtype=np.float64)
    for group_id in np.unique(groups):
        columns = np.flatnonzero(groups == group_id)
        values = stacked[:, columns].reshape(-1)
        group_mean = float(values.mean())
        group_sd = float(values.std())
        require(
            math.isfinite(group_sd) and group_sd >= SD_FLOOR,
            f"normalization group {int(group_id)} has sd {group_sd} below floor {SD_FLOOR}",
        )
        mean[columns] = group_mean
        sd[columns] = group_sd
    return mean, sd


def apply_zscore(matrix: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    values = (np.asarray(matrix, dtype=np.float64) - mean[None, :]) / sd[None, :]
    require(np.isfinite(values).all(), "z-scored behaviour contains non-finite values")
    return values


# ---------------------------------------------------------------------------
# nearest-neighbour distances
# ---------------------------------------------------------------------------


def nn_distances(query: np.ndarray, reference: np.ndarray, *, block: int = 4096) -> np.ndarray:
    """Euclidean distance from every query row to its nearest reference row."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    require(q.ndim == 2 and r.ndim == 2 and q.shape[1] == r.shape[1], "query/reference dimension mismatch")
    require(q.shape[0] > 0 and r.shape[0] > 0, "query and reference must be non-empty")
    q_sq = np.einsum("ij,ij->i", q, q)
    best = np.full(q.shape[0], np.inf, dtype=np.float64)
    for start in range(0, r.shape[0], block):
        chunk = r[start : start + block]
        d2 = q_sq[:, None] + np.einsum("ij,ij->i", chunk, chunk)[None, :] - 2.0 * (q @ chunk.T)
        np.minimum(best, d2.min(axis=1), out=best)
    np.maximum(best, 0.0, out=best)
    result = np.sqrt(best)
    require(np.isfinite(result).all(), "nearest-neighbour distances are not finite")
    return result


def stratum_labels(scalar: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Assign each query to a movement-magnitude stratum by fixed cohort-wide cut points."""
    values = np.asarray(scalar, dtype=np.float64).reshape(-1)
    edges = np.asarray(list(thresholds), dtype=np.float64)
    require(edges.ndim == 1 and edges.size >= 1, "need at least one stratum threshold")
    require(bool(np.all(np.diff(edges) > 0)), "stratum thresholds must be strictly increasing")
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def stratified_medians(
    query: np.ndarray,
    strata: np.ndarray,
    reference: np.ndarray,
    *,
    dim: int,
    n_strata: int,
) -> list[dict[str, Any]]:
    """Median NN distance per query stratum against the full, unstratified reference set."""
    strata = np.asarray(strata, dtype=np.int64).reshape(-1)
    require(strata.size == query.shape[0], "stratum label count mismatch")
    rows: list[dict[str, Any]] = []
    for index in range(n_strata):
        selected = np.flatnonzero(strata == index)
        if selected.size == 0:
            rows.append({"stratum": index, "count": 0, "median": None, "median_per_dim": None})
            continue
        median = float(np.median(nn_distances(query[selected], reference)))
        rows.append(
            {
                "stratum": index,
                "count": int(selected.size),
                "median": median,
                "median_per_dim": median / math.sqrt(dim),
            }
        )
    return rows


def centroid_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from every query row to the reference mean: the no-matching residual."""
    q = np.asarray(query, dtype=np.float64)
    centre = np.asarray(reference, dtype=np.float64).mean(axis=0)
    return np.linalg.norm(q - centre[None, :], axis=1)


def class_restricted_nn(
    query: np.ndarray,
    query_labels: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """NN distance restricted to reference rows sharing the query's discrete class.

    Returns the distances for queries whose class occurs in the reference set, and the
    fraction of queries for which such a reference exists (the discrete match rate).

    A negative label marks a sample with no usable discrete key. Such a query can never be
    paired by a key-driven mechanism, so it is counted against the match rate; such a
    reference is never offered as a partner.
    """
    query_labels = np.asarray(query_labels)
    reference_labels = np.asarray(reference_labels)
    require(query_labels.shape[0] == query.shape[0], "query label count mismatch")
    require(reference_labels.shape[0] == reference.shape[0], "reference label count mismatch")
    distances: list[np.ndarray] = []
    matched = 0
    for label in np.unique(query_labels):
        if label < 0:
            continue
        q_rows = np.flatnonzero(query_labels == label)
        r_rows = np.flatnonzero(reference_labels == label)
        if r_rows.size == 0:
            continue
        matched += int(q_rows.size)
        distances.append(nn_distances(query[q_rows], reference[r_rows]))
    match_rate = float(matched) / float(query_labels.shape[0])
    if not distances:
        return np.zeros(0, dtype=np.float64), match_rate
    return np.concatenate(distances), match_rate


# ---------------------------------------------------------------------------
# section 6: summary statistics
# ---------------------------------------------------------------------------


def distance_summary(values: np.ndarray, *, dim: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "distance summary requires at least one value")
    q25, median, q75, p90 = (float(v) for v in np.percentile(array, [25, 50, 75, 90]))
    return {
        "count": int(array.size),
        "median": median,
        "mean": float(array.mean()),
        "iqr": q75 - q25,
        "p25": q25,
        "p75": q75,
        "p90": p90,
        "max": float(array.max()),
        "median_per_dim": median / math.sqrt(dim),
        "p90_per_dim": p90 / math.sqrt(dim),
    }


def spread_over_pairs(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0, "pair spread requires at least one pair")
    require(np.isfinite(array).all(), "pair statistic contains non-finite values")
    q25, median, q75, p90 = (float(v) for v in np.percentile(array, [25, 50, 75, 90]))
    return {
        "pairs": int(array.size),
        "median": median,
        "mean": float(array.mean()),
        "iqr": q75 - q25,
        "p25": q25,
        "p75": q75,
        "p90": p90,
        "min": float(array.min()),
        "max": float(array.max()),
    }


# ---------------------------------------------------------------------------
# section 8: decision rule
# ---------------------------------------------------------------------------


def classify_ratio(median_ratio: float, p90_ratio: float) -> str:
    if median_ratio <= RATIO_TIGHT_MEDIAN_MAX and p90_ratio <= RATIO_TIGHT_P90_MAX:
        return "TIGHT"
    if median_ratio <= RATIO_MARGINAL_MEDIAN_MAX:
        return "MARGINAL"
    return "LOOSE"


def classify_absolute(median_per_dim: float) -> str:
    if median_per_dim <= ABS_TIGHT_MAX:
        return "ABS_TIGHT"
    if median_per_dim <= ABS_MARGINAL_MAX:
        return "ABS_MARGINAL"
    return "ABS_LOOSE"


def combined_verdict(ratio_verdict: str, absolute_verdict: str) -> dict[str, Any]:
    supports = ratio_verdict == "TIGHT" and absolute_verdict in {"ABS_TIGHT", "ABS_MARGINAL"}
    failing: list[str] = []
    if ratio_verdict != "TIGHT":
        failing.append(f"ratio_vs_within_session_floor={ratio_verdict}")
    if absolute_verdict == "ABS_LOOSE":
        failing.append(f"absolute_residual={absolute_verdict}")
    return {
        "ratio_verdict": ratio_verdict,
        "absolute_verdict": absolute_verdict,
        "supports_behaviour_matched_pairing": bool(supports),
        "failing_criteria": failing,
    }


# ---------------------------------------------------------------------------
# section 9: effective dimensionality
# ---------------------------------------------------------------------------


def dimensionality_report(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] > 1, f"dimensionality needs [n>1, d], got {values.shape}")
    centred = values - values.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2 / float(values.shape[0] - 1)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    require(total > 0.0, "behaviour covariance has zero total variance")
    participation = float(total**2 / float((eigenvalues**2).sum()))
    cumulative = np.cumsum(eigenvalues) / total
    thresholds = {}
    for level in (0.80, 0.90, 0.95, 0.99):
        thresholds[f"components_for_{int(level * 100)}pct"] = int(np.searchsorted(cumulative, level) + 1)
    return {
        "nominal_dim": int(values.shape[1]),
        "samples": int(values.shape[0]),
        "participation_ratio": participation,
        "eigenvalue_fractions": [float(v) for v in (eigenvalues / total)[:16].tolist()],
        "cumulative_variance_first_16": [float(v) for v in cumulative[:16].tolist()],
        **thresholds,
    }


# ---------------------------------------------------------------------------
# section 10: -1/d scaling
# ---------------------------------------------------------------------------


def fit_log_slope(sizes: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    x = np.log(np.asarray(list(sizes), dtype=np.float64))
    y = np.log(np.asarray(list(values), dtype=np.float64))
    require(x.size == y.size and x.size >= 2, "slope fit needs at least two matched points")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "slope fit inputs are not finite")
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    residual = y - (slope * x + intercept)
    denominator = float(((y - y.mean()) ** 2).sum())
    r_squared = float(1.0 - (residual**2).sum() / denominator) if denominator > 0.0 else float("nan")
    return {
        "points": int(x.size),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "implied_d_eff": float(-1.0 / slope) if slope < 0.0 else None,
    }


def classify_scaling(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`-1/d` holds when the measured d_eff tracks the participation ratio."""
    usable = [row for row in rows if row.get("fit", {}).get("points", 0) >= MIN_SWEEP_POINTS]
    if not usable:
        return {"verdict": "UNTESTABLE", "tolerance_factor": SCALING_TOLERANCE_FACTOR, "checked": []}
    checked = []
    for row in usable:
        d_eff = row["fit"].get("implied_d_eff")
        participation = float(row["participation_ratio"])
        agrees = (
            d_eff is not None
            and participation > 0.0
            and (1.0 / SCALING_TOLERANCE_FACTOR) <= (d_eff / participation) <= SCALING_TOLERANCE_FACTOR
        )
        checked.append(
            {
                "cohort": row["cohort"],
                "representation": row["representation"],
                "nominal_dim": int(row["nominal_dim"]),
                "participation_ratio": participation,
                "implied_d_eff": d_eff,
                "d_eff_over_participation_ratio": (float(d_eff / participation) if d_eff is not None else None),
                "agrees_within_tolerance": bool(agrees),
            }
        )
    agreeing = sum(1 for row in checked if row["agrees_within_tolerance"])
    if agreeing == len(checked):
        verdict = "HOLDS"
    elif agreeing > 0:
        verdict = "PARTIAL"
    else:
        verdict = "FAILS"
    return {
        "verdict": verdict,
        "tolerance_factor": SCALING_TOLERANCE_FACTOR,
        "agreeing": int(agreeing),
        "checked_total": int(len(checked)),
        "checked": checked,
    }


# ---------------------------------------------------------------------------
# per-pair evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionView:
    """One session's z-scored samples for one representation, already split and drawn."""

    cohort: str
    session: str
    representation: str
    dim: int
    query: np.ndarray
    reference: np.ndarray
    query_labels: Mapping[str, np.ndarray]
    reference_labels: Mapping[str, np.ndarray]
    query_magnitude: np.ndarray | None = None

    def usable(self) -> bool:
        return self.query.shape[0] >= MIN_POOL and self.reference.shape[0] >= MIN_POOL


def evaluate_pair(
    query_view: SessionView,
    reference_view: SessionView,
    *,
    within_floor_median: float,
    discrete_keys: Sequence[str] = (),
) -> dict[str, Any]:
    require(query_view.representation == reference_view.representation, "representation mismatch across pair")
    require(query_view.dim == reference_view.dim, "dimension mismatch across pair")
    require(within_floor_median > 0.0, "within-session floor must be positive")
    cross = nn_distances(query_view.query, reference_view.reference)
    summary = distance_summary(cross, dim=query_view.dim)
    centroid = distance_summary(
        centroid_distances(query_view.query, reference_view.reference), dim=query_view.dim
    )
    row: dict[str, Any] = {
        "cohort": query_view.cohort,
        "representation": query_view.representation,
        "query_session": query_view.session,
        "reference_session": reference_view.session,
        "dim": int(query_view.dim),
        "query_size": int(query_view.query.shape[0]),
        "reference_size": int(reference_view.reference.shape[0]),
        "cross": summary,
        "centroid": centroid,
        "within_floor_median": float(within_floor_median),
        "ratio_cross_over_within": float(summary["median"] / within_floor_median),
        "cross_over_centroid": float(summary["median"] / centroid["median"]),
    }
    for key in discrete_keys:
        restricted, match_rate = class_restricted_nn(
            query_view.query,
            query_view.query_labels[key],
            reference_view.reference,
            reference_view.reference_labels[key],
        )
        entry: dict[str, Any] = {"discrete_match_rate": match_rate}
        if restricted.size > 0:
            entry["class_restricted"] = distance_summary(restricted, dim=query_view.dim)
        row[f"discrete_{key}"] = entry
    return row


def within_session_floor(view: SessionView) -> dict[str, Any]:
    distances = nn_distances(view.query, view.reference)
    summary = distance_summary(distances, dim=view.dim)
    return {
        "cohort": view.cohort,
        "representation": view.representation,
        "session": view.session,
        "dim": int(view.dim),
        "query_size": int(view.query.shape[0]),
        "reference_size": int(view.reference.shape[0]),
        "floor": summary,
    }
