"""Pure laws for the M2 chrono4 strict-caliber early-start cell.

No data, no checkpoint, no CUDA: the chronological-first-4 selection law, the
D-opt scatter statistics, the paired selection-contribution law, the anchor
matchers and the pre-registered verdict classes (with the 1e-12 boundary band).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core

from . import plan


class Chrono4LawError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Chrono4LawError(message)


# ---------------------------------------------------------------------------
# The chronological-first-4 initialization law.
# ---------------------------------------------------------------------------


def chrono4_support(theta_first30: Any) -> dict[str, Any]:
    """The ONE support law of this cell (both families bind it identically).

    Returns the four support positions (the first four finite-direction trials
    of the first-30 pool), the strict-branch disclosure (how many of positions
    0-3 are directional), and the earliest decodable trial.
    """
    theta = np.ascontiguousarray(np.asarray(theta_first30, dtype=np.float64).reshape(-1))
    _require(theta.size >= plan.ACTIVITY_HORIZON,
             "chrono4 needs the first-30 angle pool")
    pool = theta[: plan.ACTIVITY_HORIZON]
    finite = np.flatnonzero(np.isfinite(pool)).astype(np.int64)
    _require(finite.size >= plan.M4,
             "the first-30 pool holds fewer than four directional trials")
    support = np.ascontiguousarray(finite[: plan.M4], dtype=np.int64)
    strict_directional = int(np.isfinite(pool[: plan.M4]).sum())
    last_position = int(support[-1])
    return {
        "selection": "chronological_first4_directional",
        "support_positions": support.tolist(),
        "directional_among_positions_0_to_3": strict_directional,
        "strict_branch_taken": bool(strict_directional == plan.M4),
        "fallback_branch_taken": bool(strict_directional != plan.M4),
        "last_support_position_0based": last_position,
        "earliest_decode_position_0based": last_position + 1,
        "earliest_decode_trial_1based": last_position + 2,
        "pool_depth_required_trials": last_position + 1,
    }


def dopt_support_indices(theta_first30: Any) -> np.ndarray:
    """The sealed M4 D-opt law, recomputed (the receipts' selection proof)."""
    theta = np.ascontiguousarray(np.asarray(theta_first30, dtype=np.float64).reshape(-1))
    _require(theta.size >= plan.ACTIVITY_HORIZON,
             "D-opt proof needs the first-30 angle pool")
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    candidates = np.flatnonzero(np.isfinite(theta[: plan.ACTIVITY_HORIZON])).astype(np.int64)
    _require(candidates.size >= plan.M4, "fewer than four directional first-30 candidates")
    local = greedy_forward_d_optimal_indices(theta[candidates], plan.M4)
    selected = np.sort(candidates[np.asarray(local, dtype=np.int64)]).astype(np.int64, copy=False)
    _require(selected.size == plan.M4 and int(selected.max()) < plan.ACTIVITY_HORIZON,
             "M4 D-opt selection drift")
    return np.ascontiguousarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# The angular scatter statistics (the selection-law evidence).
# ---------------------------------------------------------------------------


def scatter_statistics(theta_support: Any) -> dict[str, Any]:
    """Angular spread of a four-trial support: snapped canonical directions,
    distinct-direction count, and pairwise separations (the D-opt objective's
    own geometry)."""
    values = np.ascontiguousarray(np.asarray(theta_support, dtype=np.float64).reshape(-1))
    _require(values.size == plan.M4 and bool(np.isfinite(values).all()),
             "scatter statistics need exactly four finite directions")
    snapped = [cdm_core.nearest_canonical_direction(float(item))[0] for item in values]
    separations: list[float] = []
    for left in range(plan.M4):
        for right in range(left + 1, plan.M4):
            separations.append(float(cdm_core.circular_distance(
                float(values[left]), float(values[right]))))
    return {
        "angles_rad": [float(item) for item in values],
        "canonical_direction_indices": [int(item) for item in snapped],
        "canonical_directions_rad": [
            float(cdm_core.CANONICAL_DIRECTIONS_RAD[index]) for index in snapped
        ],
        "distinct_canonical_directions": int(len(set(snapped))),
        "min_pairwise_angular_separation_rad": float(min(separations)),
        "mean_pairwise_angular_separation_rad": float(
            sum(separations) / len(separations)),
        "max_pairwise_angular_separation_rad": float(max(separations)),
    }


def index_scatter(support: Any) -> dict[str, Any]:
    positions = np.asarray(support, dtype=np.int64).reshape(-1)
    _require(positions.size == plan.M4, "index scatter needs four positions")
    return {
        "support_positions": [int(item) for item in positions],
        "index_span": int(positions.max() - positions.min()),
        "max_position": int(positions.max()),
        "mean_position": float(positions.mean()),
    }


# ---------------------------------------------------------------------------
# Paired selection contribution and the verdict classes.
# ---------------------------------------------------------------------------


def equal_session_mean(session_values: Mapping[str, float]) -> float:
    _require(bool(session_values), "equal-session mean needs sessions")
    values = [float(item) for item in session_values.values()]
    return float(sum(values) / len(values))


def session_sd(session_values: Mapping[str, float]) -> float:
    _require(bool(session_values), "session sd needs sessions")
    values = [float(item) for item in session_values.values()]
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((item - mean) ** 2 for item in values) / len(values)))


def paired_selection_contribution(
    dopt_sessions: Mapping[str, float], chrono4_sessions: Mapping[str, float],
) -> dict[str, Any]:
    """sealed D-opt cell minus CHRONO4 cell, per session (positive = D-opt better)."""
    _require(set(dopt_sessions) == set(chrono4_sessions) and bool(dopt_sessions),
             "paired session set mismatch")
    deltas = {
        name: float(dopt_sessions[name]) - float(chrono4_sessions[name])
        for name in sorted(dopt_sessions)
    }
    values = [deltas[name] for name in sorted(deltas)]
    mean = float(sum(values) / len(values))
    return {
        "expression": "sealed_dopt_cell_minus_chrono4_cell",
        "dopt_equal_session_mean": equal_session_mean(dopt_sessions),
        "chrono4_equal_session_mean": equal_session_mean(chrono4_sessions),
        "equal_session_mean_delta": mean,
        "delta_sd": float(math.sqrt(
            sum((item - mean) ** 2 for item in values) / len(values))),
        "positive_sessions_dopt_better": int(sum(1 for item in values if item > 0.0)),
        "negative_sessions_chrono4_better": int(sum(1 for item in values if item < 0.0)),
        "tied_sessions": int(sum(1 for item in values if item == 0.0)),
        "session_count": int(len(values)),
        "breadth_denominator": int(len(values)),
        "per_session_delta": deltas,
    }


def _boundary_band(value: float, threshold: float, *, epsilon: float) -> dict[str, Any]:
    """Exact comparison with the 1e-12 disclosure band (Stage-O/P convention)."""
    within = abs(float(value) - float(threshold)) <= float(epsilon)
    return {
        "value": float(value),
        "threshold": float(threshold),
        "within_epsilon_band_of_boundary": bool(within),
        "boundary_epsilon": float(epsilon),
    }


def verdict_class(delta: float, *, epsilon: float | None = None) -> dict[str, Any]:
    """The pre-registered class of one selection-contribution delta.

    LARGE  delta > 0.02;  MODERATE 0.005 <= delta <= 0.02;  SMALL delta < 0.005.
    A delta inside the 1e-12 band STRICTLY BELOW a boundary is pulled UP into
    the higher class with an explicit band disclosure and never silently flips;
    a delta exactly ON a boundary follows its inclusive pre-registered class
    (0.02 is MODERATE, 0.005 is MODERATE).
    """
    epsilon = float(plan.VERDICT_LAW["boundary_epsilon"]) if epsilon is None else float(epsilon)
    value = float(delta)
    large_threshold = float(plan.VERDICT_LAW["thresholds"]["large_strictly_above"])
    moderate_floor = float(plan.VERDICT_LAW["thresholds"]["moderate_floor"])
    meets_large = (value > large_threshold) or (
        large_threshold - epsilon <= value < large_threshold
    )
    meets_moderate = (value >= moderate_floor) or (
        moderate_floor - epsilon <= value < moderate_floor
    )
    if meets_large:
        label = "SELECTION_CONTRIBUTION_LARGE"
    elif meets_moderate:
        label = "SELECTION_CONTRIBUTION_MODERATE"
    else:
        label = "SELECTION_CONTRIBUTION_SMALL"
    return {
        "delta": value,
        "verdict": label,
        "meets_large_threshold": bool(meets_large),
        "meets_moderate_floor": bool(meets_moderate),
        "large_threshold": large_threshold,
        "moderate_floor": moderate_floor,
        "boundary_bands": {
            "large": _boundary_band(value, large_threshold, epsilon=epsilon),
            "moderate": _boundary_band(value, moderate_floor, epsilon=epsilon),
        },
    }


# ---------------------------------------------------------------------------
# Anchor matchers.
# ---------------------------------------------------------------------------


def pure_data_pairing(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any]) -> dict[str, Any]:
    """CHRONO4 row vs sealed D-opt row: pure-data fields must match exactly.

    The sealed starts digest is read under either house spelling (the
    comparator's ``ordered_window_starts_sha256`` or the CDM screen's
    ``query_starts_sha256``); both digest the same int64 ordered-starts array.
    """
    sealed_starts = sealed_row.get(
        "ordered_window_starts_sha256", sealed_row.get("query_starts_sha256"))
    matches = {
        "window_count": int(cell_row["window_count"]) == int(sealed_row["window_count"]),
        "ordered_window_starts_sha256": (
            str(cell_row["ordered_window_starts_sha256"]) == str(sealed_starts)),
        "target_sha256": (
            str(cell_row["target_sha256"]) == str(sealed_row["target_sha256"])),
    }
    return {
        "field_matches": matches,
        "exact_match": all(bool(item) for item in matches.values()),
    }


def fidelity_anchor(
    cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any], *, r2_tolerance: float,
) -> dict[str, Any]:
    """Executor fidelity under the SEALED D-opt support: exact pure data, R2 in
    tolerance (CPU vs the sealed GPU receipt)."""
    pairing = pure_data_pairing(cell_row, sealed_row)
    r2_delta = float(cell_row["r2"]) - float(sealed_row["r2"])
    return {
        **pairing,
        "r2_cpu": float(cell_row["r2"]),
        "r2_sealed_gpu": float(sealed_row["r2"]),
        "r2_delta": r2_delta,
        "r2_within_tolerance": abs(r2_delta) <= float(r2_tolerance),
        "exact_match": bool(
            pairing["exact_match"] and abs(r2_delta) <= float(r2_tolerance)),
    }


def dopt_proof(recomputed: Any, sealed_indices: Any) -> dict[str, Any]:
    """The sealed receipt's M4 indices must equal the recomputed D-opt law."""
    left = np.asarray(recomputed, dtype=np.int64).reshape(-1).tolist()
    right = [int(item) for item in np.asarray(sealed_indices, dtype=np.int64).reshape(-1)]
    return {
        "recomputed_dopt_indices": left,
        "sealed_indices": right,
        "exact_match": left == right,
    }
