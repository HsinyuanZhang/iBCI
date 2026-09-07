"""Pure laws for the M2 labeled-pair k-curve: no data, no checkpoint, no CUDA.

The review-critical properties this module owns:

* the first-30 candidate law (the sealed isfinite mask) and the frozen greedy
  D-opt ORDER over those candidates;
* the D-opt-prefix law: the support at k is the first k of that one frozen
  order (nested by construction, asserted), fail-closed on k outside
  [4, usable];
* the usable-cap law: the grid is capped at min(per-session usable) and the
  endpoint cell is exactly that minimum;
* the act30 activity law: the direct first-30 spelling and its bitwise parity
  with the sealed ``select_activity_rows`` on the guard-admissible
  cardinalities;
* the k-curve table law (paired per-session values at every k);
* the saturation law (smallest k within 0.005 of the endpoint mean, deficit
  sense) and the ``SATURATES_AT_<k*>`` verdict strings;
* the monotonicity law (adjacent mean deltas, every dip disclosed);
* the anchor matchers re-exported from the chrono4 precedent.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.m2_chrono4_strict_v1 import laws as chrono_laws

from . import plan


class KCurveLawError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KCurveLawError(message)


# ---------------------------------------------------------------------------
# The first-30 candidate pool and the frozen greedy D-opt order.
# ---------------------------------------------------------------------------


def first30_candidates(theta_pool: Any) -> np.ndarray:
    """The finite-angle candidates within the sealed first-30 pool.

    The isfinite candidate mask is the sealed ``m2_t4_activity_budget_screen_v1``
    law; the pool container is the session's angle table (only positions 0-29
    are read).
    """
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    _require(theta.size >= plan.ACTIVITY_HORIZON,
             "the k-curve law needs the first-30 angle pool as its container")
    pool = theta[: plan.ACTIVITY_HORIZON]
    candidates = np.flatnonzero(np.isfinite(pool)).astype(np.int64)
    _require(candidates.size >= plan.M4,
             "fewer than four directional trials within the first-30 pool")
    return np.ascontiguousarray(candidates, dtype=np.int64)


def usable_count(theta_pool: Any) -> int:
    """The per-session usable (finite-angle) count inside the first-30 pool."""
    return int(first30_candidates(theta_pool).size)


def dopt_greedy_order(theta_pool: Any) -> np.ndarray:
    """The frozen greedy forward D-opt ORDER over the first-30 candidates.

    Returns the candidate POSITIONS in selection order (first chosen first);
    the k-curve support at k is this order's first k entries.
    """
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    candidates = first30_candidates(theta)
    local = np.asarray(
        greedy_forward_d_optimal_indices(theta[candidates], int(candidates.size)),
        dtype=np.int64,
    )
    _require(local.size == candidates.size and int(np.unique(local).size) == candidates.size,
             "the greedy D-opt order must be a permutation of the candidates")
    return np.ascontiguousarray(candidates[local], dtype=np.int64)


def kcurve_support(theta_pool: Any, k: int, *, order: np.ndarray | None = None) -> np.ndarray:
    """The k-curve support: the FIRST k of the frozen greedy order (sorted).

    ``k`` must lie in [4, usable] -- the pre-registered grid floor is the
    sealed M4 anchor cardinality and the ceiling is the per-session usable
    count (the cap law).
    """
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    candidates = first30_candidates(theta)
    k = int(k)
    _require(k >= plan.M4, "k never drops below the sealed M4 anchor cardinality")
    _require(k <= int(candidates.size),
             "k exceeds the session's usable first-30 candidates (the cap law)")
    sequence = dopt_greedy_order(theta) if order is None else np.asarray(order, dtype=np.int64)
    prefix = np.sort(np.ascontiguousarray(sequence[:k], dtype=np.int64))
    _require(prefix.size == k and int(np.unique(prefix).size) == k,
             "the k-curve prefix must be k distinct trials")
    _require(int(prefix.min()) >= 0 and int(prefix.max()) < plan.ACTIVITY_HORIZON,
             "the k-curve support leaves the first-30 pool")
    return np.ascontiguousarray(prefix, dtype=np.int64)


def prefix_nesting(theta_pool: Any, ks: Iterable[int]) -> dict[str, Any]:
    """The prefix law: selection(k) is the first k of the all-usable order.

    Also proves the k=4 prefix equals the sealed M4 D-opt law recomputed
    through the chrono4 precedent's own spelling.
    """
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    order = dopt_greedy_order(theta)
    supports = {int(k): kcurve_support(theta, int(k), order=order) for k in ks}
    exact = all(
        np.array_equal(supports[int(k)], np.sort(order[: int(k)])) for k in ks
    )
    nested = all(
        bool(np.isin(supports[int(a)], supports[int(b)]).all())
        for a, b in zip(sorted(supports)[:-1], sorted(supports)[1:])
    )
    sealed_m4 = chrono_laws.dopt_support_indices(theta)
    return {
        "greedy_order_positions": [int(item) for item in order],
        "usable": int(order.size),
        "supports": {str(k): [int(item) for item in supports[k]] for k in sorted(supports)},
        "prefix_exact": bool(exact),
        "prefix_nested": bool(nested),
        "k4_prefix_is_sealed_m4_law": bool(
            np.array_equal(supports.get(plan.M4, np.asarray([], dtype=np.int64)), sealed_m4)),
        "sealed_m4_positions": [int(item) for item in sealed_m4],
    }


def endpoint_support(theta_pool: Any) -> np.ndarray:
    """The all-usable support: every finite-angle position, ascending."""
    return first30_candidates(theta_pool)


# ---------------------------------------------------------------------------
# The usable-cap law over a roster.
# ---------------------------------------------------------------------------


def usable_cap(usable_by_session: Mapping[str, int], requested: Sequence[int] = plan.K_GRID_REQUESTED) -> dict[str, Any]:
    """Cap the requested k grid at min(per-session usable); disclose capping.

    The endpoint cell is exactly that minimum (k = all-usable for every
    session simultaneously).  Fail closed if any session cannot support the
    M4 anchor cardinality.
    """
    counts = {str(name): int(value) for name, value in usable_by_session.items()}
    _require(bool(counts), "the usable census is empty")
    for name, value in counts.items():
        _require(value >= plan.M4,
                 f"{name} has fewer than four usable first-30 candidates")
    cap = min(counts.values())
    requested_sorted = sorted(int(item) for item in requested)
    effective = [item for item in requested_sorted if item <= cap]
    _require(bool(effective), "the usable cap emptied the k grid")
    endpoint = int(max(effective))
    return {
        "usable_by_session": dict(sorted(counts.items())),
        "min_usable": int(cap),
        "requested_grid": requested_sorted,
        "effective_grid": effective,
        "capped": bool(effective != requested_sorted),
        "dropped_by_cap": [item for item in requested_sorted if item > cap],
        "endpoint_k": endpoint,
        "endpoint_is_all_usable": bool(endpoint == cap),
        "expected_endpoint": plan.K_ALL_USABLE_EXPECTED,
    }


# ---------------------------------------------------------------------------
# The act30 activity law (the sealed selected-M/first-30 branch, unguarded).
# ---------------------------------------------------------------------------


def act30_activity_rows(calibration: Any) -> np.ndarray:
    """The sealed act30 activity: the FULL first-30 block, unchanged by k.

    The sealed ``select_activity_rows`` returns ``values[:30]`` for every
    guard-admissible support whose activity budget is 30; this spelling is
    that branch directly (the helper guards the SUPPORT cardinality to
    M4/M10/M30, which a k=5/6/8/12/15 support would violate -- disclosed
    deviation, parity-tested on the admissible cardinalities).
    """
    values = np.asarray(calibration, dtype=np.float32)
    _require(values.ndim == 3, "calibration must be [trials,time,channels]")
    _require(values.shape[0] >= plan.ACTIVITY_HORIZON,
             "fewer than 30 calibration trials")
    return np.ascontiguousarray(values[: plan.ACTIVITY_HORIZON], dtype=np.float32)


def act30_parity_with_sealed_helper(
    calibration: Any, support: Sequence[int],
) -> dict[str, Any]:
    """On guard-admissible supports the direct spelling is the sealed helper."""
    from src.m2_t4_activity_budget_screen_v1.core import select_activity_rows

    support_array = np.asarray(support, dtype=np.int64).reshape(-1)
    _require(support_array.size in (plan.M4, plan.M10, plan.M30),
             "act30 helper parity needs a guard-admissible support size")
    sealed = select_activity_rows(
        np.asarray(calibration, dtype=np.float32),
        selected_indices=support_array, activity_budget=plan.M30,
    )
    mine = act30_activity_rows(calibration)
    return {
        "support_size": int(support_array.size),
        "bitwise_equal": bool(np.array_equal(sealed, mine)),
    }


# ---------------------------------------------------------------------------
# The k-curve table, saturation and monotonicity laws.
# ---------------------------------------------------------------------------


def equal_session_mean(session_values: Mapping[str, float]) -> float:
    return chrono_laws.equal_session_mean(session_values)


def kcurve_table(
    per_session_by_k: Mapping[int, Mapping[str, float]],
    *, endpoint_k: int,
) -> dict[str, Any]:
    """The k-curve table: paired per-session values and means at every k.

    Every k must carry the SAME session set (the pairing law); the endpoint
    mean is the endpoint_k mean of the same family/surface.
    """
    table = {int(k): dict(values) for k, values in per_session_by_k.items()}
    _require(bool(table), "the k-curve table is empty")
    _require(int(endpoint_k) in table, "the endpoint k is not in the curve")
    sessions = set(next(iter(table.values())))
    for k, values in table.items():
        _require(set(values) == sessions and bool(values),
                 f"k={k} session set disagrees with the curve's pairing")
    means = {k: equal_session_mean(values) for k, values in sorted(table.items())}
    endpoint = float(means[int(endpoint_k)])
    deficits = {str(k): float(endpoint - mean) for k, mean in sorted(means.items())}
    return {
        "k_order": [int(k) for k in sorted(table)],
        "endpoint_k": int(endpoint_k),
        "equal_session_means": {str(k): float(means[k]) for k in sorted(means)},
        "endpoint_mean": endpoint,
        "deficit_to_endpoint": deficits,
        "per_session_r2": {
            str(k): {name: float(value) for name, value in sorted(table[k].items())}
            for k in sorted(table)
        },
        "session_count": len(sessions),
    }


def saturation_point(
    per_session_by_k: Mapping[int, Mapping[str, float]],
    *,
    endpoint_k: int,
    tolerance: float = float(plan.READOUT_LAW["saturation"]["tolerance"]),
    boundary_epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """k* = the smallest grid k within ``tolerance`` of the endpoint mean.

    Deficit sense: ``endpoint_mean - mean_k <= tolerance`` (a mean above the
    endpoint has negative deficit and qualifies).  The endpoint itself always
    qualifies (deficit exactly 0), so k* exists whenever the curve holds the
    endpoint; fail closed otherwise.
    """
    table = kcurve_table(per_session_by_k, endpoint_k=endpoint_k)
    means = {int(k): float(value) for k, value in table["equal_session_means"].items()}
    endpoint = float(table["endpoint_mean"])
    qualifying = [k for k in sorted(means) if endpoint - means[k] <= float(tolerance)]
    _require(bool(qualifying),
             "the saturation law failed: not even the endpoint qualifies")
    k_star = int(qualifying[0])
    boundary = {
        str(k): bool(abs((endpoint - means[k]) - float(tolerance)) <= float(boundary_epsilon))
        for k in sorted(means)
    }
    return {
        **table,
        "tolerance": float(tolerance),
        "qualifying_ks": qualifying,
        "k_star": k_star,
        "k_star_deficit": float(endpoint - means[k_star]),
        "any_within_epsilon_band_of_boundary": bool(any(boundary.values())),
        "boundary_bands": boundary,
        "verdict": f"SATURATES_AT_{k_star}",
    }


def monotonicity(
    per_session_by_k: Mapping[int, Mapping[str, float]],
    *, endpoint_k: int,
) -> dict[str, Any]:
    """Adjacent mean deltas over the ascending grid; every dip disclosed.

    ``dips`` lists every adjacent pair whose mean DECREASES as k grows
    (non-monotone); ``per_session_dips`` counts, per session, the adjacent
    pairs whose own R2 decreased (secondary disclosure only).
    """
    table = kcurve_table(per_session_by_k, endpoint_k=endpoint_k)
    order = [int(k) for k in table["k_order"]]
    means = {int(k): float(v) for k, v in table["equal_session_means"].items()}
    adjacent = [
        {"from_k": int(a), "to_k": int(b), "delta": float(means[b] - means[a])}
        for a, b in zip(order[:-1], order[1:])
    ]
    dips = [item for item in adjacent if item["delta"] < 0.0]
    per_session = {
        name: [
            {"from_k": int(a), "to_k": int(b)}
            for a, b in zip(order[:-1], order[1:])
            if table["per_session_r2"][str(b)][name] < table["per_session_r2"][str(a)][name]
        ]
        for name in sorted(table["per_session_r2"][str(order[0])])
    }
    return {
        "adjacent_deltas": adjacent,
        "mean_monotone_nondecreasing": bool(not dips),
        "dips": dips,
        "per_session_dip_counts": {name: int(len(items)) for name, items in per_session.items()},
        "per_session_dips": per_session,
    }


# ---------------------------------------------------------------------------
# Anchor matchers (imported from the chrono4 precedent, re-exposed for tests).
# ---------------------------------------------------------------------------


def pure_data_pairing(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any]) -> dict[str, Any]:
    return chrono_laws.pure_data_pairing(cell_row, sealed_row)


def fidelity_anchor(
    cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any], *, r2_tolerance: float,
) -> dict[str, Any]:
    return chrono_laws.fidelity_anchor(cell_row, sealed_row, r2_tolerance=r2_tolerance)


def dopt_proof(recomputed: Any, sealed_indices: Any) -> dict[str, Any]:
    return chrono_laws.dopt_proof(recomputed, sealed_indices)


def cross_family_support_agreement(
    static_supports: Mapping[str, Mapping[str, Any]],
    cdm_supports: Mapping[str, Any],
) -> dict[str, Any]:
    """The two families' k-curve supports must be IDENTICAL per session.

    Keys are ``"{surface}|{session}"``; agreement covers the greedy order and
    every grid support of the two angle spellings of one roster.
    """
    agreement: dict[str, Any] = {}
    for key_static in sorted(static_supports):
        if key_static not in cdm_supports:
            continue
        left, right = static_supports[key_static], cdm_supports[key_static]
        agree = (
            left["greedy_order_positions"] == right["greedy_order_positions"]
            and left["supports"] == right["supports"]
        )
        agreement[key_static] = {
            "greedy_order": [left["greedy_order_positions"], right["greedy_order_positions"]],
            "supports": [left["supports"], right["supports"]],
            "agree": bool(agree),
        }
    _require(bool(agreement), "the cross-family agreement roster is empty")
    _require(all(item["agree"] for item in agreement.values()),
             "the two families' k-curve supports disagree")
    return agreement


__all__ = [
    "KCurveLawError",
    "first30_candidates",
    "usable_count",
    "dopt_greedy_order",
    "kcurve_support",
    "prefix_nesting",
    "endpoint_support",
    "usable_cap",
    "act30_activity_rows",
    "act30_parity_with_sealed_helper",
    "equal_session_mean",
    "kcurve_table",
    "saturation_point",
    "monotonicity",
    "pure_data_pairing",
    "fidelity_anchor",
    "dopt_proof",
    "cross_family_support_agreement",
    "chrono_laws",
]
