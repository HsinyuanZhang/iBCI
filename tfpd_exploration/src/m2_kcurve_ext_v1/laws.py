"""Pure laws for the M2 k-curve extension: no data, no checkpoint, no CUDA.

The review-critical properties this module owns:

* the first-B candidate law (the sealed isfinite mask within the leading
  block; the reblock10 first-10 precedent generalized to any B, with B=30
  equal to the sealed k-curve first-30 pool);
* the per-B greedy D-opt order and the prefix support law (the support at k
  is the FIRST k of that order, k in [4, usable_B], fail-closed outside);
* the B-column grid law: the requested template
  {4, min(5,usable_B), 8, min(10,usable_B), all-usable_B} resolved against
  the column's usable cap (min over sessions), capping disclosed;
* the decline-slope law and the Q1 verdict rule (pre-registered);
* the matched-k law for the Q2 pool-narrowing contrasts;
* the Q3 best-cell law (deterministic tie-break);
* the exact-CPU anchor matcher for same-machine receipt reproduction;
* re-exports of the sealed pure helpers (equal-session means, paired
  deltas, fidelity anchors) reused by import only.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.m2_chrono4_strict_v1 import laws as chrono_laws
from src.m2_kcurve_v1 import laws as kcurve_laws
from src.m2_memory_law_scan_v1 import gates as scan_gates

from . import plan


class KCurveExtLawError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KCurveExtLawError(message)


# ---------------------------------------------------------------------------
# The first-B candidate pool and the per-B greedy D-opt order.
# ---------------------------------------------------------------------------


def blockB_candidates(theta_pool: Any, B: int) -> np.ndarray:
    """The finite-angle candidates within the FIRST-B positions.

    The pool container is the session's angle table (a first-30 container is
    always required, the sealed roster's own metadata); only positions
    0..B-1 are read.  At B=30 this IS the sealed k-curve ``first30_candidates``
    mask; at B=10 it is the reblock10 ``first10_candidates`` mask.
    """
    horizon = int(B)
    _require(plan.M4 <= horizon <= plan.ACTIVITY_HORIZON,
             "B must lie in [4, 30] (the leading-block horizon)")
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    _require(theta.size >= plan.ACTIVITY_HORIZON,
             "the first-B law needs the first-30 angle pool as its container")
    pool = theta[:horizon]
    candidates = np.flatnonzero(np.isfinite(pool)).astype(np.int64)
    _require(candidates.size >= plan.M4,
             f"fewer than four directional trials within the first-{horizon} pool")
    return np.ascontiguousarray(candidates, dtype=np.int64)


def usable_count_B(theta_pool: Any, B: int) -> int:
    """The per-session usable (finite-angle) count within the first-B block."""
    return int(blockB_candidates(theta_pool, B).size)


def dopt_greedy_order_B(theta_pool: Any, B: int) -> np.ndarray:
    """The frozen greedy forward D-opt ORDER over the first-B candidates.

    At B=30 this IS the sealed k-curve ``dopt_greedy_order`` (same law over
    the same pool); the returned positions are in selection order.
    """
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    candidates = blockB_candidates(theta_pool, B)
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    local = np.asarray(
        greedy_forward_d_optimal_indices(theta[candidates], int(candidates.size)),
        dtype=np.int64,
    )
    _require(local.size == candidates.size
             and int(np.unique(local).size) == candidates.size,
             "the greedy D-opt order must be a permutation of the candidates")
    return np.ascontiguousarray(candidates[local], dtype=np.int64)


def support_at(theta_pool: Any, k: int, B: int, *,
               order: np.ndarray | None = None) -> np.ndarray:
    """The (B,k) support: the FIRST k of the per-B greedy order (sorted).

    ``k`` must lie in [4, usable_B] -- the grid floor is the sealed M4 anchor
    cardinality and the ceiling is the per-session usable count within B.
    """
    candidates = blockB_candidates(theta_pool, B)
    k = int(k)
    _require(k >= plan.K_FLOOR, "k never drops below the sealed M4 anchor cardinality")
    _require(k <= int(candidates.size),
             "k exceeds the session's usable candidates within B (the cap law)")
    sequence = (dopt_greedy_order_B(theta_pool, B) if order is None
                else np.asarray(order, dtype=np.int64))
    prefix = np.sort(np.ascontiguousarray(sequence[:k], dtype=np.int64))
    _require(prefix.size == k and int(np.unique(prefix).size) == k,
             "the (B,k) prefix must be k distinct trials")
    _require(int(prefix.min()) >= 0 and int(prefix.max()) < int(B),
             "the (B,k) support leaves the first-B pool")
    return np.ascontiguousarray(prefix, dtype=np.int64)


def prefix_payload(theta_pool: Any, B: int, ks: Iterable[int]) -> dict[str, Any]:
    """The per-(session,B) prefix law payload: order, usable, supports,
    nesting, and the B=30 identity with the sealed k-curve law."""
    ks = sorted({int(k) for k in ks})
    order = dopt_greedy_order_B(theta_pool, B)
    supports = {k: support_at(theta_pool, k, B, order=order) for k in ks}
    nested = all(
        bool(np.isin(supports[a], supports[b]).all())
        for a, b in zip(ks[:-1], ks[1:])
    )
    exact = all(np.array_equal(supports[k], np.sort(order[:k])) for k in ks)
    payload: dict[str, Any] = {
        "B": int(B),
        "greedy_order_positions": [int(item) for item in order],
        "usable_within_b": int(order.size),
        "candidates_within_b": [int(item) for item in blockB_candidates(theta_pool, B)],
        "supports": {str(k): [int(item) for item in supports[k]] for k in ks},
        "prefix_exact": bool(exact),
        "prefix_nested": bool(nested),
    }
    if int(B) == plan.B_REFERENCE:
        sealed_order = kcurve_laws.dopt_greedy_order(theta_pool)
        payload["b30_order_is_sealed_kcurve_law"] = bool(
            np.array_equal(order, sealed_order))
        payload["b30_usable_matches_sealed"] = bool(
            int(order.size) == kcurve_laws.usable_count(theta_pool))
    return payload


# ---------------------------------------------------------------------------
# The B-column grid law.
# ---------------------------------------------------------------------------


def requested_template(cap: int) -> list[int]:
    """The work order's requested k list resolved against the column cap.

    {4, min(5,usable_B), 8, min(10,usable_B), all-usable_B}: every entry is
    bounded by the cap by construction, so the effective grid is the sorted
    deduplication and the endpoint IS the cap.
    """
    cap = int(cap)
    _require(cap >= plan.K_FLOOR,
             "the column's usable cap cannot support the M4 anchor cardinality")
    raw = [plan.K_FLOOR, min(5, cap)]
    if 8 <= cap:  # a fixed grid card above the cap is dropped, never clamped
        raw.append(8)
    raw.extend([min(10, cap), cap])
    return sorted({int(item) for item in raw})


def b_grid(usable_by_session: Mapping[str, int], B: int) -> dict[str, Any]:
    """Resolve the column's k grid from the per-session usable census.

    The cap is the MINIMUM usable over the roster; the endpoint cell is
    exactly that minimum; capping (dropped or merged requested cards) is
    disclosed, never silent.  Fail closed if the cap drops below 4.
    """
    counts = {str(name): int(value) for name, value in usable_by_session.items()}
    _require(bool(counts), "the usable census is empty")
    for name, value in counts.items():
        _require(value >= plan.K_FLOOR,
                 f"{name} has fewer than four usable candidates within B={B}")
    cap = min(counts.values())
    requested = requested_template(cap)
    dropped = [8] if 8 > cap else []
    return {
        "B": int(B),
        "usable_by_session": dict(sorted(counts.items())),
        "min_usable": int(cap),
        "requested_template": list(plan.K_GRID_TEMPLATE),
        "effective_grid": requested,
        "endpoint_k": int(cap),
        "dropped_by_cap": dropped,
        "capped": bool(dropped or len(requested) < 5),
        "endpoint_is_all_usable": True,
    }


# ---------------------------------------------------------------------------
# Readout laws: decline slope, Q1 verdict, matched k, best cell.
# ---------------------------------------------------------------------------


def equal_session_mean(session_values: Mapping[str, float]) -> float:
    return scan_gates.equal_session_mean(session_values)


def decline_slope(per_session_by_k: Mapping[int, Mapping[str, float]], *,
                  k_low: int, k_high: int) -> float:
    """mean at k_high minus mean at k_low (negative = decline as k grows)."""
    table = {int(k): dict(v) for k, v in per_session_by_k.items()}
    _require(k_low in table and k_high in table,
             "the decline slope needs both endpoints on the curve")
    return float(equal_session_mean(table[int(k_high)])
                 - equal_session_mean(table[int(k_low)]))


def law_paired_delta(candidate: Mapping[str, float],
                     reference: Mapping[str, float]) -> dict[str, Any]:
    """The sealed paired-delta law (per-session, mean, breadth)."""
    return scan_gates.paired_delta(candidate, reference)


def q1_verdict(*, slope_fifo30: float, slope_uncapped: float,
               slope_static: float,
               tolerance: float = float(
                   plan.READOUT_LAW["q1_law_contrast"]["tolerance"])) -> dict[str, Any]:
    """The pre-registered Q1 verdict.

    ``CAPACITY_COMPETITION_CONFIRMED`` iff ``slope_uncapped >= slope_static -
    tolerance``: with the capacity cap removed, the external k-decline is not
    steeper than the no-memory static control's decline by more than the
    0.005 program tolerance (the decline has flattened to the carrier-axis
    background).  Otherwise ``CARRIER_OVERFIT_DOMINANT``: the decline
    survives unbounded accumulation.
    """
    slope_fifo30 = float(slope_fifo30)
    slope_uncapped = float(slope_uncapped)
    slope_static = float(slope_static)
    tolerance = float(tolerance)
    threshold = slope_static - tolerance
    confirmed = slope_uncapped >= threshold
    slope_range = abs(slope_fifo30 - slope_static)
    return {
        "slope_fifo30": slope_fifo30,
        "slope_uncapped": slope_uncapped,
        "slope_static": slope_static,
        "tolerance": tolerance,
        "threshold_static_minus_tolerance": float(threshold),
        "confirmed": bool(confirmed),
        "verdict": ("CAPACITY_COMPETITION_CONFIRMED" if confirmed
                    else "CARRIER_OVERFIT_DOMINANT"),
        "excess_decline_uncapped_vs_static": float(slope_uncapped - slope_static),
        "decline_removed_fraction": (
            None if slope_range == 0.0 else
            float(min(1.0, max(0.0,
                               1.0 - (slope_static - slope_uncapped) / slope_range)))
        ),
    }


def matched_ks(grid_a: Sequence[int], grid_b: Sequence[int]) -> list[int]:
    """The k values present in both columns' grids (the Q2 pairing roster)."""
    left = sorted({int(item) for item in grid_a})
    right = {int(item) for item in grid_b}
    matched = [item for item in left if item in right]
    _require(bool(matched), "the two columns share no k value")
    return matched


def best_cell(means: Mapping[str, float], *, law_order: Sequence[str] = plan.LAWS,
              b_order: Sequence[int] = plan.B_AXIS) -> dict[str, Any]:
    """The Q3 best-cell law: argmax of the external equal-session mean.

    Ties break by (law order, B ascending, k ascending) so the winner is a
    deterministic function of the table; the full ranking is the caller's.
    """
    _require(bool(means), "the best-cell table is empty")
    law_rank = {str(name): index for index, name in enumerate(law_order)}
    b_rank = {int(value): index for index, value in enumerate(b_order)}
    entries: list[tuple[str, str, int, int, float]] = []
    for key, value in means.items():
        law, b_field, k_field = str(key).split("|")
        entries.append((str(law), b_field, int(b_field.lstrip("B")),
                        int(k_field.lstrip("K")), float(value)))
    for law, _b_name, b_value, _k, _v in entries:
        _require(law in law_rank, f"unknown law on the best-cell table: {law}")
        _require(b_value in b_rank, f"unknown B on the best-cell table: {b_value}")
    ranked = sorted(entries, key=lambda item: (
        -item[4], law_rank[item[0]], b_rank[item[2]], item[3], item[0]))
    winner = ranked[0]
    return {
        "key": f"{winner[0]}|B{winner[2]}|K{winner[3]}",
        "law": winner[0],
        "B": winner[2],
        "k": winner[3],
        "value": winner[4],
        "ranking": [
            {"key": f"{law}|B{b}|K{k}", "law": law, "B": b, "k": k, "value": value}
            for law, _name, b, k, value in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Anchor matchers.
# ---------------------------------------------------------------------------


def exact_cpu_anchor(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any], *,
                     r2_tolerance: float = float(
                         plan.ANCHORS["exact_cpu_matcher"]["r2_tolerance"]),
                     ) -> dict[str, Any]:
    """The same-machine receipt reproduction matcher.

    Pure-data fields (starts digest, target digest, window count) exact, R2
    within ``r2_tolerance`` (CPU==CPU on the pinned environment; expected
    exactly 0), and the prediction digest EQUAL (the same imported executor
    on the same machine must reproduce the sealed bits).
    """
    # compare like-for-like: the first starts-digest field BOTH rows carry
    # (the sealed k-curve rows spell both; the sealed memory-scan rows spell
    # only query_starts_sha256; both digests are the same array digest)
    common_starts_field: str | None = None
    for name in ("ordered_window_starts_sha256", "query_starts_sha256"):
        if name in cell_row and name in sealed_row:
            common_starts_field = name
            break
    matches = {
        "starts_digest": (
            common_starts_field is not None
            and str(cell_row[common_starts_field]) == str(sealed_row[common_starts_field])
        ),
        "target_sha256": (str(cell_row["target_sha256"])
                          == str(sealed_row["target_sha256"])),
        "window_count": (int(cell_row["window_count"])
                         == int(sealed_row["window_count"])),
        "r2_within_tolerance": abs(
            float(cell_row["r2"]) - float(sealed_row["r2"])) <= float(r2_tolerance),
        "prediction_sha256": (str(cell_row["prediction_sha256"])
                              == str(sealed_row["prediction_sha256"])),
    }
    return {
        "starts_digest_field": common_starts_field,
        "field_matches": matches,
        "exact_match": all(bool(value) for value in matches.values()),
        "r2_cpu": float(cell_row["r2"]),
        "r2_sealed": float(sealed_row["r2"]),
        "r2_delta": float(cell_row["r2"]) - float(sealed_row["r2"]),
    }


def fidelity_anchor(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any], *,
                    r2_tolerance: float) -> dict[str, Any]:
    """The chrono4 tolerance matcher (re-exported for receipts/tests)."""
    return chrono_laws.fidelity_anchor(cell_row, sealed_row, r2_tolerance=r2_tolerance)


def support_agreement(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Two spellings of one (session,B) roster bind identical supports."""
    agree = (
        left["greedy_order_positions"] == right["greedy_order_positions"]
        and left["supports"] == right["supports"]
        and left["usable_within_b"] == right["usable_within_b"]
    )
    return {
        "greedy_order": [left["greedy_order_positions"], right["greedy_order_positions"]],
        "supports": [left["supports"], right["supports"]],
        "agree": bool(agree),
    }


__all__ = [
    "KCurveExtLawError",
    "blockB_candidates",
    "usable_count_B",
    "dopt_greedy_order_B",
    "support_at",
    "prefix_payload",
    "requested_template",
    "b_grid",
    "equal_session_mean",
    "decline_slope",
    "law_paired_delta",
    "q1_verdict",
    "matched_ks",
    "best_cell",
    "exact_cpu_anchor",
    "fidelity_anchor",
    "support_agreement",
    "kcurve_laws",
    "chrono_laws",
    "scan_gates",
]
