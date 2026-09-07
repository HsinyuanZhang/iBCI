"""Pure laws for the M2 re-blocking cell: no data, no checkpoint, no CUDA.

The review-critical properties this module owns:

* the first-10 labeled-pool law: the finite-angle candidate mask within the
  first 10 positions, the K4 (frozen greedy D-opt within those candidates)
  and KALL (all candidates) supports, and the disclosure payload;
* the OLD law's D-opt-4-from-30 selection, recomputed (the receipts' proof);
* the post-H window-partition law (the sealed post30 law with the boundary
  parameterized; horizon 10 is the re-block, horizon 30 the fidelity mode);
* the evidence-stream census arithmetic (completed query trials old vs new);
* the paired BLOCK10-minus-OLDLAW contrast with breadth;
* the pre-registered gate evaluation with the 1e-12 boundary band and the
  three verdict strings.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.m2_chrono4_strict_v1 import laws as chrono_laws
from src.m2_chrono4_strict_v1 import plan as chrono_plan

from . import plan


class ReblockLawError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReblockLawError(message)


# ---------------------------------------------------------------------------
# The first-10 labeled-pool laws.
# ---------------------------------------------------------------------------


def first10_candidates(theta_pool: Any) -> np.ndarray:
    """The finite-angle candidate mask within the FIRST-10 positions.

    The pool container is the sealed first-30 angle table (the roster's own
    metadata); only positions 0-9 are read.  The isfinite candidate mask is
    the sealed ``m2_t4_activity_budget_screen_v1`` law.
    """
    theta = np.ascontiguousarray(np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    _require(theta.size >= plan.ACTIVITY_HORIZON,
             "the first-10 law needs the first-30 angle pool as its container")
    pool = theta[: plan.BLOCK_HORIZON]
    candidates = np.flatnonzero(np.isfinite(pool)).astype(np.int64)
    _require(candidates.size >= plan.M4,
             "fewer than four directional trials within the first-10 pool")
    return np.ascontiguousarray(candidates, dtype=np.int64)


def k4_support(theta_pool: Any) -> np.ndarray:
    """The K4 support: frozen greedy forward D-opt over the first-10 candidates."""
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    theta = np.ascontiguousarray(
        np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    candidates = first10_candidates(theta)
    local = np.asarray(
        greedy_forward_d_optimal_indices(theta[candidates], plan.M4), dtype=np.int64,
    )
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    _require(selected.size == plan.M4 and int(np.unique(selected).size) == plan.M4,
             "the first-10 D-opt selection must be exactly four distinct trials")
    _require(int(selected.min()) >= 0 and int(selected.max()) < plan.BLOCK_HORIZON,
             "the K4 support leaves the first-10 pool")
    return np.ascontiguousarray(selected, dtype=np.int64)


def kall_support(theta_pool: Any) -> np.ndarray:
    """The KALL support: ALL finite-angle candidates within the first-10 pool."""
    return first10_candidates(theta_pool)


def support_payload(theta_pool: Any) -> dict[str, Any]:
    """The per-session support disclosure for the re-blocked cells."""
    theta = np.ascontiguousarray(
        np.asarray(theta_pool, dtype=np.float64).reshape(-1))
    candidates = first10_candidates(theta)
    k4 = k4_support(theta)
    kall = np.ascontiguousarray(candidates, dtype=np.int64)
    return {
        "first10_candidates": [int(item) for item in kall],
        "first10_candidate_count": int(kall.size),
        "directional_among_positions_0_to_9": int(
            np.isfinite(theta[: plan.BLOCK_HORIZON]).sum()),
        "k4_support_positions": [int(item) for item in k4],
        "k4_selection": "greedy_doptimal_4_within_first10_candidates",
        "kall_support_positions": [int(item) for item in kall],
        "kall_cardinality": int(kall.size),
        "k4_within_kall": bool(np.isin(k4, kall).all()),
        "reblock_pool_depth_required_trials": plan.BLOCK_HORIZON,
        "reblock_earliest_decode_trial_1based": plan.BLOCK_HORIZON + 1,
        "reblock_earliest_evidence_position_0based": plan.BLOCK_HORIZON,
        "oldlaw_pool_depth_required_trials": plan.OLD_HORIZON,
        "oldlaw_earliest_decode_trial_1based": plan.OLD_HORIZON + 1,
    }


def dopt30_support(theta_first30: Any) -> np.ndarray:
    """The OLD law's support: the sealed D-opt-4-from-30 law, recomputed.

    Delegated to the chrono4 precedent's own law (``laws.dopt_support_indices``),
    which is itself the sealed ``m2_t4_activity_budget_screen_v1`` /
    ``pseudo_mua_precision_cdm_v2_screen_v1`` selection law verbatim.
    """
    return chrono_laws.dopt_support_indices(theta_first30)


def dopt_proof(recomputed: Any, sealed_indices: Any) -> dict[str, Any]:
    """The sealed receipt's M4 indices must equal the recomputed D-opt law."""
    return chrono_laws.dopt_proof(recomputed, sealed_indices)


# ---------------------------------------------------------------------------
# The post-H window-partition law.
# ---------------------------------------------------------------------------


def select_post_h_window_starts(
    window_starts: Any, trial_starts: Any, horizon: int,
) -> np.ndarray:
    """The sealed ``select_common_post30_window_starts`` law, boundary parameterized.

    ``horizon=30`` must reproduce the sealed law exactly (the fidelity mode);
    ``horizon=10`` is the re-block's query partition.  A window belongs to the
    post-H partition when its start bin is at or after the first bin of trial
    H (0-based position ``horizon``).
    """
    horizon = int(horizon)
    _require(horizon >= 1, "the partition horizon must be positive")
    windows = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_starts, dtype=np.int64).reshape(-1)
    _require(trials.size > horizon,
             "the session lacks a trial-%d boundary" % horizon)
    _require(np.all(np.diff(windows) > 0), "window starts must be unique and increasing")
    _require(np.all(np.diff(trials) > 0), "trial starts must be unique and increasing")
    selected = np.ascontiguousarray(windows[windows >= trials[horizon]])
    _require(selected.size > 0, "the session has no post-%d query windows" % horizon)
    return selected


def post_h_partition_matches_sealed_post30(
    window_starts: Any, trial_starts: Any,
) -> bool:
    """At horizon 30 the parameterized law IS the sealed law (bitwise)."""
    from src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
    )

    sealed = select_common_post30_window_starts(window_starts, trial_starts)
    mine = select_post_h_window_starts(window_starts, trial_starts, plan.OLD_HORIZON)
    return bool(np.array_equal(sealed, mine))


# ---------------------------------------------------------------------------
# The evidence-stream census.
# ---------------------------------------------------------------------------


def evidence_census(total_completed_trials: int) -> dict[str, Any]:
    """Completed query trials under the old vs the new evidence boundary.

    Old stream: completed trials at 0-based positions >= 30 (the sealed
    EVIDENCE_START_POSITION law).  New stream: positions >= 10.  The extension
    ratio is the pre-registered 2-3x-on-external quantity.
    """
    total = int(total_completed_trials)
    _require(total > plan.OLD_HORIZON,
             "the census needs more than 30 completed trials")
    old_trials = total - plan.OLD_HORIZON
    new_trials = total - plan.BLOCK_HORIZON
    _require(new_trials > old_trials > 0, "the census arithmetic broke")
    return {
        "total_completed_trials": total,
        "old_evidence_trials_post30": int(old_trials),
        "new_evidence_trials_post10": int(new_trials),
        "extension_ratio_new_over_old": float(new_trials / old_trials),
        "old_evidence_start_position_0based": plan.OLD_HORIZON,
        "new_evidence_start_position_0based": plan.BLOCK_HORIZON,
    }


def window_census(window_starts: Any, trial_starts: Any) -> dict[str, Any]:
    """Window counts of the old vs new scored partitions of one session."""
    windows = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_starts, dtype=np.int64).reshape(-1)
    post30 = select_post_h_window_starts(windows, trials, plan.OLD_HORIZON)
    post10 = select_post_h_window_starts(windows, trials, plan.BLOCK_HORIZON)
    return {
        "total_windows": int(windows.size),
        "post30_windows": int(post30.size),
        "post10_windows": int(post10.size),
        "post10_over_post30_window_ratio": float(post10.size / post30.size),
    }


def cross_family_support_agreement(
    static_supports: Mapping[str, Mapping[str, Any]],
    cdm_supports: Mapping[str, Mapping[str, Any]],
    surface_pairs: tuple[tuple[str, str], ...] = (
        ("within_post10", "within_post10"),
        ("external_post10_query", "external_post10_local"),
    ),
) -> dict[str, Any]:
    """The two families must bind IDENTICAL supports per session.

    Keys are ``"{surface}|{session}"`` (the executor's storage spelling); the
    agreement compares the k4/kall/dopt30 support lists of the two angle
    spellings of one roster.  (Attempt-1 regression: the first executor built
    the lookup keys from full ``surface|session`` keys and matched nothing.)
    """
    agreement: dict[str, Any] = {}
    for surface_static, surface_cdm in surface_pairs:
        for key_static in sorted(static_supports):
            if not key_static.startswith(f"{surface_static}|"):
                continue
            session = key_static.split("|", 1)[1]
            key_cdm = f"{surface_cdm}|{session}"
            if key_cdm not in cdm_supports:
                continue
            left, right = static_supports[key_static], cdm_supports[key_cdm]
            agreement[f"{surface_static}|{session}"] = {
                "k4_support": [left["k4_support_positions"], right["k4_support_positions"]],
                "kall_support": [left["kall_support_positions"], right["kall_support_positions"]],
                "dopt30_support": [left["dopt30_support_positions"],
                                   right["dopt30_support_positions"]],
                "agree": (left["k4_support_positions"] == right["k4_support_positions"]
                          and left["kall_support_positions"] == right["kall_support_positions"]
                          and left["dopt30_support_positions"] == right["dopt30_support_positions"]),
            }
    _require(bool(agreement), "the cross-family agreement roster is empty")
    _require(all(item["agree"] for item in agreement.values()),
             "the two families' re-blocked supports disagree")
    return agreement


# ---------------------------------------------------------------------------
# The paired contrast and the gate.
# ---------------------------------------------------------------------------


def equal_session_mean(session_values: Mapping[str, float]) -> float:
    return chrono_laws.equal_session_mean(session_values)


def paired_contrast(
    candidate: Mapping[str, float], reference: Mapping[str, float],
) -> dict[str, Any]:
    """BLOCK10 cell minus OLDLAW cell, per session (positive = re-block better)."""
    _require(set(candidate) == set(reference) and bool(candidate),
             "paired session set mismatch")
    deltas = {
        name: float(candidate[name]) - float(reference[name])
        for name in sorted(candidate)
    }
    values = [deltas[name] for name in sorted(deltas)]
    mean = float(sum(values) / len(values))
    return {
        "expression": "block10_cell_minus_oldlaw_cell",
        "block10_equal_session_mean": equal_session_mean(candidate),
        "oldlaw_equal_session_mean": equal_session_mean(reference),
        "equal_session_mean_delta": mean,
        "delta_sd": float(np.sqrt(float(np.mean(np.square(
            np.asarray(values, dtype=np.float64) - mean))))),
        "positive_sessions_block10_better": int(sum(1 for item in values if item > 0.0)),
        "negative_sessions_oldlaw_better": int(sum(1 for item in values if item < 0.0)),
        "tied_sessions": int(sum(1 for item in values if item == 0.0)),
        "session_count": int(len(values)),
        "breadth_denominator": int(len(values)),
        "per_session_delta": deltas,
    }


def gate_evaluation(
    contrast: Mapping[str, Any],
    *,
    delta_floor: float = plan.GATES["primary"]["delta_floor"],
    breadth_min: int = plan.GATES["primary"]["breadth_min"],
    epsilon: float = plan.GATES["boundary_epsilon"],
) -> dict[str, Any]:
    """The pre-registered gate on one BLOCK10-minus-OLDLAW contrast.

    REBLOCK_CDM_NET_POSITIVE : delta >= +0.01 exactly AND breadth >= 4/6.
    REBLOCK_BORDERLINE       : delta within 1e-12 below +0.01 AND breadth met
                               (numbers disclosed; never a silent pass).
    REBLOCK_CDM_NOT_WORTH    : everything else.
    """
    floor = float(delta_floor)
    delta = float(contrast["equal_session_mean_delta"])
    breadth = int(contrast["positive_sessions_block10_better"])
    denominator = int(contrast["breadth_denominator"])
    minimum = int(breadth_min)
    delta_meets = delta >= floor
    delta_within_band = (floor - float(epsilon)) <= delta < floor
    breadth_meets = breadth >= minimum
    if delta_meets and breadth_meets:
        verdict = "REBLOCK_CDM_NET_POSITIVE"
    elif delta_within_band and breadth_meets:
        verdict = "REBLOCK_BORDERLINE"
    else:
        verdict = "REBLOCK_CDM_NOT_WORTH"
    return {
        "delta": delta,
        "delta_floor": floor,
        "delta_meets_floor_exactly": bool(delta_meets),
        "delta_within_epsilon_band_below_floor": bool(delta_within_band),
        "boundary_epsilon": float(epsilon),
        "positive_sessions": breadth,
        "breadth_min": minimum,
        "breadth_denominator": denominator,
        "breadth_meets": bool(breadth_meets),
        "verdict": verdict,
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


__all__ = [
    "ReblockLawError",
    "first10_candidates",
    "k4_support",
    "kall_support",
    "support_payload",
    "dopt30_support",
    "dopt_proof",
    "cross_family_support_agreement",
    "select_post_h_window_starts",
    "post_h_partition_matches_sealed_post30",
    "evidence_census",
    "window_census",
    "equal_session_mean",
    "paired_contrast",
    "gate_evaluation",
    "pure_data_pairing",
    "fidelity_anchor",
    "chrono_plan",
]
