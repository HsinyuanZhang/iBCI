"""Pure laws of the M2 prerequisite audit: no data, no Torch, no CUDA.

Every function here is deterministic and unit-testable on synthetic arrays:

* :func:`rank_auc` -- the knob-free coverage statistic for a boundary law;
* :func:`boundary_statistics` -- the pre-registered causal neural-only laws;
* :func:`support_anchor_parity` -- the Stage-O A0/b0 anchor law re-applied to
  the M2 ``fit_ridge_t4`` carrier, with the bitwise parity proof;
* :func:`boundary_verdict` -- the fail-closed contract decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


class AuditError(ValueError):
    """Fail-closed error for a malformed audit input or verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def rank_auc(statistic: Sequence[float], indicator: Sequence[bool]) -> float:
    """Rank AUC of one causal statistic against the boundary indicator.

    AUC 0.5 is chance: no monotone threshold on the statistic separates
    boundary bins from non-boundary bins.  Average ranks handle ties exactly.
    """
    scores = np.asarray(statistic, dtype=np.float64).reshape(-1)
    labels = np.asarray(indicator, dtype=bool).reshape(-1)
    require(scores.size == labels.size and scores.size >= 2, "AUC needs aligned nonempty vectors")
    require(np.isfinite(scores).all(), "AUC statistic must be finite")
    require(int(labels.sum()) >= 1 and int((~labels).sum()) >= 1, "AUC needs both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    position = 0
    while position < scores.size:
        stop = position
        while stop + 1 < scores.size and sorted_scores[stop + 1] == sorted_scores[position]:
            stop += 1
        average = 0.5 * (position + stop) + 1.0
        ranks[order[position : stop + 1]] = average
        position = stop + 1
    positive = float(ranks[labels].sum())
    n_positive = float(int(labels.sum()))
    n_negative = float(scores.size) - n_positive
    return float((positive - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative))


def boundary_statistics(neural: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    """The three pre-registered CAUSAL neural-only boundary statistics.

    Every statistic at bin ``t`` uses neural bins ``<= t`` only:

    * ``population_count``  -- total counts across channels at ``t``;
    * ``abs_delta_population_count`` -- ``|pop[t] - pop[t-1]|``;
    * ``abs_delta_population_vector`` -- ``||x[t] - x[t-1]||_2``.
    """
    values = np.asarray(neural, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 1,
            "neural stream must be [T, C]")
    population = values.sum(axis=1)
    delta_count = np.abs(np.diff(population, prepend=population[:1]))
    delta_vector = np.linalg.norm(np.diff(values, axis=0, prepend=values[:1]), axis=1)
    require(population.shape == delta_count.shape == delta_vector.shape, "statistic shape drift")
    return {
        "population_count": np.ascontiguousarray(population),
        "abs_delta_population_count": np.ascontiguousarray(delta_count),
        "abs_delta_population_vector": np.ascontiguousarray(delta_vector),
    }


def anchor_a0_b0(
    support_rates_hz: np.ndarray, support_angles_rad: np.ndarray, *,
    normalized_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The Stage-O anchor law: ``A0 = Xs^T Xs + diag(n*lambda, n*lambda, 0)``.

    The design column order and the penalty are the exact ``fit_ridge_t4``
    convention so that ``solve(A0, b0)`` and the sealed fit share arithmetic.
    """
    rates = np.ascontiguousarray(np.asarray(support_rates_hz, dtype=np.float64))
    angles = np.ascontiguousarray(np.asarray(support_angles_rad, dtype=np.float64).reshape(-1))
    require(rates.ndim == 2 and rates.shape[0] == angles.size and angles.size >= 3,
            "anchor support must be [trials, units] with >= 3 directional trials")
    require(np.isfinite(rates).all() and np.isfinite(angles).all(), "anchor support nonfinite")
    require(math.isfinite(normalized_lambda) and normalized_lambda >= 0.0,
            "normalized lambda invalid")
    design = np.column_stack((np.cos(angles), np.sin(angles), np.ones(angles.size)))
    penalty = np.diag((angles.size * normalized_lambda, angles.size * normalized_lambda, 0.0))
    a0 = np.ascontiguousarray(design.T @ design + penalty, dtype=np.float64)
    b0 = np.ascontiguousarray(design.T @ rates, dtype=np.float64)
    require(np.all(np.linalg.eigvalsh(a0) > 0.0), "anchor A0 is not positive definite")
    return a0, b0


def anchor_rebuild_t4(coefficient_blocks: Sequence[np.ndarray]) -> np.ndarray:
    """[3, U] (a, c, b) -> [U, 4] (a, c, sqrt(a*a+c*c), b), the fit_ridge_t4 law.

    The M2 sealed carrier computes the modulation column with
    ``np.sqrt(a * a + c * c)`` (NOT ``np.hypot``); the anchor mirror must use
    the same arithmetic to be bit-exact.
    """
    values = np.asarray(coefficient_blocks, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] == 3, "coefficient block must be [3, U]")
    a, c, b = values[0], values[1], values[2]
    rows = np.column_stack((a, c, np.sqrt(a * a + c * c), b))
    require(np.isfinite(rows).all(), "rebuilt carrier rows are nonfinite")
    return rows


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def support_anchor_parity(
    support_rates_hz: np.ndarray, support_angles_rad: np.ndarray, *,
    normalized_lambda: float, sealed_raw_t4: np.ndarray,
) -> dict[str, Any]:
    """Prove the anchor reproduces the sealed M2 carrier bit-for-bit."""
    sealed = np.asarray(sealed_raw_t4)
    a0, b0 = anchor_a0_b0(support_rates_hz, support_angles_rad, normalized_lambda=normalized_lambda)
    solved = np.linalg.solve(a0, b0)
    rebuilt = np.ascontiguousarray(anchor_rebuild_t4(solved), dtype=np.float32)
    require(sealed.shape == (support_rates_hz.shape[1], 4), "sealed carrier shape drift")
    bit_equal = bool(np.array_equal(rebuilt, np.ascontiguousarray(sealed, dtype=np.float32)))
    return {
        "law": "solve(A0, b0) -> [a, c, sqrt(a*a+c*c), b] float32 vs the sealed fit_ridge_t4 carrier",
        "max_abs_difference": float(np.max(np.abs(rebuilt.astype(np.float64) - np.asarray(sealed, dtype=np.float64)))),
        "rebuilt_bitwise_equal": bit_equal,
        "a0_sha256": array_digest(a0),
        "b0_columns": int(b0.shape[1]),
        "a0_eigenvalues_min": float(np.linalg.eigvalsh(a0).min()),
    }


def boundary_verdict(
    *, contract_markers_present: bool, on_done_served_to_decoder: bool,
    source_auc_by_session: Mapping[str, Mapping[str, float]], auc_floor: float,
) -> dict[str, Any]:
    """The fail-closed prerequisite-(a) decision.

    ``source_auc_by_session`` maps session -> statistic name -> AUC (the
    producer orientation).  The verdict order is exactly the work order's:
    official delivery first; otherwise a causal reconstruction law only if
    EVERY source session separates boundaries above the frozen floor under at
    least one pre-registered statistic; otherwise not evaluable.
    """
    require(auc_floor > 0.5, "AUC floor must exceed chance")
    if on_done_served_to_decoder:
        return {
            "verdict": "AVAILABLE_UNDER_OFFICIAL_CONTRACT",
            "reason": "the evaluator delivers completed-trial boundaries to the decoder",
            "reconstruction_required": False,
        }
    if not contract_markers_present:
        raise AuditError("the contract source could not be bound; the audit must fail closed")
    best_by_session: dict[str, list[float]] = {}
    for session, per_statistic in source_auc_by_session.items():
        require(bool(per_statistic), f"session {session} has no measured statistic")
        best_by_session[session] = [float(value) for value in per_statistic.values()]
    require(bool(best_by_session), "no source-session AUC was measured")
    best_per_session = {session: max(values) for session, values in best_by_session.items()}
    passing = {session: value for session, value in best_per_session.items() if value >= auc_floor}
    all_sessions_pass = len(passing) == len(best_per_session) and bool(passing)
    if all_sessions_pass:
        return {
            "verdict": "RECONSTRUCTIBLE_CAUSAL_LAW",
            "reason": "every source session separates above the frozen AUC floor",
            "reconstruction_required": True,
            "best_auc_by_source_session": best_per_session,
        }
    return {
        "verdict": "NOT_EVALUABLE_OFFICIAL_CONTRACT",
        "reason": (
            "the official M2 evaluator hides completed-trial boundaries "
            "(no on_done/observe for a continual task) and no pre-registered "
            "causal neural-only statistic separates them above the floor on "
            "every M2 source session; freezing a threshold law here would be "
            "an unfitted detector, and a decoder-output law is circular with "
            "the carrier update under test"
        ),
        "reconstruction_required": True,
        "best_auc_by_source_session": best_per_session,
        "failing_source_sessions": sorted(set(best_per_session) - set(passing)),
    }
