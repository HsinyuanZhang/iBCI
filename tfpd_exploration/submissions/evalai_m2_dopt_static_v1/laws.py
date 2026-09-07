"""Frozen selection/carrier laws for the D-opt-4 static M2 EvalAI package.

Every function below is a verbatim mirror of a sealed source, kept here so the
submission package is self-contained and auditable.  Each block carries the
exact ``file:line`` provenance of the sealed original, and
:func:`verify_against_sealed_sources` (plus the package unit tests) prove the
mirror is behaviour-identical to the sealed import.

Sealed provenance
-----------------
1. Greedy forward D-optimal core (``design_matrix_from_thetas``,
   ``log_det_gram``, ``greedy_forward_d_optimal_indices``):
   ``sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219``.
2. The M4 selection branch (finite-angle first-30 candidates, greedy
   D-optimal k=4, sorted absolute indices):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35``
   (function ``_support_indices``, ``budget == 4`` branch).
3. Ridge T4 carrier fit (``fit_ridge_t4``): imported directly from
   ``tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95`` (no
   mirror; the sealed import is used so the law cannot drift).
4. Ridge side assembly (selected-trial rates, frozen normalizer):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71``
   (function ``_ridge_side``), also imported directly from the sealed module.

This module imports numpy only.  It never imports torch and never reads data
files, so the no-data/no-CUDA unit tests can exercise it freely.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Frozen constants from tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py.
ACTIVITY_HORIZON = 30  # plan.py:19 (first-30 labelled candidate pool)
RIDGE_NORMALIZED_LAMBDA = 0.1  # plan.py:17
CHANNELS = 96  # plan.py:21
SUPPORT_BUDGET = 4  # the D-opt-4 carrier budget under deployment


# ---------------------------------------------------------------------------
# 1. Greedy forward D-optimal core.
#    Verbatim mirror of sua_exploration/mc_maze/d_optimal_calibration_design.py
#    (design_matrix_from_thetas: lines 115-121, log_det_gram: lines 176-185,
#    greedy_forward_d_optimal_indices: lines 188-219).
# ---------------------------------------------------------------------------


def design_matrix_from_thetas(thetas_rad: np.ndarray) -> np.ndarray:
    """Build ``[1, cos(theta), sin(theta)]`` rows for each trial."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    _require(theta.size > 0 and np.isfinite(theta).all(), "invalid thetas for design matrix")
    return np.column_stack(
        [np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)]
    )


def log_det_gram(design: np.ndarray, ridge: float = 1.0e-12) -> float:
    """Log-det of ``X'X`` with tiny ridge so greedy steps stay finite below rank 3."""
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    gram = matrix.T @ matrix + ridge * np.eye(3, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return -math.inf
    return float(logdet)


def greedy_forward_d_optimal_indices(
    thetas_rad: np.ndarray,
    m: int,
) -> np.ndarray:
    """Greedy forward selection maximizing ``det(X'X)`` (via log-det gain)."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    n = int(theta.size)
    _require(1 <= m <= n, f"cannot select m={m} from n={n} candidates")
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(m):
        best_index = -1
        best_score = -math.inf
        base_design = (
            design_matrix_from_thetas(theta[np.asarray(selected, dtype=np.int64)])
            if selected
            else None
        )
        base_score = log_det_gram(base_design) if base_design is not None else 0.0
        for index in remaining:
            trial_indices = selected + [index]
            trial_design = design_matrix_from_thetas(
                theta[np.asarray(trial_indices, dtype=np.int64)]
            )
            gain = log_det_gram(trial_design) - base_score
            if gain > best_score:
                best_score = gain
                best_index = index
        _require(best_index >= 0, "greedy D-optimal failed to select a trial")
        selected.append(best_index)
        remaining.remove(best_index)
    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. The M4 support-selection branch.
#    Verbatim mirror of the ``budget == 4`` branch of ``_support_indices`` in
#    tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35.
#    The sealed function receives the dataset's per-session
#    ``calib_trial_target_angles`` mapping; this mirror takes the angle row
#    directly so it stays data-free.
# ---------------------------------------------------------------------------


class LawError(RuntimeError):
    """Raised when the frozen selection law is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LawError(message)


def select_dopt4_support(target_angles: Any) -> np.ndarray:
    """Greedy D-optimal k=4 support from the finite-angle first-30 candidates.

    Mirrors ``_support_indices(dataset, session, budget=4)`` from the sealed
    screen (physical.py:20-35): candidates are the finite target angles inside
    ``angles[:30]``; the greedy law runs on the candidate subsequence; the
    returned support is the sorted absolute first-30 trial indices.
    """
    angles = np.asarray(target_angles, dtype=np.float64)
    _require(angles.size >= ACTIVITY_HORIZON, "session lacks first-30 target metadata")
    candidates = np.flatnonzero(np.isfinite(angles[:ACTIVITY_HORIZON])).astype(np.int64)
    _require(candidates.size >= SUPPORT_BUDGET, "session lacks four directional first-30 candidates")
    local = greedy_forward_d_optimal_indices(angles[candidates], SUPPORT_BUDGET)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    _require(
        selected.size == SUPPORT_BUDGET and int(selected.max()) < ACTIVITY_HORIZON,
        "M4 D-opt selection drift",
    )
    return np.ascontiguousarray(selected)


# ---------------------------------------------------------------------------
# 3./4. Sealed imports for the carrier fit (no mirror: import prevents drift).
# ---------------------------------------------------------------------------


def sealed_fit_ridge_side(dataset: Any, session: str, selected: np.ndarray):
    """Call the sealed ``_ridge_side`` (screen physical.py:38-71) directly.

    This is the exact function the sealed m2_t4_activity_budget_screen_v1 run
    used to turn the selected support into the normalized ridge T4 side
    features; importing it (rather than mirroring) keeps the deployed law
    bit-identical to the screen.
    """
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _ridge_side

    return _ridge_side(dataset, session, selected)


def sealed_support_indices(dataset: Any, session: str) -> np.ndarray:
    """Call the sealed ``_support_indices`` (screen physical.py:17-35) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _support_indices

    return _support_indices(dataset, session, SUPPORT_BUDGET)


def verify_against_sealed_sources(target_angles: Any) -> dict[str, object]:
    """Assert the mirror reproduces the sealed selection law exactly.

    Used at export time and in the unit tests.  ``target_angles`` is a plain
    angle row; the sealed function is exercised through a minimal dataset shim.
    """

    class _Shim:
        pass

    shim = _Shim()
    shim.calib_trial_target_angles = {"session": np.asarray(target_angles, dtype=np.float64)}
    mirror = select_dopt4_support(target_angles)
    sealed = sealed_support_indices(shim, "session")
    agree = np.array_equal(mirror, sealed)
    _require(agree, f"mirror/sealed D-opt selection disagree: {mirror.tolist()} vs {sealed.tolist()}")
    return {
        "mirror_equals_sealed": True,
        "selected_indices": [int(v) for v in mirror],
        "provenance": {
            "greedy_core": "sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219 (mirrored verbatim)",
            "m4_branch": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 (mirrored verbatim; sealed import cross-checked)",
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (imported, fit_ridge_t4)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (imported, _ridge_side)",
            "constants": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py:17-21",
        },
    }
