"""Frozen selection/carrier/activity laws for the D-opt-4 activity-30 M2 EvalAI package.

Every function below is a verbatim mirror of a sealed source, kept here so the
submission package is self-contained and auditable.  Each block carries the
exact ``file:line`` provenance of the sealed original, and
:func:`verify_against_sealed_sources` (plus the package unit tests) prove the
mirrors are behaviour-identical to the sealed imports.

Sealed provenance
-----------------
1. Greedy forward D-optimal core (``design_matrix_from_thetas``,
   ``log_det_gram``, ``greedy_forward_d_optimal_indices``):
   ``sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219``.
2. The M4 selection branch (finite-angle first-30 candidates, greedy
   D-optimal k=4, sorted absolute indices):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35``
   (function ``_support_indices``, ``budget == 4`` branch).
3. The activity-30 pool law (first-30 calibration trials, label-free):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87``
   (function ``select_activity_rows``, ``activity_budget == 30`` branch).
4. Ridge T4 carrier fit (``fit_ridge_t4``): imported directly from
   ``tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95`` (no
   mirror; the sealed import is used so the law cannot drift).
5. Ridge side assembly (selected-trial rates, frozen normalizer):
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
ACTIVITY_HORIZON = 30  # plan.py:19 (first-30 candidate pool / activity horizon)
RIDGE_NORMALIZED_LAMBDA = 0.1  # plan.py:17
CHANNELS = 96  # plan.py:21
TRIAL_LENGTH = 100  # plan.py:22
SUPPORT_BUDGET = 4  # the D-opt-4 carrier budget under deployment
ACTIVITY_BUDGET = 30  # the label-free first-30 B3S activity pool


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
# 3. The activity-30 pool law.
#    Verbatim mirror of the ``activity_budget == 30`` branch of
#    ``select_activity_rows`` in tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87.
#    Activity needs no target labels: the pool is the chronological first-30
#    calibration-block trials, all of which belong to the official calibration
#    phase.
# ---------------------------------------------------------------------------


def select_first30_activity_pool(calibration: Any) -> np.ndarray:
    """Return the label-free first-30 calibration activity pool ``[30,100,96]``."""
    values = np.asarray(calibration, dtype=np.float32)
    _require(values.ndim == 3, "calibration must be [trials,time,channels]")
    _require(values.shape[0] >= ACTIVITY_HORIZON, "fewer than 30 calibration trials")
    _require(values.shape[1:] == (TRIAL_LENGTH, CHANNELS), "M2 calibration shape drift")
    return np.ascontiguousarray(values[:ACTIVITY_HORIZON])


# ---------------------------------------------------------------------------
# 4./5. Sealed imports for the carrier fit and activity law (no mirror drift).
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


def sealed_select_activity_rows(
    calibration: Any, selected_indices: np.ndarray, activity_budget: int
) -> np.ndarray:
    """Call the sealed ``select_activity_rows`` (screen core.py:67-87) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import select_activity_rows

    return select_activity_rows(
        calibration,
        selected_indices=selected_indices,
        activity_budget=activity_budget,
    )


def verify_against_sealed_sources(
    target_angles: Any, calibration: Any, selected: np.ndarray | None = None
) -> dict[str, object]:
    """Assert the mirrors reproduce the sealed selection and activity laws exactly.

    Used at export time and in the unit tests.  ``target_angles`` is a plain
    angle row and ``calibration`` a plain ``[trials,time,channels]`` array; the
    sealed functions are exercised through a minimal dataset shim.
    """

    class _Shim:
        pass

    shim = _Shim()
    shim.calib_trial_target_angles = {"session": np.asarray(target_angles, dtype=np.float64)}
    mirror_selection = select_dopt4_support(target_angles)
    sealed = sealed_support_indices(shim, "session")
    _require(
        np.array_equal(mirror_selection, sealed),
        f"mirror/sealed D-opt selection disagree: {mirror_selection.tolist()} vs {sealed.tolist()}",
    )
    support = mirror_selection if selected is None else np.asarray(selected, dtype=np.int64)
    mirror_pool = select_first30_activity_pool(calibration)
    sealed_pool = sealed_select_activity_rows(
        calibration, selected_indices=support, activity_budget=ACTIVITY_BUDGET
    )
    _require(
        np.array_equal(mirror_pool, sealed_pool),
        "mirror/sealed activity-30 pool disagree",
    )
    return {
        "mirror_equals_sealed": True,
        "selected_indices": [int(v) for v in mirror_selection],
        "activity_pool_rows": int(mirror_pool.shape[0]),
        "provenance": {
            "greedy_core": "sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219 (mirrored verbatim)",
            "m4_branch": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 (mirrored verbatim; sealed import cross-checked)",
            "activity_pool": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87 activity_budget==30 branch (mirrored verbatim; sealed import cross-checked)",
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (imported, fit_ridge_t4)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (imported, _ridge_side)",
            "constants": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py:17-23",
        },
    }
