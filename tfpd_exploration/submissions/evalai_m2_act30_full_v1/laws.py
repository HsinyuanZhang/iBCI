"""Frozen selection/carrier/activity laws for the full-block m30 activity-30 M2 EvalAI package.

Every function below is a verbatim mirror of a sealed source, kept here so the
submission package is self-contained and auditable.  Each block carries the
exact ``file:line`` provenance of the sealed original, and
:func:`verify_against_sealed_sources` (plus the package unit tests) prove the
mirrors are behaviour-identical to the sealed imports.

Sealed provenance
-----------------
1. The M30 selection branch (chronological first-30 block, minimum three
   directional trials):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35``
   (function ``_support_indices``, ``budget != 4`` branch with ``budget=30``).
2. The activity-30 pool law (first-30 calibration trials, label-free):
   ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87``
   (function ``select_activity_rows``, ``activity_budget == 30`` branch).
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

from typing import Any

import numpy as np

# Frozen constants from tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py.
ACTIVITY_HORIZON = 30  # plan.py:19 (first-30 candidate pool / activity horizon)
RIDGE_NORMALIZED_LAMBDA = 0.1  # plan.py:17
CHANNELS = 96  # plan.py:21
TRIAL_LENGTH = 100  # plan.py:22
CARRIER_BUDGET = 30  # the chronological first-30 carrier block under deployment
ACTIVITY_BUDGET = 30  # the label-free first-30 B3S activity pool
MIN_DIRECTIONAL_TRIALS = 3  # sealed law: at least three finite angles in the block


# ---------------------------------------------------------------------------
# LawError / guard
# ---------------------------------------------------------------------------


class LawError(RuntimeError):
    """Raised when the frozen selection law is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LawError(message)


# ---------------------------------------------------------------------------
# 1. The M30 support-selection branch.
#    Verbatim mirror of the ``budget != 4`` branch of ``_support_indices`` in
#    tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35
#    evaluated at ``budget=30``: the chronological first-30 trial block, with
#    the sealed minimum of three directional (finite-angle) trials.
# ---------------------------------------------------------------------------


def select_m30_support(target_angles: Any) -> np.ndarray:
    """Chronological first-30 carrier support ``[0..29]``.

    Mirrors ``_support_indices(dataset, session, budget=30)`` from the sealed
    screen (physical.py:21-24): the selection is the chronological
    ``arange(30)`` block and the law fails closed when the block holds fewer
    than three finite-direction trials.
    """
    angles = np.asarray(target_angles, dtype=np.float64)
    _require(angles.size >= ACTIVITY_HORIZON, "session lacks first-30 target metadata")
    selected = np.arange(CARRIER_BUDGET, dtype=np.int64)
    _require(
        int(np.isfinite(angles[selected]).sum()) >= MIN_DIRECTIONAL_TRIALS,
        "M30 has fewer than three directional trials",
    )
    return selected


# ---------------------------------------------------------------------------
# 2. The activity-30 pool law.
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
# 3./4. Sealed imports for the carrier fit and activity law (no mirror drift).
# ---------------------------------------------------------------------------


def sealed_fit_ridge_side(dataset: Any, session: str, selected: np.ndarray):
    """Call the sealed ``_ridge_side`` (screen physical.py:38-71) directly.

    This is the exact function the sealed m2_t4_activity_budget_screen_v1 run
    used to turn the chronological m30 support into the normalized ridge T4
    side features (the fit consumes the finite-angle subset of the block);
    importing it (rather than mirroring) keeps the deployed law bit-identical
    to the screen.
    """
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _ridge_side

    return _ridge_side(dataset, session, selected)


def sealed_support_indices(dataset: Any, session: str) -> np.ndarray:
    """Call the sealed ``_support_indices`` (screen physical.py:17-35) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _support_indices

    return _support_indices(dataset, session, CARRIER_BUDGET)


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
    mirror_selection = select_m30_support(target_angles)
    sealed = sealed_support_indices(shim, "session")
    _require(
        np.array_equal(mirror_selection, sealed),
        f"mirror/sealed m30 selection disagree: {mirror_selection.tolist()} vs {sealed.tolist()}",
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
            "m30_branch": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 budget!=4 branch at budget=30 (mirrored verbatim; sealed import cross-checked)",
            "activity_pool": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87 activity_budget==30 branch (mirrored verbatim; sealed import cross-checked)",
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (imported, fit_ridge_t4)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (imported, _ridge_side)",
            "constants": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py:17-23",
        },
    }
