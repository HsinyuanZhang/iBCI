"""Carrier support-budget resampling (M5 / M8 / M10) — Phase 2 optional stub.

Design doc section 3 lists "support-resample" as an optional arm (M1 RIFT
25-condition protocol transplanted): redraw the carrier-estimation support
at budgets M5 / M8 / M10 n times each and score carrier-support stability.
It is explicitly Phase 2 and does NOT block the three main arms.

Interface-first: signatures and budget enumeration are frozen now so the
Phase 2 implementation cannot drift; every function currently raises
NotImplementedError.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import plan

SUPPORT_BUDGETS = ("M5", "M8", "M10")
BUDGET_TRIAL_COUNTS = {"M5": 5, "M8": 8, "M10": 10}
SUPPORT_RESAMPLE_PHASE = "PHASE2_OPTIONAL_NOT_BLOCKING"


def resample_carrier_support(
    session_name: str,
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    budget: str,
    n_draws: int,
    seed: int = plan.SEED,
) -> list[np.ndarray]:
    """Redraw the session carrier at a smaller support budget.

    Planned contract (Phase 2): for each of ``n_draws`` draws, re-estimate
    the [N, 4] carrier from a size-``budget`` subsample of the M10
    calibration trials used by the frozen profile estimator, returning one
    [N, 4] array per draw with the same padding geometry as ``carrier``.
    """
    if budget not in SUPPORT_BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; expected one of {SUPPORT_BUDGETS}")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    raise NotImplementedError(
        "support-resample is a Phase 2 optional arm "
        f"({SUPPORT_RESAMPLE_PHASE}); main arms t4/f0/ts4 are not blocked by it"
    )


def resample_support_matrix(
    session_rows: dict[str, dict[str, Any]],
    budget: str,
    n_draws: int,
    seed: int = plan.SEED,
) -> dict[str, list[np.ndarray]]:
    """Per-session wrapper over :func:`resample_carrier_support` (Phase 2)."""
    raise NotImplementedError(
        "support-resample is a Phase 2 optional arm "
        f"({SUPPORT_RESAMPLE_PHASE}); main arms t4/f0/ts4 are not blocked by it"
    )


def m5(session_rows: dict[str, dict[str, Any]], n_draws: int, seed: int = plan.SEED):
    return resample_support_matrix(session_rows, "M5", n_draws, seed)


def m8(session_rows: dict[str, dict[str, Any]], n_draws: int, seed: int = plan.SEED):
    return resample_support_matrix(session_rows, "M8", n_draws, seed)


def m10(session_rows: dict[str, dict[str, Any]], n_draws: int, seed: int = plan.SEED):
    return resample_support_matrix(session_rows, "M10", n_draws, seed)


__all__ = [
    "SUPPORT_BUDGETS",
    "BUDGET_TRIAL_COUNTS",
    "SUPPORT_RESAMPLE_PHASE",
    "resample_carrier_support",
    "resample_support_matrix",
    "m5",
    "m8",
    "m10",
]
