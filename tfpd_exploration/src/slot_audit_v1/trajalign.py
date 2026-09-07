"""SLOT-AUDIT V1 trajalign — re-measurement THROUGH the frozen probe law.

Work order section 4.4: the trajectory-alignment arms are re-run with the
EXACT frozen law of ``src/continuity_probe_v1`` — ``smooth_trajectory_aligned``
and its ``_trajalign_gather`` are IMPORTED, never reimplemented — on the
cached full prediction tensors (bit-identical to the probe's decode inputs),
K in {2, 4, 8, 16}, mean weighting.  Scoring follows the probe's
``_row_common`` path: float64 smoothing output cast to float32 and scored by
the house per-session variance-weighted R2 against the cached governing
targets.  Because the inputs are bit-identical, every recomputed value must
reproduce the sealed probe's trajalign rows (tolerance 1e-9; any drift is a
hard failure and fails the run).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import plan
from .scoring import house_session_r2


class SlotAuditTrajalignError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditTrajalignError(message)


def load_probe_module():
    """Import the frozen continuity probe (import-only; never edited)."""
    from src import continuity_probe_v1 as probe

    return probe


def trajalign_arm_r2(
    full_predictions: Any, targets: Any, starts: Any, valid: Any, k: int,
) -> dict[str, Any]:
    """One (session, K) trajectory-aligned row through the frozen law."""
    probe = load_probe_module()
    full32 = np.ascontiguousarray(full_predictions, dtype=np.float32)
    starts64 = np.asarray(starts, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    smoothed64, support_valid = probe.smooth_trajectory_aligned(
        np.asarray(full32, dtype=np.float64), starts64, probe.plain_weights(int(k)),
    )
    smoothed32 = np.ascontiguousarray(smoothed64[valid], dtype=np.float32)
    target32 = np.ascontiguousarray(np.asarray(targets, dtype=np.float32)[valid])
    _require(
        smoothed32.shape == target32.shape,
        "trajalign re-measurement shape drift",
    )
    _require(
        bool(np.isfinite(smoothed32).all()),
        "trajalign re-measurement nonfinite output",
    )
    return {
        "K": int(k),
        "support": int(k),
        "variance_weighted_r2": house_session_r2(smoothed32, target32),
        "n_windows": int(smoothed32.shape[0]),
    }


def sealed_trajalign_cells(sealed_probe: Mapping) -> dict[tuple[str, int, str], Mapping]:
    """The sealed probe's trajalign mean arms, keyed (surface, budget, arm)."""
    cells: dict[tuple[str, int, str], Mapping] = {}
    for cell in sealed_probe["cells"]:
        if not str(cell["arm"]).startswith("trajalign_mean_K"):
            continue
        key = (str(cell["surface"]), int(cell["budget"]), str(cell["arm"]))
        _require(key not in cells, f"sealed probe cell duplication: {key}")
        cells[key] = cell
    return cells


def compare_to_sealed(
    recomputed: Mapping[str, Mapping[str, list[Mapping]]],
    sealed_probe: Mapping, *, surfaces, budgets,
    tolerance: float = plan.TRAJALIGN_R2_TOLERANCE,
) -> dict[str, Any]:
    """Assert per-session (and, at full coverage, mean) equality to the sealed rows.

    The governing run covers the complete sealed rosters, so both the
    per-session values and the equal-session means are compared; a partial
    (smoke) roster compares only the per-session values it actually
    recomputed and records the reduced coverage.
    """
    sealed = sealed_trajalign_cells(sealed_probe)
    rows: list[dict[str, Any]] = []
    max_session_delta = 0.0
    max_mean_delta = 0.0
    full_coverage = True
    for surface in surfaces:
        for budget in budgets:
            for k in plan.K_GRID:
                arm = f"trajalign_mean_K{k}"
                key = (surface, int(budget), arm)
                _require(key in sealed, f"sealed probe lacks trajalign cell {key}")
                anchor = sealed[key]
                own = recomputed[f"{surface}_M{budget}"][f"K{k}"]
                anchor_by_session = {
                    str(row["session"]): row for row in anchor["sessions"]
                }
                session_deltas = []
                for row in own["sessions"]:
                    name = str(row["session"])
                    _require(
                        name in anchor_by_session,
                        f"trajalign anchor lacks recomputed session {name} "
                        f"at {surface} M{budget} {arm}",
                    )
                    delta = float(row["variance_weighted_r2"]) - float(
                        anchor_by_session[name]["variance_weighted_r2"]
                    )
                    session_deltas.append(delta)
                    max_session_delta = max(max_session_delta, abs(delta))
                same_roster = (
                    [str(row["session"]) for row in own["sessions"]]
                    == [str(row["session"]) for row in anchor["sessions"]]
                )
                full_coverage = full_coverage and same_roster
                mean_delta = None
                if same_roster:
                    mean_delta = float(own["equal_session_mean_r2"]) - float(
                        anchor["summary"]["equal_session_mean_r2"]
                    )
                    max_mean_delta = max(max_mean_delta, abs(mean_delta))
                rows.append({
                    "surface": surface, "budget": int(budget), "arm": arm,
                    "sessions_compared": len(session_deltas),
                    "sealed_sessions": len(anchor["sessions"]),
                    "recomputed_mean_r2": float(own["equal_session_mean_r2"]),
                    "sealed_mean_r2": float(
                        anchor["summary"]["equal_session_mean_r2"]
                    ),
                    "mean_delta": mean_delta,
                    "max_abs_session_delta": float(
                        max(abs(value) for value in session_deltas)
                    ),
                })
    _require(
        max_session_delta <= tolerance,
        f"trajalign per-session reproduction drift {max_session_delta:.3e} "
        f"exceeds {tolerance:.1e} (inputs must be bit-identical)",
    )
    if full_coverage:
        _require(
            max_mean_delta <= tolerance,
            f"trajalign mean reproduction drift {max_mean_delta:.3e} "
            f"exceeds {tolerance:.1e}",
        )
    return {
        "anchor": "sealed continuity probe trajalign mean arms",
        "tolerance": tolerance,
        "arms_checked": len(rows),
        "full_roster_coverage": full_coverage,
        "max_abs_session_delta_r2": max_session_delta,
        "max_abs_mean_delta_r2": max_mean_delta,
        "rows": rows,
    }
