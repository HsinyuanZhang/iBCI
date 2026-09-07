"""SLOT-AUDIT V1 ensembles — slot subsets of one window, stability, readings.

Work order section 4.3 / guidance section 5.2.  A subset ensemble is the
equal-weight mean of the SAME window's predictions at the subset's slots,
scored against that window's governing last-bin target (so subset {49} IS the
sealed baseline row, bit-exactly).  This is a per-window, strictly causal
transform — no cross-window pooling, no future windows.

Source selection discipline (the ONLY selection allowed): the best
single slot is chosen per budget as the argmax of the equal-session mean
per-slot R2 on the within-6 surface and then applied UNCHANGED to external.
No target (external) quantity participates in any choice.  Best-slot-per-
session rows are diagnostics only and never select anything.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import plan
from .scoring import house_session_r2


class SlotAuditEnsembleError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditEnsembleError(message)


def subset_mean(full_predictions: Any, slots: Sequence[int]) -> np.ndarray:
    """Equal-weight mean over the given slots: [W, 2] float64."""
    full = np.asarray(full_predictions, dtype=np.float64)
    indices = [int(slot) for slot in slots]
    _require(bool(indices), "empty slot subset")
    _require(
        all(0 <= slot < plan.WINDOW_BINS for slot in indices),
        "slot subset out of range",
    )
    return full[:, indices, :].mean(axis=1)


def subset_row(
    full_predictions: Any, targets: Any, valid: Any, slots: Sequence[int], *,
    name: str, session: str, source_selected: bool = False,
) -> dict[str, Any]:
    """One subset's governing row for ONE session (house float32 scorer)."""
    mean64 = subset_mean(full_predictions, slots)
    targets32 = np.asarray(targets, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    indices = [int(slot) for slot in slots]
    mean32 = np.ascontiguousarray(mean64[valid], dtype=np.float32)
    target32 = np.ascontiguousarray(targets32[valid], dtype=np.float32)
    _require(mean32.shape == target32.shape, "subset scoring shape drift")
    return {
        "subset": name,
        "session": str(session),
        "slots": indices,
        "n_slots": len(indices),
        "source_selected": bool(source_selected),
        "variance_weighted_r2": house_session_r2(mean32, target32),
    }


def best_source_slot(per_slot_by_session_within: Mapping[str, list[Mapping]]) -> int:
    """Argmax over slots of the within-surface equal-session mean per-slot R2.

    Deterministic tie-break: the LOWEST slot id.  Uses only within-surface
    rows (guidance section 5.2 / work order section 4.3).
    """
    means = np.asarray([
        [float(row["variance_weighted_r2"]) for row in per_slot_by_session_within[name]]
        for name in sorted(per_slot_by_session_within)
    ], dtype=np.float64)
    _require(means.ndim == 2 and means.shape[1] == plan.WINDOW_BINS,
             "best-source-slot table shape drift")
    return int(np.argmax(means.mean(axis=0)))


def best_slot_per_session(per_slot_by_session: Mapping[str, list[Mapping]]) -> dict[str, Any]:
    """Per-session argmax slot ids + stability diagnostics (never selects)."""
    sessions = sorted(per_slot_by_session)
    argmax = {
        name: int(np.argmax([
            float(row["variance_weighted_r2"])
            for row in per_slot_by_session[name]
        ]))
        for name in sessions
    }
    values = np.asarray([argmax[name] for name in sessions], dtype=np.int64)
    counts = np.bincount(values, minlength=plan.WINDOW_BINS)
    modal = int(np.argmax(counts))
    # Spearman-style pairwise rank overlap between sessions over all 50 slots
    matrix = np.asarray([
        [float(row["variance_weighted_r2"]) for row in per_slot_by_session[name]]
        for name in sessions
    ], dtype=np.float64)
    ranks = np.argsort(np.argsort(matrix, axis=1), axis=1).astype(np.float64)
    n = len(sessions)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            a = ranks[i] - ranks[i].mean()
            b = ranks[j] - ranks[j].mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            overlaps.append(float((a * b).sum() / denom) if denom > 0 else 0.0)
    return {
        "per_session_argmax_slot": argmax,
        "distinct_argmax_slots": int((counts > 0).sum()),
        "modal_argmax_slot": modal,
        "modal_argmax_session_count": int(counts[modal]),
        "n_sessions": n,
        "mean_pairwise_spearman_rank_overlap": (
            float(np.mean(overlaps)) if overlaps else None
        ),
    }


# ---------------------------------------------------------------------------
# the pre-registered reading classification (mechanical)
# ---------------------------------------------------------------------------


def _nonincreasing_over_tail(mean_r2: Sequence[float], tail_lo: int = plan.TAIL_LO) -> bool:
    tail = [float(value) for value in mean_r2[tail_lo:]]
    return all(a >= b for a, b in zip(tail, tail[1:]))


def _nondecreasing_over_tail(mean_r2: Sequence[float], tail_lo: int = plan.TAIL_LO) -> bool:
    tail = [float(value) for value in mean_r2[tail_lo:]]
    return all(a <= b for a, b in zip(tail, tail[1:]))


def classify_reading(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered reading classification, decided mechanically.

    ``cells`` maps ``f"{surface}_M{budget}"`` to a cell summary carrying:
      - ``per_slot_mean_r2``: the 50 equal-session mean per-slot R2 values;
      - ``subset_mean_r2``: {subset name -> equal-session mean R2};
      - ``subset_positive_sessions``: {subset name -> #sessions with paired
        delta vs {slot49} > 0};
      - ``n_sessions``.
    Rule (binding work order section 5, operationalized before results):
      (a) redundancy_ensembling iff the per-slot R2 is monotonically
          NON-INCREASING in s over the tail slots {40..49} in EVERY cell AND
          at least one averaging subset ({last5}, {last10}, all-50) beats
          {slot49} in the equal-session mean in EVERY cell;
      (b) stable_better_subset iff some candidate subset (fixed geometric or
          the per-budget source-selected single slot) beats {slot49} on
          >= 80% of sessions in BOTH surfaces at some budget;
      (c) session_dependent otherwise.  (a) takes precedence over (b).
    """
    _require(bool(cells), "classification needs cells")
    per_cell_flags: dict[str, dict[str, Any]] = {}
    for key, cell in cells.items():
        mean_r2 = [float(value) for value in cell["per_slot_mean_r2"]]
        _require(len(mean_r2) == plan.WINDOW_BINS, f"{key}: per-slot table size drift")
        baseline = float(cell["subset_mean_r2"]["slot49"])
        gains = {
            name: float(cell["subset_mean_r2"][name]) - baseline
            for name in plan.AVERAGING_SUBSETS
        }
        per_cell_flags[key] = {
            "tail_nonincreasing": _nonincreasing_over_tail(mean_r2),
            "tail_nondecreasing": _nondecreasing_over_tail(mean_r2),
            "best_averaging_subset_gain": max(gains.values()),
            "averaging_subset_gains": gains,
            "n_sessions": int(cell["n_sessions"]),
        }

    n_cells = len(cells)
    monotone_all = all(flag["tail_nonincreasing"] for flag in per_cell_flags.values())
    gain_all = all(flag["best_averaging_subset_gain"] > 0.0 for flag in per_cell_flags.values())
    monotone_majority = (
        sum(flag["tail_nonincreasing"] for flag in per_cell_flags.values())
        >= n_cells - 1
    )
    gain_majority = (
        sum(flag["best_averaging_subset_gain"] > 0.0 for flag in per_cell_flags.values())
        >= n_cells - 1
    )

    stable_candidates: list[dict[str, Any]] = []
    for budget in plan.BUDGETS:
        within = cells.get(f"within_M{budget}")
        external = cells.get(f"external_M{budget}")
        if within is None or external is None:
            continue
        for name in list(plan.AVERAGING_SUBSETS) + [plan.SOURCE_SELECTED_SUBSET]:
            if name not in within["subset_mean_r2"] or name not in external["subset_mean_r2"]:
                continue
            need_within = plan.STABLE_SUBSET_SESSION_FRACTION * int(within["n_sessions"])
            need_external = plan.STABLE_SUBSET_SESSION_FRACTION * int(external["n_sessions"])
            pos_within = float(within["subset_positive_sessions"][name])
            pos_external = float(external["subset_positive_sessions"][name])
            if pos_within >= need_within and pos_external >= need_external:
                stable_candidates.append({
                    "subset": name, "budget": int(budget),
                    "positive_sessions_within": int(pos_within),
                    "n_sessions_within": int(within["n_sessions"]),
                    "positive_sessions_external": int(pos_external),
                    "n_sessions_external": int(external["n_sessions"]),
                })

    if monotone_all and gain_all:
        classification = "redundancy_ensembling"
    elif stable_candidates:
        classification = "stable_better_subset"
    else:
        classification = "session_dependent"

    return {
        "classification": classification,
        "rule": (
            "(a) redundancy_ensembling iff per-slot R2 is monotonically "
            "non-increasing in s over the tail slots {40..49} in EVERY "
            "(surface, budget) cell AND at least one averaging subset beats "
            "{slot49} in every cell; (b) stable_better_subset iff some "
            "candidate subset beats {slot49} on >= 80% of sessions in BOTH "
            "surfaces at some budget; (c) session_dependent otherwise; "
            "(a) takes precedence over (b)"
        ),
        "operationalization_notes": [
            "the work-order phrase for (a) reads 'earlier slots individually "
            "worse, average helps', which corresponds to the OPPOSITE "
            f"monotonicity direction (non-decreasing in s); tail_nondecreasing "
            "flags are therefore reported per cell so the reading is "
            "transparent under either wording",
            "the (b) test is evaluated per budget; a subset qualifies by "
            "holding >= 80% positive sessions on BOTH surfaces at that budget",
        ],
        "redundancy_legs": {
            "tail_nonincreasing_every_cell": monotone_all,
            "averaging_gain_every_cell": gain_all,
            "tail_nonincreasing_all_but_one": monotone_majority,
            "averaging_gain_all_but_one": gain_majority,
        },
        "stable_better_subset_candidates": stable_candidates,
        "per_cell_flags": per_cell_flags,
        "target_label_leakage": False,
        "deployment_eligible": False,
        "checkpoint_selection_eligible": False,
    }
