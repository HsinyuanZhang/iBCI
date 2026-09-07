"""SLOT-AUDIT V1 alignment — the absolute-bin law and its statistics.

THE ALIGNMENT LAW (work order section 4.1, binding): a window with stream
start ``w`` and output slot ``s`` speaks about ABSOLUTE behavior bin ``w + s``.
The same absolute bin ``b`` is estimated by every window ``b-49 <= w <= b`` at
slot ``b - w``.  Slot comparisons are therefore within-absolute-bin only, and
every target below is gathered from the session-level behavior-bin array at
the ABSOLUTE bin index (never re-derived from other windows).

Two facts make the law exact on this producer (verified at materialization
time and re-asserted here per session):
  - the frozen DANDI loader draws windows strictly inside single rewarded
    trials (``range(trial.start, trial.stop - 50 + 1)``), so every bin of a
    window lies in the window's own trial — "bin w+s within the same trial"
    is a property of the loader, not a filter applied here;
  - redundant estimates of one absolute bin are pooled exactly like the
    frozen trajectory-alignment gather (containment ``starts[t'] <= b <
    starts[t']+50``), so the delta-curves and the sealed trajalign arm see
    the same estimator sets.

Residual pairs by slot separation delta: for member i = (window w, slot s)
and member j = (window w-delta, slot s+delta) — both estimate absolute bin
``w + s`` — the 2-D residual vectors r_i, r_j are centered per (session,
slot-separation column) before pooling, flattened over bins AND output dims,
and summarized per (surface, budget) by pooled correlation, covariance and
residual stds (work order section 4.2).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import plan
from .scoring import (
    house_session_r2,
    pooled_correlation_from_sums,
    residual_statistics,
    variance_weighted_r2_numpy,
)


class SlotAuditAlignmentError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditAlignmentError(message)


# ---------------------------------------------------------------------------
# the absolute-bin index law
# ---------------------------------------------------------------------------


def window_bin_matrix(starts: Any) -> np.ndarray:
    """[W, 50] absolute behavior bins: bin[w, s] = starts[w] + s."""
    starts = np.asarray(starts, dtype=np.int64)
    _require(starts.ndim == 1 and starts.size >= 1, "window starts shape drift")
    return starts[:, None] + np.arange(plan.WINDOW_BINS, dtype=np.int64)[None, :]


def slot_pair_mask(starts: Any, window_valid: Any, bin_valid: Any, slot: int) -> np.ndarray:
    """Windows whose slot-``slot`` absolute bin has an existing valid target.

    ``bin_valid`` carries BOTH target-existence conditions (finite, and not
    the loader pad sentinel -1.0 on either dimension).
    """
    starts = np.asarray(starts, dtype=np.int64)
    window_valid = np.asarray(window_valid, dtype=bool)
    bin_valid = np.asarray(bin_valid, dtype=bool)
    bins = starts + int(slot)
    _require(
        int(bins.max()) < bin_valid.size and int(bins.min()) >= 0,
        "slot bins fall outside the behavior array (alignment drift)",
    )
    return window_valid & bin_valid[bins]


def gather_slot_pairs(
    full_predictions: Any, behavior: Any, starts: Any, window_valid: Any,
    bin_valid: Any, slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(prediction[w, slot, :], behavior[w + slot]) over the valid windows."""
    full = np.asarray(full_predictions, dtype=np.float32)
    behavior = np.asarray(behavior, dtype=np.float32)
    mask = slot_pair_mask(starts, window_valid, bin_valid, slot)
    starts = np.asarray(starts, dtype=np.int64)
    pred = np.ascontiguousarray(full[mask, slot, :], dtype=np.float32)
    target = np.ascontiguousarray(behavior[starts[mask] + int(slot)], dtype=np.float32)
    return pred, target


def per_slot_rows(
    full_predictions: Any, behavior: Any, starts: Any, window_valid: Any,
    bin_valid: Any,
) -> list[dict[str, Any]]:
    """Per-slot calibration rows for ONE session (50 entries, slots 0..49).

    R2 is the house float32 scorer (identical call path to the sealed rows);
    the float64 replica is carried alongside as the audit cross-check.
    """
    rows: list[dict[str, Any]] = []
    for slot in plan.SLOTS:
        pred, target = gather_slot_pairs(
            full_predictions, behavior, starts, window_valid, bin_valid, slot,
        )
        _require(pred.shape[0] >= 3, f"slot {slot} has too few valid pairs")
        stats = residual_statistics(pred, target)
        rows.append({
            "slot": int(slot),
            "n": stats["n"],
            "variance_weighted_r2": house_session_r2(pred, target),
            "variance_weighted_r2_float64": variance_weighted_r2_numpy(pred, target),
            "mse": stats["mse"],
            "bias": stats["bias"],
            "residual_variance": stats["residual_variance"],
            "residual_std": stats["residual_std"],
        })
    return rows


def equal_session_mean_per_slot(per_slot_by_session: Mapping[str, list[Mapping]]) -> list[dict]:
    """Equal-session means of the per-slot rows (preserves slot order)."""
    sessions = sorted(per_slot_by_session)
    _require(bool(sessions), "no sessions to aggregate")
    n_slots = len(per_slot_by_session[sessions[0]])
    out: list[dict] = []
    for index in range(n_slots):
        rows = [per_slot_by_session[name][index] for name in sessions]
        _require(
            all(int(row["slot"]) == int(rows[0]["slot"]) for row in rows),
            "per-slot aggregation slot-order drift",
        )
        out.append({
            "slot": int(rows[0]["slot"]),
            "n_sessions": len(rows),
            "mean_variance_weighted_r2": float(
                np.mean([float(row["variance_weighted_r2"]) for row in rows])
            ),
            "mean_mse": float(np.mean([float(row["mse"]) for row in rows])),
            "mean_bias": [
                float(np.mean([float(row["bias"][d]) for row in rows])) for d in (0, 1)
            ],
            "mean_residual_variance": [
                float(np.mean([float(row["residual_variance"][d]) for row in rows]))
                for d in (0, 1)
            ],
            "mean_n": float(np.mean([float(row["n"]) for row in rows])),
        })
    return out


# ---------------------------------------------------------------------------
# redundant-estimate residual pairs by slot separation delta
# ---------------------------------------------------------------------------


def session_delta_sums(
    full_predictions: Any, behavior: Any, starts: Any, window_valid: Any,
    bin_valid: Any,
) -> dict[int, dict[str, float]]:
    """Per-delta pooled moment sums for ONE session (centered per slot column).

    Pair law (absolute-bin, gap-safe): member i = (window w1, slot s) and
    member j = (window w2, slot s+delta) form a redundant pair of the SAME
    absolute bin iff ``starts[w2] == starts[w1] - delta`` — i.e. the partner
    is resolved on the ABSOLUTE START GRID (searchsorted exact match), never
    on window index proximity, so trial gaps can never pair two estimates of
    different bins.  With the loader's stride-1 starts inside trials this is
    exactly the containment law of the frozen trajalign gather.

    Each slot-separation column is centered over its valid pairs before
    pooling (removes the per-slot bias term so the correlation measures the
    shared error MODE, not the shared mean offset).
    """
    full = np.asarray(full_predictions, dtype=np.float32)
    behavior = np.asarray(behavior, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    window_valid = np.asarray(window_valid, dtype=bool)
    bin_valid = np.asarray(bin_valid, dtype=bool)
    n_windows = starts.size
    _require(
        bool(np.all(np.diff(starts) > 0)),
        "window starts must be strictly increasing for absolute-bin pairing",
    )
    bins = window_bin_matrix(starts)
    _require(
        int(bins.max()) < bin_valid.size, "window bins exceed the behavior array"
    )
    ok = window_valid[:, None] & bin_valid[bins]                      # [W, 50]
    residual = (
        full.astype(np.float64) - behavior[bins].astype(np.float64)
    )                                                                # [W, 50, 2]
    out: dict[int, dict[str, float]] = {}
    for delta in range(1, plan.WINDOW_BINS):
        columns = plan.WINDOW_BINS - delta
        if columns <= 0:
            continue
        partner_start = starts - delta
        location = np.searchsorted(starts, partner_start)
        inside = location < n_windows
        location_safe = np.where(inside, location, 0)
        matched = inside & (starts[location_safe] == partner_start)
        w1 = np.flatnonzero(matched)          # later window (slot s)
        w2 = location_safe[w1]                # earlier window (slot s + delta)
        if w1.size < 2:
            continue
        columns_index = np.arange(columns, dtype=np.int64)
        x = residual[w1[:, None], columns_index[None, :], :]
        y = residual[w2[:, None], (columns_index + delta)[None, :], :]
        good = ok[w1[:, None], columns_index[None, :]] & ok[
            w2[:, None], (columns_index + delta)[None, :]
        ]
        count = good.sum(axis=0)                                     # [columns]
        usable = count > 0
        weight = good[:, :, None]
        safe_count = np.where(usable, count, 1)
        x_centered = x - np.where(
            usable[None, :, None],
            (x * weight).sum(axis=0) / safe_count[None, :, None], 0.0,
        )
        y_centered = y - np.where(
            usable[None, :, None],
            (y * weight).sum(axis=0) / safe_count[None, :, None], 0.0,
        )
        xv = x_centered[good]
        yv = y_centered[good]
        _require(int(xv.size) > 1, f"delta {delta} pool degenerated")
        out[delta] = {
            "n": float(xv.shape[0]),
            "sx": float(xv.sum()),
            "sy": float(yv.sum()),
            "sxx": float((xv * xv).sum()),
            "syy": float((yv * yv).sum()),
            "sxy": float((xv * yv).sum()),
            "n_slot_columns": int(usable.sum()),
            "n_window_pairs": int(w1.size),
        }
    return out


def pool_delta_sums(session_sums: list[Mapping[int, Mapping]]) -> dict[int, dict[str, float]]:
    """Sum the per-session moment sums (the pooled per-(surface,budget) pool)."""
    _require(bool(session_sums), "no session delta sums to pool")
    keys = set(session_sums[0])
    _require(
        all(set(entry) == keys for entry in session_sums),
        "delta support drift across sessions",
    )
    pooled: dict[int, dict[str, float]] = {}
    for delta in sorted(keys):
        total = {
            field: float(sum(float(entry[delta][field]) for entry in session_sums))
            for field in ("n", "sx", "sy", "sxx", "syy", "sxy")
        }
        total["n_sessions"] = float(len(session_sums))
        total["mean_n_per_session"] = total["n"] / float(len(session_sums))
        total["n_slot_columns"] = int(session_sums[0][delta]["n_slot_columns"])
        total["mean_window_pairs_per_session"] = float(
            sum(int(entry[delta]["n_window_pairs"]) for entry in session_sums)
        ) / float(len(session_sums))
        pooled[delta] = total
    return pooled


def delta_curve(pooled: Mapping[int, Mapping]) -> list[dict[str, Any]]:
    """The per-delta rows: pooled correlation, covariance, residual stds."""
    rows: list[dict[str, Any]] = []
    for delta in sorted(pooled):
        stats = pooled_correlation_from_sums(pooled[delta])
        rows.append({
            "delta": int(delta),
            "correlation": stats["correlation"],
            "covariance": stats["covariance"],
            "residual_std_x": stats["std_x"],
            "residual_std_y": stats["std_y"],
            "n_pairs": int(stats["n"]),
            "mean_pairs_per_session": float(pooled[delta]["mean_n_per_session"]),
        })
    return rows


# ---------------------------------------------------------------------------
# disjoint window-parity groups (context row for the prior rho_group)
# ---------------------------------------------------------------------------


def session_parity_sums(
    full_predictions: Any, behavior: Any, starts: Any, window_valid: Any,
    bin_valid: Any,
) -> dict[str, float]:
    """Moment sums of governing last-bin residuals, even vs odd window index.

    The two groups are disjoint estimator sets (no shared window).  They
    estimate ADJACENT absolute bins (b and b+1), so this is NOT the prior
    ``rho_group`` (disjoint unit groups, same per-bin velocity errors, 6 sub-C
    sessions at M4) — it is reported for context only, with the mandated
    hierarchical-variance caveat, never as a common-mode variance fraction.
    """
    pred, target = gather_slot_pairs(
        full_predictions, behavior, starts, window_valid, bin_valid,
        plan.GOVERNING_BIN,
    )
    residual = (pred - target).astype(np.float64)                    # [W, 2]
    n = residual.shape[0]
    _require(n >= 4, "parity groups need >= 4 valid windows")
    even = residual[0::2]
    odd = residual[1::2]
    m = min(even.shape[0], odd.shape[0])
    x = np.ascontiguousarray(even[:m] - even[:m].mean(axis=0, keepdims=True))
    y = np.ascontiguousarray(odd[:m] - odd[:m].mean(axis=0, keepdims=True))
    xv = x.reshape(-1)
    yv = y.reshape(-1)
    return {
        "n": float(xv.size),
        "sx": float(xv.sum()), "sy": float(yv.sum()),
        "sxx": float((xv * xv).sum()), "syy": float((yv * yv).sum()),
        "sxy": float((xv * yv).sum()),
    }


def pool_parity_sums(session_sums: list[Mapping]) -> dict[str, Any]:
    _require(bool(session_sums), "no parity sums to pool")
    total = {
        field: float(sum(float(entry[field]) for entry in session_sums))
        for field in ("n", "sx", "sy", "sxx", "syy", "sxy")
    }
    stats = pooled_correlation_from_sums(total)
    return {
        "definition": (
            "disjoint window-parity groups (even vs odd window index) of the "
            "governing last-bin residuals, centered per session per dim, "
            "pooled over bins and dims"
        ),
        "prior_rho_group_context": {
            "value": 0.9171,
            "surface": (
                "disjoint UNIT-group per-bin velocity errors, 6 sub-C source "
                "sessions at M4 (DESIGN_COMMON_MODE_BOUNDED_CARRIER_"
                "SUCCESSOR_20260829.md section 3.3) — a DIFFERENT measurement "
                "surface; the two numbers are not comparable"
            ),
        },
        **stats,
    }
