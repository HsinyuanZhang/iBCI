"""Phase 3: the 2x2 PIT-M2 factorial readout and the pre-declared gates.

Cells (first digit = model axis, second digit = deployment path):

* F00m -- SEALED static row: the ``25d7bc72`` checkpoint under the
  official-contract static surface (``external_official_query:ridge_static_m10``
  of the sealed ``m2_t4_activity_budget_screen_v1``), referenced not rerun;
* F01m -- SEALED local-FIFO row: the same checkpoint under the local FIFO
  path (``external_post30_local:m10_activity_only`` of the sealed
  ``m2_precision_cdm_v2_screen_v1``), referenced not rerun;
* F10m -- the fresh PIT-trained c1m arm scored with the sealed static law;
* F11m -- the fresh PIT-trained c1m arm scored with the sealed FIFO law.

Gate: ``F10m - F00m >= +0.01`` equal-session mean R2 on the official-contract
static surface AND >= 4/6 positive external sessions (primary);
``F11m - F01m`` is the stacking-reading secondary, reported with NO promotion
attached.  Exact ``>=`` on float64 with the 1e-12 disclosure band.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import plan


class Phase3Error(RuntimeError):
    """Fail closed for phase-3 scoring or gate drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3Error(message)


# ---------------------------------------------------------------------------
# Sealed anchors (F00m / F01m): referenced, never rerun.
# ---------------------------------------------------------------------------

def load_sealed_anchors(repo_root: Path) -> dict[str, dict[str, object]]:
    """Verify both sealed score files by sha256 and extract the anchor cells."""
    base = Path(repo_root).absolute()
    anchors: dict[str, dict[str, object]] = {}
    for cell, anchor in (("F00m", plan.F00M_ANCHOR), ("F01m", plan.F01M_ANCHOR)):
        path = base / anchor["path"]
        digest = plan.sha256_file(path)
        _require(digest == anchor["sha256"], f"pit m2 {cell} sealed anchor drift: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        sessions = _extract_sealed_cell(payload, anchor)
        _require(sorted(sessions) == sorted(anchor["per_session_r2"]),
                 f"pit m2 {cell} sealed anchor roster drift")
        anchors[cell] = {
            "source_path": anchor["path"],
            "source_sha256": anchor["sha256"],
            "cell": anchor["cell"],
            "surface": anchor["surface"],
            "per_session_r2": {name: float(sessions[name]) for name in sorted(sessions)},
            "equal_session_mean": _equal_session_mean(sessions),
            "checkpoint_sha256": anchor["checkpoint_sha256"],
            "rerun_by_this_lane": False,
        }
    return anchors


def _extract_sealed_cell(payload: Mapping[str, object],
                         anchor: Mapping[str, object]) -> dict[str, float]:
    if anchor["cell"] == "ridge_static_m10":
        summaries = payload["summaries"]
        key = f"{anchor['surface']}:{anchor['cell']}"
        _require(key in summaries, f"pit m2 sealed static cell missing: {key}")
        return dict(summaries[key]["per_session_r2"])
    rows = payload["rows"]
    selected = {
        str(row["session_id"]): float(row["r2"])
        for row in rows
        if row["cell"] == anchor["cell"] and row["surface"] == anchor["surface"]
    }
    _require(bool(selected), "pit m2 sealed FIFO cell missing")
    return selected


def _equal_session_mean(values: Mapping[str, float]) -> float:
    ordered = {name: float(values[name]) for name in sorted(values)}
    _require(bool(ordered), "pit m2 equal-session mean needs sessions")
    return float(np.asarray(list(ordered.values()), dtype=np.float64).mean())


# ---------------------------------------------------------------------------
# Fresh-arm scoring (the sealed screens' own laws, verbatim reuse).
# ---------------------------------------------------------------------------

def score_static_session(*, torch: Any, student: Any, dataset: Any, session: str,
                         surface: str, device: Any,
                         batch_size: int = 1024) -> dict[str, object]:
    """F10m/t0m static cell: the sealed ridge_static_m10 law, verbatim."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical

    cell = screen_core.CellSpec("ridge_static_m10", plan.DEPLOYMENT_BUDGET_M,
                                plan.DEPLOYMENT_BUDGET_M)
    row = screen_physical._score_session(
        torch=torch, student=student, dataset=dataset, session=session,
        surface=surface, cell=cell, device=device, batch_size=batch_size,
    )
    row.update({
        "schema": "pit_m2_phase3_static_cell_v1",
        "path": "static",
        "budget": plan.DEPLOYMENT_BUDGET_M,
    })
    return row


def score_fifo_session(*, torch: Any, litmodule: Any, dataset: Any, raw_session: Any,
                       session: str, side_mean: np.ndarray, side_std: np.ndarray,
                       device: Any, batch_size: int = 1024) -> dict[str, object]:
    """F11m/t0m local-FIFO cell: the sealed m10_activity_only law, verbatim.

    ``litmodule`` is the Lightning module (the sealed screen's ``model``
    argument: ``shared_physical._predict`` reads ``model.student``).
    """
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as shared_core

    raw_neural, raw_starts, theta30, support_rates_hz30, activities = (
        cdm_physical._native_trial_views(raw_session, session=session)
    )
    selected = cdm_physical._finite_m10_indices(theta30)
    _raw_m30_hz, valid_mask = cdm_physical._raw_fixed_ridge_m30(support_rates_hz30, theta30)
    query_rows = cdm_physical._query_trial_rows(
        dataset, session, raw_neural=raw_neural, raw_starts=raw_starts,
        activities=activities,
    )
    channels = np.arange(activities.shape[2], dtype=np.int64)
    channel_sha = cdm.channel_order_digest(channels)
    support_b3s = tuple(
        cdm.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(activities[index], dtype=np.float32),
            session_id=session, trial_id=f"{session}:trial:{index}",
            channel_order_sha256=channel_sha,
        )
        for index in range(cdm_plan.ACTIVITY_STACK_LIMIT)
    )
    memory = cdm_physical._build_memory(
        system=cdm_plan.SYSTEM_ACTIVITY, budget=plan.DEPLOYMENT_BUDGET_M,
        selected=selected, support_rates_hz30=support_rates_hz30,
        theta_first30=theta30, support_b3s=support_b3s, channel_ids=channels,
        valid_mask=valid_mask,
    )
    row = cdm_physical._score_system(
        torch=torch, model=litmodule, dataset=dataset, session=session, memory=memory,
        query_rows=query_rows, side_mean=side_mean, side_std=side_std,
        raw_neural=raw_neural, device=device, batch_size=batch_size,
    )
    row.update({
        "schema": "pit_m2_phase3_fifo_cell_v1",
        "path": "fifo",
        "budget": plan.DEPLOYMENT_BUDGET_M,
        "support_indices": selected.tolist(),
        "support_indices_sha256": shared_core.array_sha256(selected),
    })
    return row


def summarize_cells(rows: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    _require(bool(rows), "pit m2 cell summary needs rows")
    ordered = {name: float(rows[name]) for name in sorted(rows)}
    values = np.asarray(list(ordered.values()), dtype=np.float64)
    _require(np.isfinite(values).all(), "pit m2 cell r2 nonfinite")
    return {
        "session_count": int(values.size),
        "equal_session_mean": float(values.mean()),
        "per_session_r2": ordered,
    }


# ---------------------------------------------------------------------------
# The 2x2 table and the gates.
# ---------------------------------------------------------------------------

def build_table(anchors: Mapping[str, Mapping[str, object]],
                fresh: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Assemble the 2x2 (plus the reported t0m control rows).

    ``fresh`` must be keyed by ``{arm}_{path}_{surface}`` -- the attempt-3 bug
    keyed it by ``{arm}_{path}`` alone, so the within_post30 summary CLOBBERED
    the external summary and the primary contrast failed with a paired session
    set mismatch.  The gate cells bind the EXTERNAL surfaces only, fail-closed
    on the frozen external roster.
    """
    for cell in ("F00m", "F01m"):
        _require(cell in anchors, f"pit m2 table missing sealed anchor {cell}")
    primary_key = "c1m_static_external_official_query"
    secondary_key = "c1m_fifo_external_post30_local"
    _require(primary_key in fresh and secondary_key in fresh,
             f"pit m2 table missing the fresh external c1m cells "
             f"({primary_key}/{secondary_key}); fresh keys: {sorted(fresh)}")
    for key, anchor_cell in ((primary_key, "F00m"), (secondary_key, "F01m")):
        observed = sorted(fresh[key]["per_session_r2"])
        _require(observed == sorted(anchors[anchor_cell]["per_session_r2"])
                 == sorted(plan.EXTERNAL_SESSION_NAMES),
                 f"pit m2 gate cell {key} is not the external roster: {observed}")
    table = {
        "schema": "pit_m2_phase3_table_v1",
        "cells": {
            "F00m": dict(anchors["F00m"]),
            "F01m": dict(anchors["F01m"]),
            "F10m": {**dict(fresh[primary_key]), "surface": "external_official_query"},
            "F11m": {**dict(fresh[secondary_key]), "surface": "external_post30_local"},
        },
        "reported_not_gated": {
            key: dict(value) for key, value in fresh.items()
            if key not in (primary_key, secondary_key)
        },
        "metric": "variance_weighted_r2 equal-session mean",
    }
    table["contrasts"] = {
        "primary_F10m_minus_F00m": _contrast(
            fresh[primary_key]["per_session_r2"], anchors["F00m"]["per_session_r2"]),
        "secondary_F11m_minus_F01m": _contrast(
            fresh[secondary_key]["per_session_r2"], anchors["F01m"]["per_session_r2"]),
    }
    return table


def _contrast(candidate: Mapping[str, float],
              reference: Mapping[str, float]) -> dict[str, object]:
    _require(set(candidate) == set(reference) and bool(candidate),
             "pit m2 paired session set mismatch")
    differences = {name: float(candidate[name]) - float(reference[name])
                   for name in sorted(candidate)}
    values = np.asarray(list(differences.values()), dtype=np.float64)
    return {
        "candidate_minus_reference_mean": float(values.mean()),
        "per_session_delta": differences,
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "session_count": int(values.size),
    }


def evaluate_gate(*, delta_mean: float, positive_sessions: int,
                  session_count: int, floor: float, breadth_min: int) -> dict[str, object]:
    """Exact ``>=`` on float64; a miss inside the 1e-12 band is disclosed only."""
    _require(session_count > 0, "pit m2 gate needs sessions")
    exact_delta_pass = bool(delta_mean >= floor)
    within_band = bool(abs(delta_mean - floor) <= plan.GATE_BOUNDARY_EPSILON
                       and not exact_delta_pass)
    breadth_pass = bool(positive_sessions >= breadth_min and session_count >= breadth_min)
    if exact_delta_pass:
        delta_verdict = "PASSED"
    elif within_band:
        delta_verdict = "NOT_MET__WITHIN_EPSILON_BAND_OF_BOUNDARY"
    else:
        delta_verdict = "NOT_MET"
    if delta_verdict == "PASSED" and breadth_pass:
        verdict = "PASSED"
    elif delta_verdict == "PASSED":
        verdict = "NOT_MET__BREADTH"
    else:
        verdict = delta_verdict
    return {
        "delta_mean": float(delta_mean),
        "delta_floor": float(floor),
        "exact_delta_pass": exact_delta_pass,
        "within_epsilon_band_of_boundary": within_band,
        "epsilon": plan.GATE_BOUNDARY_EPSILON,
        "positive_sessions": int(positive_sessions),
        "breadth_min": int(breadth_min),
        "session_count": int(session_count),
        "breadth_pass": breadth_pass,
        "delta_verdict": delta_verdict,
        "verdict": verdict,
    }


def evaluate_gates(table: Mapping[str, object]) -> dict[str, object]:
    contrasts = table["contrasts"]
    primary_spec = plan.GATES["primary"]
    secondary_spec = plan.GATES["secondary_stacking"]
    primary = evaluate_gate(
        delta_mean=contrasts["primary_F10m_minus_F00m"]["candidate_minus_reference_mean"],
        positive_sessions=contrasts["primary_F10m_minus_F00m"]["positive_sessions"],
        session_count=contrasts["primary_F10m_minus_F00m"]["session_count"],
        floor=primary_spec["delta_floor"], breadth_min=primary_spec["breadth_min"],
    )
    secondary = evaluate_gate(
        delta_mean=contrasts["secondary_F11m_minus_F01m"]["candidate_minus_reference_mean"],
        positive_sessions=contrasts["secondary_F11m_minus_F01m"]["positive_sessions"],
        session_count=contrasts["secondary_F11m_minus_F01m"]["session_count"],
        floor=secondary_spec["delta_floor"], breadth_min=secondary_spec["breadth_min"],
    )
    return {
        "schema": "pit_m2_phase3_gates_v1",
        "primary": {
            **primary,
            "expression": primary_spec["expression"],
            "disposition_passed": primary_spec["disposition_passed"],
            "disposition_failed": primary_spec["disposition_failed"],
            "final_disposition": (primary_spec["disposition_passed"]
                                  if primary["verdict"] == "PASSED"
                                  else primary_spec["disposition_failed"]),
        },
        "secondary_stacking": {
            **secondary,
            "expression": secondary_spec["expression"],
            "role": secondary_spec["role"],
            "disposition": secondary_spec["disposition"],
            "promotion_attached": False,
        },
        "boundary_rule": plan.GATES["boundary_rule"],
        "reported_not_gated": list(plan.GATES["reported_not_gated"]),
    }


__all__ = (
    "Phase3Error", "load_sealed_anchors", "score_static_session", "score_fifo_session",
    "summarize_cells", "build_table", "evaluate_gate", "evaluate_gates",
)
