"""The work-order section 2 gates for the Part A factorial.

Boundaries are pre-registered in :mod:`.plan` and applied EXACTLY here:

* every margin comparison is an exact ``>=`` on the float64 equal-session mean;
* ``GATE_BOUNDARY_EPSILON = 1e-12`` only surfaces a disclosed
  ``within_epsilon_band_of_boundary`` flag; it never flips a verdict;
* the per-cell promotion gate is evaluated on the driving budget (external M4)
  for each candidate cell F10/F01/F11 against the F00 baseline;
* the additivity gate demands F11 >= max(F10, F01) - 0 on the same contrast;
* the safety floors are within-every-budget >= -0.02 and external M30 >= -0.02;
* M30 is never averaged with M4/M10, and nulls are reported as nulls.

``view[m<budget>][surface][cell] = {"mean_r2": ..., "per_session": {...}}``.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import plan


class CrossGateError(ValueError):
    """Fail-closed error for malformed gate inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossGateError(message)


def _budget_key(budget: int) -> str:
    return f"m{int(budget)}"


SAFETY_CONDITIONS = (
    "zero_target_updates",
    "f00_bit_anchor",
    "f01_bit_anchor",
    "causality_state_chains",
    "weight_swap_binding_verified",
    "sealed_model_digest_restored",
    "initial_carrier_invariance",
    "m30_exact_noop_per_arm",
    "hyperparameters_sealed_selection_only",
    "trust_region_no_committed_drift",
)


def margin_verdict(value: float, threshold: float) -> dict[str, Any]:
    """Exact ``>=`` verdict with the disclosed 1e-12 boundary band."""
    delta = float(value) - float(threshold)
    return {
        "value": float(value),
        "threshold": float(threshold),
        "meets_margin": bool(delta >= 0.0),
        "within_epsilon_band_of_boundary": bool(
            -plan.GATE_BOUNDARY_EPSILON <= delta < 0.0
        ),
        "boundary_epsilon": plan.GATE_BOUNDARY_EPSILON,
    }


def _cell_view(view: Mapping[str, Any], budget: int, surface: str, cell: str) -> Mapping[str, Any]:
    surface_map = view.get(_budget_key(budget), {}).get(surface, {})
    _require(cell in surface_map, f"gate input is missing cell {cell} at M{budget} {surface}")
    return surface_map[cell]


def equal_session_mean(payload: Mapping[str, Any]) -> float:
    per_session = payload.get("per_session")
    _require(isinstance(per_session, Mapping) and bool(per_session),
             "gate cell needs per-session values")
    return float(sum(float(value) for value in per_session.values())) / len(per_session)


def paired_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    left = dict(candidate["per_session"])
    right = dict(baseline["per_session"])
    _require(tuple(sorted(left)) == tuple(sorted(right)),
             "paired gate cells disagree on the session roster")
    deltas = {key: float(left[key]) - float(right[key]) for key in sorted(left)}
    return {
        "per_session": deltas,
        "equal_session_mean_delta": float(sum(deltas.values())) / len(deltas),
        "positive_sessions": sum(1 for value in deltas.values() if value > 0.0),
        "n_sessions": len(deltas),
    }


def candidate_vs_baseline(view: Mapping[str, Any], cell: str, *, budget: int,
                          surface: str = "external") -> dict[str, Any]:
    return paired_delta(
        _cell_view(view, budget, surface, cell),
        _cell_view(view, budget, surface, plan.BASELINE_CELL),
    )


def _within_bounds(view: Mapping[str, Any], cell: str) -> dict[str, Any]:
    bounds = {
        _budget_key(budget): margin_verdict(
            candidate_vs_baseline(view, cell, budget=budget, surface="within")[
                "equal_session_mean_delta"
            ],
            plan.WITHIN_EVERY_BUDGET_FLOOR,
        )
        for budget in plan.BUDGETS
    }
    return {
        "per_budget": bounds,
        "all_meet_margin": all(item["meets_margin"] for item in bounds.values()),
    }


def _m30_external_bound(view: Mapping[str, Any], cell: str) -> dict[str, Any]:
    return margin_verdict(
        candidate_vs_baseline(view, cell, budget=30, surface="external")[
            "equal_session_mean_delta"
        ],
        plan.M30_FLOOR,
    )


def evaluate_per_cell_gates(view: Mapping[str, Any]) -> dict[str, Any]:
    """Work order section 2, clause 1 and the safety floors, per candidate cell."""
    per_cell: dict[str, Any] = {}
    for cell in plan.CANDIDATE_CELLS:
        external = candidate_vs_baseline(view, cell, budget=plan.DRIVING_BUDGET)
        within = _within_bounds(view, cell)
        m30_external = _m30_external_bound(view, cell)
        margin = margin_verdict(
            external["equal_session_mean_delta"], plan.PROMOTION_EXTERNAL_DELTA,
        )
        breadth = margin_verdict(
            float(external["positive_sessions"]),
            float(plan.PROMOTION_POSITIVE_EXTERNAL_SESSIONS),
        )
        promoted = bool(
            margin["meets_margin"] and breadth["meets_margin"]
            and within["all_meet_margin"] and m30_external["meets_margin"]
        )
        per_cell[cell] = {
            "expression": plan.GATES["per_cell_promotion"]["expression"],
            "external_m4_delta": external["equal_session_mean_delta"],
            "external_m4_margin": margin,
            "positive_external_m4_sessions": int(external["positive_sessions"]),
            "breadth_margin": breadth,
            "within_bounds": within,
            "external_m30_safety": m30_external,
            "promoted": promoted,
            "disposition": (
                plan.DISPOSITION_CELL_PROMOTED if promoted else plan.DISPOSITION_CELL_FAILED
            ),
        }
    return {
        "schema": f"{plan.SCHEMA}_per_cell_gates_v1",
        "per_cell": per_cell,
        "epsilon_note": plan.GATES["boundary_rule"],
        "never_average_m30": True,
    }


def evaluate_additivity_gate(view: Mapping[str, Any]) -> dict[str, Any]:
    """Work order section 2, clause 2: F11 >= max(F10, F01) - 0 (external M4)."""
    means = {
        cell: equal_session_mean(_cell_view(view, plan.DRIVING_BUDGET, "external", cell))
        for cell in ("F10", "F01", "F11")
    }
    reference = max(means["F10"], means["F01"])
    verdict = margin_verdict(means["F11"], reference - plan.ADDITIVITY_SLACK)
    diagnostics = {
        surface: {
            _budget_key(budget): {
                cell: equal_session_mean(_cell_view(view, budget, surface, cell))
                for cell in ("F10", "F01", "F11")
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    return {
        "schema": f"{plan.SCHEMA}_additivity_gate_v1",
        "expression": plan.GATES["additivity"]["expression"],
        "external_m4_means": means,
        "reference_max_f10_f01": reference,
        "margin": verdict,
        "held": bool(verdict["meets_margin"]),
        "disposition": (
            plan.DISPOSITION_ADDITIVITY_HELD if verdict["meets_margin"]
            else plan.DISPOSITION_ADDITIVITY_VIOLATED
        ),
        "diagnostics_all_budgets_surfaces": diagnostics,
        "diagnostic_note": (
            "the gate itself binds only external M4 (the driving budget); the "
            "other budgets/surfaces are disclosed diagnostics, never averaged"
        ),
    }


def paired_contrast_tables(view: Mapping[str, Any]) -> dict[str, Any]:
    """Every candidate-vs-F00 paired delta, per budget and surface."""
    contrasts: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        for surface in plan.SURFACES:
            for cell in plan.CANDIDATE_CELLS:
                contrasts[f"{cell}_minus_F00_m{budget}_{surface}"] = candidate_vs_baseline(
                    view, cell, budget=budget, surface=surface,
                )
    return contrasts


def evaluate_cross_gate(view: Mapping[str, Any], *, safety: Mapping[str, bool]) -> dict[str, Any]:
    """The full work-order section 2 gate block."""
    _require(set(safety) == set(SAFETY_CONDITIONS),
             "gate safety block must carry exactly the pre-registered Part-A conditions")
    per_cell = evaluate_per_cell_gates(view)
    additivity = evaluate_additivity_gate(view)
    safety_effective = {
        key: bool(value) for key, value in safety.items()
    }
    safety_pass = all(safety_effective.values())
    return {
        "schema": f"{plan.SCHEMA}_gate_v1",
        "work_order_section": "2 (Part A)",
        "per_cell": per_cell["per_cell"],
        "additivity": additivity,
        "safety": safety_effective,
        "safety_all_pass": bool(safety_pass),
        "boundary_epsilon": plan.GATE_BOUNDARY_EPSILON,
        "boundary_rule": plan.GATES["boundary_rule"],
        "nulls_reported_as_nulls": True,
    }


def stop_conditions(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Honest stop/disclosure rows for Part A (nulls reported as nulls)."""
    per_cell = gate["per_cell"]
    any_promoted = any(entry["promoted"] for entry in per_cell.values())
    conditions = {
        "no_candidate_beats_f00": {
            "expression": "none of F10/F01/F11 passes the external-M4 promotion gate",
            "fired": not any_promoted,
            "evidence": {
                cell: {
                    "external_m4_delta": entry["external_m4_delta"],
                    "positive_sessions": entry["positive_external_m4_sessions"],
                    "promoted": entry["promoted"],
                }
                for cell, entry in per_cell.items()
            },
        },
        "additivity_violated": {
            "expression": plan.GATES["additivity"]["expression"],
            "fired": not bool(gate["additivity"]["held"]),
            "evidence": gate["additivity"]["external_m4_means"],
        },
        "safety_floor_breached": {
            "expression": plan.GATES["safety"]["expression"],
            "fired": any(
                (not entry["within_bounds"]["all_meet_margin"])
                or (not entry["external_m30_safety"]["meets_margin"])
                for entry in per_cell.values()
            ),
            "evidence": {
                cell: {
                    "within_all_meet_margin": entry["within_bounds"]["all_meet_margin"],
                    "external_m30_meets_margin": entry["external_m30_safety"]["meets_margin"],
                }
                for cell, entry in per_cell.items()
            },
        },
        "anchor_or_binding_failure": {
            "expression": "any bit-anchor, weight-binding or causality proof failed",
            "fired": not bool(gate["safety_all_pass"]),
            "evidence": dict(gate["safety"]),
        },
    }
    fired = [key for key, value in conditions.items() if value["fired"]]
    return {
        "schema": f"{plan.SCHEMA}_stop_conditions_v1",
        "conditions": conditions,
        "fired": fired,
        "any_fired": bool(fired),
    }
