"""The design §3.5 oracle headroom gate: GO / HOLD / STOP.

Boundaries are pre-registered in :mod:`.plan` and applied EXACTLY here:

* every margin comparison is an exact ``>=`` on the float64 equal-session mean;
* ``GATE_BOUNDARY_EPSILON = 1e-12`` only surfaces a disclosed
  ``within_epsilon_band_of_boundary`` flag; it never flips a verdict;
* M30 is never averaged with M4/M10;
* GO requires ALL conditions on at least one low budget; HOLD requires the
  primary delta in [+0.01, +0.03) with >= 9/15 positive external sessions and
  no safety failure; STOP is everything else (including any leak or safety
  failure).

Also evaluates the §3.5 interpretation rows and the §6.4 movement-ordering
diagnostic on medians of carrier movement.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from . import plan


class StageOGateError(ValueError):
    """Fail-closed error for malformed gate inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageOGateError(message)


def _budget_key(budget: int) -> str:
    return f"m{int(budget)}"


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


def _row_view(view: Mapping[str, Any], budget: int, surface: str, row: str) -> Mapping[str, Any]:
    surface_map = view.get(_budget_key(budget), {}).get(surface, {})
    _require(row in surface_map, f"gate input is missing row {row} at M{budget} {surface}")
    return surface_map[row]


def equal_session_mean(payload: Mapping[str, Any]) -> float:
    per_session = payload.get("per_session")
    _require(isinstance(per_session, Mapping) and bool(per_session), "gate row needs per-session values")
    return float(sum(float(value) for value in per_session.values())) / len(per_session)


def paired_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    left = dict(candidate["per_session"])
    right = dict(baseline["per_session"])
    _require(tuple(sorted(left)) == tuple(sorted(right)), "paired gate rows disagree on the session roster")
    deltas = {key: float(left[key]) - float(right[key]) for key in sorted(left)}
    return {
        "per_session": deltas,
        "equal_session_mean_delta": float(sum(deltas.values())) / len(deltas),
        "positive_sessions": sum(1 for value in deltas.values() if value > 0.0),
        "n_sessions": len(deltas),
    }


def _low_budget_rows(
    view: Mapping[str, Any], budget: int,
) -> dict[str, Mapping[str, Any]]:
    return {
        surface: {
            row: _row_view(view, budget, surface, row)
            for row in plan.PRIMARY_ROWS
        }
        for surface in plan.SURFACES
    }


def evaluate_oracle_gate(view: Mapping[str, Any], *, safety: Mapping[str, bool]) -> dict[str, Any]:
    """Evaluate §3.5 on ``view[m<budget>][surface][row] = {mean_r2, per_session}``."""
    _require(set(safety) == set(plan.GATES["safety_conditions"]),
             "gate safety block must carry exactly the pre-registered conditions")

    per_budget: dict[str, Any] = {}
    external_deltas: dict[int, float] = {}
    for budget in plan.BUDGETS:
        rows = _low_budget_rows(view, budget)
        external = paired_delta(rows["external"]["O2"], rows["external"]["O0"])
        within = paired_delta(rows["within"]["O2"], rows["within"]["O0"])
        per_budget[_budget_key(budget)] = {
            "external": external,
            "within": within,
            "o1_minus_o0_external": paired_delta(rows["external"]["O1"], rows["external"]["O0"]),
            "o1_minus_o0_within": paired_delta(rows["within"]["O1"], rows["within"]["O0"]),
            "o2_minus_o1_external": paired_delta(rows["external"]["O2"], rows["external"]["O1"]),
            "is_low_budget": budget in plan.LOW_BUDGETS,
        }
        if budget in plan.LOW_BUDGETS:
            external_deltas[budget] = float(external["equal_session_mean_delta"])

    within_bounds = {
        _budget_key(budget): margin_verdict(
            per_budget[_budget_key(budget)]["within"]["equal_session_mean_delta"],
            plan.WITHIN_EVERY_BUDGET_FLOOR,
        )
        for budget in plan.BUDGETS
    }
    within_pass = all(item["meets_margin"] for item in within_bounds.values())
    # The two matrix-derived conditions are evaluated HERE from the matrix; a
    # caller-supplied value for them is always overridden by the computed one.
    safety_effective = dict(safety)
    safety_effective["within_regression_bound"] = bool(within_pass)
    safety_pass = all(bool(value) for value in safety_effective.values())

    go_candidates: list[int] = []
    hold_candidates: list[int] = []
    for budget in plan.LOW_BUDGETS:
        other = next(item for item in plan.LOW_BUDGETS if item != budget)
        external = per_budget[_budget_key(budget)]["external"]
        external_margin = margin_verdict(
            external["equal_session_mean_delta"], plan.GO_EXTERNAL_DELTA,
        )
        breadth = margin_verdict(
            float(external["positive_sessions"]), float(plan.GO_POSITIVE_EXTERNAL_SESSIONS),
        )
        other_margin = margin_verdict(
            external_deltas[other], plan.OTHER_LOW_BUDGET_FLOOR,
        )
        row = {
            "external_delta": external["equal_session_mean_delta"],
            "external_margin": external_margin,
            "positive_external_sessions": int(external["positive_sessions"]),
            "breadth_margin": breadth,
            "other_low_budget_external_delta": external_deltas[other],
            "other_low_budget_margin": other_margin,
        }
        row["go_eligible"] = bool(
            external_margin["meets_margin"]
            and breadth["meets_margin"]
            and other_margin["meets_margin"]
            and within_pass
            and safety_pass
        )
        hold_margin = margin_verdict(
            external["equal_session_mean_delta"], plan.HOLD_EXTERNAL_DELTA_MIN,
        )
        hold_breadth = margin_verdict(
            float(external["positive_sessions"]), float(plan.HOLD_POSITIVE_EXTERNAL_SESSIONS),
        )
        row["hold_margin"] = hold_margin
        row["hold_breadth_margin"] = hold_breadth
        row["hold_eligible"] = bool(
            hold_margin["meets_margin"]
            and not external_margin["meets_margin"]
            and hold_breadth["meets_margin"]
            and other_margin["meets_margin"]
            and within_pass
            and safety_pass
        )
        per_budget[_budget_key(budget)]["decision_row"] = row
        if row["go_eligible"]:
            go_candidates.append(budget)
        if row["hold_eligible"]:
            hold_candidates.append(budget)

    if go_candidates:
        decision = plan.DISPOSITION_GO
        driving = f"m{max(go_candidates)}" if len(go_candidates) > 1 else f"m{go_candidates[0]}"
    elif hold_candidates:
        decision = plan.DISPOSITION_HOLD
        driving = f"m{hold_candidates[0]}"
    else:
        decision = plan.DISPOSITION_STOP
        driving = None

    return {
        "schema": "support_anchored_t4_stage_o_gate_v1",
        "decision": decision,
        "driving_budget": driving,
        "go_candidate_budgets": [f"m{item}" for item in go_candidates],
        "hold_candidate_budgets": [f"m{item}" for item in hold_candidates],
        "safety": {key: bool(value) for key, value in safety_effective.items()},
        "safety_all_pass": bool(safety_pass),
        "within_bounds": within_bounds,
        "within_regression_bound_pass": bool(within_pass),
        "per_budget": per_budget,
        "never_average_m30": True,
        "epsilon_note": plan.GATES["boundary_rule"],
        "hold_authorizes": plan.GATES["HOLD"]["authorizes"],
        "gate_scope": plan.INTERPRETATION_ROWS["gate_scope"],
    }


def interpretation_rows(view: Mapping[str, Any]) -> dict[str, Any]:
    """The §3.5 interpretation rows, evaluated per surface on equal-session means."""
    rows: dict[str, Any] = {}
    means: dict[str, dict[str, dict[str, float]]] = {}
    for budget in plan.BUDGETS:
        for surface in plan.SURFACES:
            for row in plan.PRIMARY_ROWS:
                means.setdefault(_budget_key(budget), {}).setdefault(surface, {})[row] = equal_session_mean(
                    _row_view(view, budget, surface, row),
                )

    def external(budget: int, row: str) -> float:
        return means[_budget_key(budget)]["external"][row]

    def within(budget: int, row: str) -> float:
        return means[_budget_key(budget)]["within"][row]

    rows["recursive_estimator_at_fault"] = {
        "expression": plan.INTERPRETATION_ROWS["recursive_estimator_at_fault"],
        "per_budget": {
            _budget_key(budget): bool(external(budget, "O2") > external(budget, "O0")
                                      and external(budget, "O1") <= external(budget, "O0"))
            for budget in plan.LOW_BUDGETS
        },
    }
    rows["both_help"] = {
        "expression": plan.INTERPRETATION_ROWS["both_help"],
        "per_budget": {
            _budget_key(budget): bool(external(budget, "O2") > external(budget, "O1") > external(budget, "O0"))
            for budget in plan.LOW_BUDGETS
        },
    }
    rows["close_continuous_t4"] = {
        "expression": plan.INTERPRETATION_ROWS["close_continuous_t4"],
        "per_budget": {
            _budget_key(budget): bool(external(budget, "O1") <= external(budget, "O0")
                                      and external(budget, "O2") <= external(budget, "O0"))
            for budget in plan.LOW_BUDGETS
        },
    }
    positive_low_budgets = [
        budget for budget in plan.LOW_BUDGETS
        if external(budget, "O2") > external(budget, "O0") and within(budget, "O2") > within(budget, "O0")
    ]
    rows["m4_only"] = {
        "expression": plan.INTERPRETATION_ROWS["m4_only"],
        "value": bool(positive_low_budgets == [4]),
    }
    rows["m10_only"] = {
        "expression": plan.INTERPRETATION_ROWS["m10_only"],
        "value": bool(positive_low_budgets == [10]),
    }
    rows["nonpositive_m30"] = {
        "expression": plan.INTERPRETATION_ROWS["nonpositive_m30"],
        "external_m30_o2_minus_o0": external(30, "O2") - external(30, "O0"),
        "within_m30_o2_minus_o0": within(30, "O2") - within(30, "O0"),
        "value": bool(max(external(30, "O2") - external(30, "O0"),
                          within(30, "O2") - within(30, "O0")) <= 0.0),
        "deployable_m30_law": "keep M30 carrier movement exactly disabled (alpha_M = 0)",
    }
    rows["equal_session_means"] = means
    return rows


def movement_ordering(movement_by_budget: Mapping[str, float]) -> dict[str, Any]:
    """The §6.4 diagnostic: median movement M30 <= M10 <= M4 (epsilon-guarded)."""
    required = ("m4", "m10", "m30")
    for key in required:
        _require(key in movement_by_budget, f"movement ordering needs the {key} median")
    values = {key: float(movement_by_budget[key]) for key in required}
    epsilon = plan.MOVEMENT_ORDERING["epsilon"]
    m10_le_m4 = values["m10"] <= values["m4"] + epsilon
    m30_le_m10 = values["m30"] <= values["m10"] + epsilon
    return {
        "schema": "support_anchored_t4_stage_o_movement_ordering_v1",
        "medians": values,
        "m10_le_m4": bool(m10_le_m4),
        "m30_le_m10": bool(m30_le_m10),
        "ordering_holds": bool(m10_le_m4 and m30_le_m10),
        "epsilon": epsilon,
        "status": plan.MOVEMENT_ORDERING["status"],
    }
