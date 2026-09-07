"""Gate evaluation and terminal composition for the LOCAL-M2 transfer."""

from __future__ import annotations

from typing import Any, Mapping


class M2LocalGateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2LocalGateError(message)


def _margin(value: float, threshold: float, *, epsilon: float) -> dict[str, Any]:
    meets = float(value) >= float(threshold)
    within_band = abs(float(value) - float(threshold)) <= float(epsilon)
    if within_band:
        meets = float(value) >= float(threshold) or float(value) >= float(threshold) - float(epsilon)
    return {
        "threshold": float(threshold),
        "value": float(value),
        "meets_margin": bool(meets),
        "within_epsilon_band_of_boundary": bool(within_band),
        "boundary_epsilon": float(epsilon),
    }


def equal_session_mean(session_values: Mapping[str, float]) -> float:
    _require(bool(session_values), "equal-session mean needs sessions")
    return float(sum(float(v) for v in session_values.values()) / len(session_values))


def paired_delta(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, Any]:
    _require(set(candidate) == set(reference) and bool(candidate), "paired session set mismatch")
    deltas = {name: float(candidate[name]) - float(reference[name]) for name in sorted(candidate)}
    values = [deltas[name] for name in sorted(deltas)]
    return {
        "per_session_delta": deltas,
        "equal_session_mean_delta": float(sum(values) / len(values)),
        "positive_sessions": int(sum(1 for value in values if value > 0.0)),
        "session_count": int(len(values)),
    }


def evaluate_promotion(
    *,
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    promotion_law: Mapping[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    """The Part-A promotion gate mirrored on an M2 roster (one budget/surface)."""
    delta = paired_delta(candidate, baseline)
    delta_margin = _margin(
        delta["equal_session_mean_delta"], float(promotion_law["delta_floor"]), epsilon=epsilon,
    )
    breadth_margin = _margin(
        float(delta["positive_sessions"]), float(promotion_law["breadth_min"]), epsilon=epsilon,
    )
    promoted = bool(delta_margin["meets_margin"] and breadth_margin["meets_margin"])
    return {
        "expression": str(promotion_law["expression"]),
        "delta": delta,
        "delta_margin": delta_margin,
        "breadth_margin": breadth_margin,
        "positive_session_ids": sorted(
            name for name, value in delta["per_session_delta"].items() if value > 0.0
        ),
        "promoted": promoted,
        "disposition": (
            "CELL_IMPROVES_OVER_BASELINE" if promoted else "CELL_FAILS_THE_GATE"
        ),
    }


def evaluate_m2_local_gate(
    *,
    external_m4: Mapping[str, Mapping[str, float]],
    within_by_budget: Mapping[int, Mapping[str, Mapping[str, float]]],
    external_m30: Mapping[str, Mapping[str, float]],
    safety: Mapping[str, bool],
    gates_law: Mapping[str, Any],
) -> dict[str, Any]:
    """The primary F01m gate on the LOCAL-M2 protocol."""
    epsilon = float(gates_law["boundary_epsilon"])
    promotion = evaluate_promotion(
        candidate=external_m4["F01m"], baseline=external_m4["F00m"],
        promotion_law=gates_law["promotion"], epsilon=epsilon,
    )
    within_bounds: dict[str, Any] = {}
    for budget in sorted(within_by_budget):
        surface = within_by_budget[int(budget)]
        within_bounds[f"m{int(budget)}"] = {
            "delta": paired_delta(surface["F01m"], surface["F00m"]),
            "margin": _margin(
                paired_delta(surface["F01m"], surface["F00m"])["equal_session_mean_delta"],
                float(gates_law["safety"]["within_floor"]), epsilon=epsilon,
            ),
        }
    m30_delta = paired_delta(external_m30["F01m"], external_m30["F00m"])
    external_m30_margin = _margin(
        m30_delta["equal_session_mean_delta"],
        float(gates_law["safety"]["external_m30_floor"]), epsilon=epsilon,
    )
    within_all = all(item["margin"]["meets_margin"] for item in within_bounds.values())
    safety_all = bool(
        within_all
        and external_m30_margin["meets_margin"]
        and all(
            bool(value) for key, value in safety.items()
            if key != "dandi_hyperparameters_ported_as_claims"
        )
        and safety.get("dandi_hyperparameters_ported_as_claims") is False
    )
    return {
        "schema": "cdm_p1_m2_local_v1_gate_v1",
        "work_order_section": "3 (Part B2, LOCAL protocol)",
        "driving_cell": str(gates_law["driving_cell"]),
        "driving_surface": str(gates_law["driving_surface"]),
        "promotion": promotion,
        "promoted": bool(promotion["promoted"]),
        "within_bounds": within_bounds,
        "external_m30_safety": {
            "delta": m30_delta,
            "margin": external_m30_margin,
        },
        "safety": dict(safety),
        "safety_all_pass": safety_all,
        "boundary_rule": str(gates_law["boundary_rule"]),
        "nulls_reported_as_nulls": True,
        "official_contract_disclaimer": str(gates_law["official_contract_disclaimer"]),
    }


def evaluate_secondary_gate(
    *,
    external_m4: Mapping[str, Mapping[str, float]],
    gates_law: Mapping[str, Any],
) -> dict[str, Any]:
    """The secondary G-family disclosure gate (never governs the verdict)."""
    epsilon = float(gates_law["boundary_epsilon"])
    law = {
        "expression": gates_law["secondary_full_stack"]["expression"],
        "delta_floor": gates_law["promotion"]["delta_floor"],
        "breadth_min": gates_law["promotion"]["breadth_min"],
    }
    return evaluate_promotion(
        candidate=external_m4["G01m"], baseline=external_m4["G00m"],
        promotion_law=law, epsilon=epsilon,
    )


def stop_conditions(gate: Mapping[str, Any], secondary: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "anchor_or_binding_failure": {
            "expression": "any bit-anchor, carrier-parity or causality proof failed",
            "fired": not bool(gate["safety_all_pass"]),
            "evidence": dict(gate["safety"]),
        },
        "no_candidate_beats_baseline": {
            "expression": "F01m fails the external_official_query M4 promotion gate",
            "fired": not bool(gate["promoted"]),
            "evidence": {
                "external_m4_delta": gate["promotion"]["delta"]["equal_session_mean_delta"],
                "positive_sessions": gate["promotion"]["delta"]["positive_sessions"],
            },
        },
        "safety_floor_breached": {
            "expression": (
                "F01m - F00m < -0.02 on within_post30 at any rung or on external M30"
            ),
            "fired": not (
                all(item["margin"]["meets_margin"] for item in gate["within_bounds"].values())
                and bool(gate["external_m30_safety"]["margin"]["meets_margin"])
            ),
            "evidence": {
                f"within_{key}": item["delta"]["equal_session_mean_delta"]
                for key, item in gate["within_bounds"].items()
            },
        },
        "secondary_full_stack_gate": {
            "expression": (
                "the G-family secondary disclosure (G01m - G00m on "
                "external_post30_local M4) -- informational, never governing"
            ),
            "fired": False,
            "evidence": {
                "delta": secondary["delta"]["equal_session_mean_delta"],
                "positive_sessions": secondary["delta"]["positive_sessions"],
                "promoted": bool(secondary["promoted"]),
            },
        },
    }
    governing = [name for name, item in conditions.items()
                 if item["fired"] and name != "secondary_full_stack_gate"]
    return {
        "schema": "cdm_p1_m2_local_v1_stop_conditions_v1",
        "any_fired": bool(governing),
        "fired": governing,
        "conditions": conditions,
    }
