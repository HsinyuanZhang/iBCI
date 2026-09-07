"""Gate evaluation and terminal composition for the SUA transfer (receipt-only)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


class SuaGateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuaGateError(message)


def _margin(value: float, threshold: float, *, epsilon: float) -> dict[str, Any]:
    meets = value >= threshold if threshold >= 0 else value >= threshold
    if abs(value - threshold) <= epsilon:
        within_band = True
        meets = value >= threshold or (value >= threshold - epsilon)
    else:
        within_band = False
    return {
        "threshold": float(threshold),
        "value": float(value),
        "meets_margin": bool(meets),
        "within_epsilon_band_of_boundary": bool(within_band),
        "boundary_epsilon": float(epsilon),
    }


def seed_averaged_session_means(
    values: Mapping[str, Mapping[str, Mapping[int, float]]],
) -> dict[str, float]:
    """session -> mean over seeds (equal session weighting happens downstream)."""
    means: dict[str, float] = {}
    for session, per_seed in values.items():
        _require(bool(per_seed), f"{session} has no seeded rows")
        means[session] = float(sum(float(v) for v in per_seed.values()) / len(per_seed))
    return means


def equal_session_mean(session_means: Mapping[str, float]) -> float:
    _require(bool(session_means), "equal-session mean needs sessions")
    return float(sum(float(v) for v in session_means.values()) / len(session_means))


def evaluate_sua_gate(
    *,
    external_m4: Mapping[str, Mapping[str, Mapping[int, float]]],
    within_m4: Mapping[str, Mapping[str, Mapping[int, float]]],
    external_m30: Mapping[str, Mapping[str, Mapping[int, float]]],
    within_m30: Mapping[str, Mapping[str, Mapping[int, float]]],
    safety: Mapping[str, bool],
    gates_law: Mapping[str, Any],
) -> dict[str, Any]:
    """The Part-A promotion gate mirrored on the SUA rosters."""
    epsilon = float(gates_law["boundary_epsilon"])
    promotion = gates_law["promotion"]
    f00_ext = seed_averaged_session_means(external_m4["F00s"])
    f01_ext = seed_averaged_session_means(external_m4["F01s"])
    delta = {session: f01_ext[session] - f00_ext[session] for session in f00_ext}
    positive = sorted(session for session, value in delta.items() if value > 0.0)
    delta_mean = equal_session_mean(delta)
    delta_margin = _margin(delta_mean, float(promotion["delta_floor"]), epsilon=epsilon)
    breadth_margin = _margin(
        float(len(positive)), float(promotion["breadth_min"]), epsilon=epsilon,
    )
    promoted = bool(delta_margin["meets_margin"] and breadth_margin["meets_margin"])
    within_bounds: dict[str, Any] = {}
    for label, surface in (("m4", within_m4), ("m30", within_m30)):
        f00 = seed_averaged_session_means(surface["F00s"])
        f01 = seed_averaged_session_means(surface["F01s"])
        per_session = {session: f01[session] - f00[session] for session in f00}
        within_bounds[label] = {
            "per_session_delta": per_session,
            "equal_session_mean_delta": equal_session_mean(per_session),
            "positive_sessions": int(sum(1 for value in per_session.values() if value > 0.0)),
            "margin": _margin(
                equal_session_mean(per_session),
                float(gates_law["safety"]["within_floor"]), epsilon=epsilon,
            ),
        }
    f00_m30 = seed_averaged_session_means(external_m30["F00s"])
    f01_m30 = seed_averaged_session_means(external_m30["F01s"])
    m30_delta = {session: f01_m30[session] - f00_m30[session] for session in f00_m30}
    external_m30_margin = _margin(
        equal_session_mean(m30_delta),
        float(gates_law["safety"]["external_m30_floor"]), epsilon=epsilon,
    )
    within_all = all(
        bool(within_bounds[label]["margin"]["meets_margin"]) for label in within_bounds
    )
    return {
        "schema": "cdm_p1_sua_v1_gate_v1",
        "work_order_section": "3 (Part B1)",
        "driving_cell": str(gates_law["driving_cell"]),
        "expression": str(promotion["expression"]),
        "external_m4_delta": float(delta_mean),
        "external_m4_margin": delta_margin,
        "positive_external_m4_sessions": int(len(positive)),
        "positive_external_m4_session_ids": positive,
        "breadth_margin": breadth_margin,
        "promoted": promoted,
        "disposition": (
            "CELL_IMPROVES_OVER_F00S" if promoted else "CELL_FAILS_THE_F00S_GATE"
        ),
        "within_bounds": within_bounds,
        "external_m30_safety": {
            "per_session_delta": m30_delta,
            "equal_session_mean_delta": equal_session_mean(m30_delta),
            "margin": external_m30_margin,
        },
        "safety_all_pass": bool(
            within_all
            and external_m30_margin["meets_margin"]
            and all(
                bool(value) for key, value in safety.items()
                if key != "dandi_hyperparameters_ported_as_claims"
            )
            and safety.get("dandi_hyperparameters_ported_as_claims") is False
        ),
        "safety": dict(safety),
        "boundary_rule": str(gates_law["boundary_rule"]),
        "nulls_reported_as_nulls": True,
    }


def stop_conditions(gate: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "anchor_or_binding_failure": {
            "expression": "any bit-anchor, carrier-parity or causality proof failed",
            "fired": not bool(gate["safety_all_pass"]),
            "evidence": dict(gate["safety"]),
        },
        "no_candidate_beats_f00s": {
            "expression": "F01s fails the external-M4 promotion gate",
            "fired": not bool(gate["promoted"]),
            "evidence": {
                "external_m4_delta": gate["external_m4_delta"],
                "positive_sessions": gate["positive_external_m4_sessions"],
            },
        },
        "safety_floor_breached": {
            "expression": (
                "F01s - F00s < -0.02 on the within surface at any rung or on external M30"
            ),
            "fired": not (
                all(
                    bool(gate["within_bounds"][label]["margin"]["meets_margin"])
                    for label in gate["within_bounds"]
                )
                and bool(gate["external_m30_safety"]["margin"]["meets_margin"])
            ),
            "evidence": {
                label: gate["within_bounds"][label]["equal_session_mean_delta"]
                for label in gate["within_bounds"]
            },
        },
    }
    return {
        "schema": "cdm_p1_sua_v1_stop_conditions_v1",
        "any_fired": any(item["fired"] for item in conditions.values()),
        "fired": [name for name, item in conditions.items() if item["fired"]],
        "conditions": conditions,
    }
