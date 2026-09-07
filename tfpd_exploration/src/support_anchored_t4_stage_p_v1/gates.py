"""The design §8 promotion, attribution and stopping gates for Stage P.

Boundaries are pre-registered in :mod:`.plan` and applied EXACTLY here:

* every margin comparison is an exact ``>=`` on the float64 equal-session mean;
* ``GATE_BOUNDARY_EPSILON = 1e-12`` only surfaces a disclosed
  ``within_epsilon_band_of_boundary`` flag; it never flips a verdict;
* "beats a control" is a strictly positive paired equal-session external delta
  on the driving budget (the +0.01 margins belong to the promotion and
  continuity claims), computed on the SAME prediction records;
* M30 is never averaged with M4/M10;
* the §8.1 promotion gate requires ALL conditions (including the constant,
  shuffle and no-pseudo controls) on at least one low budget;
* the §8.4 stop conditions are evaluated verbatim, each with its evidence.

Also evaluates the §8.2/§8.3 attribution rows, the §6.4 movement-ordering
diagnostic and the §8 interpretation rows.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import plan


class StagePGateError(ValueError):
    """Fail-closed error for malformed gate inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StagePGateError(message)


def _budget_key(budget: int) -> str:
    return f"m{int(budget)}"


SAFETY_CONDITIONS = (
    "zero_target_updates",
    "p0_bit_anchor",
    "causality_state_chains",
    "within_regression_bound",
    "m30_exact_noop",
    "hyperparameters_source_selected_only",
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


def _strictly_beats(delta: Mapping[str, Any]) -> dict[str, Any]:
    value = float(delta["equal_session_mean_delta"])
    return {
        "value": value,
        "beats": bool(value > 0.0),
        "within_epsilon_band_of_boundary": bool(
            0.0 >= value > -plan.GATE_BOUNDARY_EPSILON
        ),
    }


def _within_bounds(view: Mapping[str, Any], row: str) -> dict[str, Any]:
    bounds = {
        _budget_key(budget): margin_verdict(
            paired_delta(
                _row_view(view, budget, "within", row),
                _row_view(view, budget, "within", "P0"),
            )["equal_session_mean_delta"],
            plan.WITHIN_EVERY_BUDGET_FLOOR,
        )
        for budget in plan.BUDGETS
    }
    return {
        "per_budget": bounds,
        "all_meet_margin": all(item["meets_margin"] for item in bounds.values()),
    }


def evaluate_promotion_gate(view: Mapping[str, Any], *, safety: Mapping[str, bool]) -> dict[str, Any]:
    """Evaluate design §8.1 per candidate row (P1, P2) and low budget.

    ``view[m<budget>][surface][row] = {"mean_r2": ..., "per_session": {...}}``
    with the rows P0/P1/P2/P3/P4 present.  The two matrix-derived safety
    conditions (``within_regression_bound``) are computed HERE and override any
    caller-supplied value for them.
    """
    _require(set(safety) == set(SAFETY_CONDITIONS),
             "gate safety block must carry exactly the pre-registered Stage-P conditions")
    within_by_row = {row: _within_bounds(view, row) for row in plan.CANDIDATE_ROWS}
    safety_effective = dict(safety)
    safety_effective["within_regression_bound"] = bool(
        within_by_row["P1"]["all_meet_margin"] and within_by_row["P2"]["all_meet_margin"]
    )
    safety_pass = all(bool(value) for value in safety_effective.values())

    per_budget: dict[str, Any] = {}
    promoted: list[tuple[str, int]] = []
    for budget in plan.BUDGETS:
        other = next(item for item in plan.LOW_BUDGETS if item != budget)
        entry: dict[str, Any] = {"is_low_budget": budget in plan.LOW_BUDGETS}
        if budget in plan.LOW_BUDGETS:
            other_external = paired_delta(
                _row_view(view, other, "external", "P2"), _row_view(view, other, "external", "P0"),
            )["equal_session_mean_delta"]
            other_external_p1 = paired_delta(
                _row_view(view, other, "external", "P1"), _row_view(view, other, "external", "P0"),
            )["equal_session_mean_delta"]
        else:
            other_external = None
            other_external_p1 = None
        for row in plan.CANDIDATE_ROWS:
            external = paired_delta(
                _row_view(view, budget, "external", row), _row_view(view, budget, "external", "P0"),
            )
            decision: dict[str, Any] = {
                "external_delta": external["equal_session_mean_delta"],
                "external_margin": margin_verdict(
                    external["equal_session_mean_delta"], plan.PROMOTION_EXTERNAL_DELTA,
                ),
                "positive_external_sessions": int(external["positive_sessions"]),
                "breadth_margin": margin_verdict(
                    float(external["positive_sessions"]),
                    float(plan.PROMOTION_POSITIVE_EXTERNAL_SESSIONS),
                ),
                "beats_no_pseudo_evidence": _strictly_beats(external),
                "within_bounds": within_by_row[row],
            }
            if budget in plan.LOW_BUDGETS:
                other_delta = other_external if row == "P2" else other_external_p1
                decision["other_low_budget_external_delta"] = other_delta
                decision["other_low_budget_margin"] = margin_verdict(
                    float(other_delta), plan.OTHER_LOW_BUDGET_FLOOR,
                )
                beats_constant = _strictly_beats(paired_delta(
                    _row_view(view, budget, "external", row),
                    _row_view(view, budget, "external", "P3"),
                ))
                beats_shuffle = _strictly_beats(paired_delta(
                    _row_view(view, budget, "external", row),
                    _row_view(view, budget, "external", "P4"),
                ))
                decision["beats_constant_confidence"] = beats_constant
                decision["beats_deterministic_shuffle"] = beats_shuffle
                decision["promoted"] = bool(
                    decision["external_margin"]["meets_margin"]
                    and decision["breadth_margin"]["meets_margin"]
                    and decision["other_low_budget_margin"]["meets_margin"]
                    and within_by_row[row]["all_meet_margin"]
                    and beats_constant["beats"]
                    and beats_shuffle["beats"]
                    and decision["beats_no_pseudo_evidence"]["beats"]
                    and safety_pass
                )
                if decision["promoted"]:
                    promoted.append((row, budget))
            else:
                decision["promoted"] = False
            entry[row] = decision
        per_budget[_budget_key(budget)] = entry

    if promoted:
        decision_value = plan.DISPOSITION_PROMOTE
        driving = max(promoted, key=lambda item: (item[1], item[0]))
        driving_budget = f"{driving[0]}@m{driving[1]}"
    else:
        decision_value = plan.DISPOSITION_STOP
        driving_budget = None
    return {
        "schema": "support_anchored_t4_stage_p_promotion_gate_v1",
        "design_8_1_verbatim": dict(plan.DESIGN_8_1_VERBATIM),
        "decision": decision_value,
        "driving_cell": driving_budget,
        "promoted_cells": [f"{row}@m{budget}" for row, budget in promoted],
        "safety": {key: bool(value) for key, value in safety_effective.items()},
        "safety_all_pass": bool(safety_pass),
        "per_budget": per_budget,
        "never_average_m30": True,
        "epsilon_note": plan.GATES["boundary_rule"],
        "beats_control_rule": plan.GATES["beats_control_rule"],
    }


def continuity_attribution(view: Mapping[str, Any]) -> dict[str, Any]:
    """Design §8.2: the trajectory-continuity attribution claim."""
    per_budget: dict[str, Any] = {}
    for budget in plan.LOW_BUDGETS:
        delta = paired_delta(
            _row_view(view, budget, "external", "P2"), _row_view(view, budget, "external", "P1"),
        )
        beats_constant = _strictly_beats(paired_delta(
            _row_view(view, budget, "external", "P2"), _row_view(view, budget, "external", "P3"),
        ))
        beats_shuffle = _strictly_beats(paired_delta(
            _row_view(view, budget, "external", "P2"), _row_view(view, budget, "external", "P4"),
        ))
        margin = margin_verdict(delta["equal_session_mean_delta"], plan.CONTINUITY_EXTERNAL_DELTA)
        breadth = margin_verdict(
            float(delta["positive_sessions"]), float(plan.PROMOTION_POSITIVE_EXTERNAL_SESSIONS),
        )
        per_budget[_budget_key(budget)] = {
            "expression": plan.DESIGN_8_2_VERBATIM["conditions"],
            "p2_minus_p1_external": delta["equal_session_mean_delta"],
            "margin": margin,
            "positive_sessions": int(delta["positive_sessions"]),
            "breadth_margin": breadth,
            "p2_beats_constant": beats_constant,
            "p2_beats_shuffle": beats_shuffle,
            "claim_supported": bool(
                margin["meets_margin"] and breadth["meets_margin"]
                and beats_constant["beats"] and beats_shuffle["beats"]
            ),
        }
    return {
        "schema": "support_anchored_t4_stage_p_continuity_attribution_v1",
        "design_8_2_verbatim": dict(plan.DESIGN_8_2_VERBATIM),
        "per_budget": per_budget,
        "p1_positive_p2_null_note": plan.DESIGN_8_2_VERBATIM["note"],
    }


def confidence_attribution(view: Mapping[str, Any]) -> dict[str, Any]:
    """Design §8.3: the confidence-mapping attribution claim."""
    per_budget: dict[str, Any] = {}
    for budget in plan.LOW_BUDGETS:
        beats_constant = _strictly_beats(paired_delta(
            _row_view(view, budget, "external", "P2"), _row_view(view, budget, "external", "P3"),
        ))
        beats_shuffle = _strictly_beats(paired_delta(
            _row_view(view, budget, "external", "P2"), _row_view(view, budget, "external", "P4"),
        ))
        per_budget[_budget_key(budget)] = {
            "expression": plan.DESIGN_8_3_VERBATIM["conditions"],
            "p2_beats_constant": beats_constant,
            "p2_beats_deterministic_shuffle": beats_shuffle,
            "claim_supported": bool(beats_constant["beats"] and beats_shuffle["beats"]),
        }
    return {
        "schema": "support_anchored_t4_stage_p_confidence_attribution_v1",
        "design_8_3_verbatim": dict(plan.DESIGN_8_3_VERBATIM),
        "per_budget": per_budget,
        "update_count_note": plan.DESIGN_8_3_VERBATIM["note"],
    }


def stop_conditions(
    view: Mapping[str, Any], promotion: Mapping[str, Any], *, safety: Mapping[str, bool],
) -> dict[str, Any]:
    """Design §8.4, evaluated verbatim with per-condition evidence."""
    best_positive: dict[str, dict[str, Any]] = {}
    any_promotable = False
    for budget in plan.LOW_BUDGETS:
        for row in plan.CANDIDATE_ROWS:
            external = paired_delta(
                _row_view(view, budget, "external", row), _row_view(view, budget, "external", "P0"),
            )
            beats_constant = _strictly_beats(paired_delta(
                _row_view(view, budget, "external", row), _row_view(view, budget, "external", "P3"),
            ))
            beats_shuffle = _strictly_beats(paired_delta(
                _row_view(view, budget, "external", row), _row_view(view, budget, "external", "P4"),
            ))
            promotable = bool(
                external["equal_session_mean_delta"] >= plan.PROMOTION_EXTERNAL_DELTA
                and external["positive_sessions"] >= plan.PROMOTION_POSITIVE_EXTERNAL_SESSIONS
            )
            any_promotable = any_promotable or promotable
            best_positive[f"{row}@m{budget}"] = {
                "external_delta": external["equal_session_mean_delta"],
                "positive_sessions": int(external["positive_sessions"]),
                "beats_constant": beats_constant,
                "beats_shuffle": beats_shuffle,
                "promotable_on_primary_margins": promotable,
            }
    gains_disappear = any(
        entry["external_delta"] >= plan.PROMOTION_EXTERNAL_DELTA
        and not (entry["beats_constant"]["beats"] and entry["beats_shuffle"]["beats"])
        for entry in best_positive.values()
    )
    breadth_short = any(
        entry["external_delta"] > 0.0
        and entry["positive_sessions"] < plan.PROMOTION_POSITIVE_EXTERNAL_SESSIONS
        for entry in best_positive.values()
    )
    conditions = {
        "o2_stop_after_initial_run_or_permitted_hold": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][0],
            "fired": False,
            "evidence": "the sealed Stage-O terminal receipt is GO (pinned by the attempt)",
        },
        "oracle_refit_causality_leak": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][1],
            "fired": not bool(safety["causality_state_chains"]),
            "evidence": "per-trial forward state digest chain (design §3.2)",
        },
        "p1_p2_fail_to_beat_activity_only": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][2],
            "fired": not any_promotable,
            "evidence": best_positive,
        },
        "gains_disappear_under_constant_or_shuffle_controls": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][3],
            "fired": bool(gains_disappear),
            "evidence": "candidates reaching +0.01 external but failing a control",
        },
        "positive_gain_carried_by_fewer_than_10_of_15": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][4],
            "fired": bool(breadth_short),
            "evidence": "any candidate with positive aggregate external gain and breadth < 10/15",
        },
        "gains_require_target_selected_thresholds": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][5],
            "fired": not bool(safety["hyperparameters_source_selected_only"]),
            "evidence": "every selected hyperparameter comes from the within-6 folds",
        },
        "updates_require_target_gradients_or_decoder_changes": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][6],
            "fired": not bool(safety["zero_target_updates"]),
            "evidence": "optimizer/backward/parameter/normalizer/model-digest counters",
        },
        "repeated_proposals_drift_outside_support_trust_region": {
            "expression": plan.DESIGN_8_4_VERBATIM["stop_conditions"][7],
            "fired": not bool(safety["trust_region_no_committed_drift"]),
            "evidence": "committed blocks keep D2 <= c_M; proposals above c_M are rejected",
        },
    }
    fired = [key for key, value in conditions.items() if value["fired"]]
    return {
        "schema": "support_anchored_t4_stage_p_stop_conditions_v1",
        "design_8_4_verbatim": dict(plan.DESIGN_8_4_VERBATIM),
        "conditions": conditions,
        "fired": fired,
        "any_fired": bool(fired),
        "promotion_decision": promotion["decision"],
    }


def interpretation_rows(view: Mapping[str, Any], promotion: Mapping[str, Any]) -> dict[str, Any]:
    """The §8 interpretation rows on equal-session means."""
    def external(budget: int, row: str) -> float:
        return equal_session_mean(_row_view(view, budget, "external", row))

    p1_positive = any(
        external(budget, "P1") - external(budget, "P0") >= plan.PROMOTION_EXTERNAL_DELTA
        for budget in plan.LOW_BUDGETS
    )
    p2_positive = any(
        external(budget, "P2") - external(budget, "P0") >= plan.PROMOTION_EXTERNAL_DELTA
        for budget in plan.LOW_BUDGETS
    )
    return {
        "p1_positive_p2_null": {
            "expression": plan.INTERPRETATION_ROWS["p1_positive_p2_null"],
            "value": bool(p1_positive and not p2_positive),
        },
        "confidence_null": {
            "expression": plan.INTERPRETATION_ROWS["confidence_null"],
            "value": bool(
                p2_positive and not all(
                    entry["claim_supported"]
                    for entry in confidence_attribution(view)["per_budget"].values()
                )
            ),
        },
        "headroom_without_deployable_measurement": {
            "expression": plan.INTERPRETATION_ROWS["headroom_without_deployable_measurement"],
            "value": bool(promotion["decision"] == plan.DISPOSITION_STOP),
        },
        "gate_scope": plan.INTERPRETATION_ROWS["gate_scope"],
        "equal_session_means": {
            _budget_key(budget): {
                surface: {row: equal_session_mean(_row_view(view, budget, surface, row))
                          for row in plan.ROW_ORDER}
                for surface in plan.SURFACES
            }
            for budget in plan.BUDGETS
        },
    }


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
        "schema": "support_anchored_t4_stage_p_movement_ordering_v1",
        "medians": values,
        "m10_le_m4": bool(m10_le_m4),
        "m30_le_m10": bool(m30_le_m10),
        "ordering_holds": bool(m10_le_m4 and m30_le_m10),
        "m30_zero_by_construction": values["m30"] == 0.0,
        "epsilon": epsilon,
        "status": plan.MOVEMENT_ORDERING["status"],
    }
