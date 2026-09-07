"""Binding gates and stop-condition verdicts for AC3-0.

Implements exactly the pre-registered boolean semantics of ``plan.GATES``:

* the representation gate over R3/R4/R5 vs R0 (0.10 rad / 10 pp / shuffle
  degrades / not one-session-driven / zero target updates);
* the section-23-amended contrastive-claim gate vs ``max(R2, R0.5)``;
* the pre-registered E7-mirror disposition string;
* the section 19 stop conditions as receipt verdicts.

Every threshold comparison carries a numerical guard so that a boundary value
itself never flips a verdict (same convention as the P2' kill criteria).
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from . import plan


class AC3GateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3GateError(message)


EPSILON = 1.0e-12
CIRCULAR_ERROR_IMPROVEMENT_RAD = float(
    plan.GATES["REPRESENTATION_GATE"]["circular_error_improvement_over_R0_rad"]
)
SNAP_MISMATCH_IMPROVEMENT_PP = float(
    plan.GATES["REPRESENTATION_GATE"]["snap_mismatch_improvement_over_R0_pp"]
)
CONTRASTIVE_MARGIN_RAD = float(plan.GATES["CONTRASTIVE_CLAIM_GATE"]["circular_error_margin_rad"])
CONTRASTIVE_MARGIN_UTILITY = float(plan.GATES["CONTRASTIVE_CLAIM_GATE"]["coherent_utility_margin"])
E7_MIRROR_DISPOSITION = plan.GATES["E7_MIRROR"]["disposition_string"]


def _value(metrics: Mapping[str, object], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _improvement(candidate: Mapping[str, object], baseline: Mapping[str, object], key: str) -> Optional[float]:
    left = _value(candidate, key)
    right = _value(baseline, key)
    if left is None or right is None:
        return None
    return right - left


def representation_gate(
    row_metrics_by_row: Mapping[str, Mapping[str, object]],
    influence: Mapping[str, Mapping[str, object]],
    *,
    row: str,
    target_update_count: int = 0,
) -> dict[str, object]:
    """One representation-gate verdict for one of R3/R4/R5 (section 10.4)."""
    _require(row in plan.CONTRASTIVE_ROWS, f"representation gate does not apply to {row}")
    baseline = row_metrics_by_row["R0"]
    candidate = row_metrics_by_row[row]
    error_gain = _improvement(candidate, baseline, "session_mean_circular_error_rad")
    snap_rate_gain = _improvement(candidate, baseline, "session_mean_snap_mismatch_rate")
    snap_pp_gain = None if snap_rate_gain is None else 100.0 * snap_rate_gain
    shuffle = row_metrics_by_row.get("RS")
    rs_error = _value(shuffle, "session_mean_circular_error_rad") if shuffle else None
    rs_snap = _value(shuffle, "session_mean_snap_mismatch_rate") if shuffle else None
    row_error = _value(candidate, "session_mean_circular_error_rad")
    row_snap = _value(candidate, "session_mean_snap_mismatch_rate")
    shuffle_degrades = (
        None if (rs_error is None or rs_snap is None or row_error is None or row_snap is None)
        else bool((rs_error > row_error + EPSILON) and (rs_snap > row_snap + EPSILON))
    )
    influence_row = influence.get(row, {})
    min_over_drops = influence_row.get("min_over_drops")
    not_one_session = (
        None if min_over_drops is None
        else bool(float(min_over_drops) >= CIRCULAR_ERROR_IMPROVEMENT_RAD - EPSILON)
    )
    passed_parts = {
        "circular_error": bool(error_gain is not None
                               and error_gain >= CIRCULAR_ERROR_IMPROVEMENT_RAD - EPSILON),
        "snap_mismatch": bool(snap_pp_gain is not None
                              and snap_pp_gain >= SNAP_MISMATCH_IMPROVEMENT_PP - EPSILON),
        "shuffle_degrades": bool(shuffle_degrades) if shuffle_degrades is not None else False,
        "not_one_session_driven": bool(not_one_session) if not_one_session is not None else False,
        "target_updates_zero": bool(int(target_update_count) == 0),
    }
    return {
        "row": row,
        "baseline": "R0",
        "circular_error_improvement_rad": error_gain,
        "snap_mismatch_improvement_pp": snap_pp_gain,
        "shuffle_control": {
            "RS_session_mean_circular_error_rad": rs_error,
            "RS_session_mean_snap_mismatch_rate": rs_snap,
            "degrades": shuffle_degrades,
            "rule": plan.SHUFFLE_DEGRADATION_RULE["expression"],
        },
        "min_improvement_after_dropping_any_session_rad": min_over_drops,
        "not_one_session_driven": not_one_session,
        "target_update_count": int(target_update_count),
        "parts": passed_parts,
        "passed": all(passed_parts.values()),
    }


def contrastive_claim_gate(row_metrics_by_row: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Section 23 amendment 2: beat ``max(R2, R0.5)`` by the declared margin.

    ``max(R2, R-GE)`` is read STRICTLY: the maximum requirement across the two
    baselines, i.e. the row with the LOWER circular error (the stronger
    baseline).  Beating that row by the margin means beating BOTH R2 and R-GE
    by the margin, which is what "strengthens to beat max(R2, R-GE), not merely
    the raw direction baseline" requires.
    """
    r2 = _value(row_metrics_by_row.get("R2", {}), "session_mean_circular_error_rad")
    rge = _value(row_metrics_by_row.get("R0.5", {}), "session_mean_circular_error_rad")
    competitors = {name: value for name, value in (("R2", r2), ("R0.5", rge)) if value is not None}
    _require(competitors, "the contrastive-claim gate needs at least one of R2/R0.5")
    baseline_name = min(competitors, key=lambda key: competitors[key])
    baseline_value = competitors[baseline_name]
    per_row: dict[str, dict[str, object]] = {}
    for row in plan.PRIMARY_CONTRASTIVE_ROWS:
        value = _value(row_metrics_by_row.get(row, {}), "session_mean_circular_error_rad")
        margin = None if value is None else baseline_value - value
        per_row[row] = {
            "session_mean_circular_error_rad": value,
            "margin_over_baseline_rad": margin,
            "beats_baseline_by_0_05_rad": bool(margin is not None and margin >= CONTRASTIVE_MARGIN_RAD - EPSILON),
            "coherent_utility_margin": None,
        }
    passed_rows = [row for row, verdict in per_row.items() if verdict["beats_baseline_by_0_05_rad"]]
    return {
        "baseline_row": baseline_name,
        "baseline_value_rad": baseline_value,
        "baseline_definition": plan.GATES["CONTRASTIVE_CLAIM_GATE"]["baseline"],
        "margin_required_rad": CONTRASTIVE_MARGIN_RAD,
        "utility_margin_required": CONTRASTIVE_MARGIN_UTILITY,
        "utility_disjunct_status": plan.GATES["CONTRASTIVE_CLAIM_GATE"]["disjunct_status"],
        "per_row": per_row,
        "passed": bool(passed_rows),
        "passed_rows": passed_rows,
    }


def e7_mirror_verdict(
    row_metrics_by_row: Mapping[str, Mapping[str, object]],
    contrastive_gate: Mapping[str, object],
    *,
    utility_status: str = plan.UTILITY_ROWS["status_this_run"],
) -> dict[str, object]:
    """The pre-registered E7-mirror outcome (section 23 amendment 3)."""
    r2 = _value(row_metrics_by_row.get("R2", {}), "session_mean_circular_error_rad")
    rge = _value(row_metrics_by_row.get("R0.5", {}), "session_mean_circular_error_rad")
    r2_improves_direction = (
        None if (r2 is None or rge is None) else bool(r2 < rge - EPSILON)
    )
    r0 = _value(row_metrics_by_row.get("R0", {}), "session_mean_circular_error_rad")
    r2_beats_raw = None if (r2 is None or r0 is None) else bool(r2 < r0 - EPSILON)
    fired = bool(r2_improves_direction) and not bool(contrastive_gate.get("passed"))
    return {
        "fired": fired,
        "disposition": E7_MIRROR_DISPOSITION if fired else None,
        "report_exactly": (
            "a route success, not a method success" if fired else None
        ),
        "r2_improves_direction_over_R0.5": r2_improves_direction,
        "r2_beats_raw_R0": r2_beats_raw,
        "contrastive_claim_passed": bool(contrastive_gate.get("passed")),
        "utility_clause_status": (
            "NOT_EVALUATED_PROCESS_GATE" if utility_status == "NOT_RUN_PROCESS_GATE"
            else "EVALUATED"
        ),
        "pre_registered_outcome": plan.GATES["E7_MIRROR"]["pre_registered_outcome"],
    }


def stop_condition_verdicts(
    *,
    representation_gates: Mapping[str, Mapping[str, object]],
    contrastive_gate: Mapping[str, object],
    shuffle_degrades: Optional[bool],
    influence: Mapping[str, Mapping[str, object]],
    utility_status: str,
    target_update_count: int,
    chronology_proved: bool,
) -> dict[str, object]:
    """Section 19 stop conditions wired as receipt verdicts."""
    any_representation_pass = any(bool(item["passed"]) for item in representation_gates.values())
    contrastive_error_ok = bool(contrastive_gate.get("passed"))
    one_session_driven = any(
        bool(item.get("not_one_session_driven") is False) for item in representation_gates.values()
    )
    conditions = {
        "1_no_contrastive_arm_improves_R0_by_0_10_rad": not any_representation_pass,
        "2_snap_mismatch_not_improved_10pp": not any(
            bool(item["parts"]["snap_mismatch"]) for item in representation_gates.values()
        ),
        "3_R4_R5_do_not_beat_max_R2_RGE": not contrastive_error_ok,
        "4_shuffled_control_does_not_degrade": (False if shuffle_degrades is None
                                                else bool(not shuffle_degrades)),
        "5_gain_driven_by_one_source_session": bool(one_session_driven),
        "6_corrected_direction_no_coherent_utility_gain": (
            "PENDING_NOT_EVALUATED_PROCESS_GATE"
            if utility_status == "NOT_RUN_PROCESS_GATE" else False
        ),
        "7_small_gate_null_utility": "NOT_APPLICABLE_AC3_1_NOT_RUN",
        "8_larger_model_required": False,
        "9_governing_velocity_degrades": "NOT_APPLICABLE_NO_TRAINING_THIS_STAGE",
        "10_target_updates_or_target_selected_hyperparameters": bool(int(target_update_count) != 0),
        "11_chronology_not_provable": bool(not chronology_proved),
        "12_external_not_separable_from_filter": "NOT_APPLICABLE_NO_EXTERNAL_SCORE_THIS_STAGE",
    }
    fired = [name for name, value in conditions.items() if value is True]
    return {
        "conditions": conditions,
        "fired": fired,
        "any_fired": bool(fired),
    }


def screen_verdict(
    *,
    representation_gates: Mapping[str, Mapping[str, object]],
    contrastive_gate: Mapping[str, object],
    e7_mirror: Mapping[str, object],
    stop_conditions: Mapping[str, object],
) -> dict[str, object]:
    """The one-line AC3-0 verdict recorded in terminal.json.

    Under the operator review of 2026-08-29 the screen never authorizes a
    successor stage: AC3-1 stays HOLD, AC3-2 stays NO-GO, and the result goes
    to the operator.
    """
    representation_pass = any(bool(item["passed"]) for item in representation_gates.values())
    contrastive_pass = bool(contrastive_gate.get("passed"))
    if representation_pass and contrastive_pass:
        verdict = "REPRESENTATION_AND_CONTRASTIVE_GATES_PASS_UTILITY_PENDING_OPERATOR_DECISION"
    elif representation_pass:
        verdict = "REPRESENTATION_GATE_PASS_CONTRASTIVE_CLAIM_NOT_ESTABLISHED_UTILITY_PENDING"
    elif bool(e7_mirror.get("fired")):
        verdict = E7_MIRROR_DISPOSITION
    else:
        verdict = "AC3_0_NO_CONTRASTIVE_REPRESENTATION_VALUE_STOP"
    return {
        "verdict": verdict,
        "representation_gate_passed": representation_pass,
        "representation_passing_rows": [
            row for row, item in representation_gates.items() if bool(item["passed"])
        ],
        "contrastive_claim_passed": contrastive_pass,
        "e7_mirror_fired": bool(e7_mirror.get("fired")),
        "e7_mirror_disposition": e7_mirror.get("disposition"),
        "stop_conditions_fired": list(stop_conditions.get("fired", [])),
        "downstream_advance_gate": "PENDING_UTILITY_ROWS_NOT_RUN_PROCESS_GATE",
        "ac3_1_status": "HOLD (operator decision required; never set by the screen)",
        "ac3_2_status": "NO_GO (operator review 2026-08-29, regardless of screen outcome)",
    }


OPERATOR_EQUIVALENCE_BAND_RAD = float(
    plan.OPERATOR_REVIEW["operator_verdicts"]["R_GE_EQUIVALENT"]["equivalence_band_rad"]
)


def operator_verdicts(
    row_metrics_by_row: Mapping[str, Mapping[str, object]],
    contrastive_gate: Mapping[str, object],
    *,
    utility_status: str = plan.UTILITY_ROWS["status_this_run"],
) -> dict[str, object]:
    """The four pre-fixed operator interpretation strings, wired to fire.

    ``ADVANCE_AC3_1`` and ``STOP_AC3_NO_GPU`` carry a coherent-utility clause;
    with the utility rows not run under this run's process gate those two
    verdicts are PENDING and can never fire from this screen.
    """
    rge = _value(row_metrics_by_row.get("R0.5", {}), "session_mean_circular_error_rad")
    learned = {
        row: _value(row_metrics_by_row.get(row, {}), "session_mean_circular_error_rad")
        for row in plan.LEARNED_ROWS
    }
    r2 = learned.get("R2")
    best_learned_margin = (
        None if rge is None else max(
            (rge - value) for value in learned.values() if value is not None
        ) if any(value is not None for value in learned.values()) else None
    )
    rge_equivalent = (
        None if (rge is None or best_learned_margin is None)
        else bool(best_learned_margin <= OPERATOR_EQUIVALENCE_BAND_RAD + EPSILON)
    )
    utility_pending = utility_status == "NOT_RUN_PROCESS_GATE"
    e7_condition = (
        None if (r2 is None or rge is None)
        else bool(r2 < rge - EPSILON) and not bool(contrastive_gate.get("passed"))
    )
    advance = (
        False if utility_pending else None
    )
    no_gpu = (
        False if utility_pending else None
    )
    verdicts = {
        "STOP_LEARNING_KEEP_ZERO_PARAM_ENSEMBLE": {
            "fired": bool(rge_equivalent) if rge_equivalent is not None else None,
            "condition": plan.OPERATOR_REVIEW["operator_verdicts"]["R_GE_EQUIVALENT"]["condition"],
            "best_learned_margin_over_R0.5_rad": best_learned_margin,
            "equivalence_band_rad": OPERATOR_EQUIVALENCE_BAND_RAD,
        },
        "SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED": {
            "fired": bool(e7_condition) if e7_condition is not None else None,
            "condition": plan.OPERATOR_REVIEW["operator_verdicts"]["E7_MIRROR"]["condition"],
            "utility_clause_status": "NOT_EVALUATED_PROCESS_GATE",
        },
        "ADVANCE_AC3_1": {
            "fired": advance,
            "condition": plan.OPERATOR_REVIEW["operator_verdicts"]["ADVANCE"]["condition"],
            "utility_clause_status": "PENDING_NOT_EVALUATED_PROCESS_GATE",
            "note": "cannot fire from this screen: the coherent utility row is not run",
        },
        "STOP_AC3_NO_GPU": {
            "fired": no_gpu,
            "condition": plan.OPERATOR_REVIEW["operator_verdicts"]["NO_GPU"]["condition"],
            "utility_clause_status": "PENDING_NOT_EVALUATED_PROCESS_GATE",
        },
    }
    fired = [name for name, value in verdicts.items() if value["fired"] is True]
    return {
        "verdicts": verdicts,
        "fired": fired,
        "result_order": list(plan.OPERATOR_REVIEW["result_order"]),
        "advance_rule": plan.OPERATOR_REVIEW["advance_rule"],
    }
