"""The AC3-U §8 gate evaluation, with predeclared boundary semantics.

```text
ADVANCE_SMALL_M4_CARRIER_STUDY requires BOTH:
  UGE - U0 equal-session mean raw matrix R2 >= +0.01
  positive sessions (UGE > U0 paired per session) >= 4/6
U2 rescue clause: U2 may substitute only if U2 - UGE >= +0.01 under the same
  replay (and, as a substitute for UGE, the same mean margin and the same 4/6
  breadth against U0).
If UGE and U2 both fail: AC3_U_UTILITY_NULL__CLOSE_AC3_ON_FROZEN_M4_SURFACE
```

Boundary semantics: the margin is the exact ``>=`` comparison on the float64
equal-session mean, so a value exactly at +0.01 passes and a value 1e-13 below
it fails.  The program epsilon 1e-12 is applied as a disclosed boundary BAND
(``within_epsilon_band_of_boundary``) that surfaces a near-boundary miss for
operator review; it never widens the margin and never flips a verdict.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import plan


class AC3UGateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3UGateError(message)


def margin_verdict(delta: float, margin: float = plan.MARGIN_R2) -> dict[str, object]:
    """Exact ``>=`` margin test plus the disclosed program-epsilon band."""
    value = float(delta)
    meets = value >= float(margin)
    return {
        "value": value,
        "margin": float(margin),
        "meets_margin": bool(meets),
        "within_epsilon_band_of_boundary": bool(
            (not meets) and value >= float(margin) - plan.GATE_BOUNDARY_EPSILON
        ),
        "boundary_epsilon": plan.GATE_BOUNDARY_EPSILON,
        "boundary_rule": plan.GATES["boundary_rule"],
    }


def equal_session_mean(values: Sequence[float]) -> float:
    _require(bool(values), "equal-session mean of an empty roster")
    return float(sum(float(item) for item in values) / len(values))


def paired_deltas(
    row_sessions: Mapping[str, float], baseline_sessions: Mapping[str, float],
) -> dict[str, object]:
    """Per-session paired deltas plus the equal-session mean difference.

    The mean difference is ``mean(row) - mean(baseline)`` -- the same arithmetic
    the frozen ``policy._mean_delta`` uses (difference of the two row means),
    never the mean of the per-session differences.
    """
    _require(
        tuple(row_sessions) == tuple(baseline_sessions),
        "paired delta session roster drift",
    )
    return {
        "per_session": {
            session: float(row_sessions[session]) - float(baseline_sessions[session])
            for session in row_sessions
        },
        "equal_session_mean_delta": equal_session_mean(list(row_sessions.values())) - equal_session_mean(
            list(baseline_sessions.values())
        ),
        "positive_sessions": int(sum(
            1 for session in row_sessions
            if float(row_sessions[session]) > float(baseline_sessions[session])
        )),
        "n_sessions": len(row_sessions),
    }


def evaluate_gates(
    *,
    governing_r2: Mapping[str, Mapping[str, float]],
    sessions: Sequence[str],
) -> dict[str, object]:
    """Evaluate the §8 gates on the governing (raw, unfiltered) per-session R2."""
    roster = tuple(sessions)
    _require(tuple(governing_r2) == plan.ROW_ORDER, "governing R2 row topology drift")
    for row in plan.ROW_ORDER:
        _require(tuple(governing_r2[row]) == roster, f"governing R2 session roster drift for {row}")
    u0 = governing_r2["U0"]
    uge = governing_r2["UGE"]
    u2 = governing_r2["U2"]

    uge_minus_u0 = paired_deltas(uge, u0)
    u2_minus_uge = paired_deltas(u2, uge)
    u2_minus_u0 = paired_deltas(u2, u0)

    uge_mean_verdict = margin_verdict(uge_minus_u0["equal_session_mean_delta"])
    uge_breadth = bool(uge_minus_u0["positive_sessions"] >= plan.POSITIVE_SESSIONS_REQUIRED)
    uge_passed = bool(uge_mean_verdict["meets_margin"] and uge_breadth)

    rescue_mean = margin_verdict(u2_minus_uge["equal_session_mean_delta"])
    rescue_substitution_mean = margin_verdict(u2_minus_u0["equal_session_mean_delta"])
    rescue_breadth = bool(u2_minus_u0["positive_sessions"] >= plan.POSITIVE_SESSIONS_REQUIRED)
    u2_rescue = bool(
        rescue_mean["meets_margin"]
        and rescue_substitution_mean["meets_margin"]
        and rescue_breadth
    )

    if uge_passed:
        disposition = plan.DISPOSITION_ADVANCE
    elif u2_rescue:
        disposition = plan.DISPOSITION_ADVANCE_U2_RESCUE
    else:
        disposition = plan.DISPOSITION_NULL

    sensitivity_sessions = tuple(item for item in roster if item != plan.HIGH_ERROR_SESSION)
    sensitivity_u0 = {key: u0[key] for key in sensitivity_sessions}
    sensitivity_uge = {key: uge[key] for key in sensitivity_sessions}
    sensitivity_u2 = {key: u2[key] for key in sensitivity_sessions}

    return {
        "scoring": "raw_no_output_filter",
        "governing": True,
        "sessions": list(roster),
        "n_sessions": len(roster),
        "positive_sessions_required": plan.POSITIVE_SESSIONS_REQUIRED,
        "high_error_session": plan.HIGH_ERROR_SESSION,
        "high_error_session_included_in_governing": plan.HIGH_ERROR_SESSION in roster,
        "equal_session_mean_r2": {row: equal_session_mean(list(governing_r2[row].values()))
                                  for row in plan.ROW_ORDER},
        "UGE_minus_U0": uge_minus_u0,
        "U2_minus_UGE": u2_minus_uge,
        "U2_minus_U0": u2_minus_u0,
        "UGE_gate": {
            "mean_margin": uge_mean_verdict,
            "positive_sessions": int(uge_minus_u0["positive_sessions"]),
            "breadth_required": plan.POSITIVE_SESSIONS_REQUIRED,
            "breadth_passed": uge_breadth,
            "passed": uge_passed,
            "expression": plan.GATES["ADVANCE_SMALL_M4_CARRIER_STUDY"]["requires_both"],
        },
        "U2_rescue": {
            "over_UGE_margin": rescue_mean,
            "substitution_margin_over_U0": rescue_substitution_mean,
            "positive_sessions_over_U0": int(u2_minus_u0["positive_sessions"]),
            "breadth_passed": rescue_breadth,
            "passed": u2_rescue,
            "expression": plan.GATES["U2_RESCUE"]["expression"],
        },
        "disposition": disposition,
        "disposition_strings": {
            "advance": plan.DISPOSITION_ADVANCE,
            "advance_by_u2_rescue": plan.DISPOSITION_ADVANCE_U2_RESCUE,
            "null": plan.DISPOSITION_NULL,
        },
        "five_session_sensitivity_non_governing": {
            "non_governing": True,
            "excluded_session": plan.HIGH_ERROR_SESSION,
            "reason": (
                "predeclared high-error stratum; it is NOT excludable from the "
                "governing mean or the 4/6 denominator, so this summary is "
                "reported for sensitivity only"
            ),
            "sessions": list(sensitivity_sessions),
            "equal_session_mean_r2": {
                "U0": equal_session_mean(list(sensitivity_u0.values())),
                "UGE": equal_session_mean(list(sensitivity_uge.values())),
                "U2": equal_session_mean(list(sensitivity_u2.values())),
            },
            "UGE_minus_U0": paired_deltas(sensitivity_uge, sensitivity_u0),
            "U2_minus_UGE": paired_deltas(sensitivity_u2, sensitivity_uge),
        },
        "no_m10_m30_claim": plan.GATES["no_m10_m30_claim"],
        "pre_registered_expectation": dict(plan.PRE_REGISTERED_EXPECTATION),
    }
