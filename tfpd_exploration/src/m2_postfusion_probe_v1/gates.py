"""Gate and verdict evaluation for the M2 post-fusion identity probe.

The verdict law is three-way on the gated head-to-head
(POSTFUSION_MEAN - POOLED on external_post30_local):

* POSTFUSION_PROMISING  delta >= +0.005 AND positive sessions >= 4/6;
* POSTFUSION_HARMFUL    delta <= -0.005;
* POSTFUSION_NULL       otherwise (including a delta-floor pass with a
  breadth failure, disclosed as ``mean_pass_breadth_fail``).

Margins are exact ``>=``/``<=`` on float64 equal-session means with the
1e-12 disclosure band (the Stage-O/P convention); a miss inside the band
never flips the verdict.
"""

from __future__ import annotations

from typing import Any, Mapping


class PostFusionProbeGateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostFusionProbeGateError(message)


def _margin(value: float, threshold: float, *, epsilon: float) -> dict[str, Any]:
    """Exact ``>=`` on float64 with the 1e-12 disclosure band (Stage-O/P law)."""
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


def session_sd(session_values: Mapping[str, float]) -> float:
    """Population standard deviation (ddof=0) across the session R2 values."""
    _require(bool(session_values), "session sd needs sessions")
    values = [float(v) for v in session_values.values()]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return float(variance ** 0.5)


def paired_delta(
    candidate: Mapping[str, float], reference: Mapping[str, float],
) -> dict[str, Any]:
    _require(set(candidate) == set(reference) and bool(candidate),
             "paired session set mismatch")
    deltas = {name: float(candidate[name]) - float(reference[name])
              for name in sorted(candidate)}
    values = [deltas[name] for name in sorted(deltas)]
    mean = float(sum(values) / len(values))
    return {
        "per_session_delta": deltas,
        "equal_session_mean_delta": mean,
        "delta_sd": float(
            (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5),
        "positive_sessions": int(sum(1 for value in values if value > 0.0)),
        "session_count": int(len(values)),
    }


def evaluate_verdict(
    *,
    external_delta: Mapping[str, Any],
    gates_law: Mapping[str, Any],
) -> dict[str, Any]:
    """The pre-registered three-way verdict on the gated head-to-head."""
    epsilon = float(gates_law["boundary_epsilon"])
    delta_value = float(external_delta["equal_session_mean_delta"])
    promising_margin = _margin(
        delta_value, float(gates_law["primary"]["delta_floor"]), epsilon=epsilon,
    )
    breadth_value = int(external_delta["positive_sessions"])
    breadth_margin = _margin(
        float(breadth_value), float(gates_law["primary"]["breadth_min"]),
        epsilon=epsilon,
    )
    harmful_margin = _margin(
        -delta_value, -float(gates_law["harmful"]["floor"]), epsilon=epsilon,
    )
    mean_pass = bool(promising_margin["meets_margin"])
    breadth_pass = bool(breadth_margin["meets_margin"])
    harmful_pass = bool(harmful_margin["meets_margin"])
    if mean_pass and breadth_pass:
        verdict = "POSTFUSION_PROMISING"
    elif harmful_pass:
        verdict = "POSTFUSION_HARMFUL"
    else:
        verdict = "POSTFUSION_NULL"
    return {
        "verdict": verdict,
        "expression": str(gates_law["primary"]["expression"]),
        "delta": external_delta,
        "delta_margin": promising_margin,
        "breadth_margin": breadth_margin,
        "harmful_margin": harmful_margin,
        "mean_pass_breadth_fail": bool(mean_pass and not breadth_pass),
        "verdict_law": dict(gates_law["verdict_law"]),
    }
