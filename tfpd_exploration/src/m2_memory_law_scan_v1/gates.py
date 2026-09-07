"""Gate, verdict and drift-reading evaluation for the M2 memory-law scan."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class MemoryLawGateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryLawGateError(message)


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
    return float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))


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
        "delta_sd": float(math.sqrt(
            sum((v - mean) ** 2 for v in values) / len(values))),
        "positive_sessions": int(sum(1 for value in values if value > 0.0)),
        "session_count": int(len(values)),
    }


def evaluate_policy_gates(
    *,
    policy: str,
    within_m4: Mapping[str, float],
    baseline_within_m4: Mapping[str, float],
    external_by_budget: Mapping[int, Mapping[str, float]],
    baseline_external_by_budget: Mapping[int, Mapping[str, float]],
    gates_law: Mapping[str, Any],
) -> dict[str, Any]:
    """One challenger policy against UNIFORM_CAP30 under the pre-registered gates."""
    epsilon = float(gates_law["boundary_epsilon"])
    within = paired_delta(within_m4, baseline_within_m4)
    delta_margin = _margin(
        within["equal_session_mean_delta"],
        float(gates_law["primary"]["delta_floor"]), epsilon=epsilon,
    )
    breadth_margin = _margin(
        float(within["positive_sessions"]),
        float(gates_law["primary"]["breadth_min"]), epsilon=epsilon,
    )
    primary_pass = bool(delta_margin["meets_margin"] and breadth_margin["meets_margin"])
    safety_bounds: dict[str, Any] = {}
    for budget in sorted(int(item) for item in gates_law["safety"]["budgets"]):
        delta = paired_delta(
            external_by_budget[budget], baseline_external_by_budget[budget],
        )
        safety_bounds[f"m{budget}"] = {
            "delta": delta,
            "margin": _margin(
                delta["equal_session_mean_delta"],
                float(gates_law["safety"]["floor"]), epsilon=epsilon,
            ),
        }
    safety_pass = all(
        item["margin"]["meets_margin"] for item in safety_bounds.values()
    )
    return {
        "policy": str(policy),
        "primary": {
            "expression": str(gates_law["primary"]["expression"]),
            "delta": within,
            "delta_margin": delta_margin,
            "breadth_margin": breadth_margin,
            "passed": primary_pass,
            "positive_session_ids": sorted(
                name for name, value in within["per_session_delta"].items()
                if value > 0.0
            ),
        },
        "safety": {
            "expression": str(gates_law["safety"]["expression"]),
            "bounds": safety_bounds,
            "passed": safety_pass,
        },
        "selected": bool(primary_pass and safety_pass),
        "disposition": (
            "PRIMARY_PASS_SAFETY_FAIL_EXCLUDED" if primary_pass and not safety_pass
            else "SELECTED_ELIGIBLE" if primary_pass and safety_pass
            else "CHALLENGER_FAILS_THE_PRIMARY_GATE"
        ),
    }


def select_verdict(
    *,
    policy_results: Mapping[str, Mapping[str, Any]],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """The pre-registered verdict law over the evaluated challengers."""
    eligible = [name for name in policy_order
                if name in policy_results and policy_results[name]["selected"]]
    excluded = [name for name in policy_order
                if name in policy_results
                and policy_results[name]["primary"]["passed"]
                and not policy_results[name]["safety"]["passed"]]
    if not eligible:
        verdict = "MEMORY_LAW_UNIFORM_RETAINED"
        winner = None
        winning_delta = None
    else:
        winner = max(
            eligible,
            key=lambda name: (
                float(policy_results[name]["primary"]["delta"]["equal_session_mean_delta"]),
                -policy_order.index(name),
            ),
        )
        winning_delta = float(
            policy_results[winner]["primary"]["delta"]["equal_session_mean_delta"]
        )
        verdict = "UNCAPPED_WINS" if winner == "UNIFORM_UNCAPPED" else \
            f"EMA_RECENCY_WINS({winner})"
    return {
        "verdict": verdict,
        "winner": winner,
        "winning_within_m4_delta": winning_delta,
        "eligible_policies": eligible,
        "primary_pass_safety_fail_excluded": excluded,
        "tie_break": "largest within-M4 equal-session delta; ties broken by POLICY_ORDER",
    }


# ---------------------------------------------------------------------------
# The drift reading (descriptive only).
# ---------------------------------------------------------------------------


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    _require(len(x) == len(y) and len(x) >= 2, "spearman needs two equal nonempty series")
    rank_x = _average_ranks([float(item) for item in x])
    rank_y = _average_ranks([float(item) for item in y])
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    denominator_x = math.sqrt(sum((a - mean_x) ** 2 for a in rank_x))
    denominator_y = math.sqrt(sum((b - mean_y) ** 2 for b in rank_y))
    _require(denominator_x > 0.0 and denominator_y > 0.0,
             "spearman needs nonzero rank variance")
    return float(numerator / (denominator_x * denominator_y))


def drift_reading(
    *,
    session_completed_trials: Mapping[str, int],
    within_m4_deltas_by_policy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Does recency help where sessions are long?  Descriptive, never a gate."""
    sessions = sorted(session_completed_trials)
    counts = [int(session_completed_trials[name]) for name in sessions]
    median = float(sorted(counts)[len(counts) // 2]) if counts else 0.0
    long_half = [name for name in sessions
                 if int(session_completed_trials[name]) > median]
    short_half = [name for name in sessions
                  if int(session_completed_trials[name]) <= median]
    per_policy: dict[str, Any] = {}
    for policy in sorted(within_m4_deltas_by_policy):
        deltas = within_m4_deltas_by_policy[policy]
        _require(set(deltas) == set(sessions), f"{policy}: drift session set mismatch")
        series = [float(deltas[name]) for name in sessions]
        long_values = [float(deltas[name]) for name in long_half]
        short_values = [float(deltas[name]) for name in short_half]
        per_policy[policy] = {
            "spearman_rho_delta_vs_completed_trials": spearman_rho(counts, series),
            "long_half_mean_delta": (
                float(sum(long_values) / len(long_values)) if long_values else None
            ),
            "short_half_mean_delta": (
                float(sum(short_values) / len(short_values)) if short_values else None
            ),
        }
    return {
        "question": "does recency help where sessions are long?",
        "role": "descriptive only; never selects, never gates",
        "median_completed_trials": median,
        "long_half_sessions": long_half,
        "short_half_sessions": short_half,
        "per_policy": per_policy,
    }
