"""Frozen constants and factorial mechanism classification."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from h1_support_resampled_postpool_v1.plan import DATE_ORDER


SCHEMA = "h1_activity_carrier_resampling_factorial_v1"
NEW_ARMS = ("LP-AR-CF", "LP-AF-CR")
MAIN_TERMINAL_SHA256 = "c047bbf5bd6e324fcd250ef2702ab507fff54b6a13c6bc1314963983bc298a94"
EFFECT_ABS_MEAN = 0.003
EFFECT_COMMON_SIGN = 4


def _summary(values: Sequence[float], name: str) -> dict[str, Any]:
    if len(values) != len(DATE_ORDER):
        raise ValueError(f"{name} requires five date values")
    numeric = tuple(float(value) for value in values)
    if not all(value == value and abs(value) != float("inf") for value in numeric):
        raise ValueError(f"{name} contains nonfinite values")
    mean = sum(numeric) / len(numeric)
    nonnegative = sum(value >= 0.0 for value in numeric)
    nonpositive = sum(value <= 0.0 for value in numeric)
    common = max(nonnegative, nonpositive)
    return {
        "name": name,
        "values": list(numeric),
        "mean": mean,
        "nonnegative_dates": nonnegative,
        "nonpositive_dates": nonpositive,
        "common_sign_dates": common,
        "directional": abs(mean) >= EFFECT_ABS_MEAN and common >= EFFECT_COMMON_SIGN,
        "direction": "positive" if mean > 0 else "negative" if mean < 0 else "zero",
    }


def decide_factorial(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(DATE_ORDER) or tuple(str(row.get("outer_date")) for row in rows) != DATE_ORDER:
        raise ValueError("outer-date order drift")
    activity: list[float] = []
    carrier: list[float] = []
    interaction: list[float] = []
    for row in rows:
        score = row["factorial_scores"]
        ff, rr = float(score["FF"]), float(score["RR"])
        rf, fr = float(score["RF"]), float(score["FR"])
        activity.append(0.5 * ((rf - ff) + (rr - fr)))
        carrier.append(0.5 * ((fr - ff) + (rr - rf)))
        interaction.append(rr - rf - fr + ff)
    summaries = {
        "activity_resampling_main_effect": _summary(activity, "activity_resampling_main_effect"),
        "carrier_resampling_main_effect": _summary(carrier, "carrier_resampling_main_effect"),
        "matched_interaction": _summary(interaction, "matched_interaction"),
    }
    active = [name for name, value in summaries.items() if value["directional"]]
    if summaries["matched_interaction"]["directional"]:
        classification = "MATCHED_CORESAMPLING_INTERACTION"
    elif set(active) == {"activity_resampling_main_effect"} and summaries[active[0]]["direction"] == "positive":
        classification = "ACTIVITY_DIVERSITY_DOMINANT"
    elif set(active) == {"carrier_resampling_main_effect"} and summaries[active[0]]["direction"] == "positive":
        classification = "CARRIER_DIVERSITY_DOMINANT"
    elif (
        summaries["activity_resampling_main_effect"]["directional"]
        and summaries["carrier_resampling_main_effect"]["directional"]
        and summaries["activity_resampling_main_effect"]["direction"] == "positive"
        and summaries["carrier_resampling_main_effect"]["direction"] == "positive"
    ):
        classification = "BOTH_MARGINALS_POSITIVE"
    else:
        classification = "COUPLED_EFFECT_NOT_REDUCIBLE"
    return {
        "summaries": summaries,
        "classification": classification,
        "directional_components": active,
        "status": "COMPLETE_DESCRIPTIVE_MECHANISM_ONLY",
        "authorizes_model_selection": False,
        "authorizes_evalai": False,
    }


__all__ = ("MAIN_TERMINAL_SHA256", "NEW_ARMS", "SCHEMA", "decide_factorial")
