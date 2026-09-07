"""Frozen constants and source-grouped decision law."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "h1_support_resampled_postpool_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARMS = ("FROZEN-C1", "LP-F3", "LP-R3", "SRPD")
TRAIN_ARMS = ARMS[1:]
SUPPORT = 3
WINDOW = 700
TRIAL_LENGTH = 1024
UNITS = 176
OUTPUTS = 7
EPOCHS = 12
BATCH_SIZE = 32
TRAIN_STRIDE = 4
SEED = 42
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 0.0
PREDICTION_DIVISOR = 20.0
DISTILL_WEIGHT = 1.0e-5
DISTILL_FLOOR = 1.0e-8
EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
NEGATIVE_PREDECESSOR_TERMINAL_SHA256 = "5decf64d428c8c88eef72138c46c51210931376f4dbc770f497b2249546bdd81"

PRIMARY_MEAN = 0.005
PRIMARY_NONNEGATIVE = 4
PRIMARY_WORST = -0.010
DISTILL_INCREMENT_MEAN = -0.002
DISTILL_INCREMENT_WORST = -0.010
CARRIER_EXTENSION_ABS_MEAN = 0.003
CARRIER_EXTENSION_COMMON_SIGN = 4


def _finite(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if len(values) != len(DATE_ORDER):
        raise ValueError(f"{label} requires exactly five date values")
    result = tuple(float(value) for value in values)
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"{label} contains a nonfinite value")
    return result


def _gate(values: Sequence[float], *, mean_min: float, nonnegative_min: int, worst_min: float) -> dict[str, Any]:
    numeric = _finite(values, label="gate")
    mean = sum(numeric) / len(numeric)
    nonnegative = sum(value >= 0.0 for value in numeric)
    worst = min(numeric)
    return {
        "values": list(numeric),
        "mean": mean,
        "nonnegative_dates": nonnegative,
        "worst": worst,
        "required_mean": mean_min,
        "required_nonnegative_dates": nonnegative_min,
        "required_worst": worst_min,
        "pass": mean >= mean_min and nonnegative >= nonnegative_min and worst >= worst_min,
    }


def _directional_extension(values: Sequence[float], *, name: str) -> dict[str, Any]:
    numeric = _finite(values, label=name)
    mean = sum(numeric) / len(numeric)
    positive = sum(value >= 0.0 for value in numeric)
    negative = sum(value <= 0.0 for value in numeric)
    common = max(positive, negative)
    return {
        "contrast": name,
        "values": list(numeric),
        "mean": mean,
        "nonnegative_dates": positive,
        "nonpositive_dates": negative,
        "common_sign_dates": common,
        "required_abs_mean": CARRIER_EXTENSION_ABS_MEAN,
        "required_common_sign_dates": CARRIER_EXTENSION_COMMON_SIGN,
        "trigger": abs(mean) >= CARRIER_EXTENSION_ABS_MEAN and common >= CARRIER_EXTENSION_COMMON_SIGN,
    }


def decide_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(DATE_ORDER) or tuple(str(row.get("outer_date")) for row in rows) != DATE_ORDER:
        raise ValueError("outer-date order drift")
    score = {arm: tuple(float(row["scores"][arm]) for row in rows) for arm in ARMS}
    anchor = score["FROZEN-C1"]
    primary = {
        arm: _gate(
            tuple(value - base for value, base in zip(score[arm], anchor)),
            mean_min=PRIMARY_MEAN,
            nonnegative_min=PRIMARY_NONNEGATIVE,
            worst_min=PRIMARY_WORST,
        )
        for arm in TRAIN_ARMS
    }
    diversity = tuple(value - fixed for value, fixed in zip(score["LP-R3"], score["LP-F3"]))
    distill = tuple(value - random_value for value, random_value in zip(score["SRPD"], score["LP-R3"]))
    distill_safety = _gate(
        distill,
        mean_min=DISTILL_INCREMENT_MEAN,
        nonnegative_min=0,
        worst_min=DISTILL_INCREMENT_WORST,
    )
    if primary["SRPD"]["pass"] and distill_safety["pass"]:
        selected = "SRPD"
    elif primary["LP-R3"]["pass"]:
        selected = "LP-R3"
    elif primary["LP-F3"]["pass"]:
        selected = "LP-F3"
    else:
        selected = None
    extensions = (
        _directional_extension(diversity, name="LP-R3_MINUS_LP-F3"),
        _directional_extension(distill, name="SRPD_MINUS_LP-R3"),
    )
    triggered = [item for item in extensions if item["trigger"]]
    return {
        "primary_gates": primary,
        "support_diversity_by_date": list(diversity),
        "distillation_increment_by_date": list(distill),
        "distillation_safety_gate": distill_safety,
        "selected_arm": selected,
        "pass": selected is not None,
        "verdict": "PASS_FOR_SEPARATE_ALL_SOURCE_WORKORDER" if selected else "STOP_H1_LATE_POOL_C1_BRANCH",
        "carrier_extension_diagnostics": list(extensions),
        "carrier_extension_triggered": bool(triggered),
        "carrier_extension_triggered_contrasts": [item["contrast"] for item in triggered],
    }


__all__ = ("ARMS", "DATE_ORDER", "SCHEMA", "TRAIN_ARMS", "decide_oof")

