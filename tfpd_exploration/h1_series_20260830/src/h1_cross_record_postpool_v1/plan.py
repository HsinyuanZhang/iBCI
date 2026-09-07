"""Frozen numerical and selection contract for H1 cross-record memory V1."""
from __future__ import annotations

from typing import Mapping


SCHEMA = "h1_cross_record_anchored_postpool_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
G_VALUES = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
FAMILIES = ("RN", "RP")
WINDOW = 700
CHUNK_LENGTH = 768
SUPPORT_MEMBERS = 3
MAX_MEMBERS = 7
UNITS = 176
IDENTITY_LENGTH = 1024
OUTPUTS = 7
BATCH_SIZE = 24
RP_NONINFERIOR_TOLERANCE = 0.002
SMALLER_GATE_TOLERANCE = 0.001
MIN_OOF_MEAN_GAIN = 0.005
MIN_OOF_NONNEGATIVE_DATES = 4
MIN_OOF_WORST_DATE = -0.010
EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"


def arm_name(family: str, gate: float) -> str:
    if family not in FAMILIES or float(gate) not in G_VALUES:
        raise ValueError("unknown H1 cross-record arm")
    return f"{family}-G{int(round(float(gate) * 100)):03d}"


ARM_ORDER = ("A-STATIC",) + tuple(arm_name(family, gate) for family in FAMILIES for gate in G_VALUES)


def select_source_arm(mean_r2: Mapping[str, float]) -> dict[str, object]:
    """Apply the frozen RP-preferred, smaller-gate tie law."""

    if tuple(mean_r2) != ARM_ORDER:
        raise ValueError("source arm order drift")
    values = {name: float(value) for name, value in mean_r2.items()}
    if not all(value == value and value not in (float("inf"), float("-inf")) for value in values.values()):
        raise ValueError("source arm score is nonfinite")
    candidates = ARM_ORDER[1:]
    best_value = max(values[name] for name in candidates)
    rp_eligible = [
        arm_name("RP", gate) for gate in G_VALUES
        if values[arm_name("RP", gate)] >= best_value - RP_NONINFERIOR_TOLERANCE
    ]
    if rp_eligible:
        family = "RP"
        family_best = max(values[name] for name in rp_eligible)
        eligible = [
            arm_name(family, gate) for gate in G_VALUES
            if values[arm_name(family, gate)] >= family_best - SMALLER_GATE_TOLERANCE
        ]
    else:
        family = "RN"
        family_best = max(values[arm_name(family, gate)] for gate in G_VALUES)
        eligible = [
            arm_name(family, gate) for gate in G_VALUES
            if values[arm_name(family, gate)] >= family_best - SMALLER_GATE_TOLERANCE
        ]
    selected = min(eligible, key=lambda name: int(name.rsplit("G", 1)[1]))
    gate = int(selected.rsplit("G", 1)[1]) / 100.0
    return {
        "arm": selected,
        "family": family,
        "gate": gate,
        "source_mean_r2": values[selected],
        "source_gain_vs_static": values[selected] - values["A-STATIC"],
        "unconstrained_best_r2": best_value,
        "rp_preference_tolerance": RP_NONINFERIOR_TOLERANCE,
        "smaller_gate_tolerance": SMALLER_GATE_TOLERANCE,
    }


def decide_oof(rows: list[Mapping[str, object]]) -> dict[str, object]:
    if tuple(str(row["outer_date"]) for row in rows) != DATE_ORDER:
        raise ValueError("OOF date order drift")
    deltas = [float(row["selected_delta_vs_static"]) for row in rows]
    mean = sum(deltas) / len(deltas)
    nonnegative = sum(value >= 0.0 for value in deltas)
    worst = min(deltas)
    passed = mean >= MIN_OOF_MEAN_GAIN and nonnegative >= MIN_OOF_NONNEGATIVE_DATES and worst >= MIN_OOF_WORST_DATE
    return {
        "mean_gain": mean,
        "nonnegative_dates": nonnegative,
        "worst_date_gain": worst,
        "required_mean_gain": MIN_OOF_MEAN_GAIN,
        "required_nonnegative_dates": MIN_OOF_NONNEGATIVE_DATES,
        "required_worst_date_gain": MIN_OOF_WORST_DATE,
        "pass": passed,
        "verdict": "PASS_FOR_ALL_SOURCE_PACKAGE" if passed else "STOP_OR_MATCHED_TRAINING_ONLY",
    }


__all__ = (
    "ARM_ORDER", "BATCH_SIZE", "CHUNK_LENGTH", "DATE_ORDER", "EXPECTED_GPU0_UUID",
    "FAMILIES", "G_VALUES", "IDENTITY_LENGTH", "MAX_MEMBERS", "OUTPUTS", "SCHEMA",
    "SUPPORT_MEMBERS", "UNITS", "WINDOW", "arm_name", "decide_oof", "select_source_arm",
)
