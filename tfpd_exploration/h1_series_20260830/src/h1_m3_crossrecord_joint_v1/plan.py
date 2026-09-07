"""Frozen scalar contract and five-fold decision law."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "h1_m3_cross_record_joint_postpool_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARMS = ("FROZEN-C1", "N3-XR12", "J3-XR12")
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
EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
PROFILE_PREDECESSOR_TERMINAL_SHA256 = "905d0a46bda86a69a31ea290b73a90db674e958c981cfef154101fddec68e3b4"

FINAL_MEAN = 0.005
FINAL_NONNEGATIVE = 4
FINAL_WORST = -0.010
JOINT_INCREMENT_MEAN = 0.0
JOINT_INCREMENT_NONNEGATIVE = 3
JOINT_INCREMENT_WORST = -0.010


def _gate(values: Sequence[float], *, mean_min: float, positives_min: int, worst_min: float) -> dict[str, Any]:
    if len(values) != len(DATE_ORDER) or not all(isinstance(value, (float, int)) for value in values):
        raise ValueError("five finite date values are required")
    numeric = tuple(float(value) for value in values)
    if not all(value == value and abs(value) != float("inf") for value in numeric):
        raise ValueError("date gain is nonfinite")
    mean = sum(numeric) / len(numeric)
    nonnegative = sum(value >= 0.0 for value in numeric)
    worst = min(numeric)
    return {
        "mean": mean,
        "nonnegative_dates": nonnegative,
        "worst": worst,
        "required_mean": mean_min,
        "required_nonnegative_dates": positives_min,
        "required_worst": worst_min,
        "pass": mean >= mean_min and nonnegative >= positives_min and worst >= worst_min,
    }


def decide_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(DATE_ORDER) or tuple(str(row.get("outer_date")) for row in rows) != DATE_ORDER:
        raise ValueError("outer-date order drift")
    base = tuple(float(row["scores"]["FROZEN-C1"]) for row in rows)
    native = tuple(float(row["scores"]["N3-XR12"]) for row in rows)
    joint = tuple(float(row["scores"]["J3-XR12"]) for row in rows)
    native_gain = tuple(value - anchor for value, anchor in zip(native, base))
    joint_gain = tuple(value - anchor for value, anchor in zip(joint, base))
    joint_increment = tuple(value - control for value, control in zip(joint, native))
    native_gate = _gate(native_gain, mean_min=FINAL_MEAN, positives_min=FINAL_NONNEGATIVE, worst_min=FINAL_WORST)
    joint_gate = _gate(joint_gain, mean_min=FINAL_MEAN, positives_min=FINAL_NONNEGATIVE, worst_min=FINAL_WORST)
    increment_gate = _gate(
        joint_increment,
        mean_min=JOINT_INCREMENT_MEAN,
        positives_min=JOINT_INCREMENT_NONNEGATIVE,
        worst_min=JOINT_INCREMENT_WORST,
    )
    if joint_gate["pass"] and increment_gate["pass"]:
        selected, verdict = "J3-XR12", "PASS_J3_FOR_ALL_SOURCE_RETRAIN"
    elif native_gate["pass"]:
        selected, verdict = "N3-XR12", "PASS_N3_FOR_ALL_SOURCE_RETRAIN"
    else:
        selected, verdict = None, "STOP_H1_M3_CROSS_RECORD_MATCHED_TRAINING"
    return {
        "native_gain_by_date": list(native_gain),
        "joint_gain_by_date": list(joint_gain),
        "joint_increment_by_date": list(joint_increment),
        "native_gate": native_gate,
        "joint_gate": joint_gate,
        "joint_increment_gate": increment_gate,
        "selected_arm": selected,
        "pass": selected is not None,
        "verdict": verdict,
    }


__all__ = ("ARMS", "DATE_ORDER", "SCHEMA", "decide_oof")

