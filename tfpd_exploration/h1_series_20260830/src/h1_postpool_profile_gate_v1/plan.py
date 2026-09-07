"""Frozen H1 post-pool profile-gate contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "h1_postpool_profile_gate_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
PROFILE_LENGTH = 700
EPOCHS = 12
TRAIN_STRIDE = 4
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-2
WEIGHT_DECAY = 0.0
SEED = 42
MIN_OOF_MEAN_GAIN = 0.005
MIN_OOF_NONNEGATIVE_DATES = 4
MIN_OOF_WORST_DATE = -0.010
EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
DIRECT_TERMINAL_SHA256 = "ec86e26c513c3ccb9ab4ca5f8d0f0ce244e6b162b507c635f581078559a1804a"
SCALAR_TERMINAL_SHA256 = "8958985646dd801a590cd7bf5429d1be6d061e4eb95c60857e67b9ec007c0f5f"


def decide_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row["outer_date"]) for row in rows) != DATE_ORDER:
        raise ValueError("profile OOF date order drift")
    deltas = [float(row["delta_vs_static"]) for row in rows]
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
        "verdict": "PASS_FOR_ALL_SOURCE_PROFILE" if passed else "STOP_H1_POSTPOOL_PROFILE_FAMILY",
    }


__all__ = ("BATCH_SIZE", "DATE_ORDER", "DIRECT_TERMINAL_SHA256", "EPOCHS",
           "EXPECTED_GPU0_UUID", "LEARNING_RATE", "PROFILE_LENGTH", "SCALAR_TERMINAL_SHA256",
           "SCHEMA", "SEED", "TRAIN_STRIDE", "WEIGHT_DECAY", "decide_oof")

