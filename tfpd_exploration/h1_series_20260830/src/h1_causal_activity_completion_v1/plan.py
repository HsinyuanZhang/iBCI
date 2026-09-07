"""Frozen constants and decision law for H1-CAC V1."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "h1_causal_activity_completion_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARM_ORDER = ("A-STATIC", "B-TRIAL7", "C-FIX7", "D-EMED7")
CHUNK_LENGTH_BY_OUTER_DATE = {
    "19250108": 768,
    "19250113": 800,
    "19250115": 768,
    "19250119": 768,
    "19250120": 768,
}
FINAL_ALL_SOURCE_CHUNK_LENGTH = 768
STAGE0_SUPPORT = 4
STAGE1_SUPPORT = 3
MAX_MEMBERS = 7
IDENTITY_LENGTH = 1024
UNITS = 176
WINDOW = 700
BATCH_SIZE = 32
MIN_CONTENT_GAIN = 0.010
MIN_CONTENT_POSITIVE_DATES = 4
MIN_RECOVERY_FRACTION = 0.50
MIN_DEPLOYABLE_POSITIVE_DATES = 3
MIN_DEPLOYABLE_WORST_DATE = -0.010


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def decide(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the two independent, pre-registered Stage-0 gates."""

    if tuple(str(row.get("outer_date")) for row in date_rows) != DATE_ORDER:
        raise ValueError("H1-CAC date order drift")
    by_date: dict[str, dict[str, float]] = {}
    b_deltas: list[float] = []
    c_deltas: list[float] = []
    d_deltas: list[float] = []
    for row in date_rows:
        results = row.get("results")
        if not isinstance(results, Sequence):
            raise ValueError("H1-CAC date row lacks results")
        by_arm = {str(item.get("arm")): float(item["equal_recording_mean_r2"])
                  for item in results if isinstance(item, Mapping)}
        if tuple(by_arm) != ARM_ORDER or not all(_finite(value) for value in by_arm.values()):
            raise ValueError("H1-CAC arm order/value drift")
        delta = {
            "B-TRIAL7_minus_A-STATIC": by_arm["B-TRIAL7"] - by_arm["A-STATIC"],
            "C-FIX7_minus_A-STATIC": by_arm["C-FIX7"] - by_arm["A-STATIC"],
            "D-EMED7_minus_A-STATIC": by_arm["D-EMED7"] - by_arm["A-STATIC"],
        }
        by_date[str(row["outer_date"])] = delta
        b_deltas.append(delta["B-TRIAL7_minus_A-STATIC"])
        c_deltas.append(delta["C-FIX7_minus_A-STATIC"])
        d_deltas.append(delta["D-EMED7_minus_A-STATIC"])
    mean_b = sum(b_deltas) / len(b_deltas)
    mean_c = sum(c_deltas) / len(c_deltas)
    mean_d = sum(d_deltas) / len(d_deltas)
    positive_b = sum(value > 0.0 for value in b_deltas)
    positive_d = sum(value >= 0.0 for value in d_deltas)
    content_pass = mean_b >= MIN_CONTENT_GAIN and positive_b >= MIN_CONTENT_POSITIVE_DATES
    recovery = mean_d / mean_b if mean_b > 0.0 else None
    deployable_pass = bool(
        content_pass
        and recovery is not None
        and recovery >= MIN_RECOVERY_FRACTION
        and positive_d >= MIN_DEPLOYABLE_POSITIVE_DATES
        and min(d_deltas) >= MIN_DEPLOYABLE_WORST_DATE
    )
    if not content_pass:
        verdict = "STOP_H1_ACTIVITY_COMPLETION_CONTENT_DID_NOT_TRANSFER"
    elif not deployable_pass:
        verdict = "PASS_H1_ACTIVITY_CONTENT_BUT_STOP_BOUNDARY_FREE_DETECTOR"
    else:
        verdict = "PASS_H1_CAC_STAGE0_FOR_C1_SPECIFIC_M3_TO_M7"
    return {
        "per_date": by_date,
        "equal_date_mean": {
            "B-TRIAL7_minus_A-STATIC": mean_b,
            "C-FIX7_minus_A-STATIC": mean_c,
            "D-EMED7_minus_A-STATIC": mean_d,
        },
        "gate_1_content": {
            "mean_gain": mean_b,
            "required_mean_gain": MIN_CONTENT_GAIN,
            "positive_dates": positive_b,
            "required_positive_dates": MIN_CONTENT_POSITIVE_DATES,
            "pass": content_pass,
        },
        "gate_2_deployable_recovery": {
            "recovery_fraction": recovery,
            "required_recovery_fraction": MIN_RECOVERY_FRACTION,
            "positive_dates": positive_d,
            "required_positive_dates": MIN_DEPLOYABLE_POSITIVE_DATES,
            "worst_date_delta": min(d_deltas),
            "required_worst_date_delta": MIN_DEPLOYABLE_WORST_DATE,
            "pass": deployable_pass,
        },
        "verdict": verdict,
    }


def dry_plan() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_plan",
        "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
        "stage": 0,
        "dates": list(DATE_ORDER),
        "arms": list(ARM_ORDER),
        "chunk_length_by_outer_date": CHUNK_LENGTH_BY_OUTER_DATE,
        "support_members": STAGE0_SUPPORT,
        "maximum_members": MAX_MEMBERS,
        "identity_length": IDENTITY_LENGTH,
        "window": WINDOW,
        "target_updates": 0,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
    }


__all__ = (
    "ARM_ORDER", "BATCH_SIZE", "CHUNK_LENGTH_BY_OUTER_DATE", "DATE_ORDER",
    "IDENTITY_LENGTH", "MAX_MEMBERS", "SCHEMA", "STAGE0_SUPPORT", "UNITS",
    "WINDOW", "decide", "dry_plan",
)

