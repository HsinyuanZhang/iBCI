"""Pure, file-free contract for the labelled-T4 confidence-FiLM sweep."""
from __future__ import annotations

from dataclasses import dataclass


ACTIVITY_CALIBRATION_N = 30
T4_BUDGETS = (10, 15, 20, 30, 50)
COMMON_EVALUATION_START = 50
FIXED_SPLIT = (27, 6, 6)
MAX_UNITS_EXCLUSIVE = 100


@dataclass(frozen=True)
class ConfidenceFiLMProtocol:
    activity_calibration_n: int
    t4_budget: int
    evaluation_calibration_n: int
    common_evaluation_start: int
    split_counts: tuple[int, int, int]
    max_units_exclusive: int
    formal_test_evaluated: bool


def validate_protocol(protocol: ConfidenceFiLMProtocol) -> None:
    if protocol.activity_calibration_n != ACTIVITY_CALIBRATION_N:
        raise ValueError(f"activity calibration must be fixed at {ACTIVITY_CALIBRATION_N}")
    if protocol.t4_budget not in T4_BUDGETS:
        raise ValueError(f"T4 budget must be one of {T4_BUDGETS}")
    if protocol.evaluation_calibration_n != ACTIVITY_CALIBRATION_N:
        raise ValueError(
            f"evaluation activity calibration must be fixed at {ACTIVITY_CALIBRATION_N}"
        )
    if protocol.common_evaluation_start != COMMON_EVALUATION_START:
        raise ValueError(
            f"common evaluation start must be fixed at {COMMON_EVALUATION_START}"
        )
    if protocol.common_evaluation_start < max(
        protocol.activity_calibration_n, protocol.t4_budget
    ):
        raise ValueError("common evaluation start must exclude every activity/T4 calibration trial")
    if protocol.split_counts != FIXED_SPLIT or protocol.max_units_exclusive != MAX_UNITS_EXCLUSIVE:
        raise ValueError("confidence-FiLM pilot requires strict SUA CO 27/6/6 with units<100")
    if protocol.formal_test_evaluated:
        raise ValueError("confidence-FiLM Stage-0 protocol is train/validation-only")


def make_protocol(t4_budget: int) -> ConfidenceFiLMProtocol:
    protocol = ConfidenceFiLMProtocol(
        activity_calibration_n=ACTIVITY_CALIBRATION_N,
        t4_budget=t4_budget,
        evaluation_calibration_n=ACTIVITY_CALIBRATION_N,
        common_evaluation_start=COMMON_EVALUATION_START,
        split_counts=FIXED_SPLIT,
        max_units_exclusive=MAX_UNITS_EXCLUSIVE,
        formal_test_evaluated=False,
    )
    validate_protocol(protocol)
    return protocol
