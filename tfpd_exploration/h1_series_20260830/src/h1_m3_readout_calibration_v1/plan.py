"""Frozen H1-M3RC candidates and decision law."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA = "h1_m3_readout_calibration_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
FAMILIES = ("DIA7", "MAT7")
RIDGES = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
SCALE_FLOOR = 1.0e-6
MEAN_GAIN_MIN = 0.005
NONNEGATIVE_MIN = 4
WORST_MIN = -0.010
TEMPLATE_POSITIVE_MIN = 3


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 5 or tuple(str(row["outer_date"]) for row in rows) != DATE_ORDER:
        raise ValueError("outer-date order drift")
    gains = [float(row["selected_r2"] - row["base_r2"]) for row in rows]
    neural = [float(row["selected_r2"] - row["template_r2"]) for row in rows]
    mean_gain = sum(gains) / 5.0
    mean_neural = sum(neural) / 5.0
    gain_gate = mean_gain >= MEAN_GAIN_MIN and sum(x >= 0.0 for x in gains) >= NONNEGATIVE_MIN and min(gains) >= WORST_MIN
    neural_gate = mean_neural >= 0.0 and sum(x >= 0.0 for x in neural) >= TEMPLATE_POSITIVE_MIN
    return {
        "gain_by_date": gains,
        "neural_minus_template_by_date": neural,
        "mean_gain": mean_gain,
        "nonnegative_dates": sum(x >= 0.0 for x in gains),
        "worst_gain": min(gains),
        "mean_neural_minus_template": mean_neural,
        "neural_beats_template_dates": sum(x >= 0.0 for x in neural),
        "gain_gate_pass": gain_gate,
        "neural_gate_pass": neural_gate,
        "pass": gain_gate and neural_gate,
        "verdict": "PASS_H1_M3RC_FOR_ALL_SOURCE_PACKAGE" if gain_gate and neural_gate else "STOP_H1_M3RC",
    }


__all__ = ("DATE_ORDER", "FAMILIES", "RIDGES", "SCALE_FLOOR", "SCHEMA", "decide")

