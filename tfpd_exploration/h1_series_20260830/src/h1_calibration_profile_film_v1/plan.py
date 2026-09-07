"""Frozen H1 CP-FiLM constants and source-OOF decision law."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "h1_calibration_profile_film_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARMS = ("EP-ZERO", "EP-FILM", "LP-ZERO", "LP-FILM")
FILM_ARMS = ("EP-FILM", "LP-FILM")
SUPPORT = 3
TRIAL_LENGTH = 1024
UNITS = 176
WINDOW = 700
OUTPUTS = 7
HIDDEN_DIM = 32
CARRIER_DIM = 4
PROFILE_DIM = 4
CONTEXT_DIM = CARRIER_DIM + PROFILE_DIM
FILM_RANK = 8
FILM_PARAMETERS = 648
PROFILE_MASK = (1.0, 1.0, 0.0, 0.0)
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
MIN_STATE_BINS = 16

EPOCHS = 12
BATCH_SIZE = 32
TRAIN_STRIDE = 4
SEED = 42
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.0
PREDICTION_DIVISOR = 20.0

PRIMARY_MEAN = 0.005
PRIMARY_NONNEGATIVE = 4
PRIMARY_WORST = -0.010

EXPECTED_GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
PREDECESSOR_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_support_resampled_postpool_v1"
PREDECESSOR_SCORE_SHA256 = "a07ab80612de53a625b5dc29ef63bc2c24ce249979d5c3afdc43399fb27851b5"
PREDECESSOR_TERMINAL_SHA256 = "c047bbf5bd6e324fcd250ef2702ab507fff54b6a13c6bc1314963983bc298a94"
RESULT_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v1"
DESIGN_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/DESIGN_H1_CALIBRATION_PROFILE_FILM_V1_20260904.md"
WORKORDER_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CALIBRATION_PROFILE_FILM_V1_20260904.md"


def _finite(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(DATE_ORDER) or not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"{name} must contain five finite values")
    return result


def _gate(values: Sequence[float], *, name: str) -> dict[str, Any]:
    numeric = _finite(values, name=name)
    mean = sum(numeric) / len(numeric)
    nonnegative = sum(value >= 0.0 for value in numeric)
    worst = min(numeric)
    return {
        "contrast": name,
        "values": list(numeric),
        "mean": mean,
        "nonnegative_dates": nonnegative,
        "worst": worst,
        "required_mean": PRIMARY_MEAN,
        "required_nonnegative_dates": PRIMARY_NONNEGATIVE,
        "required_worst": PRIMARY_WORST,
        "pass": mean >= PRIMARY_MEAN and nonnegative >= PRIMARY_NONNEGATIVE and worst >= PRIMARY_WORST,
    }


def decide_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row.get("outer_date")) for row in rows) != DATE_ORDER:
        raise ValueError("CP-FiLM outer-date order drift")
    score = {arm: tuple(float(row["scores"][arm]) for row in rows) for arm in ARMS}
    early = tuple(a - b for a, b in zip(score["EP-FILM"], score["EP-ZERO"], strict=True))
    late = tuple(a - b for a, b in zip(score["LP-FILM"], score["LP-ZERO"], strict=True))
    zero_pool = tuple(a - b for a, b in zip(score["LP-ZERO"], score["EP-ZERO"], strict=True))
    film_pool = tuple(a - b for a, b in zip(score["LP-FILM"], score["EP-FILM"], strict=True))
    interaction = tuple(a - b for a, b in zip(late, early, strict=True))
    early_gate = _gate(early, name="EP-FILM_MINUS_EP-ZERO")
    late_gate = _gate(late, name="LP-FILM_MINUS_LP-ZERO")
    if early_gate["pass"] and late_gate["pass"]:
        label = "FILM_POOLING_ROBUST"
    elif early_gate["pass"]:
        label = "FILM_EARLY_REPLICATION_ONLY"
    elif late_gate["pass"]:
        label = "FILM_LATE_SUBSTRATE_ONLY"
    else:
        label = "H1_FILM_NULL_KEEP_LP_R3"
    return {
        "early_film_gate": early_gate,
        "late_film_gate": late_gate,
        "pooling_at_zero_by_date": list(_finite(zero_pool, name="pooling_at_zero")),
        "pooling_with_film_by_date": list(_finite(film_pool, name="pooling_with_film")),
        "interaction_by_date": list(_finite(interaction, name="interaction")),
        "pooling_at_zero_mean": sum(zero_pool) / len(zero_pool),
        "pooling_with_film_mean": sum(film_pool) / len(film_pool),
        "interaction_mean": sum(interaction) / len(interaction),
        "classification": label,
        "m2_to_h1_early_replication": bool(early_gate["pass"]),
        "late_pool_composable": bool(late_gate["pass"]),
        "selected_h1_product": "LP-FILM" if late_gate["pass"] else "LP-R3",
        "pass_any_h1_film": bool(early_gate["pass"] or late_gate["pass"]),
    }


__all__ = ("ARMS", "DATE_ORDER", "FILM_ARMS", "SCHEMA", "decide_oof")

