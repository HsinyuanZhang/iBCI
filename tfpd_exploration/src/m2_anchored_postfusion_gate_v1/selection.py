"""Pure lexical 5/2 source selection and exact-zero refit law."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import math

from . import plan


class SelectionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


def lexical_source_split(sessions: Sequence[str]) -> dict[str, tuple[str, ...]]:
    roster = tuple(sorted(str(value) for value in sessions))
    _require(len(roster) == plan.SOURCE_SESSION_COUNT and len(set(roster)) == len(roster),
             "APFG requires exactly seven unique source sessions")
    return {"fit": roster[:plan.FIT_SESSION_COUNT], "validation": roster[plan.FIT_SESSION_COUNT:]}


@dataclass(frozen=True)
class EpochValidation:
    epoch: int
    per_session_r2: Mapping[str, float]
    zero_gate_per_session_r2: Mapping[str, float]

    def equal_session_mean(self) -> float:
        values = tuple(float(self.per_session_r2[name]) for name in sorted(self.per_session_r2))
        _require(len(values) == plan.VALIDATION_SESSION_COUNT and all(math.isfinite(value) for value in values),
                 "APFG validation R2 is not a finite two-session mapping")
        return sum(values) / len(values)

    def zero_gate_mean(self) -> float:
        values = tuple(float(self.zero_gate_per_session_r2[name]) for name in sorted(self.zero_gate_per_session_r2))
        _require(set(self.per_session_r2) == set(self.zero_gate_per_session_r2),
                 "APFG validation/zero session roster drift")
        _require(all(math.isfinite(value) for value in values), "APFG zero-gate R2 is nonfinite")
        return sum(values) / len(values)


def select_earliest_best(rows: Sequence[EpochValidation], validation_roster: Sequence[str]) -> dict[str, object]:
    expected_roster = tuple(sorted(str(value) for value in validation_roster))
    _require(len(expected_roster) == plan.VALIDATION_SESSION_COUNT, "APFG validation roster size drift")
    _require({int(row.epoch) for row in rows} == set(range(1, plan.EPOCHS + 1)),
             "APFG selection requires every epoch 1..12 exactly once")
    for row in rows:
        _require(tuple(sorted(row.per_session_r2)) == expected_roster, "APFG epoch validation roster drift")
    ranked = sorted(((row.equal_session_mean(), int(row.epoch), row) for row in rows), key=lambda value: (-value[0], value[1]))
    value, epoch, row = ranked[0]
    delta = value - row.zero_gate_mean()
    session_deltas = {name: float(row.per_session_r2[name]) - float(row.zero_gate_per_session_r2[name])
                      for name in expected_roster}
    gate = delta >= -0.002 and any(item >= 0.0 for item in session_deltas.values())
    return {"selected_epoch": epoch, "equal_session_validation_r2": value,
            "zero_gate_equal_session_validation_r2": row.zero_gate_mean(), "delta_vs_zero": delta,
            "per_session_delta_vs_zero": session_deltas, "source_safety_gate_passed": gate,
            "tie_break": "earliest_epoch_at_maximum_equal_session_validation_r2",
            "refit": {"sessions": "all_seven_source_sessions", "alpha_initialization": "+0.0",
                      "epochs": epoch, "selection_metric_not_used_to_extend_refit": True}}


__all__ = ("SelectionError", "EpochValidation", "lexical_source_split", "select_earliest_best")
