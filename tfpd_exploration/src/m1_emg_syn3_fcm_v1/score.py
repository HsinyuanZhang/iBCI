"""Paired Static / CDM-A scoring contracts. No target BP."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import plan


class ScoreError(RuntimeError):
    """Fail closed for CDM-A chronology."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


@dataclass
class FIFOState:
    support: list[np.ndarray]
    completed: list[np.ndarray] = field(default_factory=list)
    next_legal_trial: int = 0
    support_trials: int = plan.SUPPORT_TRIALS


class ActivityFIFO:
    def __init__(self, support_trials: int = plan.SUPPORT_TRIALS) -> None:
        _require(int(support_trials) >= 1, "support")
        self.support_trials = int(support_trials)

    def initialize(self, support_trials_activity: list[np.ndarray]) -> FIFOState:
        _require(len(support_trials_activity) == self.support_trials, "support cardinality")
        return FIFOState(
            support=[np.asarray(item, dtype=np.float64).copy() for item in support_trials_activity],
            completed=[],
            next_legal_trial=self.support_trials,
            support_trials=self.support_trials,
        )

    def identity_for_trial(self, state: FIFOState, *, trial_index: int) -> np.ndarray:
        _require(trial_index >= 0, "trial")
        if trial_index < state.support_trials:
            selected = state.support
        else:
            _require(trial_index >= state.next_legal_trial, "current/future CDM-A identity")
            _require(trial_index == state.next_legal_trial, "CDM-A skips a query trial")
            selected = (state.support + state.completed)[-state.support_trials:]
        stacked = np.concatenate([item.reshape(1, -1) if item.ndim == 1 else item.reshape(item.shape[0], -1)
                                  for item in selected], axis=0)
        return stacked.mean(axis=0)

    def commit_completed(
        self,
        state: FIFOState,
        *,
        trial_activity: np.ndarray,
        carrier: np.ndarray,
        model_digest: str,
    ) -> tuple[FIFOState, np.ndarray, str]:
        _require(state.next_legal_trial >= state.support_trials, "commit before query")
        completed = list(state.completed) + [np.asarray(trial_activity, dtype=np.float64).copy()]
        updated = FIFOState(
            support=state.support,
            completed=completed,
            next_legal_trial=state.next_legal_trial + 1,
            support_trials=state.support_trials,
        )
        return updated, np.asarray(carrier).copy(), str(model_digest)
