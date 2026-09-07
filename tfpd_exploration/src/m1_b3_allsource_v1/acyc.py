"""Train-time calib-neural prefix cycle. Eval forwards stay M10.

Carrier side features are not sliced. Only student ``calib_trials`` shrinks
during training: M cycles through (10, 5, 2).
"""
from __future__ import annotations

from typing import Any, Sequence

import lightning.pytorch as pl

from . import plan


class AcycError(RuntimeError):
    """Fail closed for the all-source activity-prefix operator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcycError(message)


class ActivityPrefixOperator:
    """Slice ``calib_trials[:, :M]`` on training forwards only."""

    def __init__(
        self,
        cycle: Sequence[int] = plan.ACTIVITY_CYCLE,
        pool_trials: int = plan.CALIBRATION_N_TRIALS,
    ) -> None:
        values = tuple(int(item) for item in cycle)
        _require(values == plan.ACTIVITY_CYCLE, f"activity cycle drift {values}")
        _require(int(pool_trials) == plan.CALIBRATION_N_TRIALS, "prefix pool must be M10")
        _require(all(1 <= item <= int(pool_trials) for item in values), "cycle outside pool")
        self.cycle = values
        self.pool_trials = int(pool_trials)
        self.training_invocations = 0
        self.eval_invocations = 0
        self._handle: Any = None

    def m_for_index(self, index: int) -> int:
        _require(type(index) is int and index >= 0, "prefix index")
        return int(self.cycle[index % len(self.cycle)])

    def forward_pre_hook(self, module, args, kwargs):
        if not module.training:
            self.eval_invocations += 1
            return None
        scheduled_m = self.m_for_index(self.training_invocations)
        self.training_invocations += 1
        calib = kwargs.get("calib_trials")
        if calib is None:
            return None
        available = int(calib.shape[1])
        _require(available == self.pool_trials, f"training calib must be M10, saw {available}")
        kwargs = dict(kwargs)
        kwargs["calib_trials"] = calib[:, :scheduled_m]
        return args, kwargs

    def attach(self, model) -> Any:
        _require(self._handle is None, "operator already attached")
        self._handle = model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        return self._handle


class ActivityPrefixCallback(pl.Callback):
    """Attach the prefix operator to the student at fit start."""

    def __init__(
        self,
        cycle: Sequence[int] = plan.ACTIVITY_CYCLE,
        pool_trials: int = plan.CALIBRATION_N_TRIALS,
    ) -> None:
        super().__init__()
        self.operator = ActivityPrefixOperator(cycle=cycle, pool_trials=pool_trials)

    def on_fit_start(self, trainer, pl_module) -> None:
        del trainer
        self.operator.attach(pl_module.student)
