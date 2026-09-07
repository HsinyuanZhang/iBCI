"""Training-gated activity-prefix operator. Carrier bytes stay constant."""
from __future__ import annotations

from typing import Any

from . import plan


class PrefixError(RuntimeError):
    """Fail closed for S-Acyc prefix cycling."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrefixError(message)


class ActivityPrefixOperator:
    """Slice calib_trials[:, :M] during training. Eval forwards are untouched."""

    def __init__(self, arm: str) -> None:
        _require(arm in plan.STAGE1_ARMS, f"unknown Stage-1 arm {arm!r}")
        self.arm = arm
        self.cycle = tuple(plan.ACTIVITY_CYCLE)
        self.training_invocations = 0
        self.eval_invocations = 0
        self._handle: Any = None

    def m_for_index(self, index: int) -> int:
        _require(type(index) is int and index >= 0, "prefix index")
        if self.arm != "S-Acyc":
            return int(plan.SUPPORT_TRIALS)
        return int(self.cycle[index % len(self.cycle)])

    def forward_pre_hook(self, module, args, kwargs):
        if not module.training:
            self.eval_invocations += 1
            return None
        index = self.training_invocations
        scheduled_m = self.m_for_index(index)
        self.training_invocations += 1
        calib = kwargs.get("calib_trials")
        if calib is None:
            return None
        available = int(calib.shape[1])
        _require(available == plan.SUPPORT_TRIALS, f"training calib must be M10, saw {available}")
        if self.arm != "S-Acyc":
            return None
        kwargs = dict(kwargs)
        kwargs["calib_trials"] = calib[:, :scheduled_m]
        return args, kwargs

    def attach(self, model) -> Any:
        _require(self._handle is None, "operator already attached")
        self._handle = model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        return self._handle

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
