"""Stage-1 training contracts. GPU launch is refused without a capability."""
from __future__ import annotations

from . import plan


class TrainingError(RuntimeError):
    """Fail closed for Stage-1 selection/capability."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def select_fixed_last(epoch_index: int, query_r2: float | None = None) -> str:
    if query_r2 is not None:
        raise TrainingError("fixed-last selection cannot read target query")
    _require(int(epoch_index) == plan.FIXED_LAST_EPOCH_INDEX, "fixed-last epoch drift")
    return "epoch_011"


def refuse_gpu_without_capability(capability: object | None) -> None:
    if capability is None:
        raise TrainingError("Stage-1 GPU requires a separately issued capability")
    raise TrainingError("unknown GPU capability object")
