"""Thin, fail-closed wrapper over ``tfpd_lane.arm_common`` for this 50-epoch lane.

The warmup-then-cosine math is imported, not copied.  ``tfpd_lane/__init__.py``
does ``from src.tfpd_lane import ...`` and cannot be imported as
``tfpd_exploration.src.tfpd_lane.arm_common`` (frozen package; cannot edit).
This module therefore loads ``arm_common.py`` by file location so the
function bodies executed are exactly those bytes.

Importing that file pulls Torch and numpy at module level; callers that must
stay torch-free must not import this module (the plan and the public CLI
do not).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from tfpd_exploration.src.m1_t0c1_prefix_v1 import schedule as prefix_schedule

from . import plan


class LRScheduleError(RuntimeError):
    """Fail closed for the 50-epoch warmup-then-cosine law."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LRScheduleError(message)


def _load_arm_common():
    """Load ``arm_common.py`` without executing ``tfpd_lane/__init__.py``."""
    path = Path(__file__).resolve().parents[1] / "tfpd_lane" / "arm_common.py"
    _require(path.is_file(), f"arm_common.py missing at {path}")
    spec = importlib.util.spec_from_file_location(
        "m1_t0c1_prefix_v1_50ep._arm_common", path,
    )
    _require(spec is not None and spec.loader is not None, "arm_common importlib spec drift")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(callable(getattr(module, "lr_at_step", None))
             and callable(getattr(module, "warmup_steps", None))
             and callable(getattr(module, "schedule_params", None)),
             "arm_common is missing lr_at_step/warmup_steps/schedule_params")
    return module


arm_common = _load_arm_common()


def lr_at(global_step: int) -> float:
    """LR at a 0-based optimizer step over the whole 50-epoch horizon.

    Independent of arm.  Fail-closed on a non-int or an out-of-range step.
    """
    _require(type(global_step) is int, "lr_at global_step must be a Python int")
    try:
        return float(arm_common.lr_at_step(global_step, plan.EPOCHS, plan.STEPS_PER_EPOCH))
    except ValueError as error:
        raise LRScheduleError(f"lr_at out of range or illegal: {error}") from error


def schedule_law() -> dict[str, object]:
    """Runtime pin of ``arm_common.schedule_params(50, 4951)`` plus ``law_sha256``."""
    params = dict(arm_common.schedule_params(plan.EPOCHS, plan.STEPS_PER_EPOCH))
    _require(params.get("kind") == plan.LR_SCHEDULE_KIND, "arm_common schedule kind drifted")
    _require(params.get("n_epochs") == plan.EPOCHS, "arm_common n_epochs drifted")
    _require(params.get("steps_per_epoch") == plan.STEPS_PER_EPOCH, "arm_common steps_per_epoch drifted")
    _require(params.get("total_steps") == plan.TOTAL_OPTIMIZER_STEPS, "arm_common total_steps drifted")
    _require(params.get("warmup_epochs") == plan.WARMUP_EPOCHS, "arm_common warmup_epochs drifted")
    _require(params.get("warmup_steps") == plan.WARMUP_STEPS, "arm_common warmup_steps drifted")
    _require(params.get("lr_warmup_start") == plan.LR_WARMUP_START, "arm_common lr_warmup_start drifted")
    _require(params.get("lr_warmup_end") == plan.LR_WARMUP_END, "arm_common lr_warmup_end drifted")
    _require(params.get("lr_final") == plan.LR_FINAL, "arm_common lr_final drifted")
    _require(params.get("phase_local_steps") is True, "arm_common phase_local_steps drifted")
    body = {**params, "kind": plan.LR_SCHEDULE_KIND}
    return {**body, "law_sha256": plan.sha256_bytes(plan.canonical_json_bytes(body))}


def sequence_digest(values: Sequence[object]) -> str:
    """Digest a per-step LR stream.

    The predecessor ``schedule.sequence_digest`` joins on ``str(item)``.
    ``str(float)`` is not our chosen canonical form; we map each value to
    ``repr(float(x))`` first so the digest is IEEE-round-trip stable for
    CPython binary64, then reuse the predecessor joiner (which will
    ``str()`` those already-canonical strings).
    """
    canonical = [repr(float(item)) for item in values]
    return prefix_schedule.sequence_digest(canonical)


def apply(optimizer: Any, global_step: int) -> float:
    """Set every param group's ``lr`` to ``lr_at(global_step)`` and return it."""
    value = lr_at(global_step)
    groups = getattr(optimizer, "param_groups", None)
    _require(isinstance(groups, list) and groups, "optimizer has no param_groups")
    for group in groups:
        _require(isinstance(group, dict), "optimizer param group is not a dict")
        group["lr"] = value
    return value


__all__ = (
    "LRScheduleError", "lr_at", "schedule_law", "sequence_digest", "apply",
)
