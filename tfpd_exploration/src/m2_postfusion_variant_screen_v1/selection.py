"""Pure, explicitly in-sample source-monitor selection laws."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import plan


class SelectionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


def validate_source_heldin_in_sample_monitor(
        train_groups: Sequence[str], monitor_groups: Sequence[str],
        train_window_indices: Sequence[object], monitor_window_indices: Sequence[object],
) -> dict[str, object]:
    """Bind the admitted inherited monitor honestly, rather than calling it OOF.

    The sealed PIT monitor is a fixed subset of its source-training windows.
    This exact geometry is useful for reproducibility and endpoint selection,
    but is never a source-generalization claim.
    """
    train = tuple(sorted(str(value) for value in train_groups))
    monitor = tuple(sorted(str(value) for value in monitor_groups))
    _require(bool(train) and bool(monitor), "source train/monitor groups must both be nonempty")
    _require(len(set(train)) == len(train) and len(set(monitor)) == len(monitor),
             "source group roster contains duplicates")
    _require(train == monitor and len(train) == plan.SOURCE_MONITOR_SESSION_COUNT,
             "inherited source monitor roster must exactly equal the seven-session train roster")

    def _coordinates(values: Sequence[object], label: str) -> tuple[tuple[str, int], ...]:
        output: list[tuple[str, int]] = []
        for value in values:
            _require(isinstance(value, (tuple, list)) and len(value) == 2,
                     f"{label} window coordinate shape drift")
            output.append((str(value[0]), int(value[1])))
        _require(len(set(output)) == len(output), f"{label} window coordinates contain duplicates")
        return tuple(output)

    train_windows = _coordinates(train_window_indices, "source train")
    monitor_windows = _coordinates(monitor_window_indices, "source monitor")
    _require(len(train_windows) == plan.SOURCE_TRAIN_WINDOW_COUNT,
             "source train window count drift")
    _require(len(monitor_windows) == plan.SOURCE_MONITOR_WINDOW_COUNT,
             "source monitor window count drift")
    _require(set(monitor_windows).issubset(set(train_windows)),
             "source monitor must be an exact subset of source-training coordinates")
    return {
        "train_groups": list(train), "monitor_groups": list(monitor),
        "monitor_name": "source_heldin_in_sample_monitor",
        "monitor_is_in_sample": True,
        "monitor_window_count": len(monitor_windows), "train_window_count": len(train_windows),
        "monitor_windows_contained_in_train": len(monitor_windows),
    }


def equal_group_mean(per_group_r2: Mapping[str, float]) -> float:
    _require(bool(per_group_r2), "source grouped metric is empty")
    values = np.asarray([float(per_group_r2[name]) for name in sorted(per_group_r2)], dtype=np.float64)
    _require(np.isfinite(values).all(), "source grouped metric is nonfinite")
    return float(values.mean())


@dataclass(frozen=True)
class ArmSelection:
    arm: str
    best_source_metric: float
    best_observed_epoch: int
    added_parameters: int


def choose_screen_winner(curves: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    """Highest source metric through the common 12 epochs, then simple tie rule."""
    _require(set(curves) == set(plan.ARMS), "screen winner needs exactly PF-MEAN/PF-R1/PF-R50")
    candidates: list[ArmSelection] = []
    for arm in plan.ARMS:
        rows = list(curves[arm])
        _require(rows and {int(row["epoch"]) for row in rows} == set(range(1, plan.SOURCE_EPOCHS_SCREEN + 1)),
                 "screen curve has invalid epoch topology")
        ranked = sorted(((float(row["source_heldin_in_sample_monitor_mean"]), int(row["epoch"])) for row in rows),
                        key=lambda item: (-item[0], item[1]))
        value, epoch = ranked[0]
        candidates.append(ArmSelection(arm, value, epoch, plan.ARM_ADDED_PARAMETERS[arm]))
    max_value = max(item.best_source_metric for item in candidates)
    near = [item for item in candidates if max_value - item.best_source_metric <= plan.SOURCE_TIE_EPSILON]
    winner = sorted(near, key=lambda item: (item.added_parameters, plan.ARM_TIEBREAK_ORDER.index(item.arm)))[0]
    return {
        "winner": winner.arm,
        "winner_best_source_monitor_through_epoch12": winner.best_source_metric,
        "winner_best_observed_epoch": winner.best_observed_epoch,
        "selection_metric": "source_heldin_in_sample_monitor_equal_session_mean",
        "selection_is_descriptive_only": True,
        "tie_epsilon": plan.SOURCE_TIE_EPSILON,
        "tie_break": "fewest_added_parameters_then_PF-MEAN_PF-R1_PF-R50",
        "candidates": [
            {"arm": item.arm, "best_source_monitor_through_epoch12": item.best_source_metric,
             "best_observed_epoch": item.best_observed_epoch, "added_parameters": item.added_parameters}
            for item in candidates
        ],
    }


def choose_screen_checkpoint(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Best source grouped metric over the fixed 12 epochs; earlier breaks ties."""
    _require(bool(rows) and all(1 <= int(row["epoch"]) <= plan.SOURCE_EPOCHS_SCREEN for row in rows),
             "screen checkpoint needs valid source epoch rows")
    ranking = sorted(((float(row["source_heldin_in_sample_monitor_mean"]), int(row["epoch"]), row)
                      for row in rows), key=lambda item: (-item[0], item[1]))
    value, epoch, row = ranking[0]
    return {"epoch": epoch, "source_heldin_in_sample_monitor_mean": value,
            "tie_break": "higher_source_heldin_in_sample_monitor_then_earlier_epoch",
            "row": dict(row)}


__all__ = ("SelectionError", "validate_source_heldin_in_sample_monitor", "equal_group_mean", "choose_screen_winner", "choose_screen_checkpoint")
