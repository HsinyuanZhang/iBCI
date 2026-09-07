"""CDM x P1 cross-dataset factorial, Part A (V1) -- additive package.

Work order: ``docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`` section 2.
See :mod:`.plan` for the frozen contract and :mod:`.replay` for the driver.
"""

from . import gates, plan, replay, weights

__all__ = ["gates", "plan", "replay", "weights"]
