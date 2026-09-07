"""Shared V4-aligned constants and uncertainty helpers for this scaffold.

These are intentionally local copies so the package stays reviewable without
importing heavy aggregators. Behaviour matches
``aggregate_side_feature_ablation_v2.sigma_delta_paired`` for the paired SE.
"""
from __future__ import annotations

import math
from typing import Sequence

EXPECTED_EPOCH_WINDOW = list(range(5, 13))
EXPECTED_TOTAL_EPOCHS = 12
# Intentionally unset: handoff forbids writing GPU thresholds before P0N freezes them.
PRACTICAL_EFFECT_FLOOR: float | None = None


def sample_std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError(f"sample_std requires n>=2, got {n}")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def sigma_delta_paired(per_seed_deltas: Sequence[float]) -> float:
    """SE of the seed-mean paired delta; requires >=2 seeds."""
    n = len(per_seed_deltas)
    if n < 2:
        raise ValueError(
            "sigma_delta_paired requires at least 2 seeds "
            f"(refusing silent fallback), got {n}"
        )
    return sample_std(per_seed_deltas) / math.sqrt(n)


def two_sigma_paired(per_seed_deltas: Sequence[float]) -> float:
    return 2.0 * sigma_delta_paired(per_seed_deltas)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)
