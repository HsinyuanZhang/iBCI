"""Fold0 source-only windows via M1-TEMPORAL-v2 / rSyn3-refit-v1. Outer unread."""

from __future__ import annotations

from tfpd_exploration.src.m1_temporal_v2.data import (
    build_source_only_datamodule,
    materialize_source_banks,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

from .config import TRAIN_SESSIONS


def build_fold0_train_datamodule():
    return build_source_only_datamodule()


def materialize_source_banks_for_train() -> dict[str, M1Bank]:
    banks = materialize_source_banks()
    missing = [name for name in TRAIN_SESSIONS if name not in banks]
    if missing:
        raise RuntimeError(f"missing source banks {missing}")
    return banks
