"""Local Falcon lifecycle adapter for the exact M2 cold-start runtime.

This is a repository-backed candidate, not a self-contained submission or a
change to an existing image. No query labels or calibration fitting are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.interface import BCIDecoder

from .m2_cold_start import ColdStartLinearConvM2Decoder
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Error, OFFICIAL_BATCH


class FamilyM2FalconDecoder(BCIDecoder):
    """Preserve Falcon tag, padded-batch, observe, and continual M2 semantics."""

    def __init__(self, task_config, model_path: str | Path, batch_size: int = 7):
        if task_config.task != FalconTask.m2 or task_config.n_channels != 96:
            raise RuntimeV3Error("M2 adapter requires Falcon m2 with 96 channels")
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.runtime = ColdStartLinearConvM2Decoder(model_path, batch_size=batch_size)
        payload = self.runtime._ref.load_payload(model_path)
        if payload.get("task") != task_config.task:
            raise RuntimeV3Error("M2 payload task does not match evaluator task")
        self.batch_size = self.runtime.batch_size
        self._active = 0

    def reset(self, dataset_tags: Iterable[str | Path]) -> None:
        tags = [self.task_config.hash_dataset(Path(tag).stem) for tag in dataset_tags]
        # Runtime reset is transactional: validate the entire new roster before
        # publishing it, leaving the previous stream usable on a rejected reset.
        self.runtime.reset(tags)
        self._active = len(tags)

    def _observations(self, observations: np.ndarray) -> np.ndarray:
        if not self._active:
            raise RuntimeV3Error("reset must precede M2 observations")
        value = np.asarray(observations)
        if (value.dtype.kind not in "biuf" or value.ndim != 2 or value.shape[1] != 96
                or value.shape[0] not in (self._active, self.batch_size)):
            raise RuntimeV3Error("expected real numeric [active or configured batch,96]")
        if not np.isfinite(value).all():
            raise RuntimeV3Error("M2 observations must be finite")
        if value.shape[0] > self._active and np.any(value[self._active:] != 0):
            raise RuntimeV3Error("inactive evaluator padding must be zero")
        # Falcon NWB spike binning produces uint8 counts. Match the frozen
        # adapter's conversion; the underlying runtime stays strict float32.
        converted = np.ascontiguousarray(value[:self._active], dtype=np.float32)
        if not np.isfinite(converted).all():
            raise RuntimeV3Error("M2 observations overflow float32")
        return converted

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        return self.runtime.predict(self._observations(neural_observations))

    def observe(self, neural_observations: np.ndarray) -> None:
        self.runtime.observe(self._observations(neural_observations))

    def on_done(self, dones: np.ndarray) -> None:
        value = np.asarray(dones)
        if not self._active or value.ndim != 1 or value.shape[0] not in (self._active, self.batch_size):
            raise RuntimeV3Error("M2 done flags must match the active or padded roster")
        # M2 is continual: trial flags never clear its W50 history.
        self.runtime.on_done(value[:self._active])

    def set_batch_size(self, batch_size: int) -> None:
        size = int(batch_size)
        if size != batch_size or not 1 <= size <= OFFICIAL_BATCH:
            raise RuntimeV3Error("M2 batch size must be an integer in 1..7")
        self.runtime.batch_size = size
        self.runtime._engine = None
        self.runtime._active = 0
        self.runtime._channels = None
        self.batch_size = size
        self._active = 0
