"""Exact zero-history reset for the isolated M2 linear-conv runtime.

Before temporal positional encoding, every frontend token of an all-zero
raw history is identical within a session: causal Conv1d sees only zeros
at every age, including the left boundary. Compute that token once per
session and materialize its W50 copies. Reset returns no prediction, so
the temporal forward whose output the base reset discards is unnecessary.
Nonzero-history/mutation rebuild paths remain the inherited full paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Error, WINDOW
from .m2_linear_conv import LinearConvFiveTokenM2Decoder, _LinearConvFiveTokenExactE


class ColdStartLinearConvM2Decoder(LinearConvFiveTokenM2Decoder):
    @torch.no_grad()
    def reset(self, dataset_tags) -> None:
        self._require_eval()
        reference = next(self.model.parameters())
        if reference.device.type != "cpu" or reference.dtype != torch.float32:
            raise RuntimeV3Error("zero-history reset requires CPU float32 model state")
        tags = [Path(tag).stem for tag in dataset_tags]
        if not tags or len(tags) > self.batch_size:
            raise RuntimeV3Error("active dataset count must be 1..configured batch")
        missing = [tag for tag in tags if tag not in self._banks]
        if missing:
            raise RuntimeV3Error(f"unknown frozen bank tag(s): {missing}")
        rows = [self._banks[tag] for tag in tags]
        channels = int(np.asarray(rows[0]["E0"]).shape[0])
        if any(np.asarray(row["E0"]).shape != (channels, 50)
               or np.asarray(row["T"]).shape != (channels, 4) for row in rows):
            raise RuntimeV3Error("heterogeneous unit roster is not vectorizable")
        masks = [np.asarray(row["unit_mask"], dtype=np.bool_) for row in rows]
        if any(mask.shape != (channels,) or not mask.any() for mask in masks):
            raise RuntimeV3Error("unit-mask roster drift or empty unit row")
        bank = self._ref.SessionBank(
            E0=torch.stack([torch.as_tensor(row["E0"], dtype=torch.float32) for row in rows]),
            T=torch.stack([torch.as_tensor(row["T"], dtype=torch.float32) for row in rows]),
            unit_mask=torch.stack([torch.as_tensor(mask, dtype=torch.bool) for mask in masks]),
        )
        engine = _LinearConvFiveTokenExactE(self.model, bank)
        raw = torch.zeros((len(rows), WINDOW, channels), dtype=torch.float32)
        zero_token = engine._frontend(raw[:, :1])
        if not bool(torch.isfinite(zero_token).all()):
            raise RuntimeV3Error("non-finite zero-history frontend")
        engine.raw = raw
        engine.frontend = zero_token.expand(-1, WINDOW, -1).clone()
        self._engine = engine
        self._active, self._channels = len(rows), channels
