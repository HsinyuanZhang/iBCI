"""BCIDecoder: predict([1,85,27000]) -> [158,880], dotted/undotted tags, six M3 payloads."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from .constants import (
    ALL_DATES,
    CARRIER_DIM,
    N_CHANNELS,
    N_FREQ,
    N_MS_BINS,
    N_NEURAL_SAMPLES,
    N_SPEC_FRAMES,
)
from .data import bin_tx_counts
from .memory import GrowingMemory
from .model import B1SpintSFCJ
from .sfc import zero9


class TagError(ValueError):
    pass


def assert_predict_shape(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 2:
        raise ValueError(f"predict 3-D return is forbidden; got ndim={out.ndim} shape={out.shape}")
    if out.shape != (N_FREQ, N_SPEC_FRAMES):
        raise ValueError(f"predict must return [158,880], got {out.shape}")
    return out


def resolve_dataset_tag(tag) -> str:
    if tag is None:
        raise TagError("empty dataset tag")
    if isinstance(tag, Path):
        text = tag.name
    else:
        text = str(tag)
    dotted = re.search(r"2021\.(\d{2})\.(\d{2})", text)
    if dotted:
        undotted = f"2021{dotted.group(1)}{dotted.group(2)}"
        if undotted in ALL_DATES:
            return undotted
        raise TagError(f"unknown dotted tag {text}")
    undotted = re.search(r"2021\d{4}", text.replace("-", ""))
    if undotted and undotted.group(0) in ALL_DATES:
        return undotted.group(0)
    # hash_dataset style already undotted
    if text in ALL_DATES:
        return text
    raise TagError(f"unknown dataset tag {tag!r}")


class B1SFCJDecoder(BCIDecoder):
    def __init__(
        self,
        model: B1SpintSFCJ,
        payloads: dict,
        *,
        law: str = "GROWING",
        memory_mode: str = "running_sum",
        task_config: Optional[FalconConfig] = None,
        batch_size: int = 1,
    ):
        super().__init__(task_config or FalconConfig(FalconTask.b1), batch_size=batch_size)
        if batch_size != 1:
            raise ValueError("B1 decoder batch size must be 1")
        self.model = model
        self.model.eval()
        self.payloads = payloads
        self.law = law
        self.memory_mode = memory_mode
        self.memory = GrowingMemory(model, law=law, mode=memory_mode, dtype=next(model.parameters()).dtype)
        self.active_date: Optional[str] = None
        self.query_label_access_count = 0
        self._last_digest = None

    def reset(self, dataset_tags: List = [""]):
        resolved = []
        for tag in dataset_tags:
            resolved.append(resolve_dataset_tag(tag))
        if len(set(resolved)) != 1:
            raise TagError(f"duplicate/conflicting tags {dataset_tags} -> {resolved}")
        date = resolved[0]
        if date not in self.payloads:
            raise TagError(f"no M3 payload for {date}")
        self.memory.reset_state()
        payload = self.payloads[date]
        carrier = np.asarray(payload["carrier"], dtype=np.float64)
        if carrier.shape != (N_CHANNELS, CARRIER_DIM):
            raise TagError(f"carrier shape {carrier.shape}")
        activities = payload["activity"]
        self.memory.seed_m3(activities, carrier)
        self.active_date = date
        self._last_digest = self.memory.pool_digest()
        return date

    def _current_to_counts(self, neural_observations: np.ndarray) -> np.ndarray:
        arr = np.asarray(neural_observations)
        if arr.shape != (1, N_CHANNELS, N_NEURAL_SAMPLES):
            raise ValueError(f"predict expects [1,85,27000], got {arr.shape}")
        tx = np.transpose(arr[0], (1, 0))
        return bin_tx_counts(tx).astype(np.float64)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self.active_date is None or self.memory.K == 0:
            raise RuntimeError("decoder.reset must load M3 before predict")
        counts = self._current_to_counts(neural_observations)
        digest_before = self.memory.pool_digest()
        ident, _ = self.memory.identity(with_grad=False)
        x = torch.as_tensor(counts, dtype=ident.dtype, device=ident.device).unsqueeze(0)
        src = x.permute(0, 2, 1) + ident.unsqueeze(0)
        tokens = self.model.fc_in(src)
        query = self.model.freq_queries.expand(1, -1, -1)
        hidden = query
        pad = torch.zeros(1, N_CHANNELS, dtype=torch.bool, device=src.device)
        for layer in self.model.transformer:
            hidden = layer(hidden, tokens, key_padding_mask=pad)
        std_log = self.model.head(hidden)
        log_spec = std_log * self.model.log_std + self.model.log_mean
        raw = torch.exp(log_spec)[0].detach().cpu().numpy()
        raw = assert_predict_shape(raw)
        self.memory.commit(counts, member_id=f"q:{self.memory.K}")
        self._last_digest = self.memory.pool_digest()
        if self.law == "GROWING" and digest_before == self._last_digest:
            raise RuntimeError("commit did not change pool digest")
        return np.asarray(raw, dtype=np.float64)

    def on_done(self, dones: np.ndarray):
        # B1 evaluator does not call on_done; keep a no-op that does not commit.
        return None


def dummy_payloads(dates=ALL_DATES) -> dict:
    out = {}
    for date in dates:
        out[date] = {
            "activity": [np.zeros((N_MS_BINS, N_CHANNELS), dtype=np.float64) for _ in range(3)],
            "carrier": zero9(),
        }
    return out
