"""CPU streaming predict probe. Continual: on_done is a no-op. No KV cache."""

from __future__ import annotations

import time

import numpy as np
import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank, _H1TemporalBase


class H1TemporalStreamer:
    """W=700 rolling history. Eval-mask gaps do not reset. on_done is no-op."""

    def __init__(self, model: _H1TemporalBase, bank: H1Bank, *, batch_size: int = 1) -> None:
        self.model = model
        self.bank = bank
        self.batch_size = int(batch_size)
        self.window = H1_TEMPORAL.window
        self.n_units = H1_TEMPORAL.n_units
        self.divisor = H1_TEMPORAL.prediction_divisor
        self.buffer = np.zeros((self.window, self.batch_size, self.n_units), dtype=np.float32)
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.history_count.fill(0)

    def on_done(self, dones: np.ndarray | None = None) -> None:
        return None

    def observe(self, neural: np.ndarray) -> None:
        values = np.asarray(neural, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.n_units:
            raise ValueError("observations must be [B,176]")
        active = values.shape[0]
        if active <= 0 or active > self.batch_size:
            raise ValueError("observation batch drift")
        if active < self.batch_size:
            values = np.pad(values, ((0, self.batch_size - active), (0, 0)))
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = values
        self.history_count[:active] += 1

    def predict(self, neural: np.ndarray) -> np.ndarray:
        active = int(np.asarray(neural).shape[0])
        self.observe(neural)
        x = torch.as_tensor(self.buffer.transpose(1, 0, 2), dtype=torch.float32)
        self.model.eval()
        with torch.inference_mode():
            native = self.model.forward_last(x, self.bank)
        out = (native.detach().cpu().numpy().astype(np.float32) / np.float32(self.divisor))[:active]
        if not np.isfinite(out).all():
            raise RuntimeError("nonfinite H1 temporal stream prediction")
        return np.ascontiguousarray(out)


def probe_cpu_stream(
    model: _H1TemporalBase,
    bank: H1Bank,
    *,
    n_windows: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    streamer = H1TemporalStreamer(model, bank, batch_size=1)
    streamer.reset()
    rng = np.random.default_rng(seed)
    warmup = 4
    times: list[float] = []
    last = None
    for step in range(n_windows + warmup):
        neural = rng.standard_normal((1, H1_TEMPORAL.n_units), dtype=np.float32)
        start = time.perf_counter()
        pred = streamer.predict(neural)
        elapsed = time.perf_counter() - start
        if step >= warmup:
            times.append(elapsed)
        last = pred
        if streamer.history_count[0] < H1_TEMPORAL.window:
            # History must accumulate; on_done must not clear it.
            streamer.on_done(np.ones((1,), dtype=bool))
            if streamer.history_count[0] == 0:
                raise RuntimeError("on_done reset the continual history")
    assert last is not None and last.shape == (1, 7)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_windows": float(len(times)),
        "cpu_ms_per_window_mean": float(arr.mean() * 1000.0),
        "cpu_ms_per_window_p50": float(np.median(arr) * 1000.0),
        "cpu_ms_per_window_p95": float(np.quantile(arr, 0.95) * 1000.0),
        "last_pred_abs_max": float(np.abs(last).max()),
        "kv_cache_enabled": 0.0,
    }
