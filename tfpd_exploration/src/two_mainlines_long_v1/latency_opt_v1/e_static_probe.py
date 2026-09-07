"""E-static parity + 128-call H1 FLAT compare. One formal CPU bench."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import (
    _index_split,
    load_session_arrays,
    materialize_banks,
)
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.streaming import H1TemporalStreamer
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1TemporalFlatDecoder
from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v1.e0_baseline import RESULT, WARMUP, MICRO, _stats
from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v1.e_static import compile_static, forward_with_static, parity_max_abs

REPO = Path("/home/xinyuan/Work_host/SPINT")


def _load_flat():
    ckpt = torch.load(
        REPO / "tfpd_exploration/results/h1_temporal_decoder_quick_product_v1/flat/epoch_005.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = H1TemporalFlatDecoder(seed=42)
    model.load_state_dict(ckpt["ema"]["shadow"], strict=True)
    model.eval()
    return model


def main() -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    session = HELDIN_SESSIONS[0]
    path = _index_split("held-in-minival")[session]
    arrays = load_session_arrays(path, session, skip_first3=False)
    bank = materialize_banks({session: arrays})[session]
    model = _load_flat()
    local = torch.randn(1, 32, 176, 16)
    parity = parity_max_abs(model.frontend.token_mlp, local, bank)
    cache = compile_static(model.frontend.token_mlp, bank)
    original = model.frontend.token_mlp.forward

    def cached_forward(local_t, e0, hc):
        return forward_with_static(model.frontend.token_mlp, local_t, cache)

    rows = np.ascontiguousarray(arrays.neural[: WARMUP + MICRO], dtype=np.float32)

    def time_model() -> list[float]:
        streamer = H1TemporalStreamer(model, bank, batch_size=1)
        streamer.reset()
        times = []
        for i, row in enumerate(rows):
            start = time.perf_counter()
            pred = streamer.predict(row.reshape(1, -1))
            elapsed = time.perf_counter() - start
            if not np.isfinite(pred).all():
                raise RuntimeError("nonfinite")
            if i >= WARMUP:
                times.append(elapsed)
        return times

    base = time_model()
    model.frontend.token_mlp.forward = cached_forward  # type: ignore[method-assign]
    cached = time_model()
    model.frontend.token_mlp.forward = original  # type: ignore[method-assign]
    report = {
        "schema": "latency_opt_v1_e_static_h1_flat",
        "status": "E_STATIC_MICROPROBE",
        "updated": datetime.now(timezone.utc).isoformat(),
        "parity_max_abs": parity,
        "baseline": _stats(base),
        "e_static": _stats(cached),
        "p95_ratio": float(np.percentile(cached, 95) / np.percentile(base, 95)),
        "note": "Not a product claim. Cache is inference-only. No 2048x3.",
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    dest = RESULT / "e_static_h1_flat.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if parity > 1.0e-5:
        raise RuntimeError(f"E-static parity failed: {parity}")
    return report


if __name__ == "__main__":
    main()
