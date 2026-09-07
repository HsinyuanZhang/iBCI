"""E0 same-machine baseline. No speedup claim. New root only."""

from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path("/home/xinyuan/Work_host/SPINT")
STAMP = "20260905_181900"
RESULT = REPO / "tfpd_exploration/results/latency_opt_v1" / STAMP
MICRO = 128
WARMUP = 8
LONG_CALLS = 2048
LONG_REPEATS = 3


def _env() -> dict[str, Any]:
    cpu = ""
    try:
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        model = next(line.split(":", 1)[1].strip() for line in cpu.splitlines() if "model name" in line)
    except Exception:
        model = platform.processor()
    return {
        "hostname": platform.node(),
        "cpu_model": model,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": "cpu",
    }


def _stats(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
        "max_s": float(arr.max()),
        "deadline_20ms_miss": int((arr > 0.020).sum()),
        "deadline_20ms_miss_rate": float((arr > 0.020).mean()),
    }


def _time_stream(predict, rows: np.ndarray, warmup: int) -> list[float]:
    times: list[float] = []
    for i, row in enumerate(rows):
        start = time.perf_counter()
        pred = predict(row)
        elapsed = time.perf_counter() - start
        if not np.isfinite(pred).all():
            raise RuntimeError("nonfinite predict")
        if i >= warmup:
            times.append(elapsed)
    return times


def _m2_small() -> dict[str, Any]:
    from tfpd_exploration.src.m2_dual_track_v1 import data, plan
    from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import load_s1_ema
    from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import constants as C

    model, meta = load_s1_ema(device="cpu")
    session = plan.HELDIN_SESSIONS[0]
    bank = data.load_session_bank("source_minival", session, device="cpu")
    store = np.asarray(bank.X_store)
    if store.ndim == 3:
        stream = np.ascontiguousarray(store.reshape(-1, store.shape[-1]), dtype=np.float32)
    else:
        stream = np.ascontiguousarray(store, dtype=np.float32)
    need = WARMUP + MICRO
    if stream.shape[0] < need:
        raise RuntimeError(f"M2 stream too short: {stream.shape}")
    rows = stream[:need]
    window = np.zeros((C.WINDOW, C.CHANNELS), dtype=np.float32)

    def predict(row: np.ndarray) -> np.ndarray:
        nonlocal window
        window = np.roll(window, -1, axis=0)
        window[-1] = row
        x = torch.from_numpy(window).unsqueeze(0)
        with torch.inference_mode():
            out = model.forward_last(x, bank, bank.unit_mask)
        return (out.detach().cpu().numpy() / C.BEHAVIOR_SCALE).astype(np.float32)

    times = _time_stream(predict, rows, WARMUP)
    mean = float(np.mean(times))
    return {
        "cell": "M2_SMALL_unoptimized_B",
        "weight_sha256": meta["weight_sha256"],
        "session": session,
        "window": C.WINDOW,
        "n_units": C.CHANNELS,
        "source": "held-in source_minival public-dev",
        "microprobe": _stats(times),
        "eta_2048x3_s": mean * LONG_CALLS * LONG_REPEATS,
        "uses_old_spint_decoder": False,
    }


def _h1_flat() -> dict[str, Any]:
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import (
        _index_split,
        load_session_arrays,
        materialize_banks,
    )
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.streaming import H1TemporalStreamer
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1TemporalFlatDecoder

    ckpt_path = (
        REPO
        / "tfpd_exploration/results/h1_temporal_decoder_quick_product_v1/flat/epoch_005.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = H1TemporalFlatDecoder(seed=42)
    model.load_state_dict(ckpt["ema"]["shadow"], strict=True)
    model.eval()
    session = HELDIN_SESSIONS[0]
    path = _index_split("held-in-minival")[session]
    arrays = load_session_arrays(path, session, skip_first3=False)
    bank = materialize_banks({session: arrays})[session]
    need = WARMUP + MICRO
    neural = np.ascontiguousarray(arrays.neural[:need], dtype=np.float32)
    if neural.shape[0] < need:
        raise RuntimeError(f"H1 stream too short: {neural.shape}")
    streamer = H1TemporalStreamer(model, bank, batch_size=1)
    streamer.reset()

    def predict(row: np.ndarray) -> np.ndarray:
        return streamer.predict(row.reshape(1, -1))

    times = _time_stream(predict, neural, WARMUP)
    mean = float(np.mean(times))
    return {
        "cell": "H1_FLAT_unoptimized_B_ema_e5",
        "ckpt": str(ckpt_path),
        "epoch": 5,
        "view": "EMA",
        "session": session,
        "nwb": str(path),
        "window": 700,
        "n_units": 176,
        "source": "held-in-minival public-dev",
        "microprobe": _stats(times),
        "eta_2048x3_s": mean * LONG_CALLS * LONG_REPEATS,
        "uses_old_c2_decoder": False,
    }


def main() -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "latency_opt_v1_e0_microprobe",
        "status": "MICROPROBE_OK",
        "stamp": STAMP,
        "updated": datetime.now(timezone.utc).isoformat(),
        "env": _env(),
        "note": "128-call public-dev microprobe only. Not a speedup claim. 2048x3 not started.",
        "spint_reference": "deferred; this receipt is unoptimized new-B only",
        "cells": {},
    }
    report["cells"]["m2_small"] = _m2_small()
    (RESULT / "e0_microprobe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["cells"]["h1_flat"] = _h1_flat()
    dest = RESULT / "e0_microprobe.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
