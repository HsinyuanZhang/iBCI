"""Host training-decoder vs Falcon stream parity, plus smoke window export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from tfpd_exploration.src.m2_dual_track_v1 import data, plan

from . import constants as C
from .container_decoder import TrfFalconDecoder
from .load_weights import load_pick


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _heldin_window() -> tuple[str, str, np.ndarray, object]:
    session = plan.HELDIN_SESSIONS[0]
    bank = data.load_session_bank("source_minival", session, device="cpu")
    store = np.asarray(bank.X_store)
    if store.ndim == 3:
        window = np.ascontiguousarray(store[0], dtype=np.float32)
    else:
        start = int(bank.eligible_starts[0])
        window = np.ascontiguousarray(store[start : start + plan.WINDOW], dtype=np.float32)
    require(window.shape == (C.WINDOW, C.CHANNELS), f"window shape {window.shape}")
    return session, "source_minival", window, bank


def host_stream_parity(kind: str, payload_path: Path) -> dict[str, Any]:
    session, surface, window, bank = _heldin_window()
    model, meta = load_pick(kind, device="cpu")
    x = torch.from_numpy(window).unsqueeze(0)
    with torch.inference_mode():
        raw = model.forward_last(x, bank, bank.unit_mask)
    host_native = (raw.detach().cpu().numpy() / C.BEHAVIOR_SCALE).astype(np.float32)
    require(bool(np.isfinite(host_native).all()), "host forward nonfinite")

    from .banks import session_tag_map

    tags = session_tag_map()
    tag = tags[session]
    stem = f"sub-MonkeyNRun1_{tag.split('_', 1)[1]}_held_in_eval" if tag.startswith("Run1_") else f"sub-MonkeyN{tag}_held_in_eval"
    # Official remote stems look like sub-MonkeyNRun1_20201019_held_in_eval
    date = tag.split("_", 1)[1]
    run = tag.split("_", 1)[0]
    stem = f"sub-MonkeyN{run}_{date}_held_in_eval"

    config = FalconConfig(task=FalconTask.m2)
    require(config.hash_dataset(stem) == tag, f"remote-path hash {config.hash_dataset(stem)} != {tag}")
    decoder = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=[stem])
    pred = None
    started = time.perf_counter()
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
    stream_s = time.perf_counter() - started
    require(pred is not None, "empty stream")
    require(bool(np.isfinite(pred).all()), "falcon stream nonfinite")
    max_abs = float(np.max(np.abs(pred - host_native)))
    require(max_abs < 1.0e-5, f"host vs falcon stream drift {max_abs}")
    require(set(decoder.bank_by_dataset_tag) == set(C.OFFICIAL_TAGS), "payload missing official tags")

    smoke_path = payload_path.parent / f"{kind}_smoke_window.npz"
    np.savez(
        smoke_path,
        window=window,
        expected=pred,
        tag_stem=np.asarray(stem),
    )
    return {
        "kind": kind,
        "session": session,
        "surface": surface,
        "tag": tag,
        "remote_stem": stem,
        "max_abs_host_vs_stream": max_abs,
        "host_native": host_native.tolist(),
        "stream_native": pred.tolist(),
        "finite": True,
        "all_official_tags": True,
        "stream_seconds_50bins": stream_s,
        "ms_per_bin": 1000.0 * stream_s / C.WINDOW,
        "smoke_window": str(smoke_path),
        "weight_sha256": meta["weight_sha256"],
        "kv_cache_across_windows": False,
    }


def estimate_cpu_hours(ms_per_bin: float, *, bins_per_wave: int = 25000, waves: int = 2) -> float:
    """Conservative official-test wall estimate from a single-slot CPU bin time."""
    return (ms_per_bin / 1000.0) * bins_per_wave * waves / 3600.0
