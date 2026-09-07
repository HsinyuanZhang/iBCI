"""Host training-decoder vs Falcon stream parity, plus smoke window export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import (
    load_session_arrays,
    materialize_banks,
)
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import H1_DATA_DIR, HELDIN_SESSIONS
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from . import constants as C
from .container_decoder import H1TemporalFalconDecoder
from .load_weights import load_ema


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _heldin_window() -> tuple[str, str, np.ndarray, H1Bank]:
    session = HELDIN_SESSIONS[0]
    root = H1_DATA_DIR / "sub-HumanPitt-held-in-minival"
    found = sorted(root.rglob(f"*{session}*.nwb"))
    require(len(found) == 1, f"minival count for {session}: {found}")
    arrays = load_session_arrays(found[0], session, skip_first3=False)
    require(arrays.neural.shape[1] == C.CHANNELS, f"channels {arrays.neural.shape}")
    start = 0
    window = np.ascontiguousarray(arrays.neural[start : start + C.WINDOW], dtype=np.float32)
    require(window.shape == (C.WINDOW, C.CHANNELS), f"window shape {window.shape}")
    banks = materialize_banks({session: arrays})
    return session, "held-in-minival", window, banks[session]


def host_stream_parity(kind: str, payload_path: Path) -> dict[str, Any]:
    session, surface, window, bank = _heldin_window()
    model, meta = load_ema(kind, device="cpu")
    x = torch.from_numpy(window).unsqueeze(0)
    with torch.inference_mode():
        raw = model.forward_last(x, bank, bank.unit_mask)
    host_native = (raw.detach().cpu().numpy() / C.BEHAVIOR_SCALE).astype(np.float32)
    require(bool(np.isfinite(host_native).all()), "host forward nonfinite")

    stem = f"sub-HumanPitt-held-in-minival_{session}"
    config = FalconConfig(task=FalconTask.h1)
    tag = config.hash_dataset(stem)
    require(tag in C.OFFICIAL_TAGS, f"hashed tag {tag} is not official")
    decoder = H1TemporalFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=[stem])
    require(set(decoder.bank_by_dataset_tag) == set(C.OFFICIAL_TAGS), "payload missing official tags")
    pred = None
    started = time.perf_counter()
    history_before = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
        history_before = int(decoder.history_count[0])
        decoder.on_done(np.ones((1,), dtype=bool))
        require(int(decoder.history_count[0]) == history_before, "on_done reset continual history")
    stream_s = time.perf_counter() - started
    require(pred is not None, "empty stream")
    require(bool(np.isfinite(pred).all()), "falcon stream nonfinite")
    max_abs = float(np.max(np.abs(pred - host_native)))
    require(max_abs < 1.0e-5, f"host vs falcon stream drift {max_abs}")
    require(int(decoder.decoder.temporal.pe.size(0)) >= C.WINDOW, "PE must cover 700")

    smoke_path = payload_path.parent / f"{kind}_smoke_window.npz"
    np.savez(smoke_path, window=window, expected=pred, tag_stem=np.asarray(stem))
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
        "on_done_noop": True,
        "pe_max_len": int(decoder.decoder.temporal.pe.size(0)),
        "stream_seconds_700bins": stream_s,
        "ms_per_window": 1000.0 * stream_s / C.WINDOW,
        "smoke_window": str(smoke_path),
        "weight_sha256": meta["weight_sha256"],
        "kv_cache_across_windows": False,
        "epoch": C.EPOCH,
        "epochs_target": C.EPOCHS_TARGET,
        "view": C.VIEW,
    }


def estimate_cpu_hours(
    ms_per_window: float,
    *,
    heldin_minival_bins: int = 20920,
    heldout_files: int = 14,
    heldout_bins_proxy: int = 1600,
) -> dict[str, float]:
    """Honest official-test wall estimate. Hidden test files are not opened.

    Profiled stream was ~482 ms/window. Official H1 recommended batch is 8;
    this runtime runs one full W=700 forward per active session per bin, so
    CPU work scales with total bins, not with a claimed cache speedup.
    """
    heldout_proxy = heldout_files * heldout_bins_proxy
    total = heldin_minival_bins + heldout_proxy
    hours = (ms_per_window / 1000.0) * total / 3600.0
    hours_if_heldout_2500 = (ms_per_window / 1000.0) * (heldin_minival_bins + heldout_files * 2500) / 3600.0
    return {
        "ms_per_window": ms_per_window,
        "heldin_minival_bins": float(heldin_minival_bins),
        "heldout_files": float(heldout_files),
        "heldout_bins_proxy": float(heldout_bins_proxy),
        "cpu_hours_heldin_plus_heldout_proxy": hours,
        "cpu_hours_if_heldout_2500bins": hours_if_heldout_2500,
        "six_hour_risk": float(hours >= 6.0 or hours_if_heldout_2500 >= 6.0),
        "note": (
            "482 ms/window class cost; hidden test not opened; 14 held-out files "
            "proxied. Official batch=8 still does one forward per session per bin. "
            "6h EvalAI wall risk is possible."
        ),
    }
