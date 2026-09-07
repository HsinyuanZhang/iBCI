"""Frozen source-data routing plus an explicit native/runtime unit bridge."""
from __future__ import annotations

import numpy as np
import torch

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import *  # frozen legal source routing
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import SessionArrays, _index_split

TARGET_MULTIPLIER = 20.0

def build_window_manifest() -> dict:
    """Read the frozen source routing without mutating the legacy result root."""
    calib_paths = _index_split("held-in-calib")
    mini_paths = _index_split("held-in-minival")
    train_sessions = {name: load_session_arrays(calib_paths[name], name, skip_first3=True) for name in HELDIN_SESSIONS}
    mini_sessions = {name: load_session_arrays(mini_paths[name], name, skip_first3=False) for name in HELDIN_SESSIONS}
    return {"train_sessions": train_sessions, "mini_sessions": mini_sessions}

def collate_runtime_target(sessions: dict[str, SessionArrays], items: list[tuple[str, int]]):
    session = items[0][0]
    if any(name != session for name, _ in items):
        raise RuntimeError("batch must remain session-pure")
    rec = sessions[session]
    xs = np.stack([rec.neural[start:start + WINDOW] for _, start in items]).astype(np.float32, copy=False)
    native = np.stack([rec.velocity[start + WINDOW - 1] for _, start in items]).astype(np.float32, copy=False)
    target = native * TARGET_MULTIPLIER
    # This contract is intentionally loud: model output is divided by 20 at runtime.
    if not np.allclose(target / TARGET_MULTIPLIER, native, rtol=0.0, atol=1e-7):
        raise AssertionError("runtime target bridge lost native velocity units")
    return session, torch.as_tensor(xs), torch.as_tensor(target), torch.as_tensor(native), [s for _, s in items]
