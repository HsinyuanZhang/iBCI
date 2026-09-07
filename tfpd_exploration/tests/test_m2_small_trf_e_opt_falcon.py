"""Host parity: exact-E Falcon adapter vs naive full-window on the sealed SMALL payload."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import session_tag_map
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.container_decoder import (
    TrfFalconDecoder as NaiveFalcon,
)

NEW = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m2_small_trf_e_opt_v1")
sys.path.insert(0, str(NEW))
from trf_falcon_decoder import TrfFalconDecoder as ExactEFalcon  # noqa: E402

PAYLOAD = NEW / "artifacts/m2_small_trf_s1_ema_e19.pkl"


def _stem(session: str) -> str:
    tag = session_tag_map()[session]
    date = tag.split("_", 1)[1]
    run = tag.split("_", 1)[0]
    return f"sub-MonkeyN{run}_{date}_held_in_eval"


def _trace(session: str, bins: int) -> np.ndarray:
    bank = data.load_session_bank("source_minival", session, device="cpu")
    store = np.asarray(bank.X_store, dtype=np.float32)
    if store.ndim == 3:
        store = store.reshape(-1, store.shape[-1])
    return np.ascontiguousarray(store[:bins], dtype=np.float32)


def test_exact_e_matches_naive_b1_and_b7_streams():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    config = FalconConfig(task=FalconTask.m2)
    sessions = list(plan.HELDIN_SESSIONS[:7])
    bins = 80
    traces = [_trace(session, bins) for session in sessions]
    stems = [_stem(session) for session in sessions]

    naive = NaiveFalcon(task_config=config, model_path=str(PAYLOAD), batch_size=7)
    exact = ExactEFalcon(task_config=config, model_path=str(PAYLOAD), batch_size=7)
    naive.reset(dataset_tags=stems)
    exact.reset(dataset_tags=stems)

    max_abs = 0.0
    for step in range(bins):
        row = np.stack([trace[step] for trace in traces], axis=0)
        ref = naive.predict(row)
        got = exact.predict(row)
        max_abs = max(max_abs, float(np.max(np.abs(got - ref))))
    assert max_abs < 1.0e-5, max_abs

    naive.reset(dataset_tags=stems[:1])
    exact.reset(dataset_tags=stems[:1])
    for step in range(bins):
        row = traces[0][step].reshape(1, -1)
        ref = naive.predict(row)
        got = exact.predict(row)
        assert float(np.max(np.abs(got - ref))) < 1.0e-5
