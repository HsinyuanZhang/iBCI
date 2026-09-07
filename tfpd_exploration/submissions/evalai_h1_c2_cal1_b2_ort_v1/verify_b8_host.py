#!/usr/bin/env python3
"""Host B=8 ORT vs packed exact-E gate for official H1 batch size."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask

DEST = Path(__file__).resolve().parent
os.environ.setdefault("RT_PACKED_DECODER", str(DEST / "h1_trf_falcon_decoder.py"))
sys.path.insert(0, str(DEST))
sys.path.insert(0, "/home/xinyuan/Work_host/SPINT")
sys.path.insert(0, "/home/xinyuan/Work_host/SPINT/btransform_unified_v1/src")

from btransform_unified_v1 import adapters
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
from h1_exacte_ort import OrtH1ProjAddFalconDecoder
from h1_trf_falcon_decoder import H1ProjAddFalconDecoder

PAYLOAD = DEST / "artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl"
GRAPHS = DEST / "artifacts/ort_graphs"
GATE = 1.0e-5
B = 8
N_STEPS = 400


def main() -> None:
    config = FalconConfig(task=FalconTask.h1)
    cache = adapters._h1_source_cache()
    sessions = list(HELDIN_SESSIONS[:B])
    stems = [f"sub-HumanPitt-held-in-minival_{s}" for s in sessions]
    streams = [np.ascontiguousarray(cache["minival"][s]["neural"][:N_STEPS], dtype=np.float32) for s in sessions]
    packed = H1ProjAddFalconDecoder(task_config=config, model_path=str(PAYLOAD), batch_size=B)
    packed.reset(dataset_tags=stems)
    ort = OrtH1ProjAddFalconDecoder(
        task_config=config, model_path=str(PAYLOAD), batch_size=B, graph_dir=GRAPHS, intra_op=2, inter_op=1
    )
    ort.reset(dataset_tags=stems)
    max_abs = 0.0
    last = None
    for t in range(N_STEPS):
        batch = np.stack([s[t] for s in streams], axis=0)
        p = packed.predict(batch)
        o = ort.predict(batch)
        last = o
        max_abs = max(max_abs, float(np.max(np.abs(o - p))))
        if max_abs > GATE:
            raise RuntimeError(f"B8 gate fail t={t} max_abs={max_abs}")
    report = {
        "status": "HOST_B8_GATE_PASS",
        "batch": B,
        "steps": N_STEPS,
        "max_abs": max_abs,
        "last_shape": list(last.shape),
        "sessions": sessions,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
