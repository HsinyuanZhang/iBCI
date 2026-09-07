"""Materialize E0 (700-d C2 fused identity) + 4-d H-C for every official H1 tag.

Reads the sealed C2 payload for public M3 activity/carrier only. Does not copy
C2 decoder weights into the deployed payload.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import (
    load_frozen_c2_materializer,
    sha256_file,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL

from . import constants as C


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def collect_official_banks() -> dict[str, dict[str, Any]]:
    require(C.C2_PAYLOAD.is_file(), f"missing C2 calibration payload {C.C2_PAYLOAD}")
    require(sha256_file(C.C2_PAYLOAD) == C.C2_PAYLOAD_SHA256, "C2 payload SHA drift")
    materializer = load_frozen_c2_materializer(expected_sha256=C.C2_CKPT_SHA256)
    require(materializer.checkpoint_sha256 == C.C2_CKPT_SHA256, "C2 ckpt SHA drift")
    payload = torch.load(C.C2_PAYLOAD, map_location="cpu", weights_only=False)
    sessions = payload["sessions"]
    require(set(sessions) == set(C.OFFICIAL_TAGS), f"official tag drift {sorted(sessions)}")
    require(len(sessions) == C.EXPECTED_SESSION_COUNT, "H1 must cover 27 tags")
    banks: dict[str, dict[str, Any]] = {}
    for tag in C.OFFICIAL_TAGS:
        row = sessions[tag]
        activity = torch.as_tensor(np.asarray(row["identity"], dtype=np.float32))
        carrier = torch.as_tensor(np.asarray(row["carrier"], dtype=np.float32))
        if activity.ndim == 3:
            activity = activity.unsqueeze(0)
        if carrier.ndim == 2:
            carrier = carrier.unsqueeze(0)
        e0, hc = materializer.materialize_bank(activity, carrier)
        e0_np = np.ascontiguousarray(e0.detach().cpu().numpy(), dtype=np.float32)
        hc_np = np.ascontiguousarray(hc.detach().cpu().numpy(), dtype=np.float32)
        require(e0_np.shape == (C.CHANNELS, H1_TEMPORAL.e0_dim), f"{tag} E0 {e0_np.shape}")
        require(hc_np.shape == (C.CHANNELS, H1_TEMPORAL.hc_dim), f"{tag} H-C {hc_np.shape}")
        require(bool(np.isfinite(e0_np).all() and np.isfinite(hc_np).all()), f"{tag} nonfinite bank")
        banks[tag] = {
            "E0": e0_np,
            "T": hc_np,
            "unit_mask": np.ones(C.CHANNELS, dtype=np.bool_),
            "session": str(row["session"]),
            "e0_sha256": array_sha256(e0_np),
            "hc_sha256": array_sha256(hc_np),
        }
    return banks


def bank_receipt(banks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_tags": len(banks),
        "official_tags": list(C.OFFICIAL_TAGS),
        "c2_ckpt_sha256": C.C2_CKPT_SHA256,
        "c2_payload_sha256": C.C2_PAYLOAD_SHA256,
        "e0_is_fused_identity": True,
        "e0_dim": 700,
        "hc_dim": 4,
        "per_tag": {
            tag: {
                "session": banks[tag]["session"],
                "e0_sha256": banks[tag]["e0_sha256"],
                "hc_sha256": banks[tag]["hc_sha256"],
            }
            for tag in C.OFFICIAL_TAGS
        },
    }
