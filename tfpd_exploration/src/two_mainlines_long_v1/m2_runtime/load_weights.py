"""Load frozen S1 EMA / S2 RAW decoder weights. Host-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder
from tfpd_exploration.src.m2_b_small_stability_v1.score import apply_view
from tfpd_exploration.src.m2_dual_track_v1.decoders import BTransformerDecoder
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256

from . import constants as C


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_s1_ema(*, device: str | torch.device = "cpu") -> tuple[SmallTransformerDecoder, dict[str, Any]]:
    device = torch.device(device)
    require(C.S1_CKPT.is_file(), f"missing S1 ckpt {C.S1_CKPT}")
    ckpt = torch.load(C.S1_CKPT, map_location="cpu", weights_only=False)
    require(ckpt.get("schema") == C.S1_CKPT_SCHEMA, f"S1 schema {ckpt.get('schema')}")
    require(int(ckpt.get("epoch", -1)) == 19, f"S1 epoch {ckpt.get('epoch')}")
    require(int(ckpt.get("seed", -1)) == 42, f"S1 seed {ckpt.get('seed')}")
    require("ema" in ckpt and "shadow" in ckpt["ema"], "S1 missing ema.shadow")
    model = SmallTransformerDecoder(seed=42)
    apply_view(model, ckpt, "EMA")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = {
        "kind": "small",
        "cell": "S1-SMALL-COS",
        "seed": 42,
        "epoch": 19,
        "view": "EMA",
        "ckpt": str(C.S1_CKPT),
        "ckpt_schema": C.S1_CKPT_SCHEMA,
        "weight_sha256": tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()}),
        "ema_n_updates": int(ckpt["ema"].get("n_updates", -1)),
        "uses_old_spint_decoder": False,
    }
    return model, meta


def load_s2_raw(*, device: str | torch.device = "cpu") -> tuple[BTransformerDecoder, dict[str, Any]]:
    device = torch.device(device)
    require(C.S2_CKPT.is_file(), f"missing S2 ckpt {C.S2_CKPT}")
    ckpt = torch.load(C.S2_CKPT, map_location="cpu", weights_only=False)
    require(ckpt.get("schema") == C.S2_CKPT_SCHEMA, f"S2 schema {ckpt.get('schema')}")
    epoch = int(ckpt.get("epoch_one_based", ckpt.get("epoch", -1)))
    require(epoch == 20, f"S2 epoch {epoch}")
    model = BTransformerDecoder(seed=42)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    require(not missing and not unexpected, f"S2 state mismatch {missing} {unexpected}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = {
        "kind": "large",
        "cell": "B-TRANSFORMER",
        "seed": 42,
        "epoch": 20,
        "view": "RAW",
        "ckpt": str(C.S2_CKPT),
        "ckpt_schema": C.S2_CKPT_SCHEMA,
        "weight_sha256": tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()}),
        "disclosure": "architecture probe; worst-session 0.2466 on ses-2020-11-19-Run1",
        "uses_old_spint_decoder": False,
    }
    return model, meta


def load_pick(kind: str, *, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    if kind == "small":
        return load_s1_ema(device=device)
    if kind == "large":
        return load_s2_raw(device=device)
    raise ValueError(f"unknown kind {kind}")


def export_state_numpy(model: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        out[name] = tensor.detach().cpu().contiguous().numpy()
    return out
