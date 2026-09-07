"""Load EMA snapshot epoch 5. RAW is diagnostic only and is not exported."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
)

from . import constants as C


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_ema(kind: str, *, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    require(kind in {"flat", "route"}, f"unknown kind {kind}")
    ckpt_path = C.FLAT_CKPT if kind == "flat" else C.ROUTE_CKPT
    require(ckpt_path.is_file(), f"missing snapshot {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    require(ckpt.get("schema") == C.CKPT_SCHEMA, f"schema {ckpt.get('schema')}")
    require(str(ckpt.get("arm")) == kind, f"arm {ckpt.get('arm')} != {kind}")
    require(int(ckpt.get("epoch", -1)) == C.EPOCH, f"epoch {ckpt.get('epoch')} is not 5")
    require(int(ckpt.get("seed", -1)) == C.SEED, f"seed {ckpt.get('seed')}")
    require("ema" in ckpt and "shadow" in ckpt["ema"], "missing ema.shadow")
    require(int(ckpt["epoch"]) != C.EPOCHS_TARGET, "refusing to label epoch 5 as the 12-epoch product")
    if kind == "flat":
        model = H1TemporalFlatDecoder(seed=C.SEED)
    else:
        model = H1TemporalRouteDecoder(seed=C.SEED)
    missing, unexpected = model.load_state_dict(ckpt["ema"]["shadow"], strict=True)
    require(not missing and not unexpected, f"EMA load mismatch {missing} {unexpected}")
    require(int(model.temporal.pe.size(0)) >= C.WINDOW, "PE must cover 700")
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = {
        "kind": kind,
        "arm": kind,
        "seed": C.SEED,
        "epoch": C.EPOCH,
        "epochs_target": C.EPOCHS_TARGET,
        "view": C.VIEW,
        "ckpt": str(ckpt_path),
        "ckpt_schema": C.CKPT_SCHEMA,
        "ema_n_updates": int(ckpt["ema"].get("n_updates", -1)),
        "weight_sha256": tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()}),
        "uses_old_c2_decoder": False,
        "c2_identity_swap": False,
        "disclosure": "known-source development; snapshot epoch 5 of intended 12; not clean LODO",
    }
    return model, meta


def export_state_numpy(model: nn.Module) -> dict[str, Any]:
    return {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in model.state_dict().items()}
