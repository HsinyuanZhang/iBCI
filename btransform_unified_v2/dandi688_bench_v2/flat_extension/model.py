"""Flat temporal variant of the base DANDI decoder, without changing base code."""
from __future__ import annotations

import torch

from learnable_recency_v1.flat_control import flat_config

from ..model import B3SIdentityEncoder, DandiRiftDecoder


class FlatDandiRiftDecoder(DandiRiftDecoder):
    def __init__(self, arm: str = "full", n_pad: int = 100, seed: int = 42,
                 encoder: B3SIdentityEncoder | None = None, freeze_encoder: bool = True) -> None:
        super().__init__(arm=arm, n_pad=n_pad, seed=seed, encoder=encoder,
                         freeze_encoder=freeze_encoder, recency_cfg=flat_config("m2"))


def validate_flat_model(model: DandiRiftDecoder, *, require_frozen_full: bool = False) -> dict:
    cfg = model.learnable_cfg
    if cfg.tier != "fixed" or tuple(cfg.half_life_seconds) != (None,) * 8:
        raise ValueError("Flat decoder must use fixed all-None M2 recency configuration")
    slopes = model.temporal.recency_slopes.detach().cpu()
    expanded = slopes.reshape(1, 8).expand(4, 8)
    if tuple(expanded.shape) != (4, 8) or int(expanded.count_nonzero()) != 0:
        raise ValueError("Flat decoder has nonzero temporal recency slopes")
    forbidden = [name for name, _ in model.named_parameters()
                 if "slope_log" in name or "recency" in name or "cable" in name or "gate" in name]
    if forbidden:
        raise ValueError(f"Flat decoder has learnable recency parameters: {forbidden}")
    frozen = None
    if model.arm == "full":
        if model.encoder is None or model.encoder.side_dim != 4:
            raise ValueError("Flat Full requires the base representation-matched concat encoder")
        frozen = sum(p.numel() for p in model.encoder.parameters())
        if frozen != 18290:
            raise ValueError(f"Flat Full encoder parameter count drift: {frozen}")
        if require_frozen_full and any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError("Flat Full encoder must remain frozen")
    return {"temporal_variant": "flat", "tier": "fixed", "half_life_seconds": [None] * 8,
            "effective_bias": "allzero", "slopes_shape": [4, 8], "slopes_zero_count": 32,
            "learnable_recency_parameters": [], "full_encoder_parameters": frozen,
            "full_encoder_frozen": frozen is not None and not any(p.requires_grad for p in model.encoder.parameters())}


def make_model(arm: str, *, seed: int, encoder: B3SIdentityEncoder | None = None,
               freeze_encoder: bool = True, n_pad: int = 100) -> FlatDandiRiftDecoder:
    model = FlatDandiRiftDecoder(arm=arm, seed=seed, encoder=encoder,
                                 freeze_encoder=freeze_encoder, n_pad=n_pad)
    validate_flat_model(model, require_frozen_full=arm == "full" and freeze_encoder)
    return model


__all__ = ["FlatDandiRiftDecoder", "make_model", "validate_flat_model"]
