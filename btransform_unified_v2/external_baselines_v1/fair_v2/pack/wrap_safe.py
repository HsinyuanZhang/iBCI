"""Container-safe learnable wrap. Joint/concat modules are omitted."""
from __future__ import annotations

from math import log

import torch
from torch import Tensor, nn

from btransform_unified_v2.model import RiftDecoder
from btransform_unified_v2.temporal import RiftTemporal

from .config import LearnableRecencyConfig
from .temporal import LearnableRecencyTemporal

RIFT_TEMPORAL_SEED_OFFSET = 0x52494654


def _architecture_key(config) -> tuple:
    return (config.layers, config.windows, config.width, config.heads, config.ffn_width)


def _slopes_from_config(config, device: torch.device | None = None) -> Tensor:
    slopes = [
        0.0 if half is None else log(2.0) * config.bin_seconds / half
        for half in config.half_life_seconds
    ]
    tensor = torch.tensor(slopes, dtype=torch.float32)
    return tensor if device is None else tensor.to(device=device)


def reinit_rift_linears(temporal: RiftTemporal, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + RIFT_TEMPORAL_SEED_OFFSET)
        for module in temporal.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def seeded_rift_temporal(config, seed: int, *, device: torch.device | str | None = None) -> RiftTemporal:
    rng = torch.get_rng_state()
    try:
        temporal = RiftTemporal(config)
        reinit_rift_linears(temporal, seed)
        if device is not None:
            temporal = temporal.to(device=torch.device(device))
        return temporal
    finally:
        torch.set_rng_state(rng)


def _shell_with_blocks(existing: RiftTemporal, config) -> RiftTemporal:
    temporal = RiftTemporal.__new__(RiftTemporal)
    nn.Module.__init__(temporal)
    temporal.config = config
    temporal.add_module("blocks", existing.blocks)
    temporal.register_buffer(
        "recency_slopes",
        _slopes_from_config(config, existing.recency_slopes.device),
        persistent=True,
    )
    temporal._attention_backend = existing._attention_backend
    return temporal


def adopt_temporal_config(existing: RiftTemporal, config, seed: int) -> RiftTemporal:
    if _architecture_key(existing.config) != _architecture_key(config):
        device = next(existing.parameters()).device
        return seeded_rift_temporal(config, seed, device=device)
    if existing.config.half_life_seconds == config.half_life_seconds and existing.config.bin_seconds == config.bin_seconds:
        return existing
    return _shell_with_blocks(existing, config)


def attach_learnable_recency(temporal: RiftTemporal, recency_cfg: LearnableRecencyConfig) -> LearnableRecencyTemporal:
    if recency_cfg.tier == "fixed":
        raise ValueError("fixed tier must stay a plain RiftTemporal")
    if isinstance(temporal, LearnableRecencyTemporal):
        raise TypeError("temporal is already a LearnableRecencyTemporal")
    return LearnableRecencyTemporal.from_initialized(temporal, recency_cfg)


def install_temporal(decoder: nn.Module, recency_cfg: LearnableRecencyConfig, seed: int) -> nn.Module:
    existing = decoder.temporal
    if not isinstance(existing, RiftTemporal):
        raise TypeError("decoder.temporal must be a RiftTemporal")
    rebuilt = adopt_temporal_config(existing, recency_cfg.temporal_config, seed)
    decoder.temporal = rebuilt if recency_cfg.tier == "fixed" else attach_learnable_recency(rebuilt, recency_cfg)
    decoder.temporal_config = recency_cfg.temporal_config
    decoder.learnable_cfg = recency_cfg
    decoder.bias_mode = "recency" if recency_cfg.tier == "fixed" else f"learnable_{recency_cfg.tier}"
    return decoder


class LearnableRiftDecoder(RiftDecoder):
    def __init__(
        self,
        task: str,
        recency_cfg: LearnableRecencyConfig,
        *,
        context_bins: int | None = None,
        seed: int = 42,
        proj_dim: int = 16,
    ) -> None:
        super().__init__(
            task,
            context_bins=context_bins if context_bins is not None else recency_cfg.context_bins,
            bias_mode="recency",
            seed=seed,
            proj_dim=proj_dim,
        )
        install_temporal(self, recency_cfg, seed)
