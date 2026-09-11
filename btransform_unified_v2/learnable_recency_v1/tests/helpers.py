from __future__ import annotations

import torch

from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.temporal import RiftTemporal
from learnable_recency_v1.config import LearnableRecencyConfig
from learnable_recency_v1.temporal import LearnableRecencyTemporal

TIERS = ("learned_slope", "fox_gate", "cable")


def small_half_lives(heads: int) -> tuple[float | None, ...]:
    base: list[float | None] = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, None, None]
    return tuple(base[:heads])


def temporal_config(context_bins: int, *, width: int = 16, heads: int = 4, layers: int = 4) -> RiftTemporalConfig:
    return RiftTemporalConfig.for_context(
        context_bins,
        width=width,
        heads=heads,
        ffn_width=width * 2,
        layers=layers,
        half_life_seconds=small_half_lives(heads),
    )


def recency_config(tier: str, context_bins: int, *, heads: int = 4, **overrides: object) -> LearnableRecencyConfig:
    values: dict[str, object] = dict(
        tier=tier,
        per_layer=False,
        learn_flat_heads=False,
        cable_nw=False,
        cable_hidden=8,
        half_life_seconds=small_half_lives(heads),
        context_bins=context_bins,
        layers=int(overrides.get("layers", 4)),
        task="m2" if context_bins == 50 else "h1" if context_bins == 300 else "m1",
    )
    values.update(overrides)
    return LearnableRecencyConfig(**values)  # type: ignore[arg-type]


def make_pair(context_bins: int, tier: str, *, seed: int = 0, width: int = 16, heads: int = 4, **overrides: object):
    torch.manual_seed(seed)
    config = temporal_config(context_bins, width=width, heads=heads)
    fixed = RiftTemporal(config).eval()
    torch.manual_seed(seed)
    sibling = RiftTemporal(config)
    learnable = LearnableRecencyTemporal.from_initialized(
        sibling, recency_config(tier, context_bins, heads=heads, **overrides)
    ).eval()
    return fixed, learnable


def stream(model: LearnableRecencyTemporal | RiftTemporal, z: torch.Tensor, valid_mask: torch.Tensor | None = None):
    state = model.init_state(z.shape[0], z.device, z.dtype)
    out = []
    for time in range(z.shape[1]):
        mask = None if valid_mask is None else valid_mask[:, time]
        hidden, state = model.step(z[:, time], state, mask)
        out.append(hidden)
    return torch.stack(out, dim=1)


def scramble_new_params(model: LearnableRecencyTemporal, generator: torch.Generator | None = None) -> None:
    with torch.no_grad():
        for name in model.new_parameter_names():
            param = getattr(model, name)
            param.add_(torch.randn(param.shape, generator=generator) * 0.05)
