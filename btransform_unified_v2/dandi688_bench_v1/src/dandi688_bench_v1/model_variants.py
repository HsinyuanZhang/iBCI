"""Bench-local decoder geometry variants (user directive 2026-09-10, D2).

The rift_v1 main package (btransform_unified_v2.model) is NEVER modified.
The shallow ablation (temporal depth 4 -> 1) subclasses RiftDecoder inside
this bench package and rebuilds only the temporal stack:

  RiftShallowDecoder  a single local-attention layer (layers=1) spanning the
                      SAME total receptive field as the stock D4 stack
                      (RiftTemporalConfig.for_context distributes
                      context_bins - kernel over the layer count, so
                      context=50/k5/D1 -> windows=(46,)).  Weight init replays
                      the exact RiftDecoder.__init__ RIFT-only discipline
                      (fork_rng + manual_seed(seed + 0x52494654) + xavier/
                      LayerNorm init over the temporal modules), so the
                      subclass stays deterministic and byte-reproducible.

D1 (shortwin) needs no subclass: RiftDecoder already takes ``context_bins``,
so the runner builds it with context_bins=10 directly.
"""
from __future__ import annotations

from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.model import RiftDecoder
from btransform_unified_v2.temporal import RiftTemporal

# RIFT-only temporal init seed offset, mirrored from
# btransform_unified_v2.model.RiftDecoder.__init__ (the constant is part of
# the frozen init contract; it is duplicated here only because the stock
# constructor does not expose it).
TEMPORAL_INIT_SEED_OFFSET = 0x52494654


class RiftShallowDecoder(RiftDecoder):
    """D1 (single temporal layer) bench-local subclass of RiftDecoder.

    Everything except the temporal stack is inherited unchanged from the
    stock construction: same frontend (proj_add), same readout, same seed
    discipline.  The temporal stack is rebuilt with
    plan.SHALLOW_TEMPORAL_LAYERS layers over the same context_bins, then
    re-initialized under the same fork_rng/manual_seed discipline so the
    run is deterministic given the seed.
    """

    TEMPORAL_LAYERS = 1  # bound to plan.SHALLOW_TEMPORAL_LAYERS (asserted)

    def __init__(
        self,
        task,
        context_bins: int | None = None,
        bias_mode: str = "recency",
        seed: int = 42,
        proj_dim: int = 16,
        temporal_layers: int | None = None,
    ) -> None:
        super().__init__(
            task, context_bins=context_bins, bias_mode=bias_mode,
            seed=seed, proj_dim=proj_dim,
        )
        import torch
        from torch import nn

        layers = self.TEMPORAL_LAYERS if temporal_layers is None else int(temporal_layers)
        if layers < 1:
            raise ValueError("temporal_layers must be >= 1")
        self.temporal_layers = layers
        config = RiftTemporalConfig.for_context(
            self.context_bins, bias_mode=bias_mode, layers=layers,
        )
        self.temporal = RiftTemporal(config)
        self.temporal_config = config
        # replay the stock RIFT-only init discipline over the rebuilt stack
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + TEMPORAL_INIT_SEED_OFFSET)
            for module in self.temporal.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


__all__ = ["RiftShallowDecoder", "TEMPORAL_INIT_SEED_OFFSET"]
