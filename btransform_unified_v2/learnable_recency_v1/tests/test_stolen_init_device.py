"""New learnable parameters must follow a stolen temporal's device.

A cuda-built decoder (the H1 path) swaps in the learnable stack after
``.to(device)``; ``register_parameter`` never moves tensors, so params created
on cpu would crash ``effective_slopes`` at the first cuda forward. Smoke runs
never catch this because they force ``--device cpu``.
"""

from __future__ import annotations

import torch

from btransform_unified_v2.temporal import RiftTemporal
from learnable_recency_v1.temporal import LearnableRecencyTemporal

from helpers import recency_config, temporal_config


def test_stolen_temporal_keeps_new_params_on_module_device() -> None:
    if not torch.cuda.is_available():
        meta = torch.device("meta")
        temporal = RiftTemporal(temporal_config(50, width=16, heads=4)).to(meta)
        learnable = LearnableRecencyTemporal.from_initialized(
            temporal, recency_config("learned_slope", 50, heads=4)
        )
        assert learnable.slope_log.device.type == "meta"
        assert learnable._learn_mask.device.type == "meta"
        return
    device = torch.device("cuda")
    temporal = RiftTemporal(temporal_config(50, width=16, heads=4)).to(device)
    learnable = LearnableRecencyTemporal.from_initialized(
        temporal, recency_config("learned_slope", 50, heads=4)
    )
    assert learnable.slope_log.device == device
    z = torch.randn(2, 12, 16, device=device)
    mask = torch.ones(2, 12, dtype=torch.bool, device=device)
    out = learnable(z, mask)
    assert out.device == device and torch.isfinite(out).all()
