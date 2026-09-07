"""Compute only the five frontend tokens invalidated by a k=5 window shift."""
from __future__ import annotations

import torch


def repaired_local_features(local_conv, raw: torch.Tensor) -> torch.Tensor:
    """Return local features for raw indices (0,1,2,3,W-1), in that order.

    The first four tokens acquire a new zero-padded left boundary on every
    finite-window shift. The newest token needs exactly the last five raw bins.
    Stitching prefix4 and suffix5 is safe ONLY because all four intermediate
    suffix outputs are discarded: the last output's k=5 receptive field lies
    wholly inside suffix5. Nonlinear token/slot work then runs on 5, not 9, bins.
    This does not cache contextual temporal state or change the raw window.
    """
    if raw.ndim != 3 or raw.shape[1] < 9:
        raise ValueError("repair requires [B,W,N] with W >= 9")
    if int(local_conv.cfg.conv_kernel) != 5:
        raise ValueError("five-token repair is defined only for causal k=5")
    stitched = torch.cat((raw[:, :4], raw[:, -5:]), dim=1)
    local = local_conv(stitched)
    return torch.cat((local[:, :4], local[:, -1:]), dim=1)
