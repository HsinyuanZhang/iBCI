"""Rank-8 proj_add injection (P8).

The v1 proj_add operator fuses ``proj.view(..., groups, 16)`` into the 16-dim
local conv channels, so its projection width must be a multiple of 16 — P8
cannot be built directly. ``P8`` here therefore means a rank-8 injection: the
decoder is built with ``proj_dim=16`` (byte-identical init path) and its
``frontend.e0_proj`` is replaced by a bias-free ``Linear(e0_dim, 8)`` holding
the first 8 rows of the initialized P16 weight, zero-padded to 16 at forward
time. Only the first 8 local conv channels receive identity information.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class PaddedProj8(nn.Module):
    """``pad(Linear(e0_dim -> 8)(x), (0, 8))`` -> [N, 16]."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.out_features != 8 or linear.bias is not None:
            raise ValueError("PaddedProj8 requires a bias-free Linear with out_features=8")
        self.linear = linear

    def forward(self, e0: Tensor) -> Tensor:
        return functional.pad(self.linear(e0), (0, 8))


def maybe_truncate_p8(module: nn.Module, proj_dim: int) -> nn.Module:
    """Apply the rank-8 truncation when ``proj_dim == 8``; else validate only.

    The caller must have built the decoder with ``proj_dim=16``. The first 8
    rows of the initialized P16 ``e0_proj`` weight are kept verbatim, so the
    truncation is deterministic given the build seed.
    """
    proj = getattr(getattr(module, "frontend", None), "e0_proj", None)
    if proj_dim != 8:
        if int(getattr(module, "proj_dim", proj_dim)) != int(proj_dim):
            raise RuntimeError("proj_dim attribute drift")
        return module
    if not isinstance(proj, nn.Linear) or proj.out_features != 16 or proj.bias is not None:
        raise RuntimeError("P8 truncation requires a bias-free P16 frontend.e0_proj Linear")
    # Stay on the source module's device/dtype; the decoder may already be on cuda.
    linear = nn.Linear(
        proj.in_features, 8, bias=False, device=proj.weight.device, dtype=proj.weight.dtype
    )
    with torch.no_grad():
        linear.weight.copy_(proj.weight[:8])
    module.frontend.e0_proj = PaddedProj8(linear)
    module.proj_dim = 8
    return module


def p8_new_parameter_summary(module: nn.Module) -> dict[str, int]:
    proj = getattr(getattr(module, "frontend", None), "e0_proj", None)
    if isinstance(proj, PaddedProj8):
        return {"frontend.e0_proj.linear.weight": int(proj.linear.weight.numel())}
    return {}


__all__ = ["PaddedProj8", "maybe_truncate_p8", "p8_new_parameter_summary"]
