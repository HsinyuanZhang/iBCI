"""The sole CRST-B4-FLAT -> CRST-B4-ROUTE structural delta for M2 SMALL.

This is intentionally a small, independently testable component rather than a
second decoder.  The integration point is the *pre-softmax* slot-to-unit logits
in ``m2_b_small_stability_v1.decoder.SharedSetFrontend``.  It must not be used
to change tokens, unit masks, temporal layers, positional encoding, loss, or
output calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class M2RoutingConfig:
    """Frozen M2 SMALL dimensions from CRST-B4's shared frontend."""

    e0_dim: int = 50
    functional_dim: int = 4
    heads: int = 8
    slots: int = 8
    route_key_dim: int = 32


class M2SlotRoutingBias(nn.Module):
    """Additive calibration-only route bias.

    For calibration carrier ``c_n = LN([E0_n; T_n])`` this returns
    ``tanh(g_h) * q_cal[h,s] dot P_h(c_n) / sqrt(32)`` with shape
    ``[B,H,S,N]``.  No sample spike, local convolution output, dynamic attention
    query/key, time index, mask, or temporal state enters this calculation.

    ``g`` starts at zero.  Thus a route decoder whose shared weights have been
    copied from FLAT is exactly FLAT at initialization, while ``g`` still has a
    useful gradient whenever the independently initialized route score is
    nonzero.
    """

    def __init__(self, cfg: M2RoutingConfig = M2RoutingConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        carrier_dim = cfg.e0_dim + cfg.functional_dim
        self.route_proj = nn.ModuleList(
            nn.Linear(carrier_dim, cfg.route_key_dim) for _ in range(cfg.heads)
        )
        self.q_cal = nn.Parameter(torch.empty(cfg.heads, cfg.slots, cfg.route_key_dim))
        self.g = nn.Parameter(torch.zeros(cfg.heads))
        nn.init.xavier_uniform_(self.q_cal)
        for projection in self.route_proj:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, e0: torch.Tensor, functional: torch.Tensor) -> torch.Tensor:
        """Return a [batch, heads, slots, units] bias from static calibration."""
        if e0.ndim == 2:
            e0 = e0.unsqueeze(0)
        if functional.ndim == 2:
            functional = functional.unsqueeze(0)
        if e0.ndim != 3 or functional.ndim != 3:
            raise ValueError("E0 and T must be [N,D] or [B,N,D]")
        if e0.shape[:2] != functional.shape[:2]:
            raise ValueError("E0/T batch and unit axes must agree")
        if e0.shape[-1] != self.cfg.e0_dim or functional.shape[-1] != self.cfg.functional_dim:
            raise ValueError("unexpected M2 calibration carrier widths")
        carrier = torch.cat([e0, functional], dim=-1)
        carrier = F.layer_norm(carrier, (carrier.shape[-1],), weight=None, bias=None)
        projected = torch.stack([layer(carrier) for layer in self.route_proj], dim=1)
        score = torch.einsum("hsr,bhnr->bhsn", self.q_cal, projected)
        score = score * (self.cfg.route_key_dim ** -0.5)
        return score * torch.tanh(self.g).view(1, self.cfg.heads, 1, 1)


def add_route_bias(flat_logits: torch.Tensor, routing: M2SlotRoutingBias, e0: torch.Tensor, functional: torch.Tensor) -> torch.Tensor:
    """Apply the only legal ROUTE delta to pre-softmax logits.

    ``flat_logits`` is [B,H,S,N]; callers retain their pre-existing mask and
    softmax sequence.  In particular mask handling stays identical to FLAT.
    """
    if flat_logits.ndim != 4:
        raise ValueError("slot-to-unit logits must be [B,H,S,N]")
    bonus = routing(e0, functional)
    if bonus.shape != flat_logits.shape:
        raise ValueError(f"routing/attention shape mismatch: {tuple(bonus.shape)} != {tuple(flat_logits.shape)}")
    return flat_logits + bonus
