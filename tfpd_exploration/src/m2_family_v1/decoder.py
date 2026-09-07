"""Trainable M2 SMALL CRST-B4 decoder factory, isolated from frozen ext6 code."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfpd_exploration.src.m2_b_small_stability_v1.config import SMALL, SmallConfig, require
from tfpd_exploration.src.m2_b_small_stability_v1.decoder import (
    SmallTransformerDecoder,
    _expand_unit_mask,
)
from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank
from tfpd_exploration.src.m2_dual_track_v1.decoders import whole_unit_dropout

from .routing import M2SlotRoutingBias, add_route_bias


class ExplicitSlotUnitAttention(nn.Module):
    """Numerically mapped `nn.MultiheadAttention` for batch-first self-attention.

    Parameter layout intentionally follows PyTorch MHA's concatenated q/k/v
    projection so a frozen FLAT MHA can be copied losslessly before introducing
    the ROUTE-only bias.
    """

    def __init__(self, dim: int = 256, heads: int = 8, *, routed: bool = False) -> None:
        super().__init__()
        require(dim % heads == 0, "attention head divisibility")
        self.dim, self.heads, self.head_dim, self.routed = dim, heads, dim // heads, routed
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj_weight = nn.Parameter(torch.empty(dim, dim))
        self.out_proj_bias = nn.Parameter(torch.empty(dim))
        if routed:
            self.routing = M2SlotRoutingBias()

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, *, routed: bool) -> "ExplicitSlotUnitAttention":
        module = cls(mha.embed_dim, mha.num_heads, routed=routed)
        require(mha.dropout == 0.0 and mha.batch_first, "only frozen M2 batch_first/dropout=0 MHA is legal")
        with torch.no_grad():
            module.in_proj_weight.copy_(mha.in_proj_weight)
            module.in_proj_bias.copy_(mha.in_proj_bias)
            module.out_proj_weight.copy_(mha.out_proj.weight)
            module.out_proj_bias.copy_(mha.out_proj.bias)
        return module

    def copy_shared_from(self, flat: "ExplicitSlotUnitAttention") -> None:
        require(not flat.routed, "source must be FLAT")
        with torch.no_grad():
            self.in_proj_weight.copy_(flat.in_proj_weight)
            self.in_proj_bias.copy_(flat.in_proj_bias)
            self.out_proj_weight.copy_(flat.out_proj_weight)
            self.out_proj_bias.copy_(flat.out_proj_bias)

    def forward(
        self, slots: torch.Tensor, tokens: torch.Tensor, key_padding_mask: torch.Tensor,
        *, e0: torch.Tensor | None = None, functional: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, slots_n, _ = slots.shape
        units = tokens.size(1)
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(slots, q_weight, q_bias).view(batch, slots_n, self.heads, self.head_dim).transpose(1, 2)
        k = F.linear(tokens, k_weight, k_bias).view(batch, units, self.heads, self.head_dim).transpose(1, 2)
        v = F.linear(tokens, v_weight, v_bias).view(batch, units, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.routed:
            if e0 is None or functional is None:
                raise ValueError("ROUTE requires E0 and T calibration")
            logits = add_route_bias(logits, self.routing, e0, functional)
        logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attended = torch.softmax(logits, dim=-1)
        values = torch.matmul(attended, v).transpose(1, 2).contiguous().view(batch, slots_n, self.dim)
        return F.linear(values, self.out_proj_weight, self.out_proj_bias)


class FamilySetFrontend(nn.Module):
    """M2's existing frontend with an explicit mapped attention implementation."""

    def __init__(self, source: nn.Module, *, routed: bool) -> None:
        super().__init__()
        self.cfg = source.cfg
        self.local_conv = copy.deepcopy(source.local_conv)
        self.token_mlp = copy.deepcopy(source.token_mlp)
        self.slots = nn.Parameter(source.slots.detach().clone())
        self.slot_norm = copy.deepcopy(source.slot_norm)
        self.token_norm = copy.deepcopy(source.token_norm)
        self.attn = ExplicitSlotUnitAttention.from_mha(source.mha, routed=routed)
        self.slot_ffn_norm = copy.deepcopy(source.slot_ffn_norm)
        self.slot_ffn = copy.deepcopy(source.slot_ffn)
        self.slot_proj = copy.deepcopy(source.slot_proj)
        self.routed = routed

    def forward(self, x: torch.Tensor, bank: SessionBank, unit_keep: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch, width, units = x.shape
        local = self.local_conv(x)
        e0 = bank.E0.to(device=x.device, dtype=x.dtype)
        functional = bank.T.to(device=x.device, dtype=x.dtype)
        e0_expand = (e0.view(1, 1, units, cfg.identity_dim).expand(batch, width, units, cfg.identity_dim)
                     if e0.ndim == 2 else e0.unsqueeze(1).expand(batch, width, units, cfg.identity_dim))
        t_expand = (functional.view(1, 1, units, cfg.t4_dim).expand(batch, width, units, cfg.t4_dim)
                    if functional.ndim == 2 else functional.unsqueeze(1).expand(batch, width, units, cfg.t4_dim))
        tokens = self.token_norm(self.token_mlp(torch.cat([local, e0_expand, t_expand], dim=-1)))
        slots = self.slot_norm(self.slots).view(1, 1, cfg.slots, cfg.set_dim).expand(batch, width, cfg.slots, cfg.set_dim)
        q = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, units, cfg.set_dim)
        pad = (~unit_keep).unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        # Carrier is static for every time bin; repeat only the batch axis.
        carrier_e0 = e0 if e0.ndim == 3 else e0.unsqueeze(0).expand(batch, -1, -1)
        carrier_t = functional if functional.ndim == 3 else functional.unsqueeze(0).expand(batch, -1, -1)
        carrier_e0 = carrier_e0.unsqueeze(1).expand(batch, width, units, cfg.identity_dim).reshape(batch * width, units, cfg.identity_dim)
        carrier_t = carrier_t.unsqueeze(1).expand(batch, width, units, cfg.t4_dim).reshape(batch * width, units, cfg.t4_dim)
        out = self.attn(q, k, pad, e0=carrier_e0 if self.routed else None, functional=carrier_t if self.routed else None)
        out = q + out
        out = out + self.slot_ffn(self.slot_ffn_norm(out))
        return self.slot_proj(out.reshape(batch, width, cfg.slots * cfg.set_dim))


class M2FamilyDecoder(SmallTransformerDecoder):
    """Causal SMALL FLAT/ROUTE decoder initialized from one common base state."""

    def __init__(self, *, routed: bool, seed: int = 42, cfg: SmallConfig = SMALL) -> None:
        super().__init__(seed=seed, cfg=cfg)
        original = self.frontend
        self.frontend = FamilySetFrontend(original, routed=routed)
        self.routed = routed
        self.name = "CRST-B4-ROUTE" if routed else "CRST-B4-FLAT"

    def _fuse(self, x: torch.Tensor, bank: SessionBank, unit_mask: torch.Tensor | None, *, dropout_keep=None, dropout_generator=None) -> torch.Tensor:
        if dropout_keep is not None:
            keep = dropout_keep if dropout_keep.ndim != 1 else dropout_keep.unsqueeze(0).expand(x.size(0), -1)
        else:
            keep = _expand_unit_mask(unit_mask, bank, x.size(0))
            if self.training and self.unit_dropout_p > 0:
                keep = whole_unit_dropout(keep, p=self.unit_dropout_p, generator=dropout_generator)
        return self.frontend(x, bank, keep)


def make_paired_decoders(seed: int = 42, cfg: SmallConfig = SMALL) -> tuple[M2FamilyDecoder, M2FamilyDecoder]:
    """Make FLAT/ROUTE with all shared tensors byte-identical and ROUTE g=0."""
    flat = M2FamilyDecoder(routed=False, seed=seed, cfg=cfg)
    route = M2FamilyDecoder(routed=True, seed=seed, cfg=cfg)
    flat_state, route_state = flat.state_dict(), route.state_dict()
    with torch.no_grad():
        for name, value in route_state.items():
            if ".routing." not in name:
                require(name in flat_state, f"shared ROUTE key absent in FLAT: {name}")
                value.copy_(flat_state[name])
        route.frontend.attn.routing.g.zero_()
    return flat, route


def shared_parameter_max_abs_diff(flat: M2FamilyDecoder, route: M2FamilyDecoder) -> float:
    left, right = flat.state_dict(), route.state_dict()
    shared = [name for name in left if name in right and ".routing." not in name]
    return max(float((left[name] - right[name]).abs().max()) for name in shared)
