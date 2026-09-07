"""Exact lifted spatial attention for common unscaled H1 FLAT/ROUTE.

An immutable inference snapshot: its owning streaming engine reconstructs it
after model/bank mutation. No change to weights, mode, masking or temperature.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .grouped_value import grouped_value_projection
from .linear_conv import repaired_local_features_linear
from tfpd_exploration.src.h1_family_v1.model import UnscaledDotSlotUnitAttention
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


class H1LiftedSpatialFrontend:
    @torch.no_grad()
    def __init__(self, model, bank: H1Bank):
        self.model, self.bank, self.frontend = model, bank, model.frontend
        first_parameter = next(model.parameters())
        if model.training or first_parameter.device.type != "cpu" or first_parameter.dtype != torch.float32:
            raise RuntimeError("H1 spatial lift requires CPU float32 eval model")
        front, cfg = self.frontend, self.frontend.cfg
        attn = front.attn
        if not isinstance(attn, UnscaledDotSlotUnitAttention):
            raise TypeError("common unscaled H1 attention required")
        if (cfg.local_dim, cfg.e0_dim, cfg.hc_dim, cfg.conv_kernel) != (16, 700, 4, 5):
            raise ValueError("H1 frontend geometry drift")
        if bank.E0.shape[-2:] != (cfg.n_units, cfg.e0_dim) or bank.T.shape[-2:] != (cfg.n_units, cfg.hc_dim):
            raise ValueError("H1 bank geometry drift")
        if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in (bank.E0, bank.T)):
            raise TypeError("H1 bank must be CPU float32")
        if bank.unit_mask.device.type != "cpu" or bank.unit_mask.dtype != torch.bool:
            raise TypeError("H1 mask must be CPU bool")
        self.heads, self.head_dim, self.slots, self.dim = attn.n_heads, attn.head_dim, cfg.slots, cfg.set_dim
        self.local_weight, e0_weight, hc_weight = front.token_mlp.fc1.weight.split([16, 700, 4], dim=1)
        # Preserve native left-associated additions by retaining separate terms.
        self.e0_term = F.linear(bank.E0, e0_weight, None)
        self.hc_term = F.linear(bank.T, hc_weight, None)
        self.slot_query = front.slot_norm(front.slots)
        query = attn.q_proj(self.slot_query).view(self.slots, self.heads, self.head_dim).transpose(0, 1)
        key_weight = attn.k_proj.weight.view(self.heads, self.head_dim, self.dim)
        key_bias = attn.k_proj.bias.view(self.heads, self.head_dim)
        self.qwk = torch.matmul(query, key_weight)
        self.qbk = torch.einsum("hsd,hd->hs", query, key_bias)
        self.value_weight = attn.v_proj.weight.view(self.heads, self.head_dim, self.dim).transpose(-1, -2)
        self.value_bias = attn.v_proj.bias.view(self.heads, 1, self.head_dim)
        bank_batch = bank.E0.shape[0] if bank.E0.ndim == 3 else 1
        self.route_bonus = attn.routing_bonus(bank.E0, bank.T, bank_batch) if attn.routed else None

    @staticmethod
    def _broadcast_static(value):
        return value[None, None] if value.ndim == 2 else value[:, None]

    @torch.no_grad()
    def encode_local(self, local):
        batch, width, units, local_dim = local.shape
        if units != self.frontend.cfg.n_units or local_dim != 16:
            raise ValueError("H1 local feature shape drift")
        front, attn = self.frontend, self.frontend.attn
        hidden = F.linear(local, self.local_weight, None)
        hidden = hidden + self._broadcast_static(self.e0_term)
        hidden = hidden + self._broadcast_static(self.hc_term) + front.token_mlp.fc1.bias
        tokens = front.token_norm(front.token_mlp.fc2(F.gelu(hidden)))
        tokens = tokens.reshape(batch * width, units, self.dim)
        logits = F.linear(tokens, self.qwk.reshape(self.heads * self.slots, self.dim))
        logits = logits.view(batch * width, units, self.heads, self.slots).permute(0, 2, 3, 1)
        logits = (logits + self.qbk[None, :, :, None]) * (self.head_dim ** -0.5)
        if self.route_bonus is not None:
            bonus = self.route_bonus.expand(batch, -1, -1, -1)
            logits = logits + bonus[:, None].expand(-1, width, -1, -1, -1).reshape_as(logits)
        # Retain both scale operations, with ROUTE bonus between them.
        logits = logits * (self.head_dim ** 0.5)
        keep = self.bank.unit_mask
        if keep.ndim == 1:
            keep = keep[None].expand(batch, -1)
        if keep.shape != (batch, units):
            raise ValueError("H1 batch/mask roster drift")
        pad = (~keep)[:, None].expand(batch, width, units).reshape(batch * width, units)
        weights = torch.softmax(logits.masked_fill(pad[:, None, None], float("-inf")), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attended = torch.bmm(weights.reshape(batch * width, self.heads * self.slots, units), tokens)
        attended = attended.reshape(batch * width, self.heads, self.slots, self.dim)
        values = grouped_value_projection(attended, self.value_weight)
        values = values + weights.sum(-1, keepdim=True) * self.value_bias[None]
        merged = values.transpose(1, 2).contiguous().reshape(batch * width, self.slots, self.dim)
        query = self.slot_query[None].expand(batch * width, -1, -1)
        slots = query + attn.out_proj(merged)
        slots = slots + front.slot_ffn(front.slot_ffn_norm(slots))
        return front.slot_proj(slots.reshape(batch, width, self.slots * self.dim))

    @torch.no_grad()
    def full(self, raw):
        if raw.ndim != 3 or raw.dtype != torch.float32 or raw.device.type != "cpu":
            raise TypeError("CPU float32 [B,T,N] required")
        return self.encode_local(self.frontend.local_conv(raw))

    @torch.no_grad()
    def repair(self, raw):
        if raw.ndim != 3 or raw.shape[1] < 9:
            raise ValueError("five-token spatial repair requires width at least 9")
        return self.encode_local(repaired_local_features_linear(self.frontend.local_conv, raw))
