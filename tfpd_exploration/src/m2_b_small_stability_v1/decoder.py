"""Local small Transformer decoder with explicit widths.

Adapted from ``m2_dual_track_v1.decoders`` (read-only). Widths come from
``SmallConfig`` only — this module never assigns ``plan.B_TEMPORAL_WIDTH``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank
from tfpd_exploration.src.m2_dual_track_v1.decoders import initialize_decoder, whole_unit_dropout

from .config import SHARED_CALIB_PARAMS, SMALL, SmallConfig, require

# Re-export the frozen historical init domains (pure functions from the old package).
FRONTEND_RNG_DOMAIN = 0
TEMPORAL_RNG_DOMAIN = 1_000_003


def _expand_unit_mask(unit_mask: torch.Tensor | None, bank: SessionBank, batch: int) -> torch.Tensor:
    if unit_mask is None:
        unit_mask = bank.unit_mask
    if unit_mask.dtype != torch.bool:
        unit_mask = unit_mask.bool()
    if unit_mask.dim() == 1:
        unit_mask = unit_mask.unsqueeze(0).expand(batch, -1)
    return unit_mask.contiguous()


def _sinusoidal_pe(max_len: int, width: int) -> torch.Tensor:
    pe = torch.zeros(max_len, width)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _sdpa_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int) -> torch.Tensor:
    q_len = q.size(2)
    k_len = k.size(2)
    if past_len == 0 and q_len == k_len:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    q_pos = torch.arange(q_len, device=q.device) + past_len
    k_pos = torch.arange(k_len, device=q.device)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    bias = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
    bias.masked_fill_(~allow, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)


class SharedCausalConv(nn.Module):
    def __init__(self, cfg: SmallConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(1, cfg.conv_channels, kernel_size=cfg.conv_kernel, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = cfg.conv_kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n_units = x.shape
        h = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        h = F.pad(h, (self.left_pad, 0))
        h = self.act(self.conv(h))
        return h.reshape(batch, n_units, self.cfg.conv_channels, width).permute(0, 3, 1, 2)


class SharedSetFrontend(nn.Module):
    def __init__(self, cfg: SmallConfig) -> None:
        super().__init__()
        self.cfg = cfg
        require(cfg.slot_proj_in == cfg.slots * cfg.set_dim, "slot_proj_in must be slots*set_dim")
        require(cfg.slot_proj_out == cfg.temporal_width, "slot_proj_out must equal temporal_width")
        self.local_conv = SharedCausalConv(cfg)
        self.token_mlp = nn.Sequential(
            nn.Linear(cfg.token_in, cfg.set_dim),
            nn.GELU(),
            nn.Linear(cfg.set_dim, cfg.set_dim),
        )
        self.slots = nn.Parameter(torch.zeros(cfg.slots, cfg.set_dim))
        self.slot_norm = nn.LayerNorm(cfg.set_dim)
        self.token_norm = nn.LayerNorm(cfg.set_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.set_dim,
            num_heads=cfg.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.slot_ffn_norm = nn.LayerNorm(cfg.set_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(cfg.set_dim, 4 * cfg.set_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.set_dim, cfg.set_dim),
        )
        self.slot_proj = nn.Linear(cfg.slot_proj_in, cfg.slot_proj_out)

    def forward(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_keep: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch, width, n_units = x.shape
        local = self.local_conv(x)
        e0 = bank.E0.to(device=x.device, dtype=x.dtype)
        t4 = bank.T.to(device=x.device, dtype=x.dtype)
        if e0.dim() == 2:
            e0 = e0.view(1, 1, n_units, cfg.identity_dim).expand(batch, width, n_units, cfg.identity_dim)
        else:
            e0 = e0.unsqueeze(1).expand(batch, width, n_units, cfg.identity_dim)
        if t4.dim() == 2:
            t4 = t4.view(1, 1, n_units, cfg.t4_dim).expand(batch, width, n_units, cfg.t4_dim)
        else:
            t4 = t4.unsqueeze(1).expand(batch, width, n_units, cfg.t4_dim)
        tokens = self.token_mlp(torch.cat([local, e0, t4], dim=-1))
        tokens = self.token_norm(tokens)
        slots = self.slot_norm(self.slots).view(1, 1, cfg.slots, cfg.set_dim).expand(
            batch, width, cfg.slots, cfg.set_dim
        )
        q = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, n_units, cfg.set_dim)
        pad = (~unit_keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
        attn_out, _ = self.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        fused = slots_out.reshape(batch, width, cfg.slots * cfg.set_dim)
        return self.slot_proj(fused)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SmallConfig) -> None:
        super().__init__()
        self.n_heads = cfg.heads
        self.head_dim = cfg.temporal_width // cfg.heads
        require(self.head_dim * self.n_heads == cfg.temporal_width, "transformer head dim")
        self.qkv = nn.Linear(cfg.temporal_width, 3 * cfg.temporal_width)
        self.proj = nn.Linear(cfg.temporal_width, cfg.temporal_width)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, width, dim = x.shape
        qkv = self.qkv(x).view(batch, width, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = _sdpa_causal(q, k, v, past_len=past_len)
        out = out.transpose(1, 2).contiguous().view(batch, width, dim)
        return self.proj(out), (k, v)


class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: SmallConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.temporal_width)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.temporal_width)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.ffn),
            nn.GELU(),
            nn.Linear(cfg.ffn, cfg.temporal_width),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, cache = self.attn(self.norm1(x), kv_cache=kv_cache, past_len=past_len)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, cache


class CausalTransformerStack(nn.Module):
    def __init__(self, cfg: SmallConfig, max_len: int = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalTransformerBlock(cfg) for _ in range(cfg.layers))
        self.register_buffer("pe", _sinusoidal_pe(max_len, cfg.temporal_width), persistent=False)
        self.max_len = max_len

    def _add_pe(self, x: torch.Tensor, start: int) -> torch.Tensor:
        width = x.size(1)
        require(start + width <= self.pe.size(0), "positional encoding overflow")
        return x + self.pe[start : start + width].unsqueeze(0).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._add_pe(x, 0)
        for block in self.blocks:
            h, _ = block(h, kv_cache=None, past_len=0)
        return h


class SmallTransformerDecoder(nn.Module):
    """S0/S1 decoder: 256-wide temporal stack, 8 slots, 70-d tokens."""

    name = "S-SMALL-TRANSFORMER"
    training_target_space = "decoder_raw"

    def __init__(self, seed: int = 42, cfg: SmallConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SMALL
        require(self.cfg.temporal_width == 256, "small decoder must be 256-wide")
        require(self.cfg.ffn == 512, "small FFN must be 512")
        self.frontend = SharedSetFrontend(self.cfg)
        self.final_norm = nn.LayerNorm(self.cfg.temporal_width)
        hidden_in, hidden_mid, out_dim = self.cfg.readout
        require(hidden_in == self.cfg.temporal_width, "readout in-width")
        self.readout = nn.Sequential(
            nn.Linear(hidden_in, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, out_dim),
        )
        self.temporal = CausalTransformerStack(self.cfg)
        self.unit_dropout_p = self.cfg.unit_dropout
        initialize_decoder(self, seed)

    def _fuse(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None,
        *,
        dropout_keep: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if dropout_keep is not None:
            keep = dropout_keep
            if keep.dim() == 1:
                keep = keep.unsqueeze(0).expand(x.size(0), -1)
        else:
            keep = _expand_unit_mask(unit_mask, bank, x.size(0))
            if self.training and self.unit_dropout_p > 0.0:
                keep = whole_unit_dropout(keep, p=self.unit_dropout_p, generator=dropout_generator)
        return self.frontend(x, bank, keep)

    def _readout_all(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(self.final_norm(hidden))

    def forward_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        *,
        dropout_keep: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.temporal(
            self._fuse(
                x, bank, unit_mask, dropout_keep=dropout_keep, dropout_generator=dropout_generator
            )
        )

    def forward_last(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        *,
        dropout_keep: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(
            x, bank, unit_mask, dropout_keep=dropout_keep, dropout_generator=dropout_generator
        )
        return self._readout_all(hidden)[:, -1, :]

    def forward_scores(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        *,
        dropout_keep: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._readout_all(
            self.forward_hidden(
                x, bank, unit_mask, dropout_keep=dropout_keep, dropout_generator=dropout_generator
            )
        )

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        return {name: param for name, param in self.named_parameters() if param.requires_grad}


def count_decoder_parameters(module: nn.Module) -> dict[str, int]:
    named = dict(module.named_parameters())
    total = int(sum(p.numel() for p in named.values()))
    temporal = int(sum(p.numel() for n, p in named.items() if n.startswith("temporal.")))
    front = int(sum(p.numel() for n, p in named.items() if not n.startswith("temporal.")))
    return {
        "decoder_total": total,
        "temporal": temporal,
        "frontend_plus_readout": front,
        "shared_calib": SHARED_CALIB_PARAMS,
        "decoder_plus_shared_calib": total + SHARED_CALIB_PARAMS,
        "named_count": int(len(named)),
    }


def init_state_sha256(module: nn.Module) -> str:
    return tensor_state_sha256({name: param.detach() for name, param in module.named_parameters()})


__all__ = [
    "SmallTransformerDecoder",
    "count_decoder_parameters",
    "init_state_sha256",
    "whole_unit_dropout",
]
