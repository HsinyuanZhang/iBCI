"""H1 W=700 temporal Transformer: FLAT slot attention and ROUTE calibration routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .h1_config import (
    FRONTEND_RNG_DOMAIN,
    H1_TEMPORAL,
    H1TemporalConfig,
    ROUTING_RNG_DOMAIN,
    TEMPORAL_RNG_DOMAIN,
    require,
)


def _sinusoidal_pe(max_len: int, width: int) -> torch.Tensor:
    pe = torch.zeros(max_len, width)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _kaiming_uniform_(tensor: torch.Tensor, generator: torch.Generator, a: float = math.sqrt(5)) -> None:
    if tensor.ndim < 2:
        bound = 1.0 / math.sqrt(max(tensor.numel(), 1))
    else:
        fan_in = tensor.size(1)
        for size in tensor.shape[2:]:
            fan_in *= size
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain / math.sqrt(max(fan_in, 1))
        bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def _reinit_param(name: str, param: torch.Tensor, generator: torch.Generator) -> None:
    if param.ndim >= 2:
        _kaiming_uniform_(param, generator)
        return
    fan = param.numel()
    bound = 1.0 / math.sqrt(max(fan, 1))
    with torch.no_grad():
        if "norm" in name and name.endswith("weight"):
            param.fill_(1.0)
        elif "norm" in name and name.endswith("bias"):
            param.zero_()
        elif name.endswith("bias"):
            param.uniform_(-bound, bound, generator=generator)
        else:
            param.uniform_(-bound, bound, generator=generator)


def whole_unit_dropout(
    unit_mask: torch.Tensor,
    p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if unit_mask.dtype != torch.bool:
        unit_mask = unit_mask.bool()
    orig = unit_mask if unit_mask.dim() == 2 else unit_mask.unsqueeze(0)
    batch, n_units = orig.shape
    if p <= 0.0:
        return orig.clone()
    if generator is None:
        rnd = torch.rand(batch, n_units, device=orig.device)
    else:
        rnd = torch.rand(batch, n_units, generator=generator).to(device=orig.device)
    keep = orig & ~(rnd < p)
    empty = keep.sum(dim=-1) == 0
    if empty.any():
        first = orig.to(dtype=torch.int64).argmax(dim=-1)
        rows = empty.nonzero(as_tuple=False).squeeze(-1)
        keep = keep.clone()
        keep[rows] = False
        keep[rows, first[rows]] = True
    return keep


class SharedCausalConv(nn.Module):
    def __init__(self, cfg: H1TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(1, cfg.conv_channels, kernel_size=cfg.conv_kernel, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = cfg.conv_kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n_units = x.shape
        hidden = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        hidden = F.pad(hidden, (self.left_pad, 0))
        hidden = self.act(self.conv(hidden))
        return hidden.reshape(batch, n_units, self.cfg.conv_channels, width).permute(0, 3, 1, 2)

    def forward_tiled(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Exact causal-conv tile with kernel-1 left overlap. Does not reset state."""
        left = self.left_pad
        raw_start = max(0, start - left)
        chunk = x[:, raw_start:end]
        if start == 0:
            local = self.forward(chunk)
            return local[:, : end - start]
        # Provide left context from previous raw bins; drop the overlap outputs.
        batch, width, n_units = chunk.shape
        hidden = chunk.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        hidden = self.act(self.conv(hidden))
        hidden = hidden.reshape(batch, n_units, self.cfg.conv_channels, width).permute(0, 3, 1, 2)
        drop = start - raw_start
        return hidden[:, drop:]


class FactoredTokenMLP(nn.Module):
    """Exact concat(local, E0, H-C) then Linear+GELU+Linear, without materializing concat."""

    def __init__(self, cfg: H1TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.token_in, cfg.set_dim)
        self.fc2 = nn.Linear(cfg.set_dim, cfg.set_dim)

    def forward(self, local: torch.Tensor, e0: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        weight = self.fc1.weight
        w_loc, w_e0, w_hc = weight.split([cfg.local_dim, cfg.e0_dim, cfg.hc_dim], dim=1)
        hidden = F.linear(local, w_loc, None)
        e0_term = F.linear(e0, w_e0, None)
        hc_term = F.linear(hc, w_hc, None)
        if e0_term.dim() == 2:
            e0_term = e0_term.unsqueeze(0).unsqueeze(1)
        elif e0_term.dim() == 3:
            e0_term = e0_term.unsqueeze(1)
        if hc_term.dim() == 2:
            hc_term = hc_term.unsqueeze(0).unsqueeze(1)
        elif hc_term.dim() == 3:
            hc_term = hc_term.unsqueeze(1)
        hidden = hidden + e0_term + hc_term + self.fc1.bias
        return self.fc2(F.gelu(hidden))

    def naive_concat(self, local: torch.Tensor, e0: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
        batch, width, n_units, _ = local.shape
        if e0.dim() == 2:
            e0 = e0.view(1, 1, n_units, self.cfg.e0_dim).expand(batch, width, n_units, self.cfg.e0_dim)
        elif e0.dim() == 3:
            e0 = e0.unsqueeze(1).expand(batch, width, n_units, self.cfg.e0_dim)
        if hc.dim() == 2:
            hc = hc.view(1, 1, n_units, self.cfg.hc_dim).expand(batch, width, n_units, self.cfg.hc_dim)
        elif hc.dim() == 3:
            hc = hc.unsqueeze(1).expand(batch, width, n_units, self.cfg.hc_dim)
        tokens = torch.cat([local, e0, hc], dim=-1)
        return self.fc2(F.gelu(self.fc1(tokens)))


class SlotUnitAttention(nn.Module):
    """Dynamic slot-to-unit attention, optional static calibration routing on logits."""

    def __init__(self, cfg: H1TemporalConfig, routed: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.routed = routed
        self.n_heads = cfg.heads
        self.head_dim = cfg.set_dim // cfg.heads
        require(self.head_dim * self.n_heads == cfg.set_dim, "set head dim")
        self.q_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.k_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.v_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.out_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        if routed:
            self.route_proj = nn.ModuleList(
                nn.Linear(cfg.e0_dim + cfg.hc_dim, cfg.route_key_dim) for _ in range(cfg.heads)
            )
            self.q_cal = nn.Parameter(torch.zeros(cfg.heads, cfg.slots, cfg.route_key_dim))
            self.g = nn.Parameter(torch.zeros(cfg.heads))

    def routing_bonus(self, e0: torch.Tensor, hc: torch.Tensor, batch: int) -> torch.Tensor:
        require(self.routed, "routing_bonus on FLAT")
        if e0.dim() == 2:
            e0 = e0.unsqueeze(0).expand(batch, -1, -1)
        if hc.dim() == 2:
            hc = hc.unsqueeze(0).expand(batch, -1, -1)
        joined = torch.cat([e0, hc], dim=-1)
        calibrated = F.layer_norm(joined, (joined.shape[-1],), weight=None, bias=None)
        parts = [proj(calibrated) for proj in self.route_proj]
        projected = torch.stack(parts, dim=1)  # [B, H, N, 32]
        scale = self.cfg.route_key_dim ** -0.5
        bonus = torch.einsum("hks,bhns->bhkn", self.q_cal, projected) * scale
        gate = torch.tanh(self.g).view(1, self.n_heads, 1, 1)
        return gate * bonus

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        unit_keep: torch.Tensor,
        e0: torch.Tensor | None = None,
        hc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_width, n_slots, dim = slots.shape
        n_units = tokens.size(1)
        query = self.q_proj(slots).view(batch_width, n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        if self.routed:
            require(e0 is not None and hc is not None, "ROUTE needs E0 and H-C")
            batch = unit_keep.size(0)
            width = batch_width // batch
            bonus = self.routing_bonus(e0, hc, batch)
            logits = logits + bonus.unsqueeze(1).expand(-1, width, -1, -1, -1).reshape(
                batch_width, self.n_heads, n_slots, n_units
            )
        pad = (~unit_keep).unsqueeze(1).expand(-1, batch_width // unit_keep.size(0), -1)
        pad = pad.reshape(batch_width, n_units)
        logits = logits.masked_fill(pad.unsqueeze(1).unsqueeze(2), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        hidden = torch.matmul(weights, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_width, n_slots, dim)
        return self.out_proj(hidden)


class SharedSetFrontend(nn.Module):
    def __init__(self, cfg: H1TemporalConfig, routed: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.routed = routed
        self.local_conv = SharedCausalConv(cfg)
        self.token_mlp = FactoredTokenMLP(cfg)
        self.slots = nn.Parameter(torch.zeros(cfg.slots, cfg.set_dim))
        self.slot_norm = nn.LayerNorm(cfg.set_dim)
        self.token_norm = nn.LayerNorm(cfg.set_dim)
        self.attn = SlotUnitAttention(cfg, routed=routed)
        self.slot_ffn_norm = nn.LayerNorm(cfg.set_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(cfg.set_dim, 4 * cfg.set_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.set_dim, cfg.set_dim),
        )
        self.slot_proj = nn.Linear(cfg.slots * cfg.set_dim, cfg.temporal_width)

    def _set_from_local(
        self,
        local: torch.Tensor,
        e0: torch.Tensor,
        hc: torch.Tensor,
        unit_keep: torch.Tensor,
    ) -> torch.Tensor:
        batch, width, n_units, _ = local.shape
        tokens = self.token_norm(self.token_mlp(local, e0, hc))
        slots = self.slot_norm(self.slots).view(1, 1, self.cfg.slots, self.cfg.set_dim)
        slots = slots.expand(batch, width, self.cfg.slots, self.cfg.set_dim)
        query = slots.reshape(batch * width, self.cfg.slots, self.cfg.set_dim)
        key = tokens.reshape(batch * width, n_units, self.cfg.set_dim)
        attended = self.attn(query, key, unit_keep, e0=e0, hc=hc)
        slots_out = query + attended
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        fused = slots_out.reshape(batch, width, self.cfg.slots * self.cfg.set_dim)
        return self.slot_proj(fused)

    def forward(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        hc: torch.Tensor,
        unit_keep: torch.Tensor,
        *,
        tile_size: int | None = None,
        use_checkpoint: bool = False,
        conv_full_window: bool = True,
    ) -> torch.Tensor:
        width = x.size(1)
        if tile_size is None or tile_size >= width:
            local = self.local_conv(x)
            return self._set_from_local(local, e0, hc, unit_keep)
        local_full = self.local_conv(x) if conv_full_window else None
        chunks: list[torch.Tensor] = []
        for start in range(0, width, tile_size):
            end = min(width, start + tile_size)
            if conv_full_window:
                assert local_full is not None
                local = local_full[:, start:end]
            else:
                local = self.local_conv.forward_tiled(x, start, end)
            if use_checkpoint and local.requires_grad:
                chunk = activation_checkpoint(
                    self._set_from_local, local, e0, hc, unit_keep, use_reentrant=False
                )
            else:
                chunk = self._set_from_local(local, e0, hc, unit_keep)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)


def _sdpa_causal(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, past_len: int) -> torch.Tensor:
    q_len = query.size(2)
    k_len = key.size(2)
    if past_len == 0 and q_len == k_len:
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=True)
    q_pos = torch.arange(q_len, device=query.device) + past_len
    k_pos = torch.arange(k_len, device=query.device)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    bias = torch.zeros(q_len, k_len, device=query.device, dtype=query.dtype)
    bias.masked_fill_(~allow, float("-inf"))
    return F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: H1TemporalConfig) -> None:
        super().__init__()
        self.n_heads = cfg.heads
        self.head_dim = cfg.temporal_width // cfg.heads
        require(self.head_dim * self.n_heads == cfg.temporal_width, "transformer head dim")
        self.qkv = nn.Linear(cfg.temporal_width, 3 * cfg.temporal_width)
        self.proj = nn.Linear(cfg.temporal_width, cfg.temporal_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, dim = x.shape
        qkv = self.qkv(x).view(batch, width, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = _sdpa_causal(query, key, value, past_len=0)
        return self.proj(out.transpose(1, 2).contiguous().view(batch, width, dim))


class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: H1TemporalConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.temporal_width)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.temporal_width)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.ffn),
            nn.GELU(),
            nn.Linear(cfg.ffn, cfg.temporal_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class CausalTransformerStack(nn.Module):
    def __init__(self, cfg: H1TemporalConfig) -> None:
        super().__init__()
        require(cfg.pe_max_len >= cfg.window, "PE length must cover W=700")
        self.blocks = nn.ModuleList(CausalTransformerBlock(cfg) for _ in range(cfg.layers))
        self.register_buffer("pe", _sinusoidal_pe(cfg.pe_max_len, cfg.temporal_width), persistent=False)
        self.max_len = cfg.pe_max_len

    def forward(self, x: torch.Tensor, *, use_checkpoint: bool = False) -> torch.Tensor:
        width = x.size(1)
        require(width <= self.pe.size(0), f"positional encoding overflow: {width}>{self.pe.size(0)}")
        hidden = x + self.pe[:width].unsqueeze(0).to(dtype=x.dtype)
        for block in self.blocks:
            if use_checkpoint and hidden.requires_grad:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return hidden


@dataclass
class H1Bank:
    E0: torch.Tensor
    T: torch.Tensor
    unit_mask: torch.Tensor


def _expand_keep(
    unit_mask: torch.Tensor | None,
    bank: H1Bank,
    batch: int,
    *,
    dropout_p: float,
    training: bool,
    dropout_generator: torch.Generator | None,
) -> torch.Tensor:
    keep = bank.unit_mask if unit_mask is None else unit_mask
    if keep.dtype != torch.bool:
        keep = keep.bool()
    if keep.dim() == 1:
        keep = keep.unsqueeze(0).expand(batch, -1)
    keep = keep.contiguous()
    if training and dropout_p > 0.0:
        keep = whole_unit_dropout(keep, p=dropout_p, generator=dropout_generator)
    return keep


class _H1TemporalBase(nn.Module):
    name = "H1-TEMPORAL"
    training_target_space = "decoder_native"

    def __init__(self, cfg: H1TemporalConfig, routed: bool, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.routed = routed
        require(cfg.pe_max_len >= cfg.window, "PE must cover the configured window")
        require(cfg.out_dim > 0, "output dim")
        self.frontend = SharedSetFrontend(cfg, routed=routed)
        self.temporal = CausalTransformerStack(cfg)
        self.final_norm = nn.LayerNorm(cfg.temporal_width)
        self.readout = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(cfg.readout_hidden, cfg.out_dim),
        )
        self.unit_dropout_p = cfg.unit_dropout
        self._tile_size: int | None = None
        self._use_checkpoint = False

    def set_memory_policy(self, tile_size: int | None, use_checkpoint: bool) -> None:
        if tile_size is not None:
            require(tile_size > 0, "tile_size")
            require(self.cfg.slots == 8, "tile path must keep 8 slots")
        self._tile_size = tile_size
        self._use_checkpoint = bool(use_checkpoint)

    def _fuse(
        self,
        x: torch.Tensor,
        bank: H1Bank,
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
            keep = _expand_keep(
                unit_mask,
                bank,
                x.size(0),
                dropout_p=self.unit_dropout_p,
                training=self.training,
                dropout_generator=dropout_generator,
            )
        e0 = bank.E0.to(device=x.device, dtype=x.dtype)
        hc = bank.T.to(device=x.device, dtype=x.dtype)
        return self.frontend(
            x,
            e0,
            hc,
            keep,
            tile_size=self._tile_size,
            use_checkpoint=self._use_checkpoint,
        )

    def forward_hidden(
        self,
        x: torch.Tensor,
        bank: H1Bank,
        unit_mask: torch.Tensor | None = None,
        *,
        dropout_keep: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        fused = self._fuse(
            x, bank, unit_mask, dropout_keep=dropout_keep, dropout_generator=dropout_generator
        )
        return self.temporal(fused, use_checkpoint=self._use_checkpoint)

    def forward_last(
        self,
        x: torch.Tensor,
        bank: H1Bank,
        unit_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(x, bank, unit_mask, **kwargs)
        return self.readout(self.final_norm(hidden))[:, -1, :]

    def forward_scores(
        self,
        x: torch.Tensor,
        bank: H1Bank,
        unit_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(x, bank, unit_mask, **kwargs)
        return self.readout(self.final_norm(hidden))

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def routing_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad and ("routing" in name or ".attn.route_" in name or ".attn.q_cal" in name or ".attn.g" in name)
        }


class H1TemporalFlatDecoder(_H1TemporalBase):
    name = "H1-TEMPORAL-TRF-FLAT"

    def __init__(self, seed: int = 42, cfg: H1TemporalConfig | None = None) -> None:
        super().__init__(cfg or H1_TEMPORAL, routed=False, seed=seed)
        initialize_h1_flat(self, seed)


class H1TemporalRouteDecoder(_H1TemporalBase):
    name = "H1-TEMPORAL-TRF-ROUTE"

    def __init__(
        self,
        seed: int = 42,
        cfg: H1TemporalConfig | None = None,
        *,
        flat_template: H1TemporalFlatDecoder | None = None,
    ) -> None:
        super().__init__(cfg or H1_TEMPORAL, routed=True, seed=seed)
        if flat_template is not None:
            copy_flat_into_route(flat_template, self)
            initialize_route_only(self, seed)
        else:
            initialize_h1_flat(self, seed)
            initialize_route_only(self, seed)


def initialize_h1_flat(module: nn.Module, seed: int) -> None:
    front_g = torch.Generator(device="cpu").manual_seed(int(seed) + FRONTEND_RNG_DOMAIN)
    temp_g = torch.Generator(device="cpu").manual_seed(int(seed) + TEMPORAL_RNG_DOMAIN)
    for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
        if _is_routing_name(name):
            continue
        generator = temp_g if name.startswith("temporal.") else front_g
        if param.device.type != "cpu":
            cpu = param.detach().cpu().clone()
            _reinit_param(name, cpu, generator)
            with torch.no_grad():
                param.copy_(cpu.to(device=param.device, dtype=param.dtype))
        else:
            _reinit_param(name, param, generator)
    if hasattr(module, "frontend") and hasattr(module.frontend, "slots"):
        slot_g = torch.Generator(device="cpu").manual_seed(int(seed) + FRONTEND_RNG_DOMAIN + 17)
        slots = module.frontend.slots
        noise = torch.randn(slots.shape, generator=slot_g, dtype=torch.float32)
        with torch.no_grad():
            slots.copy_((noise * 0.02).to(device=slots.device, dtype=slots.dtype))


def initialize_route_only(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + ROUTING_RNG_DOMAIN)
    for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
        if not _is_routing_name(name):
            continue
        if name.endswith(".attn.g") or name.endswith(".g"):
            with torch.no_grad():
                param.zero_()
            continue
        if param.device.type != "cpu":
            cpu = param.detach().cpu().clone()
            _reinit_param(name, cpu, generator)
            with torch.no_grad():
                param.copy_(cpu.to(device=param.device, dtype=param.dtype))
        else:
            _reinit_param(name, param, generator)


def _is_routing_name(name: str) -> bool:
    return (
        ".attn.route_proj" in name
        or name.endswith(".attn.q_cal")
        or name.endswith(".attn.g")
        or ".routing." in name
    )


def copy_flat_into_route(flat: H1TemporalFlatDecoder, route: H1TemporalRouteDecoder) -> None:
    flat_state = flat.state_dict()
    route_state = route.state_dict()
    copied = 0
    with torch.no_grad():
        for name, tensor in route_state.items():
            if _is_routing_name(name):
                continue
            require(name in flat_state, f"ROUTE missing FLAT key {name}")
            tensor.copy_(flat_state[name])
            copied += 1
    require(copied > 0, "copied no FLAT keys")


def shared_parameter_max_abs_diff(flat: nn.Module, route: nn.Module) -> float:
    worst = 0.0
    route_params = dict(route.named_parameters())
    for name, param in flat.named_parameters():
        if _is_routing_name(name):
            continue
        require(name in route_params, f"shared name missing on ROUTE: {name}")
        worst = max(worst, float((param.detach() - route_params[name].detach()).abs().max().item()))
    return worst


def prove_zero_gate_equals_flat(
    flat: H1TemporalFlatDecoder,
    route: H1TemporalRouteDecoder,
    x: torch.Tensor,
    bank: H1Bank,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, float]:
    require(float(route.frontend.attn.g.abs().max().item()) == 0.0, "g_h must be zero for the proof")
    was_train = flat.training
    flat.eval()
    route.eval()
    try:
        with torch.no_grad():
            left = flat.forward_last(x, bank)
            right = route.forward_last(x, bank)
        delta = float((left - right).abs().max().item())
        require(torch.allclose(left, right, atol=atol, rtol=rtol), f"zero-gate ROUTE != FLAT maxabs={delta}")
        return {"max_abs_diff": delta, "atol": atol, "rtol": rtol}
    finally:
        flat.train(was_train)
        route.train(was_train)


def count_h1_decoder_parameters(module: nn.Module) -> dict[str, int]:
    named = dict(module.named_parameters())
    total = int(sum(p.numel() for p in named.values()))
    trainable = int(sum(p.numel() for p in named.values() if p.requires_grad))
    temporal = int(sum(p.numel() for n, p in named.items() if n.startswith("temporal.")))
    routing = int(sum(p.numel() for n, p in named.items() if _is_routing_name(n)))
    return {
        "decoder_total": total,
        "trainable": trainable,
        "temporal": temporal,
        "routing": routing,
        "frontend_plus_readout": total - temporal,
        "named_count": int(len(named)),
    }


def adamw_param_groups(module: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    nodecay: list[nn.Parameter] = []
    for name, param in module.trainable_parameters().items():
        if not param.requires_grad:
            continue
        is_bias = name.endswith("bias")
        is_norm = "norm" in name.lower()
        if is_bias or is_norm:
            nodecay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
