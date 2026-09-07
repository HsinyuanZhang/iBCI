"""B-track temporal decoders: shared set frontend + Mamba2 or causal Transformer.

Initialization domains (seed-identical D0/D1 frontend)
------------------------------------------------------
``FRONTEND_RNG_DOMAIN = 0`` and ``TEMPORAL_RNG_DOMAIN = 1_000_003``.

``initialize_decoder(module, seed)`` draws two isolated ``torch.Generator``s:

* frontend + shared readout: ``Generator.manual_seed(seed + FRONTEND_RNG_DOMAIN)``
* temporal stack:            ``Generator.manual_seed(seed + TEMPORAL_RNG_DOMAIN)``

Parameters are visited in sorted ``named_parameters()`` order so sibling
construction order cannot leak across domains. D0 (Transformer) and D1
(Mamba) therefore receive bitwise-identical frontend/readout tensors when
constructed with the same ``seed``.

Mamba2 mixer dynamics (``A_log``, ``dt_bias``, ``D``) keep the official
reference initialization from the Mamba2 constructor. Those parameters are
created under a temporarily seeded global RNG at ``seed + TEMPORAL_RNG_DOMAIN``
and are **not** overwritten by the generic Linear/Conv reinit. Official
``_no_weight_decay`` tags on those tensors are preserved.

No internal-state clamp is applied to any linear SSM path.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import plan
from .contracts import SessionBank
from .ssm_backend import (
    BLOCK_REASON,
    MAMBA_AVAILABLE,
    InferenceParams,
    Mamba2,
)

FRONTEND_RNG_DOMAIN = 0
TEMPORAL_RNG_DOMAIN = 1_000_003

FRONTEND_PREFIXES = ("frontend.", "final_norm.", "readout.")
TEMPORAL_PREFIXES = ("temporal.",)

_LOCAL_DIM = plan.B_CONV_CHANNELS
_TOKEN_IN = _LOCAL_DIM + plan.IDENTITY_DIM + plan.T4_DIM  # 16+50+4=70


def whole_unit_dropout(
    unit_mask: torch.Tensor,
    p: float = plan.B_UNIT_DROPOUT,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop whole units for one window; the mask has no time axis.

    ``unit_mask`` is ``[N]`` or ``[B, N]`` (True = eligible). Each window
    drops units independently with probability ``p``. If every remaining
    unit is dropped, restore the lowest originally-valid index. If the
    incoming mask is empty, restore index 0.
    """
    if unit_mask.dtype != torch.bool:
        unit_mask = unit_mask.bool()
    orig = unit_mask if unit_mask.dim() == 2 else unit_mask.unsqueeze(0)
    batch, n_units = orig.shape
    if p <= 0.0:
        return orig.clone()
    if p >= 1.0:
        drop = torch.ones(batch, n_units, dtype=torch.bool, device=orig.device)
    else:
        # Draw on CPU when a Generator is supplied so CUDA tensors stay valid.
        if generator is None:
            rnd = torch.rand(batch, n_units, device=orig.device)
        else:
            rnd = torch.rand(batch, n_units, generator=generator).to(device=orig.device)
        drop = rnd < p
    keep = orig & ~drop
    empty = keep.sum(dim=-1) == 0
    if empty.any():
        # Lowest originally-valid index; argmax on bool is first True, else 0.
        first = orig.to(dtype=torch.int64).argmax(dim=-1)
        rows = empty.nonzero(as_tuple=False).squeeze(-1)
        keep = keep.clone()
        keep[rows] = False
        keep[rows, first[rows]] = True
    return keep


def _expand_unit_mask(unit_mask: torch.Tensor | None, bank: SessionBank, batch: int) -> torch.Tensor:
    if unit_mask is None:
        unit_mask = bank.unit_mask
    if unit_mask.dtype != torch.bool:
        unit_mask = unit_mask.bool()
    if unit_mask.dim() == 1:
        unit_mask = unit_mask.unsqueeze(0).expand(batch, -1)
    return unit_mask.contiguous()


def _kaiming_uniform_(tensor: torch.Tensor, generator: torch.Generator, a: float = math.sqrt(5)) -> None:
    if tensor.ndim < 2:
        bound = 1.0 / math.sqrt(max(tensor.numel(), 1))
    else:
        fan_in = tensor.size(1)
        for s in tensor.shape[2:]:
            fan_in *= s
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain / math.sqrt(max(fan_in, 1))
        bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def _reinit_param(name: str, param: torch.Tensor, generator: torch.Generator) -> None:
    if param.ndim >= 2:
        _kaiming_uniform_(param, generator)
        return
    if name.endswith("bias") or param.ndim == 1:
        if "norm" in name or name.endswith("weight") and param.numel() > 1 and "slots" not in name:
            # LayerNorm / RMS-style affine: weight=1, bias=0 when 1-D and named as such.
            pass
        fan = param.numel()
        bound = 1.0 / math.sqrt(max(fan, 1))
        with torch.no_grad():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.0)
            elif "norm" in name and name.endswith("bias"):
                param.zero_()
            else:
                param.uniform_(-bound, bound, generator=generator)
        return


def _is_frontend_name(name: str) -> bool:
    return name.startswith(FRONTEND_PREFIXES)


def _is_temporal_name(name: str) -> bool:
    return name.startswith(TEMPORAL_PREFIXES)


def _is_mamba_dynamics(name: str) -> bool:
    return name.endswith("A_log") or name.endswith(".D") or name.endswith("dt_bias") or name.endswith(".D")


def initialize_decoder(module: nn.Module, seed: int) -> None:
    """Re-initialize frontend/readout and (non-dynamics) temporal params.

    Mamba2 ``A_log`` / ``dt_bias`` / ``D`` keep constructor values.
    """
    front_g = torch.Generator(device="cpu").manual_seed(int(seed) + FRONTEND_RNG_DOMAIN)
    temp_g = torch.Generator(device="cpu").manual_seed(int(seed) + TEMPORAL_RNG_DOMAIN)
    for name, param in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        if _is_mamba_dynamics(name):
            continue
        if param.device.type != "cpu":
            cpu = param.detach().cpu().clone()
            if _is_frontend_name(name):
                _reinit_param(name, cpu, front_g)
            elif _is_temporal_name(name):
                _reinit_param(name, cpu, temp_g)
            else:
                _reinit_param(name, cpu, front_g)
            with torch.no_grad():
                param.copy_(cpu.to(device=param.device, dtype=param.dtype))
            continue
        if _is_frontend_name(name):
            _reinit_param(name, param, front_g)
        elif _is_temporal_name(name):
            _reinit_param(name, param, temp_g)
        else:
            _reinit_param(name, param, front_g)
    # Learned slots: small Gaussian in the frontend domain (re-draw with a
    # dedicated sub-seed so the Linear stream stays independent of slot scale).
    if hasattr(module, "frontend") and hasattr(module.frontend, "slots"):
        slot_g = torch.Generator(device="cpu").manual_seed(int(seed) + FRONTEND_RNG_DOMAIN + 17)
        slots = module.frontend.slots
        cpu = torch.randn(slots.shape, generator=slot_g, dtype=torch.float32)
        with torch.no_grad():
            slots.copy_((cpu * 0.02).to(device=slots.device, dtype=slots.dtype))


def count_trainable_parameters(module: nn.Module) -> dict[str, int]:
    named = dict(module.named_parameters())
    allow = module.trainable_parameters() if hasattr(module, "trainable_parameters") else named
    total = sum(p.numel() for p in allow.values())
    front = sum(p.numel() for n, p in allow.items() if _is_frontend_name(n))
    temporal = sum(p.numel() for n, p in allow.items() if _is_temporal_name(n))
    return {
        "trainable_total": int(total),
        "frontend_plus_readout": int(front),
        "temporal": int(temporal),
        "named_count": int(len(allow)),
    }


def adamw_param_groups(module: nn.Module) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    nodecay: list[nn.Parameter] = []
    for name, param in module.trainable_parameters().items():
        if not param.requires_grad:
            continue
        no_wd = bool(getattr(param, "_no_weight_decay", False))
        is_bias = name.endswith("bias")
        is_norm = "norm" in name.lower()
        if no_wd or is_bias or is_norm:
            nodecay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": plan.B_WEIGHT_DECAY},
        {"params": nodecay, "weight_decay": 0.0},
    ]


class SharedCausalConv(nn.Module):
    """Shared causal Conv1d(1, 16, k=5)+SiLU applied independently per unit."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, plan.B_CONV_CHANNELS, kernel_size=plan.B_CONV_KERNEL, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = plan.B_CONV_KERNEL - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] -> [B, T, N, 16]
        batch, width, n_units = x.shape
        h = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        h = F.pad(h, (self.left_pad, 0))
        h = self.act(self.conv(h))
        return h.reshape(batch, n_units, plan.B_CONV_CHANNELS, width).permute(0, 3, 1, 2)


class SharedSetFrontend(nn.Module):
    """Per-bin set encoder: local conv + E0/T fusion + 8-slot cross-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.local_conv = SharedCausalConv()
        self.token_mlp = nn.Sequential(
            nn.Linear(_TOKEN_IN, plan.B_SET_DIM),
            nn.GELU(),
            nn.Linear(plan.B_SET_DIM, plan.B_SET_DIM),
        )
        self.slots = nn.Parameter(torch.zeros(plan.B_SLOTS, plan.B_SET_DIM))
        self.slot_norm = nn.LayerNorm(plan.B_SET_DIM)
        self.token_norm = nn.LayerNorm(plan.B_SET_DIM)
        self.mha = nn.MultiheadAttention(
            embed_dim=plan.B_SET_DIM,
            num_heads=plan.B_SLOT_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.slot_ffn_norm = nn.LayerNorm(plan.B_SET_DIM)
        self.slot_ffn = nn.Sequential(
            nn.Linear(plan.B_SET_DIM, 4 * plan.B_SET_DIM),
            nn.GELU(),
            nn.Linear(4 * plan.B_SET_DIM, plan.B_SET_DIM),
        )
        self.slot_proj = nn.Linear(plan.B_SLOTS * plan.B_SET_DIM, plan.B_TEMPORAL_WIDTH)

    def forward(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_keep: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, N]; unit_keep: [B, N] True = attend
        batch, width, n_units = x.shape
        local = self.local_conv(x)
        e0 = bank.E0.to(device=x.device, dtype=x.dtype)
        t4 = bank.T.to(device=x.device, dtype=x.dtype)
        if e0.dim() == 2:
            e0 = e0.view(1, 1, n_units, plan.IDENTITY_DIM).expand(batch, width, n_units, plan.IDENTITY_DIM)
        else:
            e0 = e0.unsqueeze(1).expand(batch, width, n_units, plan.IDENTITY_DIM)
        if t4.dim() == 2:
            t4 = t4.view(1, 1, n_units, plan.T4_DIM).expand(batch, width, n_units, plan.T4_DIM)
        else:
            t4 = t4.unsqueeze(1).expand(batch, width, n_units, plan.T4_DIM)
        tokens = self.token_mlp(torch.cat([local, e0, t4], dim=-1))
        tokens = self.token_norm(tokens)
        slots = self.slot_norm(self.slots).view(1, 1, plan.B_SLOTS, plan.B_SET_DIM).expand(batch, width, plan.B_SLOTS, plan.B_SET_DIM)
        q = slots.reshape(batch * width, plan.B_SLOTS, plan.B_SET_DIM)
        k = tokens.reshape(batch * width, n_units, plan.B_SET_DIM)
        pad = (~unit_keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
        # If a row is all-padded, MultiheadAttention errors; keep-one already prevents this.
        attn_out, _ = self.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        fused = slots_out.reshape(batch, width, plan.B_SLOTS * plan.B_SET_DIM)
        return self.slot_proj(fused)


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
    # Additive mask: -inf blocks future keys. Bool SDPA masks are inverted
    # (True = drop) and have been version-sensitive.
    bias = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
    bias.masked_fill_(~allow, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = plan.B_TRANSFORMER_HEADS
        self.head_dim = plan.B_TEMPORAL_WIDTH // plan.B_TRANSFORMER_HEADS
        plan.require(self.head_dim * self.n_heads == plan.B_TEMPORAL_WIDTH, "transformer head dim")
        self.qkv = nn.Linear(plan.B_TEMPORAL_WIDTH, 3 * plan.B_TEMPORAL_WIDTH)
        self.proj = nn.Linear(plan.B_TEMPORAL_WIDTH, plan.B_TEMPORAL_WIDTH)

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
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(plan.B_TEMPORAL_WIDTH)
        self.attn = CausalSelfAttention()
        self.norm2 = nn.LayerNorm(plan.B_TEMPORAL_WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(plan.B_TEMPORAL_WIDTH, plan.B_TRANSFORMER_FFN),
            nn.GELU(),
            nn.Linear(plan.B_TRANSFORMER_FFN, plan.B_TEMPORAL_WIDTH),
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
    def __init__(self, max_len: int = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalTransformerBlock() for _ in range(plan.B_TRANSFORMER_LAYERS))
        self.register_buffer("pe", _sinusoidal_pe(max_len, plan.B_TEMPORAL_WIDTH), persistent=False)
        self.max_len = max_len

    def _add_pe(self, x: torch.Tensor, start: int) -> torch.Tensor:
        width = x.size(1)
        plan.require(start + width <= self.pe.size(0), "positional encoding overflow")
        return x + self.pe[start : start + width].unsqueeze(0).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._add_pe(x, 0)
        for block in self.blocks:
            h, _ = block(h, kv_cache=None, past_len=0)
        return h

    def forward_cached(
        self,
        x: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
        past_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        h = self._add_pe(x, past_len)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache = None if caches is None else caches[i]
            h, cache = block(h, kv_cache=cache, past_len=past_len)
            new_caches.append(cache)
        return h, new_caches


class PrenormMamba2Block(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE or Mamba2 is None:
            raise RuntimeError(BLOCK_REASON or "ENGINEERING_BLOCKED: Mamba2 unavailable")
        self.norm = nn.LayerNorm(plan.B_TEMPORAL_WIDTH)
        self.mixer = Mamba2(
            d_model=plan.B_TEMPORAL_WIDTH,
            d_state=plan.B_MAMBA_D_STATE,
            d_conv=plan.B_MAMBA_D_CONV,
            expand=plan.B_MAMBA_EXPAND,
            headdim=plan.B_MAMBA_HEADDIM,
            ngroups=plan.B_MAMBA_NGROUPS,
            chunk_size=plan.B_MAMBA_CHUNK,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))

    def step(
        self,
        x_t: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, conv_state, ssm_state = self.mixer.step(self.norm(x_t), conv_state, ssm_state)
        return x_t + y, conv_state, ssm_state


class Mamba2Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE or Mamba2 is None:
            raise RuntimeError(BLOCK_REASON or "ENGINEERING_BLOCKED: Mamba2 unavailable")
        self.blocks = nn.ModuleList(PrenormMamba2Block(i) for i in range(plan.B_MAMBA_LAYERS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def allocate_state(self, batch: int, max_seqlen: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [block.mixer.allocate_inference_cache(batch, max_seqlen) for block in self.blocks]

    def reset_state(self, state: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for conv, ssm in state:
            conv.zero_()
            ssm.zero_()

    def step_all(
        self,
        x_t: torch.Tensor,
        state: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        new_state: list[tuple[torch.Tensor, torch.Tensor]] = []
        h = x_t
        for block, (conv, ssm) in zip(self.blocks, state):
            h, conv, ssm = block.step(h, conv, ssm)
            new_state.append((conv, ssm))
        return h, new_state

    def forward_fill_state(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Parallel prefix that also writes inference caches (non-mem-eff path)."""
        if InferenceParams is None:
            raise RuntimeError(BLOCK_REASON or "ENGINEERING_BLOCKED: InferenceParams missing")
        batch, width, _ = x.shape
        params = InferenceParams(max_seqlen=width, max_batch_size=batch)
        h = x
        for block in self.blocks:
            h = h + block.mixer(block.norm(h), inference_params=params)
        state = [params.key_value_memory_dict[i] for i in range(len(self.blocks))]
        return h, state


class _BDecoderBase(nn.Module):
    name: str
    training_target_space: str = plan.TRAINING_TARGET_SPACE

    def __init__(self) -> None:
        super().__init__()
        self.frontend = SharedSetFrontend()
        self.final_norm = nn.LayerNorm(plan.B_TEMPORAL_WIDTH)
        self.readout = nn.Sequential(
            nn.Linear(plan.B_TEMPORAL_WIDTH, plan.B_READOUT_HIDDEN),
            nn.GELU(),
            nn.Linear(plan.B_READOUT_HIDDEN, plan.OUT_DIM),
        )
        self.unit_dropout_p = plan.B_UNIT_DROPOUT

    def _fuse(self, x: torch.Tensor, bank: SessionBank, unit_mask: torch.Tensor | None) -> torch.Tensor:
        keep = _expand_unit_mask(unit_mask, bank, x.size(0))
        if self.training and self.unit_dropout_p > 0.0:
            keep = whole_unit_dropout(keep, p=self.unit_dropout_p)
        return self.frontend(x, bank, keep)

    def _readout_all(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(self.final_norm(hidden))

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map temporal features ``[B, T, 512]`` to decoder_raw ``[B, T, 2]``."""
        return self._readout_all(hidden)

    def forward_last(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(x, bank, unit_mask)
        return self._readout_all(hidden)[:, -1, :]

    def forward_scores(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._readout_all(self.forward_hidden(x, bank, unit_mask))

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        # Calibration encoder is not a submodule; bank.E0 / bank.T are frozen inputs.
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def reset_temporal_state(self) -> None:
        return None

    def forward_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def step_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def chunk_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        chunk_size: int = plan.B_MAMBA_CHUNK,
    ) -> torch.Tensor:
        raise NotImplementedError


class BTransformerDecoder(_BDecoderBase):
    """D0: shared frontend + 4 pre-norm causal Transformer blocks."""

    name = "B-TRANSFORMER"

    def __init__(self, seed: int = plan.SEED_PRIMARY) -> None:
        super().__init__()
        self.temporal = CausalTransformerStack()
        initialize_decoder(self, seed)

    def forward_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.temporal(self._fuse(x, bank, unit_mask))

    def step_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self._fuse(x, bank, unit_mask)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        outs = []
        for t in range(fused.size(1)):
            h, caches = self.temporal.forward_cached(fused[:, t : t + 1], caches, past_len=t)
            outs.append(h)
        return torch.cat(outs, dim=1)

    def chunk_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        chunk_size: int = plan.B_MAMBA_CHUNK,
    ) -> torch.Tensor:
        fused = self._fuse(x, bank, unit_mask)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        outs = []
        width = fused.size(1)
        for start in range(0, width, chunk_size):
            chunk = fused[:, start : start + chunk_size]
            h, caches = self.temporal.forward_cached(chunk, caches, past_len=start)
            outs.append(h)
        return torch.cat(outs, dim=1)


class BMambaDecoder(_BDecoderBase):
    """D1: shared frontend + 4 pre-norm residual Mamba2 blocks.

    Raises ENGINEERING_BLOCKED if official Mamba2 is not usable. This class
    never falls back to DiagSSM / Mamba1 / a handwritten scan.
    """

    name = "B-MAMBA"

    def __init__(self, seed: int = plan.SEED_PRIMARY) -> None:
        if not MAMBA_AVAILABLE or Mamba2 is None:
            raise RuntimeError(BLOCK_REASON or "ENGINEERING_BLOCKED: Mamba2 unavailable")
        super().__init__()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(seed) + TEMPORAL_RNG_DOMAIN)
        try:
            self.temporal = Mamba2Stack()
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        initialize_decoder(self, seed)

    def forward_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.temporal(self._fuse(x, bank, unit_mask))

    def step_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self._fuse(x, bank, unit_mask)
        batch, width, _ = fused.shape
        state = self.temporal.allocate_state(batch, width)
        outs = []
        for t in range(width):
            h, state = self.temporal.step_all(fused[:, t : t + 1], state)
            outs.append(h)
        return torch.cat(outs, dim=1)

    def chunk_hidden(
        self,
        x: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        chunk_size: int = plan.B_MAMBA_CHUNK,
    ) -> torch.Tensor:
        fused = self._fuse(x, bank, unit_mask)
        width = fused.size(1)
        if width <= chunk_size:
            # One chunk: parallel fill + no leftover steps.
            hidden, _ = self.temporal.forward_fill_state(fused)
            return hidden
        first = fused[:, :chunk_size]
        hidden_first, state = self.temporal.forward_fill_state(first)
        outs = [hidden_first]
        for t in range(chunk_size, width):
            h, state = self.temporal.step_all(fused[:, t : t + 1], state)
            outs.append(h)
        return torch.cat(outs, dim=1)

    def reset_temporal_state(self) -> None:
        # Each forward / step / chunk allocates a fresh zero state; no clamp.
        return None


def build_candidate(kind: str, seed: int = plan.SEED_PRIMARY) -> _BDecoderBase:
    key = kind.strip().upper().replace("_", "-")
    if key in {"B-TRANSFORMER", "D0", "TRANSFORMER"}:
        return BTransformerDecoder(seed=seed)
    if key in {"B-MAMBA", "D1", "MAMBA", "MAMBA2"}:
        return BMambaDecoder(seed=seed)
    raise ValueError(f"unknown B candidate: {kind}")


__all__ = [
    "FRONTEND_RNG_DOMAIN",
    "TEMPORAL_RNG_DOMAIN",
    "whole_unit_dropout",
    "initialize_decoder",
    "count_trainable_parameters",
    "adamw_param_groups",
    "SharedSetFrontend",
    "BTransformerDecoder",
    "BMambaDecoder",
    "build_candidate",
]
