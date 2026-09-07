"""Isolated exact ROUTE five-token current-query stream for M1.

This intentionally does not alter the FLAT-only V3 constructor.  It mirrors
its immutable lifecycle and finite W=100/k=5 repair law, but admits a routed
frontend and caches the bank-only routing logit term.  The cached term is
added at the native attention location: after dot-product/scaling and before
the unit mask and softmax.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F

from tfpd_exploration.src.m1_runtime_v3.runtime import (
    BankBatch, HeterogeneousCurrentQueryStream, _unique_storage_bytes,
)
from .repair import repaired_local_features


@dataclass(frozen=True)
class _LiftedRouteAttention:
    """Constant-slot attention with native ROUTE logits inserted pre-mask."""
    qwk: torch.Tensor
    key_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor
    output_weight: torch.Tensor
    output_bias: torch.Tensor
    head_dim: int

    @classmethod
    @torch.no_grad()
    def from_attention(cls, attn, slots):
        heads, d = attn.n_heads, attn.head_dim
        q = attn.q_proj(slots).reshape(slots.shape[-2], heads, d).transpose(0, 1)
        wk = attn.k_proj.weight.reshape(heads, d, -1)
        bk = attn.k_proj.bias.reshape(heads, d)
        return cls(
            torch.matmul(q, wk), torch.einsum("hsd,hd->hs", q, bk),
            attn.v_proj.weight.reshape(heads, d, -1).transpose(-1, -2),
            attn.v_proj.bias.reshape(heads, 1, d), attn.out_proj.weight,
            attn.out_proj.bias, d,
        )

    def __call__(self, tokens, keep, routing_bonus):
        # qWk @ X + qbk, then native scale, then the static routed logit;
        # no per-unit K/V projection is evaluated on this path.
        b, n, width = tokens.shape
        h, k, _ = self.qwk.shape
        logits = F.linear(tokens, self.qwk.reshape(h * k, width)).reshape(b, n, h, k).permute(0, 2, 3, 1)
        logits = (logits + self.key_bias[None, :, :, None]) * (self.head_dim ** -0.5)
        logits = logits + routing_bonus
        weights = torch.nan_to_num(torch.softmax(logits.masked_fill(~keep[:, None, None, :], float("-inf")), dim=-1), nan=0.0)
        # A @ (X Wv^T + bv) = (A @ X) Wv^T + sum(A) bv.
        content = torch.bmm(weights.reshape(b, h * k, n), tokens).reshape(b, h, k, width)
        values = torch.matmul(content, self.value_weight.unsqueeze(0))
        values = values + weights.sum(-1, keepdim=True) * self.value_bias.unsqueeze(0)
        merged = values.transpose(1, 2).contiguous().reshape(b, k, h * self.head_dim)
        return F.linear(merged, self.output_weight, self.output_bias)

    @property
    def owned_tensors(self):
        return (self.qwk, self.key_bias)


class RouteLiftedFiveTokenCurrentQueryStream(HeterogeneousCurrentQueryStream):
    """ROUTE-only counterpart to the isolated M1 five-token stream."""

    def __init__(self, model, bank: BankBatch, *, window: int | None = None, kernel: int = 5):
        # Do not call the parent: its prohibition on routed models is a useful
        # FLAT contract and must remain true.  This is the explicit sibling
        # constructor with the same validation/lifecycle initialization.
        if model.training:
            raise ValueError("model must be eval before stream construction")
        if not getattr(model, "routed", False):
            raise ValueError("RouteLiftedFiveTokenCurrentQueryStream requires routed=True")
        self.model = model
        self.window, self.kernel = int(model.cfg.window), int(kernel)
        if ((window is not None and int(window) != self.window) or self.window != 100
                or self.kernel != 5 or self.kernel != int(model.cfg.conv_kernel)):
            raise ValueError("fixed M1 W=100/k=5 contract drift")
        self.bank = bank
        self.batch, self.units = bank.E0.shape[:2]
        bank.validate(self.batch, self.units)
        self.raw = self.z = None
        self.memories, self._static, self._lifecycle = [], {}, None
        self.refresh_state(bank=bank)

    @torch.no_grad()
    def _build_static(self):
        super()._build_static()
        attn = self.model.frontend.attn
        # [B,H,K,N], including the native LayerNorm, q_cal, route projections,
        # route-key scaling, and tanh(g).  It depends only on legal E0/T and
        # is invalidated by the inherited model/bank lifecycle token.
        bonus = attn.routing_bonus(self.bank.E0, self.bank.T, self.batch)
        expected = (self.batch, attn.n_heads, self.model.frontend.cfg.slots, self.units)
        if tuple(bonus.shape) != expected or not bool(torch.isfinite(bonus).all()):
            raise RuntimeError("invalid static routing bonus")
        self._static["routing_bonus"] = bonus
        self._static["lifted"] = _LiftedRouteAttention.from_attention(attn, self._static["slots"])

    @torch.no_grad()
    def _encode_local(self, local: torch.Tensor, rows) -> torch.Tensor:
        """One route-aware frontend used for both W=100 rebuild and repair."""
        f = self.model.frontend
        b, t, n, _ = local.shape
        affine = self._static["affine"][rows]
        keep = self.bank.unit_mask[rows]
        bonus = self._static["routing_bonus"][rows]
        hidden = F.linear(local, self._static["wloc"], None) + affine.unsqueeze(1)
        tokens = f.token_norm(f.token_mlp.fc2(F.gelu(hidden))).reshape(b * t, n, f.cfg.set_dim)
        expanded_keep = keep[:, None].expand(-1, t, -1).reshape(b * t, n)
        expanded_bonus = bonus[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, f.attn.n_heads, f.cfg.slots, n)
        attended = self._static["lifted"](tokens, expanded_keep, expanded_bonus)
        query = self._static["slots"].unsqueeze(1).expand(b, t, -1, -1).reshape(b * t, f.cfg.slots, f.cfg.set_dim)
        out = query + attended
        out = out + f.slot_ffn(f.slot_ffn_norm(out))
        return f.slot_proj(out.reshape(b, t, f.cfg.slots * f.cfg.set_dim))

    @torch.no_grad()
    def _frontend(self, x, *, rows=None):
        # Inherited refresh_state calls this for its initial/history W=100
        # build.  It must never take the FLAT-only parent implementation.
        rows = slice(0, self.batch) if rows is None else rows
        return self._encode_local(self.model.frontend.local_conv(x), rows)

    @torch.no_grad()
    def _repair_frontend(self, raw: torch.Tensor, active: int) -> torch.Tensor:
        f = self.model.frontend
        local = repaired_local_features(f.local_conv, raw)
        return self._encode_local(local, slice(0, active))

    @torch.no_grad()
    def observe(self, observations, *, active: int | None = None) -> None:
        self._check_immutable()
        x = torch.as_tensor(observations, device=self.raw.device, dtype=self.raw.dtype)
        active = self.batch if active is None else int(active)
        if (not 0 < active <= self.batch or x.ndim != 2 or x.shape != (active, self.units)
                or not bool(torch.isfinite(x).all())):
            raise ValueError("observations must be finite [active,N], 1<=active<=B")
        raw = self.raw.clone()
        raw[:active, :-1] = self.raw[:active, 1:]
        raw[:active, -1] = x
        repair = self._repair_frontend(raw[:active], active)
        z = self.z.clone()
        z[:active, :4] = repair[:, :4]
        z[:active, 4:-1] = self.z[:active, 5:]
        z[:active, -1:] = repair[:, -1:]
        self.raw, self.z = raw, z
        self._advance_memory((0, 1, 2, 3, self.window - 1), active)
        self.observation_counts[:active] += 1

    @property
    def static_cache_bytes(self):
        static = (
            self._static.get("affine"), self._static.get("slots"), self._static.get("q"),
            self._static.get("routing_bonus"), *self._static["lifted"].owned_tensors,
            *self._static.get("age_biases", []),
        )
        return _unique_storage_bytes(static, excluded=(*self.model.parameters(), *self.model.buffers()))


__all__ = ("RouteLiftedFiveTokenCurrentQueryStream",)
