"""Read-only algebraic slot attention optimization for constant learned queries.

For each head, q @ (X Wk^T + bk)^T = (q Wk) X^T + q bk.
Also A @ (X Wv^T + bv) = (A X) Wv^T + sum(A) bv.
This removes per-unit dense K/V projections, without low-rank approximation.
FP32 reassociation still requires the ordinary native-output parity gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LiftedSlotAttention:
    queries: torch.Tensor
    key_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor
    output_weight: torch.Tensor
    output_bias: torch.Tensor
    head_dim: int

    @classmethod
    @torch.no_grad()
    def from_split(cls, attn, slots):
        heads, head_dim = attn.n_heads, attn.head_dim
        q = attn.q_proj(slots).reshape(slots.shape[-2], heads, head_dim).transpose(0, 1)
        wk = attn.k_proj.weight.reshape(heads, head_dim, -1)
        bk = attn.k_proj.bias.reshape(heads, head_dim)
        lifted = torch.matmul(q, wk)
        bias = torch.einsum("hsd,hd->hs", q, bk)
        wv = attn.v_proj.weight.reshape(heads, head_dim, -1).transpose(-1, -2)
        bv = attn.v_proj.bias.reshape(heads, 1, head_dim)
        return cls(lifted, bias, wv, bv, attn.out_proj.weight, attn.out_proj.bias, head_dim)

    def __call__(self, tokens, keep):
        batch, units, dim = tokens.shape
        heads, slots, _ = self.queries.shape
        # One large GEMM instead of broadcasting X across H tiny GEMMs.
        logits = F.linear(tokens, self.queries.reshape(heads * slots, dim))
        logits = logits.reshape(batch, units, heads, slots).permute(0, 2, 3, 1)
        logits = (logits + self.key_bias[None, :, :, None]) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits.masked_fill(~keep[:, None, None, :], float("-inf")), dim=-1)
        content = torch.bmm(weights.reshape(batch, heads * slots, units), tokens)
        content = content.reshape(batch, heads, slots, dim)
        values = torch.matmul(content, self.value_weight.unsqueeze(0))
        values = values + weights.sum(-1, keepdim=True) * self.value_bias.unsqueeze(0)
        merged = values.transpose(1, 2).contiguous().reshape(tokens.shape[0], self.queries.shape[1], -1)
        return F.linear(merged, self.output_weight, self.output_bias)

    @property
    def owned_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.queries, self.key_bias))
