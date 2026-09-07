"""Finer short-lag resolution via query-time log age bias, without changing K/V.

Only age lookup differs from current_query_v2. Tokens are still independently
projected, no positional information is baked into cached K/V, and all W raw
history positions remain legal. This is a NEW trained operator, not equivalent
to a V2 model after its age biases become nonzero.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ..current_query_v2.core import QueryReadBlock, QueryTemporalStack


def log_age_indices(window: int, age_buckets: int) -> torch.Tensor:
    if min(window, age_buckets) < 1:
        raise ValueError("positive window and bucket count required")
    age = torch.arange(window, dtype=torch.float64)
    return torch.floor(torch.log1p(age) * age_buckets / math.log1p(window)).long().clamp(max=age_buckets - 1)


class LogAgeReadBlock(QueryReadBlock):
    def __init__(self, width, heads, ffn, window, age_buckets):
        super().__init__(width, heads, ffn, window, age_buckets)
        self.register_buffer("age_bucket_by_age", log_age_indices(window, age_buckets))

    def read(self, query, memory, valid_mask=None):
        k, v = memory
        b, _, t, _ = k.shape
        if not 0 < t <= self.window:
            raise ValueError("memory length outside finite window")
        q = self.q_proj(self.query_norm(query)).reshape(b, 1, self.heads, self.head_dim).transpose(1, 2)
        # Ordered memory is oldest..newest; this immutable table is indexed at
        # READ time only. Boundary-crossing ages never modify cached K/V.
        buckets = self.age_bucket_by_age[:t].flip(0)
        bias = self.age_bias[:, buckets].to(dtype=query.dtype).unsqueeze(0).unsqueeze(2)
        if valid_mask is not None:
            if valid_mask.shape != (b, t) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [batch, window]")
            if not bool(valid_mask.any(dim=1).all()) or not bool(valid_mask[:, -1].all()):
                raise ValueError("current query must be a valid observation")
            bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        context = context.transpose(1, 2).reshape(b, 1, self.width)
        hidden = query + self.out_proj(context)
        return hidden + self.ffn(self.ffn_norm(hidden))


class LogAgeQueryTemporalStack(QueryTemporalStack):
    """V2-compatible cache API with explicit, strict-loadable new age semantics."""
    def __init__(self, *, width=256, heads=8, layers=4, ffn=512, window=700, age_buckets=16, seed=42):
        super().__init__(width=width, heads=heads, layers=layers, ffn=ffn, window=window, age_buckets=age_buckets, seed=seed)
        # Preserve both the V2 parameter initialization and the caller RNG.
        with torch.random.fork_rng(devices=[]):
            for index, original in enumerate(self.blocks):
                replacement = LogAgeReadBlock(width, heads, ffn, window, age_buckets)
                state = dict(original.state_dict())
                state["age_bucket_by_age"] = replacement.age_bucket_by_age.clone()
                replacement.load_state_dict(state, strict=True)
                self.blocks[index] = replacement
        self.register_buffer("temporal_contract_version", torch.tensor(3, dtype=torch.int64))
