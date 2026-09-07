"""A new trained reader: fixed multi-timescale recency plus learned log-age bias.

The prior is applied only at query read time. Memory K/V still derive
independently from the original frontend tokens; no contextualized history or
age information is baked into cross-window cached values. W never changes.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ..current_query_v3.core import LogAgeQueryTemporalStack, LogAgeReadBlock


DEFAULT_HEAD_TAUS = (2., 4., 8., 16., 32., 64., 128., float("inf"))


class RecencyPriorReadBlock(LogAgeReadBlock):
    def __init__(self, width, heads, ffn, window, age_buckets, head_taus):
        super().__init__(width, heads, ffn, window, age_buckets)
        age = torch.arange(window, dtype=torch.float32)
        prior = torch.stack([torch.zeros_like(age) if math.isinf(tau) else -age / tau for tau in head_taus])
        self.register_buffer("recency_prior_by_age", prior)

    def read(self, query, memory, valid_mask=None):
        k, v = memory
        batch, _, length, _ = k.shape
        if not 0 < length <= self.window:
            raise ValueError("memory length outside finite window")
        q = self.q_proj(self.query_norm(query)).reshape(batch, 1, self.heads, self.head_dim).transpose(1, 2)
        buckets = self.age_bucket_by_age[:length].flip(0)
        # Both tables are indexed at read time against oldest..newest memory.
        # The final head has exactly zero prior and retains global access.
        bias = self.age_bias[:, buckets] + self.recency_prior_by_age[:, :length].flip(-1)
        bias = bias.to(dtype=query.dtype).unsqueeze(0).unsqueeze(2)
        if valid_mask is not None:
            if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [batch, window]")
            if not bool(valid_mask.any(dim=1).all()) or not bool(valid_mask[:, -1].all()):
                raise ValueError("current query must be a valid observation")
            bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0., is_causal=False)
        context = context.transpose(1, 2).reshape(batch, 1, self.width)
        hidden = query + self.out_proj(context)
        return hidden + self.ffn(self.ffn_norm(hidden))


class RecencyPriorQueryTemporalStack(LogAgeQueryTemporalStack):
    """Strict new operator with seven local scales and one global head by default."""
    def __init__(self, *, width=256, heads=8, layers=4, ffn=512, window=700,
                 age_buckets=16, seed=42, head_taus=DEFAULT_HEAD_TAUS):
        head_taus = tuple(float(value) for value in head_taus)
        if len(head_taus) != heads or any(math.isnan(value) or value <= 0 for value in head_taus):
            raise ValueError("one positive finite/infinite time scale required for each head")
        if not any(math.isinf(value) for value in head_taus):
            raise ValueError("at least one unbiased global head is required")
        super().__init__(width=width, heads=heads, layers=layers, ffn=ffn, window=window, age_buckets=age_buckets, seed=seed)
        with torch.random.fork_rng(devices=[]):
            for index, original in enumerate(self.blocks):
                replacement = RecencyPriorReadBlock(width, heads, ffn, window, age_buckets, head_taus)
                state = dict(original.state_dict())
                state["recency_prior_by_age"] = replacement.recency_prior_by_age.clone()
                replacement.load_state_dict(state, strict=True)
                self.blocks[index] = replacement
        self.temporal_contract_version.fill_(4)
        self.register_buffer("recency_tau_bins", torch.tensor(head_taus, dtype=torch.float32))
