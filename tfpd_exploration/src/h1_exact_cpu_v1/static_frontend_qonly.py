"""Static-carrier exact E plus an algebraic final-query projection slice."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .static_frontend import StaticCarrierExactFullWindowStream


class StaticCarrierQOnlyExactFullWindowStream(StaticCarrierExactFullWindowStream):
    """Exact E with Q computed only for the final normalized row.

    K and V are still computed for every finite-window row, from contiguous
    slices of the original ``attn.qkv`` weights/bias.  This is algebraically
    the final-query slice already used by E; it merely avoids unused Q rows.
    No temporal output is retained after a public call.
    """
    def _full_window_last_hidden(self) -> torch.Tensor:
        z = self.front.current
        temporal = self.model.temporal
        hidden = z + temporal.pe[:z.size(1)].to(z).unsqueeze(0)
        for block in temporal.blocks[:-1]:
            hidden = block(hidden)
        block = temporal.blocks[-1]
        normalized = block.norm1(hidden)
        batch, width, dim = normalized.shape
        attention = block.attn
        weight, bias = attention.qkv.weight, attention.qkv.bias
        query = F.linear(normalized[:, -1:], weight[:dim], bias[:dim])
        kv = F.linear(normalized, weight[dim:], bias[dim:]).reshape(
            batch, width, 2, attention.n_heads, attention.head_dim
        )
        key, value = kv.unbind(dim=2)
        query = query.reshape(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        result = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        result = attention.proj(result.transpose(1, 2).reshape(batch, 1, dim))
        last = hidden[:, -1:] + result
        return last + block.ffn(block.norm2(last))
