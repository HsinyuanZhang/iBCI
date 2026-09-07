"""Read fixed, independently encoded history with a depth-updated current query.

This is NOT ordinary Transformer KV caching. Each layer's memory depends only
on its original frontend token z; historical tokens never contextualize one
another. The current query is the only representation updated across layers.
The deployment cache preserves exactly the same finite-window operator,
including the caller's explicitly encoded startup and left-boundary tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class QueryReadBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int, window: int, age_buckets: int) -> None:
        super().__init__()
        self.width, self.heads = width, heads
        self.head_dim = width // heads
        self.window, self.age_buckets = window, age_buckets
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.out_proj = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.age_bias = nn.Parameter(torch.zeros(heads, age_buckets))

    def project_memory(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = z.shape
        kv = self.kv_proj(self.memory_norm(z)).reshape(b, t, 2, self.heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def read(
        self,
        query: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k, v = memory
        b, _, t, _ = k.shape
        q = self.q_proj(self.query_norm(query)).reshape(b, 1, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        age = torch.arange(t - 1, -1, -1, device=query.device)
        buckets = torch.div(age * self.age_buckets, self.window, rounding_mode="floor")
        buckets = buckets.clamp(max=self.age_buckets - 1)
        bias = self.age_bias[:, buckets].to(dtype=query.dtype).unsqueeze(0).unsqueeze(2)
        if valid_mask is not None:
            if valid_mask.shape != (b, t) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [batch, window]")
            if not bool(valid_mask.any(dim=1).all()) or not bool(valid_mask[:, -1].all()):
                raise ValueError("current query must be a valid observation")
            bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        # All memory positions precede/equal the current query. is_causal=True
        # on this non-square [1,T] attention would wrongly allow only key zero.
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        context = context.transpose(1, 2).reshape(b, 1, self.width)
        hidden = query + self.out_proj(context)
        return hidden + self.ffn(self.ffn_norm(hidden))

    def forward(self, query: torch.Tensor, z: torch.Tensor, valid_mask=None) -> torch.Tensor:
        return self.read(query, self.project_memory(z), valid_mask)


class QueryTemporalStack(nn.Module):
    """Four (configurable) current-query updates over one finite history.

    ``forward(z)`` returns [B,1,D], intentionally not [B,W,D]. Window W is
    retained; no future inputs or contextualized pre-window memory are used.
    Seeded construction does not perturb the caller's global random stream.
    """

    def __init__(
        self, *, width: int = 256, heads: int = 8, layers: int = 4,
        ffn: int = 512, window: int = 700, age_buckets: int = 16, seed: int = 42,
    ) -> None:
        super().__init__()
        if min(width, heads, layers, ffn, window, age_buckets) < 1 or width % heads:
            raise ValueError("positive dimensions and width divisible by heads required")
        self.width, self.heads, self.window = width, heads, window
        self.max_len, self.age_buckets = window, age_buckets
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1_000_003)
            self.blocks = nn.ModuleList([
                QueryReadBlock(width, heads, ffn, window, age_buckets) for _ in range(layers)
            ])

    def _validate(self, z: torch.Tensor) -> None:
        if z.ndim != 3 or z.shape[-1] != self.width or not 0 < z.shape[1] <= self.window:
            raise ValueError(f"expected [B,T,{self.width}] with 0<T<={self.window}")

    def forward(self, z, *, use_checkpoint: bool = False, valid_mask=None):
        self._validate(z)
        query = z[:, -1:]
        for block in self.blocks:
            if use_checkpoint and torch.is_grad_enabled() and z.requires_grad:
                query = checkpoint(block, query, z, valid_mask, use_reentrant=False)
            else:
                query = block(query, z, valid_mask)
        return query

    def project_memory(self, z: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._validate(z)
        return [block.project_memory(z) for block in self.blocks]

    def forward_cached(self, last_z, memories, *, valid_mask=None):
        if last_z.ndim != 3 or last_z.shape[1:] != (1, self.width):
            raise ValueError("last_z must be [B,1,D]")
        if len(memories) != len(self.blocks):
            raise ValueError("memory depth mismatch")
        query = last_z
        for block, memory in zip(self.blocks, memories, strict=True):
            query = block.read(query, memory, valid_mask)
        return query

    @torch.no_grad()
    def initialize_from_full_window(self, old_stack: nn.Module) -> None:
        """Copy compatible initialization, without claiming operator equality."""
        if len(old_stack.blocks) != len(self.blocks):
            raise ValueError("cannot match unequal depth")
        for new, old in zip(self.blocks, old_stack.blocks, strict=True):
            new.query_norm.load_state_dict(old.norm1.state_dict(), strict=True)
            new.memory_norm.load_state_dict(old.norm1.state_dict(), strict=True)
            new.ffn_norm.load_state_dict(old.norm2.state_dict(), strict=True)
            new.ffn.load_state_dict(old.ffn.state_dict(), strict=True)
            new.q_proj.weight.copy_(old.attn.qkv.weight[:self.width])
            new.q_proj.bias.copy_(old.attn.qkv.bias[:self.width])
            new.kv_proj.weight.copy_(old.attn.qkv.weight[self.width:])
            new.kv_proj.bias.copy_(old.attn.qkv.bias[self.width:])
            new.out_proj.load_state_dict(old.attn.proj.state_dict(), strict=True)
            new.age_bias.zero_()


def _weight_token(module: nn.Module) -> tuple:
    return tuple(
        (name, p.data_ptr(), p._version, tuple(p.shape), str(p.dtype), str(p.device))
        for name, p in module.named_parameters()
    )


class QueryMemoryCache:
    """Inference-only independent-memory cache with checked rollover reuse.

    Caller supplies the current *frontend* window and changed token indices.
    Unlisted changes automatically rebuild the entire cache. Thus a bad
    boundary claim cannot silently reuse stale K/V. Weight/device/dtype/shape
    changes also rebuild; training-mode use is rejected.
    """

    def __init__(self, model: QueryTemporalStack) -> None:
        self.model = model
        self.z: torch.Tensor | None = None
        self.memories: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.token: tuple | None = None
        self.last_rebuild_reason: str | None = None

    def _require_inference(self) -> None:
        if self.model.training:
            raise RuntimeError("QueryMemoryCache requires model.eval(); never cache across training updates")

    @torch.no_grad()
    def reset(self, z: torch.Tensor, *, reason: str = "reset") -> None:
        self._require_inference()
        self.model._validate(z)
        self.memories = [(k.detach().clone(), v.detach().clone()) for k, v in self.model.project_memory(z)]
        self.z = z.detach().clone()
        self.token = _weight_token(self.model)
        self.last_rebuild_reason = reason

    @torch.no_grad()
    def advance(self, z: torch.Tensor, changed_indices: Sequence[int]) -> None:
        self._require_inference()
        if self.z is None:
            self.reset(z, reason="uninitialized")
            return
        if self.token != _weight_token(self.model):
            self.reset(z, reason="weights_changed")
            return
        if z.shape != self.z.shape or z.device != self.z.device or z.dtype != self.z.dtype:
            self.reset(z, reason="topology_changed")
            return
        t = z.shape[1]
        changed = sorted(set(int(i) for i in changed_indices))
        if not changed or changed[0] < 0 or changed[-1] >= t or t - 1 not in changed:
            raise ValueError("changed indices must include newest token and stay within window")
        unchanged = [i for i in range(t - 1) if i not in changed]
        if unchanged and not torch.equal(z[:, unchanged], self.z[:, [i + 1 for i in unchanged]]):
            self.reset(z, reason="unlisted_token_change")
            return
        updates = self.model.project_memory(z[:, changed])
        new_memories = []
        for (old_k, old_v), (update_k, update_v) in zip(self.memories, updates, strict=True):
            new_k, new_v = torch.empty_like(old_k), torch.empty_like(old_v)
            new_k[:, :, :-1] = old_k[:, :, 1:]
            new_v[:, :, :-1] = old_v[:, :, 1:]
            new_k[:, :, changed] = update_k
            new_v[:, :, changed] = update_v
            new_memories.append((new_k, new_v))
        self.memories = new_memories
        self.z = z.detach().clone()
        self.last_rebuild_reason = None

    @torch.no_grad()
    def predict(self, *, valid_mask=None) -> torch.Tensor:
        self._require_inference()
        if self.z is None:
            raise RuntimeError("reset/advance required before predict")
        if self.token != _weight_token(self.model):
            self.reset(self.z, reason="weights_changed")
        return self.model.forward_cached(self.z[:, -1:], self.memories, valid_mask=valid_mask)

    @property
    def state_bytes(self) -> int:
        tensors = [item for pair in self.memories for item in pair]
        if self.z is not None:
            tensors.append(self.z)
        return sum(t.numel() * t.element_size() for t in tensors)
