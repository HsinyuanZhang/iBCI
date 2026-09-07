"""Exact cached-memory sibling of the CPU M2 QueryAge runtime.

Only QueryTemporalStack's independent memory K/V projections are reused.  The
M2 raw window, lifted spatial repair, current query updates, and readout all
remain the trained operator used by :mod:`m2_family_queryage`.
"""
from __future__ import annotations

import torch

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryMemoryCache

from .m2_family_causal import _tensor_signature
from .m2_family_queryage import M2FamilyQueryAgeRuntime, M2RuntimeBank


class M2FamilyQueryAgeCachedRuntime(M2FamilyQueryAgeRuntime):
    """M2 W50 stream with exact five-token K/V cache rollover."""

    _CHANGED = (0, 1, 2, 3, 49)

    @staticmethod
    def _cache_signature(cache: QueryMemoryCache) -> tuple:
        tensors = [cache.z] if cache.z is not None else []
        tensors.extend(value for pair in cache.memories for value in pair)
        return _tensor_signature(tensors)

    @torch.no_grad()
    def _reset_memory(self, *, reason: str) -> None:
        temporal = self.model.temporal
        self.memory = QueryMemoryCache(temporal)
        self.memory.reset(self.frontend, reason=reason)
        self._memory_signature = self._cache_signature(self.memory)

    @torch.no_grad()
    def reset(self, bank=None, *, batch=None):
        super().reset(bank, batch=batch)
        self._reset_memory(reason="reset")

    @torch.no_grad()
    def _audit(self):
        before = getattr(self, "_state_signature", None)
        super()._audit()
        # Parent refresh reconstructs frontend/spatial state after any model,
        # bank, or mask mutation.  It must never leave K/V for the old state.
        if (not hasattr(self, "memory") or self.memory.model is not self.model.temporal
                or before != self._state_signature):
            self._reset_memory(reason="model_bank_or_mask_changed")
            return
        if self._cache_signature(self.memory) != self._memory_signature:
            raise RuntimeError("cached QueryAge K/V or frontend memory mutated; reset required")

    @torch.no_grad()
    def _last(self):
        self._audit()
        if self.memory.model is not self.model.temporal:
            self._reset_memory(reason="temporal_replaced")
        self.memory.advance(self.frontend, changed_indices=self._CHANGED)
        hidden = self.memory.predict()
        if hidden.shape != (self._active, 1, 256) or hidden.dtype != torch.float32:
            raise RuntimeError("cached QueryAge temporal output contract drift")
        self._memory_signature = self._cache_signature(self.memory)
        return self.model.readout(self.model.final_norm(hidden))[:, 0]

    def state_bytes(self):
        return {**super().state_bytes(), "query_memory_cache": self.memory.state_bytes}


__all__ = ["M2FamilyQueryAgeCachedRuntime", "M2RuntimeBank"]
