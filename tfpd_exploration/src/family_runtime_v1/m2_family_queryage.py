"""Safe CPU streaming wrapper for fresh M2 FW-QueryAge16 FLAT/ROUTE models.

The lifted five-token spatial repair and lifecycle checks are inherited from
``M2FamilyCausalRuntime``.  Temporal work is intentionally *not* inherited:
each public call recomputes the actual finite W50 QueryTemporalStack over the
current frontend window.  This makes no claim that causal KV/context states
would be interchangeable with current-query independent memory.
"""

from __future__ import annotations

import torch

from tfpd_exploration.src.m2_queryage_family_v1.model import M2QueryAgeFamilyDecoder
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack

from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank


class M2FamilyQueryAgeRuntime(M2FamilyCausalRuntime):
    """CPU-FP32 W50 stream with exact full-window QueryAge temporal recompute."""

    def _validate(self, bank, batch):
        super()._validate(bank, batch)
        if not isinstance(self.model, M2QueryAgeFamilyDecoder):
            raise TypeError("actual M2QueryAgeFamilyDecoder is required; causal M2 is rejected")
        temporal = self.model.temporal
        if not isinstance(temporal, QueryTemporalStack):
            raise TypeError("actual QueryTemporalStack is required; causal temporal stack is rejected")
        if (temporal.width, temporal.heads, temporal.window, temporal.age_buckets, len(temporal.blocks)) != (256, 8, 50, 16, 4):
            raise RuntimeError("exact FW-QueryAge16 W50 topology required")
        if any(block.ffn[0].out_features != 512 for block in temporal.blocks):
            raise RuntimeError("exact FW-QueryAge16 FFN512 topology required")
        if hasattr(temporal, "pe"):
            raise RuntimeError("QueryAge runtime forbids positional encoding")

    @torch.no_grad()
    def _last(self):
        # This invokes all four actual QueryReadBlocks.  It is deliberately a
        # full W50 independent-memory recomputation rather than a causal cache.
        hidden = self.model.temporal(self.frontend)
        if hidden.shape != (self._active, 1, 256) or hidden.dtype != torch.float32:
            raise RuntimeError("QueryAge temporal output contract drift")
        return self.model.readout(self.model.final_norm(hidden))[:, 0]


__all__ = ["M2FamilyQueryAgeRuntime", "M2RuntimeBank"]
