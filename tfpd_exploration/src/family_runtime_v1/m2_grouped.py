"""Isolated exact M2 lifted variant with head-grouped value projection."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .grouped_value import grouped_value_projection
from .m2_lifted import LiftedFiveTokenM2Decoder, _LiftedFiveTokenExactE, _M2LiftedMHA


class _M2GroupedValueMHA(_M2LiftedMHA):
    def __call__(self, tokens: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        batch, units, dim = tokens.shape
        slots = self.qwk.size(1)
        logits = F.linear(tokens, self.qwk.reshape(self.heads * slots, dim))
        logits = logits.view(batch, units, self.heads, slots).permute(0, 2, 3, 1)
        logits = (logits + self.qbk[None, :, :, None]) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits.masked_fill(~keep[:, None, None, :], float("-inf")), dim=-1)
        attended = torch.bmm(weights.reshape(batch, self.heads * slots, units), tokens)
        attended = attended.reshape(batch, self.heads, slots, dim)
        values = grouped_value_projection(attended, self.wv)
        values = values + weights.sum(-1, keepdim=True) * self.bv.unsqueeze(0)
        merged = values.transpose(1, 2).contiguous().reshape(batch, slots, self.heads * self.head_dim)
        return F.linear(merged, self.ow, self.ob)


class _GroupedValueFiveTokenExactE(_LiftedFiveTokenExactE):
    @torch.no_grad()
    def _rebuild_lifted(self) -> None:
        front = self.model.frontend
        self._lifted = _M2GroupedValueMHA.from_mha(front.mha, front.slot_norm(front.slots))
        self._lifted_versions = (int(self._lifted.qwk._version), int(self._lifted.qbk._version))


class GroupedValueFiveTokenM2Decoder(LiftedFiveTokenM2Decoder):
    """Same public lifecycle as lifted M2; only the value GEMM grouping differs."""
    def reset(self, dataset_tags) -> None:
        super().reset(dataset_tags)
        old = self._engine
        replacement = _GroupedValueFiveTokenExactE(self.model, old.bank)
        replacement.raw = old.raw
        replacement.frontend = old.frontend
        self._engine = replacement
