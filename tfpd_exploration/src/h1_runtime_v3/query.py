"""Exact static-signed frontend specialization for the finite query reader.

Only bank-dependent carrier mixing weights are compiled at reset/rebuild.
The W700 boundary repair, independent-memory K/V law and read-time recency
tables are inherited unchanged. Unlike the FULL specialization, this class
retains the legal current-query memory cache and never substitutes FULL.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream


class StaticSignedQueryStream(CurrentQueryStream):
    def __init__(self, model, bank, **kwargs):
        if int(getattr(model, "frontend_contract_version", -1)) != 4:
            raise ValueError("static signed query requires the unchanged V4 signed frontend")
        self._static_weight = self._static_mask = None
        self.static_rebuild_count = 0
        super().__init__(model, bank, **kwargs)

    @torch.no_grad()
    def _rebuild(self, raw, reason):
        self._require_eval()
        if self.bank.unit_mask.ndim != 1:
            raise ValueError("static signed query currently requires one canonical one-dimensional bank mask")
        ref = self._parameter()
        prepared = raw.to(device=ref.device, dtype=ref.dtype)
        # The exact model's no-dropout operation, including LN/tanh, unit
        # centering and L2 normalization, is run once per valid binding.
        self._static_weight = self.model.frontend.weights(self.bank, prepared).detach().clone()
        self._static_mask = self.bank.unit_mask.to(prepared).unsqueeze(0).detach().clone()
        self.static_rebuild_count += 1
        super()._rebuild(prepared, reason)

    def _frontend(self, x):
        if self._static_weight is None or self._static_mask is None:
            raise RuntimeError("static signed carrier not prepared")
        frontend = self.model.frontend
        masked = x * self._static_mask
        signed = torch.einsum("bwn,nd->bwd", masked, self._static_weight)
        population = masked.sum(-1, keepdim=True) / self._static_mask.sum(-1, keepdim=True).clamp_min(1.0)
        mixed = signed + frontend.pop_projection(population)
        hidden = frontend.depthwise(F.pad(mixed.transpose(1, 2), (4, 0))).transpose(1, 2)
        return F.layer_norm(F.gelu(frontend.pointwise(hidden)), (hidden.shape[-1],),
                            weight=None, bias=None, eps=1e-5)

    @property
    def static_carrier_bytes(self):
        return sum(value.numel()*value.element_size() for value in (self._static_weight, self._static_mask)
                   if value is not None)

    @property
    def state_bytes(self):
        return super().state_bytes + self.static_carrier_bytes
