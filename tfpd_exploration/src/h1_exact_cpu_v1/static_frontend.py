"""Exact static-carrier frontend specialization for V4 H1 FULL on CPU."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import (
    ExactFullWindowStream,
)


class StaticCarrierExactFullWindowStream(ExactFullWindowStream):
    """Exact E stream that caches only V4's bank-only carrier weights.

    ``SignedCarrierFrontend.weights(bank, x)`` has no dependence on neural
    observations in its no-dropout branch.  The parent class's virtual
    ``_rebuild`` is invoked before a frontend rebuild whenever model parameters,
    buffers, E0/T, unit mask, dtype, device, or roster change.  We calculate
    the static weight and mask immediately before delegating to that rebuild;
    this preserves those parent invalidation rules rather than inventing a
    weaker cache identity.

    Temporal state is still fully recomputed by ExactFullWindowStream.  This
    class does not retain temporal K/V or any contextualized history.
    """

    def __init__(self, model, bank, **kwargs) -> None:
        self._static_weight: torch.Tensor | None = None
        self._static_mask: torch.Tensor | None = None
        self.static_rebuild_count = 0
        super().__init__(model, bank, **kwargs)

    @torch.no_grad()
    def _rebuild(self, raw: torch.Tensor, reason: str) -> None:
        ref = self._parameter()
        prepared = raw.to(device=ref.device, dtype=ref.dtype)
        frontend = self.model.frontend
        # Use the model's original public weight method for the exact same
        # carrier concatenation, layer norm, phi/tanh, mask centering, and L2
        # normalization.  The cached tensors are detached only after that
        # original calculation has completed.
        self._static_weight = frontend.weights(self.bank, prepared).detach().clone()
        self._static_mask = self.bank.unit_mask.to(
            device=prepared.device, dtype=prepared.dtype
        ).unsqueeze(0).detach().clone()
        self.static_rebuild_count += 1
        super()._rebuild(prepared, reason)

    def _frontend(self, x: torch.Tensor) -> torch.Tensor:
        if self._static_weight is None or self._static_mask is None:
            raise RuntimeError("static carrier was not prepared before frontend evaluation")
        frontend = self.model.frontend
        # This is exactly SignedCarrierFrontend.forward(..., dropout_keep=None)
        # with its bank-only `weights` and `bank_mask` supplied from `_rebuild`.
        masked = x * self._static_mask
        signed = torch.einsum("bwn,nd->bwd", masked, self._static_weight)
        population = masked.sum(-1, keepdim=True) / self._static_mask.sum(-1, keepdim=True).clamp_min(1.0)
        mixed = signed + frontend.pop_projection(population)
        hidden = frontend.depthwise(F.pad(mixed.transpose(1, 2), (4, 0))).transpose(1, 2)
        return F.layer_norm(
            F.gelu(frontend.pointwise(hidden)), (hidden.shape[-1],), weight=None, bias=None, eps=1e-5
        )

    @property
    def static_carrier_bytes(self) -> int:
        return sum(value.numel() * value.element_size()
                   for value in (self._static_weight, self._static_mask) if value is not None)

    @property
    def state_bytes(self) -> int:
        # Parent reports rolling raw/features, not this specialization's two
        # separately allocated immutable carrier tensors.
        return super().state_bytes + self.static_carrier_bytes
