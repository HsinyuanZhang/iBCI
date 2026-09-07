"""Experimental M2 runtime with a precomposed frozen frontend affine term.

This module is intentionally not wired into any selected runtime, proof, or
benchmark.  It replaces, only for this opt-in adapter, the repeated spatial
expression ``local @ W + E0 @ We + T @ Wt + bias`` with a reset-time cached
``local @ W + (E0 @ We + T @ Wt + bias)``.  That is algebraically equivalent
over real arithmetic but changes FP32 addition association, so its contract is
native-output agreement at tolerance, never bit identity.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .h1_causal import _tensor_signature
from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank, W
from .m2_family_spatial import M2FamilyLiftedSpatial
from .grouped_value import grouped_value_projection


STATUS = "EXPERIMENTAL_NOT_PROMOTED"
EQUIVALENCE_ATOL = 1e-5
EQUIVALENCE_RTOL = 1e-5


class M2FamilyStaticAffineSpatial(M2FamilyLiftedSpatial):
    """Lifted spatial frontend with a bank/model-snapshot static affine cache."""

    @torch.no_grad()
    def __init__(self, model, bank):
        super().__init__(model, bank)
        # Preserve the individual E0/T projections inherited above so their
        # signatures remain independently auditable.  This extra tensor is
        # deliberately included in derived_tensors(), making direct mutation
        # a runtime error rather than silently changing future predictions.
        self.static_affine = (self.e + self.t + self.b).contiguous()
        if not bool(torch.isfinite(self.static_affine).all()):
            raise RuntimeError("nonfinite experimental M2 static affine cache")

    def derived_tensors(self):
        return super().derived_tensors() + (self.static_affine,)

    @torch.no_grad()
    def encode(self, local):
        batch, width, units, _ = local.shape
        keep = self.bank.unit_mask
        if keep.ndim == 1:
            keep = keep[None].expand(batch, -1)
        if keep.shape != (batch, units) or not bool(keep.any(1).all()):
            raise ValueError("nonempty BxN mask required")
        z = F.linear(local, self.lw) + self._static(self.static_affine)
        tok = self.front.token_norm(self.front.token_mlp[2](F.gelu(z))).reshape(batch * width, units, self.dim)
        logits = F.linear(tok, self.qwk.reshape(self.h * self.s, self.dim)).view(batch * width, units, self.h, self.s).permute(0, 2, 3, 1)
        logits = (logits + self.qbk[None, :, :, None]) * (self.d ** -.5)
        if self.route is not None:
            route = self.route
            route = route.expand(batch, -1, -1, -1) if route.shape[0] == 1 else route
            logits = logits + route[:, None].expand(-1, width, -1, -1, -1).reshape_as(logits)
        pad = (~keep)[:, None].expand(batch, width, units).reshape(batch * width, units)
        weight = torch.softmax(logits.masked_fill(pad[:, None, None], float("-inf")), -1)
        attended = torch.bmm(weight.reshape(batch * width, self.h * self.s, units), tok).reshape(batch * width, self.h, self.s, self.dim)
        value = grouped_value_projection(attended, self.vw) + weight.sum(-1, keepdim=True) * self.vb[None]
        merged = value.transpose(1, 2).reshape(batch * width, self.s, self.dim)
        slots = self.slot[None].expand(batch * width, -1, -1) + F.linear(merged, self.front.attn.out_proj_weight, self.front.attn.out_proj_bias)
        slots = slots + self.front.slot_ffn(self.front.slot_ffn_norm(slots))
        return self.front.slot_proj(slots.reshape(batch, width, self.s * self.dim))


class M2FamilyStaticAffineRuntime(M2FamilyCausalRuntime):
    """Opt-in tolerance-equivalent runtime; never the selected default path."""

    status = STATUS
    equivalence = {"kind": "real_arithmetic_equivalent_fp32_tolerance_not_bit_exact", "atol": EQUIVALENCE_ATOL, "rtol": EQUIVALENCE_RTOL}

    @torch.no_grad()
    def reset(self, bank: M2RuntimeBank | None = None, *, batch: int | None = None):
        bank = self.bank if bank is None else bank
        batch = getattr(self, "_active", self.batch_size) if batch is None else batch
        self._validate(bank, batch)
        spatial = M2FamilyStaticAffineSpatial(self.model, bank)
        raw = torch.zeros(batch, W, 96, dtype=torch.float32)
        # Initializing through the experimental operator keeps the retained
        # frontend internally self-consistent after an explicit reset.
        token = spatial.full(raw[:, :1])
        frontend = token.expand(-1, W, -1).clone()
        if not bool(torch.isfinite(frontend).all()):
            raise RuntimeError("nonfinite experimental M2 zero-history frontend")
        self.bank, self._active, self.spatial = bank, batch, spatial
        self.raw, self.frontend = raw, frontend
        self._state_signature = self._signature()
        self._derived_signature = _tensor_signature(self.spatial.derived_tensors())

    @torch.no_grad()
    def _audit(self):
        self._validate(self.bank, self._active)
        signature = self._signature()
        if signature != self._state_signature:
            spatial = M2FamilyStaticAffineSpatial(self.model, self.bank)
            frontend = spatial.full(self.raw)
            if not bool(torch.isfinite(frontend).all()):
                raise RuntimeError("nonfinite experimental M2 mutation refresh")
            self.spatial, self.frontend = spatial, frontend
            self._state_signature = self._signature()
            self._derived_signature = _tensor_signature(spatial.derived_tensors())
        elif _tensor_signature(self.spatial.derived_tensors()) != self._derived_signature:
            raise RuntimeError("derived experimental M2 static tensor mutated; reset required")

    @property
    def static_affine_bytes(self) -> int:
        """Informational breakdown; already included in ``derived_static``."""
        static = self.spatial.static_affine
        return static.numel() * static.element_size()
