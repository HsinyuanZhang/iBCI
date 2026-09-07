"""Exact streaming runtime for the newly trained explicit FLAT/ROUTE M2 pair.

This accepts an actual M2FamilyDecoder and a frozen bank, not the historical
e8 submission payload. No checkpoint selection, calibration fitting or labels
are involved. The first three temporal blocks retain full W50 evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .h1_causal import _tensor_signature
from .m2_family_spatial import M2FamilyLiftedSpatial

W = 50


@dataclass(frozen=True)
class M2RuntimeBank:
    E0: torch.Tensor
    T: torch.Tensor
    unit_mask: torch.Tensor


class M2FamilyCausalRuntime:
    """CPU FP32 [active,96] bins to owning native [configured,2] output."""

    def __init__(self, model, bank, *, batch_size=1):
        if int(batch_size) != batch_size or not 1 <= int(batch_size) <= 7:
            raise ValueError("M2 configured batch must be an integer in 1..7")
        self.model, self.batch_size = model, int(batch_size)
        self.bank = self.raw = self.frontend = None
        self.reset(bank, batch=self.batch_size)

    def _validate(self, bank, batch):
        if int(batch) != batch or not 1 <= int(batch) <= self.batch_size:
            raise ValueError("M2 active batch outside configured roster")
        cfg = self.model.cfg
        if (cfg.window, cfg.channels, cfg.identity_dim, cfg.t4_dim, cfg.layers, cfg.out_dim) != (50, 96, 50, 4, 4, 2):
            raise RuntimeError("actual family W50/96/E050/T4/L4/out2 required")
        if any(m.training for m in self.model.modules()):
            raise RuntimeError("M2 runtime requires eval mode throughout")
        if bank.E0.shape not in ((96, 50), (batch, 96, 50)) or bank.T.shape not in ((96, 4), (batch, 96, 4)):
            raise ValueError("M2 E0/T bank shape drift")
        if bank.unit_mask.shape not in ((96,), (batch, 96)):
            raise ValueError("M2 bank mask shape drift")
        for value in (*self.model.parameters(), *self.model.buffers(), bank.E0, bank.T):
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise RuntimeError("M2 runtime requires CPU float32 model and bank")
        if bank.unit_mask.device.type != "cpu" or bank.unit_mask.dtype != torch.bool:
            raise RuntimeError("M2 runtime requires CPU bool mask")
        if not bool(bank.unit_mask.any(dim=-1).all()):
            raise ValueError("native M2 attention requires at least one unit per stream")

    def _signature(self):
        return id(self.model), _tensor_signature((*self.model.parameters(), *self.model.buffers(),
                                                 self.bank.E0, self.bank.T, self.bank.unit_mask))

    def _native_frontend(self, raw, bank):
        mask = bank.unit_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(raw.shape[0], -1)
        return self.model.frontend(raw, bank, mask)

    @torch.no_grad()
    def reset(self, bank=None, *, batch=None):
        bank = self.bank if bank is None else bank
        batch = getattr(self, "_active", self.batch_size) if batch is None else batch
        self._validate(bank, batch)
        spatial = M2FamilyLiftedSpatial(self.model, bank)
        raw = torch.zeros(batch, W, 96, dtype=torch.float32)
        token = self._native_frontend(raw[:, :1], bank)
        frontend = token.expand(-1, W, -1).clone()
        if not bool(torch.isfinite(frontend).all()):
            raise RuntimeError("nonfinite M2 zero-history frontend")
        self.bank, self._active, self.spatial = bank, batch, spatial
        self.raw, self.frontend = raw, frontend
        self._state_signature = self._signature()
        self._derived_signature = _tensor_signature(self.spatial.derived_tensors())

    @torch.no_grad()
    def _audit(self):
        self._validate(self.bank, self._active)
        signature = self._signature()
        if signature != self._state_signature:
            spatial = M2FamilyLiftedSpatial(self.model, self.bank)
            frontend = self._native_frontend(self.raw, self.bank)
            if not bool(torch.isfinite(frontend).all()):
                raise RuntimeError("nonfinite M2 mutation refresh")
            self.spatial, self.frontend = spatial, frontend
            self._state_signature = self._signature()
            self._derived_signature = _tensor_signature(spatial.derived_tensors())
        elif _tensor_signature(self.spatial.derived_tensors()) != self._derived_signature:
            raise RuntimeError("derived M2 static tensor mutated; reset required")

    @torch.no_grad()
    def _last(self):
        hidden = self.frontend + self.model.temporal.pe[:W][None]
        # Actual family training blocks return (hidden, cache), unlike the
        # historical serialized deployment module's tensor-only blocks.
        for block in self.model.temporal.blocks[:-1]:
            hidden, _ = block(hidden, kv_cache=None, past_len=0)
        block = self.model.temporal.blocks[-1]
        normalized = block.norm1(hidden)
        attn = block.attn
        batch, width, dim = normalized.shape
        query = F.linear(normalized[:, -1:], attn.qkv.weight[:dim], attn.qkv.bias[:dim])
        query = query.view(batch, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        kv = F.linear(normalized, attn.qkv.weight[dim:], attn.qkv.bias[dim:])
        key, value = kv.view(batch, width, 2, attn.n_heads, attn.head_dim).unbind(2)
        output = F.scaled_dot_product_attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                               dropout_p=0.0, is_causal=False)
        output = attn.proj(output.transpose(1, 2).contiguous().view(batch, 1, dim))
        last = hidden[:, -1:] + output
        last = last + block.ffn(block.norm2(last))
        return self.model.readout(self.model.final_norm(last))[:, 0]

    @torch.no_grad()
    def predict(self, observations):
        value = np.asarray(observations)
        if value.shape != (self._active, 96) or value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError("expected finite contiguous float32 [active,96]")
        self._audit()
        self.raw = torch.cat((self.raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        repair = self.spatial.repair(self.raw)
        self.frontend = torch.cat((repair[:, :4], self.frontend[:, 5:], repair[:, -1:]), dim=1)
        output = (self._last().numpy() / 5).copy()
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite native M2 prediction")
        if self._active == self.batch_size:
            return output
        padded = np.zeros((self.batch_size, 2), dtype=np.float32)
        padded[:self._active] = output
        return padded

    def observe(self, observations):
        self.predict(observations)

    def on_done(self, dones):
        if np.asarray(dones).shape != (self._active,):
            raise ValueError("one M2 done flag per active stream required")
        self._audit()  # Continual M2 flags do not reset or advance history.

    def state_bytes(self):
        def count(values):
            return sum(x.numel() * x.element_size() for x in values)
        return {"raw_and_frontend": count((self.raw, self.frontend)),
                "model_parameters_and_buffers": count((*self.model.parameters(), *self.model.buffers())),
                "active_banks": count((self.bank.E0, self.bank.T, self.bank.unit_mask)),
                "derived_static": count(self.spatial.derived_tensors())}
