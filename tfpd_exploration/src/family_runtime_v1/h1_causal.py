"""Exact W700 causal H1 streaming engine for common FLAT/ROUTE models.

The finite-window spatial cache repairs four left-boundary tokens plus the
newest one. Temporal layers 1..3 remain full-window; only the unused query and
FFN rows in layer 4 are omitted. No cross-window temporal KV cache is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .h1_lifted_frontend import H1LiftedSpatialFrontend
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

W = 700
DIVISOR = 20.0


def _tensor_signature(values):
    entries = []
    for value in values:
        try:
            version = int(value._version)
        except RuntimeError as exc:
            raise RuntimeError("auditable tensors must be created outside inference_mode") from exc
        entries.append((value.data_ptr(), tuple(value.shape), value.dtype, value.device, version))
    return tuple(entries)


@dataclass(frozen=True)
class H1StateBytes:
    raw_and_frontend: int
    model_parameters_and_buffers: int
    active_banks: int
    derived_static: int


class H1CausalRuntime:
    """CPU float32 model runtime; observations are contiguous float32 [B,176]."""

    def __init__(self, model, bank: H1Bank, *, batch_size=1):
        if int(batch_size) != batch_size or not 1 <= int(batch_size) <= 8:
            raise ValueError("configured batch must be an integer in 1..8")
        self.model = model
        self.batch_size = int(batch_size)
        self.bank = None
        self.raw = self.frontend = None
        self.reset(bank, batch=self.batch_size)

    def _validate(self, bank, batch):
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("H1 runtime requires eval mode throughout")
        cfg = self.model.cfg
        if (cfg.window, cfg.n_units, cfg.layers, cfg.out_dim, cfg.prediction_divisor) != (W, 176, 4, 7, DIVISOR):
            raise RuntimeError("formal H1 W700/176/four-layer/seven-output geometry required")
        if float(getattr(self.model, "activity_scale", 1.0)) != 1.0:
            raise RuntimeError("formal H1 activity scale must remain one")
        for value in (*self.model.parameters(), *self.model.buffers(), bank.E0, bank.T):
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise RuntimeError("H1 runtime supports only CPU float32 model and bank tensors")
        if bank.unit_mask.device.type != "cpu" or bank.unit_mask.dtype != torch.bool:
            raise RuntimeError("H1 mask must be CPU bool")
        if bank.E0.shape not in ((176, 700), (batch, 176, 700)) or bank.T.shape not in ((176, 4), (batch, 176, 4)):
            raise RuntimeError("H1 bank batch/geometry drift")
        if bank.unit_mask.shape not in ((176,), (batch, 176)):
            raise RuntimeError("H1 unit-mask batch/geometry drift")
        if int(batch) != batch or not 1 <= int(batch) <= self.batch_size:
            raise RuntimeError("active H1 batch exceeds configured roster")

    def _signature(self):
        return (id(self.model), _tensor_signature(
            (*self.model.parameters(), *self.model.buffers(), self.bank.E0, self.bank.T, self.bank.unit_mask)))

    def _derived_tensors(self):
        lift = self.spatial
        return (lift.e0_term, lift.hc_term, lift.slot_query, lift.qwk, lift.qbk,
                *((lift.route_bonus,) if lift.route_bonus is not None else ()))

    @torch.no_grad()
    def reset(self, bank=None, *, batch=None):
        candidate_bank = self.bank if bank is None else bank
        candidate_batch = getattr(self, "_active", self.batch_size) if batch is None else batch
        self._validate(candidate_bank, candidate_batch)
        spatial = H1LiftedSpatialFrontend(self.model, candidate_bank)
        raw = torch.zeros(candidate_batch, W, 176, dtype=torch.float32)
        # Every pre-PE token of an all-zero causal history is identical.
        token = self.model.encode_frontend(raw[:, :1], candidate_bank)
        frontend = token.expand(-1, W, -1).clone()
        if not bool(torch.isfinite(frontend).all()):
            raise RuntimeError("nonfinite H1 zero-history reset")
        self.bank, self._active, self.spatial = candidate_bank, candidate_batch, spatial
        self.raw, self.frontend = raw, frontend
        self._state_signature = self._signature()
        self._derived_signature = _tensor_signature(self._derived_tensors())

    @torch.no_grad()
    def _audit(self):
        self._validate(self.bank, self._active)
        signature = self._signature()
        if signature != self._state_signature:
            # Rebuild all cached rows from the actual retained W700 history,
            # then rebuild constant queries, projections and route bonuses.
            spatial = H1LiftedSpatialFrontend(self.model, self.bank)
            frontend = self.model.encode_frontend(self.raw, self.bank)
            if not bool(torch.isfinite(frontend).all()):
                raise RuntimeError("nonfinite H1 mutation refresh")
            self.spatial, self.frontend = spatial, frontend
            self._state_signature = self._signature()
            self._derived_signature = _tensor_signature(self._derived_tensors())
        elif _tensor_signature(self._derived_tensors()) != self._derived_signature:
            raise RuntimeError("H1 derived static tensor mutated; reset required")

    @torch.no_grad()
    def _last(self):
        hidden = self.frontend + self.model.temporal.pe[:W][None]
        for block in self.model.temporal.blocks[:-1]:
            hidden = block(hidden)
        block = self.model.temporal.blocks[-1]
        normalized = block.norm1(hidden)
        attn = block.attn
        batch, width, dim = normalized.shape
        query = F.linear(normalized[:, -1:], attn.qkv.weight[:dim], attn.qkv.bias[:dim])
        query = query.view(batch, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        kv = F.linear(normalized, attn.qkv.weight[dim:], attn.qkv.bias[dim:])
        kv = kv.view(batch, width, 2, attn.n_heads, attn.head_dim)
        key, value = kv.unbind(dim=2)
        output = F.scaled_dot_product_attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                               dropout_p=0.0, is_causal=False)
        output = attn.proj(output.transpose(1, 2).contiguous().view(batch, 1, dim))
        last = hidden[:, -1:] + output
        last = last + block.ffn(block.norm2(last))
        return self.model.readout(self.model.final_norm(last))[:, 0]

    @torch.no_grad()
    def predict(self, observations):
        value = np.asarray(observations)
        if (value.shape != (self._active, 176) or value.dtype != np.float32
                or not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError("expected finite contiguous float32 [active,176]")
        self._audit()
        self.raw = torch.cat((self.raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        repair = self.spatial.repair(self.raw)
        self.frontend = torch.cat((repair[:, :4], self.frontend[:, 5:], repair[:, -1:]), dim=1)
        output = (self._last().numpy() / DIVISOR).copy()
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite H1 native output")
        if self._active == self.batch_size:
            return output
        padded = np.zeros((self.batch_size, 7), dtype=np.float32)
        padded[:self._active] = output
        return padded

    def observe(self, observations):
        self.predict(observations)

    def on_done(self, dones):
        flags = np.asarray(dones)
        if flags.shape != (self._active,):
            raise ValueError("one H1 done flag per active stream required")
        self._audit()  # Continual H1: flags do not reset or advance the history.

    def state_bytes(self):
        def count(values):
            return sum(value.numel() * value.element_size() for value in values)
        return H1StateBytes(count((self.raw, self.frontend)),
                            count((*self.model.parameters(), *self.model.buffers())),
                            count((self.bank.E0, self.bank.T, self.bank.unit_mask)),
                            count(self._derived_tensors()))
