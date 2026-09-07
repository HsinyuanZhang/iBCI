"""M2 exact causal decoder with sparse boundary frontend recomputation."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder, RuntimeV3Error, _ExactEFinalQ
from .repair import repaired_local_features


class _FiveTokenExactE(_ExactEFinalQ):
    def _repair_frontend(self, raw: torch.Tensor) -> torch.Tensor:
        front = self.model.frontend
        cfg = front.cfg
        local = repaired_local_features(front.local_conv, raw)
        batch, width, n_units, _ = local.shape
        first, activation, second = front.token_mlp
        tokens = F.linear(local, first.weight[:, :16], None) + self.static_token.unsqueeze(1)
        tokens = front.token_norm(second(activation(tokens)))
        slots = front.slot_norm(front.slots).view(1, 1, cfg.slots, cfg.set_dim)
        q = slots.expand(batch, width, -1, -1).reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, n_units, cfg.set_dim)
        keep = self.bank.unit_mask
        if keep.ndim == 1:
            keep = keep.unsqueeze(0).expand(batch, -1)
        pad = (~keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
        attended, _ = front.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        out = q + attended
        out = out + front.slot_ffn(front.slot_ffn_norm(out))
        return front.slot_proj(out.reshape(batch, width, cfg.slots * cfg.set_dim))

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        self._refresh_if_mutated()
        if self.raw is None or self.frontend is None:
            raise RuntimeV3Error("advance before rebuild")
        if next_bin.shape != (self.raw.shape[0], 1, self.raw.shape[2]):
            raise RuntimeV3Error("new bin shape / roster drift")
        self.raw = torch.cat((self.raw[:, 1:], next_bin), dim=1)
        repair = self._repair_frontend(self.raw)
        self.frontend = torch.cat((repair[:, :4], self.frontend[:, 5:], repair[:, -1:]), dim=1)
        return self._last(self.frontend)


class FiveTokenM2Decoder(RuntimeV3Decoder):
    """Local candidate; original public API and mutation-refresh law retained."""

    def reset(self, dataset_tags) -> None:
        super().reset(dataset_tags)
        original = self._engine
        replacement = _FiveTokenExactE(self.model, original.bank)
        replacement.raw = original.raw
        replacement.frontend = original.frontend
        self._engine = replacement
