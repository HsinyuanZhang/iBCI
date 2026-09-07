"""Independent M1 candidate: five-token repair plus algebraically lifted K/V."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .m1 import FiveTokenCurrentQueryStream
from .lifted_attention import LiftedSlotAttention
from .repair import repaired_local_features


class LiftedFiveTokenCurrentQueryStream(FiveTokenCurrentQueryStream):
    @torch.no_grad()
    def _build_static(self):
        super()._build_static()
        self._lifted = LiftedSlotAttention.from_split(self.model.frontend.attn, self._static["slots"])

    @torch.no_grad()
    def _repair_frontend(self, raw, active):
        f = self.model.frontend
        local = repaired_local_features(f.local_conv, raw)
        b, t, n, _ = local.shape
        hidden = F.linear(local, self._static["wloc"], None) + self._static["affine"][:active].unsqueeze(1)
        tokens = f.token_norm(f.token_mlp.fc2(F.gelu(hidden))).reshape(b * t, n, f.cfg.set_dim)
        keep = self.bank.unit_mask[:active, None].expand(-1, t, -1).reshape(b * t, n)
        attended = self._lifted(tokens, keep)
        query = self._static["slots"].unsqueeze(1).expand(b, t, -1, -1).reshape(b * t, f.cfg.slots, f.cfg.set_dim)
        out = query + attended
        out = out + f.slot_ffn(f.slot_ffn_norm(out))
        return f.slot_proj(out.reshape(b, t, f.cfg.slots * f.cfg.set_dim))

    @property
    def static_cache_bytes(self):
        return super().static_cache_bytes + self._lifted.owned_bytes
