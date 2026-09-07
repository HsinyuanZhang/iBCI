"""M1 exact current-query stream with one 5-token nonlinear frontend call."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from tfpd_exploration.src.m1_runtime_v3.runtime import HeterogeneousCurrentQueryStream
from .repair import repaired_local_features


class FiveTokenCurrentQueryStream(HeterogeneousCurrentQueryStream):
    """Preserve V3 guards/state/weights; eliminate four unused slot evaluations."""

    @torch.no_grad()
    def _repair_frontend(self, raw: torch.Tensor, active: int) -> torch.Tensor:
        f = self.model.frontend
        local = repaired_local_features(f.local_conv, raw)
        b, t, n, _ = local.shape
        a = f.attn
        affine = self._static["affine"][:active]
        keep = self.bank.unit_mask[:active]
        hidden = F.linear(local, self._static["wloc"], None) + affine.unsqueeze(1)
        tokens = f.token_norm(f.token_mlp.fc2(F.gelu(hidden)))
        slots = self._static["slots"].unsqueeze(1).expand(b, t, -1, -1)
        query = slots.reshape(b * t, f.cfg.slots, f.cfg.set_dim)
        q = self._static["q"].unsqueeze(1).expand(b, t, -1, -1, -1)
        q = q.reshape(b * t, a.n_heads, f.cfg.slots, a.head_dim)
        key = a.k_proj(tokens).view(b * t, n, a.n_heads, a.head_dim).transpose(1, 2)
        value = a.v_proj(tokens).view(b * t, n, a.n_heads, a.head_dim).transpose(1, 2)
        logits = torch.matmul(q, key.transpose(-2, -1)) * (a.head_dim ** -0.5)
        pad = (~keep).unsqueeze(1).expand(-1, t, -1).reshape(b * t, n)
        weights = torch.nan_to_num(torch.softmax(logits.masked_fill(pad[:, None, None, :], float("-inf")), dim=-1), nan=0.0)
        attended = a.out_proj(torch.matmul(weights, value).transpose(1, 2).contiguous().view(b * t, f.cfg.slots, f.cfg.set_dim))
        out = query + attended
        out = out + f.slot_ffn(f.slot_ffn_norm(out))
        return f.slot_proj(out.reshape(b, t, f.cfg.slots * f.cfg.set_dim))

    @torch.no_grad()
    def observe(self, observations, *, active: int | None = None) -> None:
        self._check_immutable()
        x = torch.as_tensor(observations, device=self.raw.device, dtype=self.raw.dtype)
        active = self.batch if active is None else int(active)
        if not 0 < active <= self.batch or x.ndim != 2 or x.shape != (active, self.units) or not bool(torch.isfinite(x).all()):
            raise ValueError("observations must be finite [active,N], 1<=active<=B")
        raw = self.raw.clone()
        raw[:active, :-1] = self.raw[:active, 1:]
        raw[:active, -1] = x
        repair = self._repair_frontend(raw[:active], active)
        z = self.z.clone()
        z[:active, :4] = repair[:, :4]
        z[:active, 4:-1] = self.z[:active, 5:]
        z[:active, -1:] = repair[:, -1:]
        self.raw, self.z = raw, z
        self._advance_memory((0, 1, 2, 3, self.window - 1), active)
        self.observation_counts[:active] += 1
