"""M2 grouped-value runtime with algebraic five-token Conv1d repair."""
from __future__ import annotations

from torch.nn import functional as F

from .linear_conv import repaired_local_features_linear
from .m2_grouped import GroupedValueFiveTokenM2Decoder, _GroupedValueFiveTokenExactE


class _LinearConvFiveTokenExactE(_GroupedValueFiveTokenExactE):
    def _repair_frontend(self, raw):
        front = self.model.frontend
        cfg = front.cfg
        local = repaired_local_features_linear(front.local_conv, raw)
        batch, width, units, _ = local.shape
        first, activation, second = front.token_mlp
        tokens = front.token_norm(second(activation(
            F.linear(local, first.weight[:, :16], None) + self.static_token.unsqueeze(1)
        ))).reshape(batch * width, units, cfg.set_dim)
        query = front.slot_norm(front.slots).view(1, 1, cfg.slots, cfg.set_dim)
        query = query.expand(batch, width, -1, -1).reshape(batch * width, cfg.slots, cfg.set_dim)
        keep = self.bank.unit_mask
        if keep.ndim == 1:
            keep = keep.unsqueeze(0).expand(batch, -1)
        keep = keep.unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        out = query + self._lifted(tokens, keep)
        out = out + front.slot_ffn(front.slot_ffn_norm(out))
        return front.slot_proj(out.reshape(batch, width, cfg.slots * cfg.set_dim))


class LinearConvFiveTokenM2Decoder(GroupedValueFiveTokenM2Decoder):
    """No learned-state, stream-window, temporal, or public-API changes."""
    def reset(self, dataset_tags) -> None:
        super().reset(dataset_tags)
        old = self._engine
        replacement = _LinearConvFiveTokenExactE(self.model, old.bank)
        replacement.raw = old.raw
        replacement.frontend = old.frontend
        self._engine = replacement
