"""Template-anchored B1 SPINT with padded causal activity histories."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_MS_BINS, N_SPEC_FRAMES
from tfpd_exploration.src.b1_sfcj_v1.model import B1SpintSFCJ
from tfpd_exploration.src.b1_sfcj_v1.util import ieee_positive_zero

from .plan import ARMS, D_MODEL, N_HEADS, N_LAYERS


class TemplateAnchoredB1(nn.Module):
    """Predict only a standardized-log residual around the target M3 template.

    The identity operator matches the native/J-R1 SPINT algebra.  Histories are
    padded only for batching; ``history_mask`` supplies the exact causal K for
    every sample.  The residual head is zero initialized, so all arms start at
    the exact same TPL-M3 prediction.
    """

    def __init__(
        self,
        *,
        fusion: str,
        profile_kind: str,
        log_mean,
        log_std,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        if profile_kind not in ("zero", "spsfc9"):
            raise ValueError(profile_kind)
        self.fusion = fusion
        self.profile_kind = profile_kind
        self.core = B1SpintSFCJ(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            fusion=fusion,
            carrier_kind="sfc9" if profile_kind == "spsfc9" else "zero",
            dropout_rate=0.0,
            log_mean=torch.as_tensor(log_mean, dtype=torch.float32),
            log_std=torch.as_tensor(log_std, dtype=torch.float32),
        )
        # Exact TPL-M3 anchor at initialization.  Once the head receives its
        # first gradient, the rest of the SPINT path becomes trainable too.
        with torch.no_grad():
            self.core.head.weight.zero_()
            self.core.head.bias.zero_()

    @property
    def alpha(self):
        return self.core.alpha

    def identity_from_padded(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        carrier: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if history.ndim != 4 or history.shape[2:] != (N_MS_BINS, N_CHANNELS):
            raise ValueError(f"history must be [B,K,900,85], got {tuple(history.shape)}")
        bsz, kmax = history.shape[:2]
        if history_mask.shape != (bsz, kmax):
            raise ValueError("history mask shape mismatch")
        if carrier.shape != (bsz, N_CHANNELS, 9):
            raise ValueError("carrier shape mismatch")
        if unit_mask.shape != (bsz, N_CHANNELS):
            raise ValueError("unit mask shape mismatch")
        if torch.any(history_mask.sum(dim=1) < 1):
            raise ValueError("every sample needs at least one completed history trial")

        if self.profile_kind == "zero":
            carrier = torch.zeros_like(carrier)
        carrier = carrier * unit_mask.unsqueeze(-1)

        trials = history.permute(0, 1, 3, 2).reshape(bsz * kmax, N_CHANNELS, N_MS_BINS)
        u = self.core.pre_pool(trials).reshape(bsz, kmax, N_CHANNELS, -1)
        u = u * unit_mask[:, None, :, None]
        valid = history_mask[:, :, None, None].to(dtype=u.dtype)
        denom = history_mask.sum(dim=1).to(dtype=u.dtype).reshape(bsz, 1, 1)
        u_mean = (u * valid).sum(dim=1) / denom
        h_native = self.core.post_pool(torch.cat([u_mean, carrier], dim=-1))

        c_each = carrier[:, None].expand(-1, kmax, -1, -1)
        post_each = self.core.post_pool(torch.cat([u, c_each], dim=-1))
        h_post = (post_each * valid).sum(dim=1) / denom
        if self.fusion == "jr1":
            h = h_native + torch.tanh(self.core.alpha) * (h_post - h_native)
        else:
            h = h_native
        return h, {"h_native": h_native, "h_post": h_post, "K": history_mask.sum(dim=1)}

    def forward(
        self,
        *,
        current: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        carrier: torch.Tensor,
        template_stdlog: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if current.shape[1:] != (N_MS_BINS, N_CHANNELS):
            raise ValueError(f"current must be [B,900,85], got {tuple(current.shape)}")
        if template_stdlog.shape[1:] != (N_FREQ, N_SPEC_FRAMES):
            raise ValueError("template shape mismatch")
        if torch.any(unit_mask.sum(dim=1) < 1):
            raise ValueError("whole-unit dropout removed every channel")

        identity, parts = self.identity_from_padded(history, history_mask, carrier, unit_mask)
        src = (current.permute(0, 2, 1) + identity) * unit_mask.unsqueeze(-1)
        tokens = self.core.fc_in(src)
        hidden = self.core.freq_queries.expand(current.size(0), -1, -1)
        dropped = unit_mask.eq(0)
        for layer in self.core.transformer:
            hidden = layer(hidden, tokens, key_padding_mask=dropped)
        delta_stdlog = self.core.head(hidden)
        pred_stdlog = template_stdlog + delta_stdlog
        log_spec = pred_stdlog * self.core.log_std + self.core.log_mean
        raw = torch.exp(log_spec)
        return {
            "raw": raw,
            "pred_stdlog": pred_stdlog,
            "delta_stdlog": delta_stdlog,
            **parts,
        }


def build_four_arms(*, log_mean, log_std, seed: int) -> dict[str, TemplateAnchoredB1]:
    arms = {}
    shared = None
    for name, fusion, profile in ARMS:
        torch.manual_seed(int(seed))
        model = TemplateAnchoredB1(
            fusion=fusion,
            profile_kind=profile,
            log_mean=log_mean,
            log_std=log_std,
        )
        if shared is None:
            shared = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("alpha")}
        else:
            own = model.state_dict()
            overlap = {k: v for k, v in shared.items() if k in own and own[k].shape == v.shape}
            model.load_state_dict(overlap, strict=False)
        model.core.zero_carrier_columns()
        if model.alpha is not None:
            with torch.no_grad():
                model.alpha.copy_(torch.tensor(0.0, dtype=model.alpha.dtype))
            if not ieee_positive_zero(model.alpha):
                raise RuntimeError(f"{name} alpha is not IEEE +0")
        arms[name] = model
    return arms


def parameter_count(model: nn.Module) -> int:
    return int(sum(math.prod(p.shape) for p in model.parameters()))
