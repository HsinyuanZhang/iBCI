"""M2-style last-frame decoder for B1 spectrogram residuals."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_MS_BINS, VALID_END, VALID_START
from tfpd_exploration.src.b1_sfcj_v1.data import spec_frame_to_bin
from tfpd_exploration.src.b1_sfcj_v1.model import CrossAttentionLayer
from tfpd_exploration.src.b1_sfcj_v1.util import ieee_positive_zero

from .plan import ARMS, D_MODEL, N_HEADS, N_LAYERS, WINDOW


def causal_window_indices(window: int = WINDOW) -> torch.Tensor:
    rows = []
    for frame in range(VALID_START, VALID_END):
        endpoint = spec_frame_to_bin(frame)
        start = endpoint - int(window) + 1
        if start < 0 or endpoint >= N_MS_BINS:
            raise ValueError(f"invalid causal window frame={frame} [{start},{endpoint}]")
        rows.append(list(range(start, endpoint + 1)))
    return torch.tensor(rows, dtype=torch.long)


class FrameTARM(nn.Module):
    """SPINT unit-set decoder evaluated at each official B1 frame."""

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
        window: int = WINDOW,
    ):
        super().__init__()
        if fusion not in ("native", "jr1"):
            raise ValueError(fusion)
        if profile_kind not in ("zero", "spsfc9"):
            raise ValueError(profile_kind)
        self.fusion = fusion
        self.profile_kind = profile_kind
        self.window = int(window)
        self.d_model = int(d_model)
        self.pre_pool = nn.Sequential(nn.Linear(N_MS_BINS, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.post_pool = nn.Sequential(nn.Linear(d_model + 9, d_model), nn.ReLU(), nn.Linear(d_model, self.window))
        self.fc_in = nn.Sequential(nn.Linear(self.window, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.freq_queries = nn.Parameter(torch.randn(1, N_FREQ, d_model) / math.sqrt(d_model))
        self.transformer = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 1)
        if fusion == "jr1":
            self.alpha = nn.Parameter(torch.zeros(()))
            if not ieee_positive_zero(self.alpha):
                raise RuntimeError("alpha is not IEEE +0")
        else:
            self.register_parameter("alpha", None)
        self.register_buffer("log_mean", torch.as_tensor(log_mean, dtype=torch.float32).reshape(1, N_FREQ))
        self.register_buffer("log_std", torch.as_tensor(log_std, dtype=torch.float32).reshape(1, N_FREQ))
        self.register_buffer("window_indices", causal_window_indices(self.window), persistent=True)
        # Exact neural-free template at step zero, while random carrier columns
        # make profile-vs-zero head gradients different on the first update.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def identity(self, history: torch.Tensor, carrier: torch.Tensor, unit_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # history [K,900,85], carrier [85,9]
        if history.ndim != 3 or history.shape[1:] != (N_MS_BINS, N_CHANNELS):
            raise ValueError(f"history shape {tuple(history.shape)}")
        if carrier.shape != (N_CHANNELS, 9):
            raise ValueError("carrier shape")
        if self.profile_kind == "zero":
            carrier = torch.zeros_like(carrier)
        carrier = carrier * unit_mask[:, None]
        trials = history.permute(0, 2, 1)
        u = self.pre_pool(trials) * unit_mask[None, :, None]
        u_mean = u.mean(dim=0)
        h_native = self.post_pool(torch.cat([u_mean, carrier], dim=-1))
        c = carrier[None].expand(history.size(0), -1, -1)
        h_post = self.post_pool(torch.cat([u, c], dim=-1)).mean(dim=0)
        if self.fusion == "jr1":
            h = h_native + torch.tanh(self.alpha) * (h_post - h_native)
        else:
            h = h_native
        return h, {"h_native": h_native, "h_post": h_post}

    def forward_trial(
        self,
        *,
        current: torch.Tensor,
        history: torch.Tensor,
        carrier: torch.Tensor,
        template_stdlog: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if current.shape != (N_MS_BINS, N_CHANNELS):
            raise ValueError(f"current shape {tuple(current.shape)}")
        if template_stdlog.shape != (N_FREQ, 880):
            raise ValueError(f"template shape {tuple(template_stdlog.shape)}")
        if unit_mask.shape != (N_CHANNELS,) or torch.count_nonzero(unit_mask) == 0:
            raise ValueError("invalid unit mask")
        h, parts = self.identity(history, carrier, unit_mask)
        windows = current[self.window_indices]  # [700,W,N]
        src = (windows.permute(0, 2, 1) + h[None]) * unit_mask[None, :, None]
        tokens = self.fc_in(src)
        hidden = self.freq_queries.expand(src.size(0), -1, -1)
        padding = unit_mask.eq(0)[None].expand(src.size(0), -1)
        for layer in self.transformer:
            hidden = layer(hidden, tokens, key_padding_mask=padding)
        delta = self.head(hidden).squeeze(-1)  # [700,158]
        template = template_stdlog[:, VALID_START:VALID_END].T
        pred_stdlog = template + delta
        raw = torch.exp(pred_stdlog * self.log_std + self.log_mean)
        return {"raw_valid": raw, "delta_stdlog": delta, "pred_stdlog": pred_stdlog, **parts}


def _carrier_init(model: FrameTARM, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 99173)
    weight = model.post_pool[0].weight
    fan_in = weight.shape[1]
    fan_out = weight.shape[0]
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    values = torch.empty((fan_out, 9), dtype=weight.dtype)
    values.uniform_(-bound, bound, generator=generator)
    with torch.no_grad():
        weight[:, model.d_model :].copy_(values)


def build_arms(*, log_mean, log_std, seed: int) -> dict[str, FrameTARM]:
    arms = {}
    shared = None
    for name, fusion, profile in ARMS:
        torch.manual_seed(int(seed))
        model = FrameTARM(fusion=fusion, profile_kind=profile, log_mean=log_mean, log_std=log_std)
        _carrier_init(model, seed)
        if shared is None:
            shared = {k: v.detach().clone() for k, v in model.state_dict().items() if k != "alpha"}
        else:
            own = model.state_dict()
            model.load_state_dict({k: v for k, v in shared.items() if k in own and own[k].shape == v.shape}, strict=False)
        if model.alpha is not None:
            with torch.no_grad():
                model.alpha.zero_()
        arms[name] = model
    return arms


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
