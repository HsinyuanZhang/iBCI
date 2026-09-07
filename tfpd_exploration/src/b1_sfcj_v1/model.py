"""B1-SPINT substrate with common 9-D carrier interface and J-R1 fusion."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    ARM_SPECS,
    CARRIER_DIM,
    D_MODEL,
    N_CHANNELS,
    N_FREQ,
    N_HEADS,
    N_LAYERS,
    N_MS_BINS,
    N_SPEC_FRAMES,
)
from .util import ieee_positive_zero


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        qn = self.norm1(query)
        kn = self.norm1(key_value)
        attn_out, _ = self.cross_attn(query=qn, key=kn, value=kn, key_padding_mask=key_padding_mask)
        x = query + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class B1SpintSFCJ(nn.Module):
    """Six arms share this class. Native substate is the non-alpha modules."""

    def __init__(
        self,
        *,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        fusion: str = "native",
        carrier_kind: str = "zero",
        dropout_rate: float = 0.0,
        log_mean: Optional[torch.Tensor] = None,
        log_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if fusion not in ("native", "jr1"):
            raise ValueError(fusion)
        if carrier_kind not in ("zero", "sfc4", "sfc9"):
            raise ValueError(carrier_kind)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.fusion = fusion
        self.carrier_kind = carrier_kind
        self.dropout_rate = float(dropout_rate)
        self.query_label_access_count = 0
        self.target_optimizer_state = None

        self.pre_pool = nn.Sequential(
            nn.Linear(N_MS_BINS, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(d_model + CARRIER_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, N_MS_BINS),
        )
        self.fc_in = nn.Sequential(
            nn.Linear(N_MS_BINS, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.freq_queries = nn.Parameter(torch.randn(1, N_FREQ, d_model) / math.sqrt(d_model))
        self.transformer = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, N_SPEC_FRAMES)
        if fusion == "jr1":
            self.alpha = nn.Parameter(torch.zeros(()))
            if not ieee_positive_zero(self.alpha):
                raise RuntimeError("alpha is not IEEE +0")
        else:
            self.register_parameter("alpha", None)

        mean = torch.zeros(N_FREQ) if log_mean is None else torch.as_tensor(log_mean, dtype=torch.float32)
        std = torch.ones(N_FREQ) if log_std is None else torch.as_tensor(log_std, dtype=torch.float32)
        self.register_buffer("log_mean", mean.reshape(1, N_FREQ, 1))
        self.register_buffer("log_std", std.reshape(1, N_FREQ, 1))
        self.zero_carrier_columns()

    def zero_carrier_columns(self) -> None:
        with torch.no_grad():
            self.post_pool[0].weight[:, self.d_model :].zero_()

    def native_state_dict(self) -> dict:
        skip = {"alpha"}
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k not in skip}

    def identity_from_stack(self, calib: torch.Tensor, carrier: torch.Tensor, unit_mask: Optional[torch.Tensor] = None):
        """calib [B,K,900,85], carrier [B,85,9] -> identity [B,85,900] via sequential whole-stack sums."""
        bsz, k, _, _ = calib.shape
        trials = calib.permute(0, 1, 3, 2)
        if unit_mask is not None:
            c = carrier * unit_mask.unsqueeze(-1)
        else:
            c = carrier
        u_list = []
        post_list = []
        for idx in range(k):
            u_i = self.pre_pool(trials[:, idx])
            if unit_mask is not None:
                u_i = u_i * unit_mask.unsqueeze(-1)
            u_list.append(u_i)
            post_list.append(self.post_pool(torch.cat([u_i, c], dim=-1)))
        sum_u = u_list[0]
        sum_post = post_list[0]
        for idx in range(1, k):
            sum_u = sum_u + u_list[idx]
            sum_post = sum_post + post_list[idx]
        k_f = float(k)
        u_mean = sum_u / k_f
        h_n = self.post_pool(torch.cat([u_mean, c], dim=-1))
        h_p = sum_post / k_f
        if self.fusion == "jr1":
            h = h_n + torch.tanh(self.alpha) * (h_p - h_n)
        else:
            h = h_n
        u = torch.stack(u_list, dim=1)
        return h, u, h_n, h_p

    def forward(
        self,
        x: torch.Tensor,
        calib: torch.Tensor,
        carrier: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        return_raw: bool = True,
    ):
        """
        x: [B,900,85] current trial
        calib: [B,K,900,85] activity pool (does not include current trial)
        carrier: [B,85,9]
        returns standardized-log [B,158,880] or raw via inverse-std+exp
        """
        if self.carrier_kind == "zero":
            carrier = torch.zeros_like(carrier)
        identity, _, _, _ = self.identity_from_stack(calib, carrier, unit_mask=unit_mask)
        src = x.permute(0, 2, 1) + identity
        if unit_mask is None:
            unit_mask = torch.ones(src.size(0), src.size(1), device=src.device, dtype=src.dtype)
            if self.training and self.dropout_rate > 0:
                unit_mask = F.dropout(unit_mask, p=self.dropout_rate, training=True)
        src = src * unit_mask.unsqueeze(-1)
        tokens = self.fc_in(src)
        dropped = unit_mask == 0
        query = self.freq_queries.expand(src.size(0), -1, -1)
        hidden = query
        for layer in self.transformer:
            hidden = layer(hidden, tokens, key_padding_mask=dropped.bool())
        std_log = self.head(hidden)
        if not return_raw:
            return std_log
        log_spec = std_log * self.log_std + self.log_mean
        return torch.exp(log_spec)


def build_six_arms(*, d_model: int = D_MODEL, n_heads: int = N_HEADS, seed: int = 0, **kwargs) -> dict:
    torch.manual_seed(seed)
    proto = B1SpintSFCJ(d_model=d_model, n_heads=n_heads, fusion="native", carrier_kind="zero", **kwargs)
    native = proto.native_state_dict()
    arms = {}
    for name, fusion, kind in ARM_SPECS:
        torch.manual_seed(seed)
        model = B1SpintSFCJ(d_model=d_model, n_heads=n_heads, fusion=fusion, carrier_kind=kind, **kwargs)
        overlap = {k: v for k, v in native.items() if k in model.state_dict()}
        model.load_state_dict(overlap, strict=False)
        model.zero_carrier_columns()
        if model.alpha is not None:
            with torch.no_grad():
                model.alpha.copy_(torch.zeros((), dtype=model.alpha.dtype))
            if not ieee_positive_zero(model.alpha):
                raise RuntimeError(f"{name} alpha is not IEEE +0")
        arms[name] = model
    return arms


def permute_units(x, calib, carrier, perm):
    """Keep carrier rows aligned with unit permutation of activity."""
    return x[:, :, perm], calib[:, :, :, perm], carrier[:, perm, :]
