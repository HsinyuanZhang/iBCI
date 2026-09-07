"""Large TFPD rung: carrier-conditioned set attention over task-frame unit tokens.

Scale-matched to the SPINT decoder (~4.6M trainable parameters).  This is the
handoff's Route A / structural candidate #4, minimized to stay inside the
task-frame discipline:

    token_it  = W_a f(x_i[t-W:t])  (.)  W_c g(c_i)        (bilinear task-frame token)
    z_t       = CrossAttn(Q = learned slots, K = V = {token_it}_i)  (+ mass channel)
    y_hat     = GRU(z) -> head

Design freezes:
- The carrier enters ONLY through the multiplicative token construction; there
  is no additive identity port and no per-unit embedding table.
- Queries are LEARNED STATIC slots (multi-slot read-out, handoff factor 4).
  State-conditioned queries are a later, separately contracted rung.
- Softmax pooling changes weights when N changes, so the aggregation carries an
  explicit population-mass channel: [log1p(N), log1p(mean activity)] appended
  to the slot outputs (handoff section 5.6 requirement).
- Causal by the same trailing-window f as the frozen Priority-1 model;
  permutation invariance is structural (attention over a set of unit tokens).

Placed OUTSIDE src/tfpd/ on purpose: running cell closures bind src/tfpd/*.py,
and this file must not perturb them.
"""

from __future__ import annotations

import torch
from torch import nn

from src.tfpd.bilinear_readin import CausalActivityEncoder


class _PreNormCrossAttentionSlotRead(nn.Module):
    """Cross-attention from learned query slots to a variable-size token set."""

    def __init__(self, token_dim: int, num_slots: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.queries = nn.Parameter(torch.randn(num_slots, token_dim) * token_dim**-0.5)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, token_dim)
        )
        self.norm_out = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D] -> slot outputs [B, num_slots, D]."""
        batch = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        q = self.norm_q(q)
        kv = self.norm_kv(tokens)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        x = attended + self.dropout(self.ffn(self.norm_out(attended)))
        return x


class TaskFrameSetAttentionDecoder(nn.Module):
    """Large TFPD: task-frame tokens + slot cross-attention + causal GRU."""

    def __init__(
        self,
        window_size: int = 20,
        carrier_dim: int = 4,
        feature_dim: int = 64,
        embed_dim: int = 64,
        token_dim: int = 512,
        num_slots: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        gru_hidden: int = 256,
        num_covariates: int = 2,
        activity_hidden: int = 256,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.activity_encoder = CausalActivityEncoder(window_size, activity_hidden, feature_dim)
        self.carrier_map = nn.Linear(carrier_dim, embed_dim)  # g, affine with bias (frozen fallback semantics)
        self.token_activity = nn.Linear(feature_dim, token_dim, bias=False)  # W_a
        self.token_carrier = nn.Linear(embed_dim, token_dim, bias=False)  # W_c
        self.slot_read = _PreNormCrossAttentionSlotRead(token_dim, num_slots, num_heads, ffn_dim)
        slot_width = num_slots * token_dim + 2  # + population-mass channels
        self.temporal = nn.GRU(slot_width, gru_hidden, batch_first=True)  # causal
        self.head = nn.Linear(gru_hidden, num_covariates)

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N] counts, carrier: [B, N, carrier_dim] -> behaviour [B, T, C]."""
        if x.ndim != 3 or carrier.ndim != 3:
            raise ValueError(f"expected x [B,T,N] and carrier [B,N,d], got {tuple(x.shape)}, {tuple(carrier.shape)}")
        batch, length, num_units = x.shape
        if carrier.shape[1] != num_units:
            raise ValueError(f"carrier unit count {carrier.shape[1]} != activity unit count {num_units}")

        activity = self.activity_encoder(x)  # [B, T, N, r]
        embedded = self.carrier_map(carrier)  # [B, N, k]
        tokens = self.token_activity(activity) * self.token_carrier(embedded).unsqueeze(1)  # [B, T, N, D]

        # Flatten (B, T) into the batch axis so attention runs per time step over
        # the unit set.  [B,T,N,D] is contiguous: reshape directly — a permute
        # here would scramble time into the unit axis (the exact bug that broke
        # G1/G4 in the first large-gate attempt).
        slots = self.slot_read(tokens.reshape(batch * length, num_units, -1))  # [B*T, S, D]
        slots = slots.reshape(batch, length, -1)

        # Population-mass channels, CAUSAL: the activity magnitude is reduced over
        # the unit and feature axes only (dims 2,3), never over time.
        mass = torch.stack(
            [
                torch.full((batch, length), float(torch.log1p(torch.tensor(float(num_units)))), device=x.device),
                torch.log1p(activity.abs().mean(dim=(2, 3))),
            ],
            dim=-1,
        )  # [B, T, 2]
        features = torch.cat([slots, mass], dim=-1)

        recurrent, _ = self.temporal(features)
        return self.head(recurrent)

    def zero_carrier(self, carrier: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(carrier)

    @staticmethod
    def wrong_pair_carrier(carrier: torch.Tensor, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(carrier.shape[1], generator=generator)
        return carrier[:, perm]
