"""§5.3 static rung: full low-rank carrier-generated read-in B_i = H(c_i).

The frozen Priority-1 model is the diagonal special case
``B_i = diag(V g(c_i)) U`` of the parent handoff's generic per-unit read-in.
This module implements the next static rung:

    z_t = (1/N) * sum_i U1 ( g(c_i) (.) (U2 a_it) )
    B_i = U1 diag(g(c_i)) U2        (rank k, still static, still multilinear)

No softmax, no state-dependence, no time axis in the carrier: this arm isolates
"full low-rank static operator" against "diagonal static operator" (post-bilinear
consumer extension §5.3).  ``g`` keeps its bias so the zero-carrier fallback
semantics match the frozen model: ``g(0) = bias_g`` gives a constant modulation.

Initialization is parent-exact by construction when ``U1`` is initialized to the
identity layout and ``U2`` absorbs ``U``: with ``k = d`` and ``U1 = I``,
``z = (1/N) sum (U2 a) (.) g(c)`` — but we standard-initialize per the
Stage-1 contract family; exactness is checked structurally by G5, not assumed.
"""

from __future__ import annotations

import torch
from torch import nn

from src.tfpd.bilinear_readin import CausalActivityEncoder, CarrierMap


class LowRankCarrierReadIn(nn.Module):
    """B_i = U1 diag(g(c_i)) U2, applied as U1( g(c) (.) (U2 a) )."""

    def __init__(self, carrier_dim: int, feature_dim: int, embed_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.carrier_map = CarrierMap(carrier_dim, embed_dim)  # g: 4 -> k, affine with bias
        self.readin_activity = nn.Linear(feature_dim, embed_dim, bias=False)  # U2: r -> k
        self.readout_latent = nn.Linear(embed_dim, latent_dim, bias=False)  # U1: k -> d

    def forward(self, activity: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """activity: [B, T, N, r], carrier: [B, N, 4] -> [B, T, d]."""
        embedded = self.carrier_map(carrier).unsqueeze(1)  # [B, 1, N, k]
        projected = self.readin_activity(activity)  # [B, T, N, k]
        return self.readout_latent(projected * embedded)  # [B, T, N, k] -> [B, T, N, d]


class BilinearLowRankTaskFrameDecoder(nn.Module):
    """Static full-low-rank TFPD rung (§5.3). Interface matches the frozen model."""

    def __init__(
        self,
        window_size: int = 20,
        carrier_dim: int = 4,
        feature_dim: int = 16,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        gru_hidden: int = 64,
        num_covariates: int = 2,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.activity_encoder = CausalActivityEncoder(window_size, hidden_dim, feature_dim)
        self.readin = LowRankCarrierReadIn(carrier_dim, feature_dim, embed_dim, latent_dim)
        self.temporal = nn.GRU(latent_dim, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, num_covariates)

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or carrier.ndim != 3:
            raise ValueError(f"expected x [B,T,N] and carrier [B,N,d], got {tuple(x.shape)}, {tuple(carrier.shape)}")
        num_units = x.shape[-1]
        if carrier.shape[1] != num_units:
            raise ValueError(f"carrier unit count {carrier.shape[1]} != activity unit count {num_units}")
        activity = self.activity_encoder(x)  # [B, T, N, r]
        z = self.readin(activity, carrier).sum(dim=2) / num_units  # [B, T, d]
        recurrent, _ = self.temporal(z)
        return self.head(recurrent)

    def zero_carrier(self, carrier: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(carrier)

    @staticmethod
    def wrong_pair_carrier(carrier: torch.Tensor, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(carrier.shape[1], generator=generator)
        return carrier[:, perm]
