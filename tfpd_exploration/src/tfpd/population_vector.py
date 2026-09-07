"""Priority-2 TFPD baseline: learned population vector with a transparent PV core.

Frozen math (Stage-0 contract, amendment block):

    act_i(t)  = softplus(f(x_i[t-W:t]))          (non-negative by freeze)
    conf_i    = sigmoid(MLP(c_i))                in (0, 1)
    dir_i     = [a_i, c_i] / max(m_i, eps)       carrier's own task direction
    mass_t    = sum_i conf_i * act_i(t)          >= 0 by construction
    pv_t      = sum_i conf_i * act_i(t) * dir_i / (mass_t + eps)
    pop_conf_t = log(1 + mass_t)                 (explicit count statistic)

pv and pop_conf share the single mass statistic with the same eps — there is
exactly one denominator in the model.  The zero-carrier control gives
``dir = 0`` and a constant confidence, so the fallback is the population-rate
channel alone: pv = 0, pop_conf = log(1 + const * sum act).  A learned
task-space basis and the same causal GRU temporal decoder family as the
Priority-1 model map [basis(pv), pop_conf] to behaviour.
"""

from __future__ import annotations

import torch
from torch import nn

from src.tfpd.bilinear_readin import CausalActivityEncoder

PV_EPS = 1e-6


class LearnedPopulationVectorDecoder(nn.Module):
    def __init__(
        self,
        window_size: int = 20,
        carrier_dim: int = 4,
        hidden_dim: int = 64,
        basis_dim: int = 16,
        gru_hidden: int = 64,
        num_covariates: int = 2,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if carrier_dim < 2:
            raise ValueError("PV core requires carrier_dim >= 2 (a, c)")
        self.window_size = window_size
        self.activity = CausalActivityEncoder(window_size, hidden_dim, feature_dim=1)
        self.confidence = nn.Sequential(
            nn.Linear(carrier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.basis = nn.Linear(2, basis_dim)
        self.temporal = nn.GRU(basis_dim + 1, gru_hidden, batch_first=True)  # causal
        self.head = nn.Linear(gru_hidden, num_covariates)

    @staticmethod
    def _carrier_direction(carrier: torch.Tensor) -> torch.Tensor:
        """[a, c] / max(m, eps) from the analytic carrier [B, N, 4]."""
        direction = carrier[..., 0:2]
        magnitude = carrier[..., 2:3].clamp_min(PV_EPS)
        return direction / magnitude

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N] counts, carrier: [B, N, carrier_dim] -> behaviour [B, T, C]."""
        if x.ndim != 3 or carrier.ndim != 3:
            raise ValueError(f"expected x [B,T,N] and carrier [B,N,d], got {tuple(x.shape)}, {tuple(carrier.shape)}")
        num_units = x.shape[-1]
        if carrier.shape[1] != num_units:
            raise ValueError(f"carrier unit count {carrier.shape[1]} != activity unit count {num_units}")

        activity = torch.nn.functional.softplus(self.activity(x).squeeze(-1))  # [B, T, N]
        confidence = self.confidence(carrier).squeeze(-1)  # [B, N]
        direction = self._carrier_direction(carrier)  # [B, N, 2]

        weighted = activity * confidence.unsqueeze(1)  # [B, T, N]
        mass = weighted.sum(dim=2, keepdim=True)  # [B, T, 1], >= 0
        pv = torch.einsum("btn,bnd->btd", weighted, direction) / (mass + PV_EPS)  # [B, T, 2]
        pop_conf = torch.log1p(mass)  # [B, T, 1]

        features = torch.cat([self.basis(pv), pop_conf], dim=-1)  # [B, T, basis+1]
        recurrent, _ = self.temporal(features)
        return self.head(recurrent)

    def zero_carrier(self, carrier: torch.Tensor) -> torch.Tensor:
        """Zero-content control: direction 0, constant confidence, rate fallback."""
        return torch.zeros_like(carrier)

    @staticmethod
    def wrong_pair_carrier(carrier: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Row-permuted carrier: valid directions, wrong unit pairing."""
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(carrier.shape[1], generator=generator)
        return carrier[:, perm]
