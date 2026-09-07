"""§5.4 cheapest state-dependence rung: state-conditioned diagonal gain on h.

    z_t = (1/N) * sum_i (U a_it) (.) (V g(c_i)) (.) (1 + tanh(W h_{t-1}))

Factor 3 (state-dependence) with no softmax, no competition, and no variable-N
normalization hazard.  ``W`` is zero-initialized, so at initialization the gain
is identically 1 and the model is EXACTLY the frozen Priority-1 bilinear model
(the house zero-init discipline; cf. carrier_post_pool in h1_carrierid_spint).

The GRU is unrolled with GRUCell so that the read-in at step t sees h_{t-1};
the query/support causality and permutation-invariance structure are inherited
unchanged from the frozen model (gates G1/G4 re-verify).
"""

from __future__ import annotations

import torch
from torch import nn

from src.tfpd.bilinear_readin import CausalActivityEncoder, CarrierMap


class BilinearStateGainTaskFrameDecoder(nn.Module):
    """Frozen bilinear read-in + state-conditioned diagonal gain (§5.4 rung 1)."""

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
        self.latent_dim = latent_dim
        self.activity_encoder = CausalActivityEncoder(window_size, hidden_dim, feature_dim)
        self.carrier_map = CarrierMap(carrier_dim, embed_dim)
        self.readin_activity = nn.Linear(feature_dim, latent_dim, bias=False)  # U
        self.readin_carrier = nn.Linear(embed_dim, latent_dim, bias=False)  # V
        # State-conditioned gain. Zero-init => gain == 1 at start => parent-exact.
        self.state_gain = nn.Linear(gru_hidden, latent_dim, bias=False)  # W
        nn.init.zeros_(self.state_gain.weight)
        self.cell = nn.GRUCell(latent_dim, gru_hidden)
        self.head = nn.Linear(gru_hidden, num_covariates)

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or carrier.ndim != 3:
            raise ValueError(f"expected x [B,T,N] and carrier [B,N,d], got {tuple(x.shape)}, {tuple(carrier.shape)}")
        batch, length, num_units = x.shape
        if carrier.shape[1] != num_units:
            raise ValueError(f"carrier unit count {carrier.shape[1]} != activity unit count {num_units}")

        activity = self.activity_encoder(x)  # [B, T, N, r]
        embedded = self.carrier_map(carrier)  # [B, N, k]
        u_a = self.readin_activity(activity)  # [B, T, N, d]
        v_g = self.readin_carrier(embedded).unsqueeze(1)  # [B, 1, N, d]
        static_z = (u_a * v_g).sum(dim=2) / num_units  # [B, T, d]

        hidden = x.new_zeros(batch, self.cell.hidden_size)
        outputs = []
        for t in range(length):
            gain = 1.0 + torch.tanh(self.state_gain(hidden))  # [B, d], == 1 at init
            z_t = static_z[:, t, :] * gain
            hidden = self.cell(z_t, hidden)
            outputs.append(self.head(hidden))
        return torch.stack(outputs, dim=1)  # [B, T, C]

    def zero_carrier(self, carrier: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(carrier)

    @staticmethod
    def wrong_pair_carrier(carrier: torch.Tensor, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(carrier.shape[1], generator=generator)
        return carrier[:, perm]
