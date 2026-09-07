"""Priority-1 TFPD model: explicit low-rank bilinear activity-carrier read-in.

Frozen math (Stage-0 contract, amendment block):

    z_t = (1/N) * sum_i (U f(x_i[t-W:t])) (.) (V g(c_i))
    y_hat_{1:T} = D_psi(z_{1:T})

Frozen interface decisions:

- ``g`` is an affine map WITH bias and ``U``/``V`` are bias-free.  Under the
  zero-carrier control ``g(0) = bias_g`` is a constant per-unit embedding, so
  ``V g(0)`` reduces the read-in to a carrier-content-free population mean of
  activity features — this is the intended, explicit Z4 activity fallback.
  A ``V`` bias would add a carrier-independent constant latent and is excluded.
- Mean normalization ``1/N`` (not the handoff's illustrative ``1/sqrt(N)``):
  shared ``f`` makes unit features positively correlated, the sum's variance
  grows ~N, and only the mean keeps the latent scale N-stable (gate G2).
- ``f`` ends in Tanh; ``D_psi`` is a causal GRU + linear head.  Standard
  initialization; no teacher anywhere.

Arms are inputs, not flags.  The zero-carrier control feeds ``torch.zeros_like``
; the wrong-pair control feeds a row-permuted carrier.  Both reuse trained
weights unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class CausalActivityEncoder(nn.Module):
    """f_theta: shared per-unit encoder over the trailing W activity bins."""

    def __init__(self, window_size: int, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N] spike counts -> causal features [B, T, N, r]."""
        b, t, n = x.shape
        pad = x.new_zeros(b, self.window_size - 1, n)
        windows = torch.cat([pad, x], dim=1).unfold(dimension=1, size=self.window_size, step=1)
        return self.net(windows)  # [B, T, N, W] -> [B, T, N, r]


class CarrierMap(nn.Module):
    """g: affine map from the analytic carrier [a, c, m, b] to R^k, WITH bias.

    The bias is load-bearing: under the zero-carrier control it defines the
    constant fallback direction V(g(0)).
    """

    def __init__(self, carrier_dim: int = 4, embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(carrier_dim, embed_dim)

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        """carrier: [B, N, 4] -> [B, N, k]."""
        return self.proj(carrier)


class BilinearTaskFrameDecoder(nn.Module):
    """Task-Frame Population Decoder, Priority-1 instantiation."""

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
        self.carrier_map = CarrierMap(carrier_dim, embed_dim)
        self.readin_activity = nn.Linear(feature_dim, latent_dim, bias=False)  # U
        self.readin_carrier = nn.Linear(embed_dim, latent_dim, bias=False)  # V, bias-free by freeze
        self.temporal = nn.GRU(latent_dim, gru_hidden, batch_first=True)  # causal D_psi
        self.head = nn.Linear(gru_hidden, num_covariates)

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N] counts, carrier: [B, N, carrier_dim] -> behaviour [B, T, C].

        Permutation invariance over units is structural: the population read-in
        is a single mean over the unit axis.
        """
        if x.ndim != 3 or carrier.ndim != 3:
            raise ValueError(f"expected x [B,T,N] and carrier [B,N,d], got {tuple(x.shape)}, {tuple(carrier.shape)}")
        num_units = x.shape[-1]
        if carrier.shape[1] != num_units:
            raise ValueError(f"carrier unit count {carrier.shape[1]} != activity unit count {num_units}")

        activity = self.activity_encoder(x)  # [B, T, N, r]
        embedded = self.carrier_map(carrier)  # [B, N, k]
        u_a = self.readin_activity(activity)  # [B, T, N, d]
        v_g = self.readin_carrier(embedded).unsqueeze(1)  # [B, 1, N, d]
        z = (u_a * v_g).sum(dim=2) / num_units  # [B, T, d]
        recurrent, _ = self.temporal(z)
        return self.head(recurrent)  # [B, T, C]

    def zero_carrier(self, carrier: torch.Tensor) -> torch.Tensor:
        """Zero-content control input (constant g-bias fallback remains)."""
        return torch.zeros_like(carrier)

    @staticmethod
    def wrong_pair_carrier(carrier: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Row-permuted carrier: valid content, wrong unit pairing."""
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(carrier.shape[1], generator=generator)
        return carrier[:, perm]
