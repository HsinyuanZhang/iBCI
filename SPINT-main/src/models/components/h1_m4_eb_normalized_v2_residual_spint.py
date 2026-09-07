"""V2-named copy of the frozen H1 M=4 residual SPINT topology.

The only behavioral difference from the V1 topology is the scale of the
carrier supplied by the V2 DataModule.  The residual remains [4,700], bias-free
in the carrier path, zero initialized, and inserted after ``fc_id_out``.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from src.models.components.spint import SpintModel


class H1M4EBNormalizedV2ResidualSpint(SpintModel):
    """Matched base/joint topology for source-normalized V2 carriers."""

    def __init__(self, *, train_residual: bool, **kwargs):
        super().__init__(**kwargs)
        # Constructing zeros does not consume RNG; both arms therefore retain
        # identical shared initialization under synchronized materialization.
        self.eb_residual = nn.Parameter(torch.zeros(4, self.window_size), requires_grad=train_residual)

    def identity_projection(self, calib_trialized_neural_features: torch.Tensor) -> torch.Tensor:
        """Return the original ``fc_id_out`` representation before EB residual."""

        identity = calib_trialized_neural_features.permute(0, 1, 3, 2)
        identity = self.fc_id_in(identity).mean(dim=1, keepdim=False)
        return self.fc_id_out(identity)

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("H1 M=4 EB normalized V2 SPINT requires identity and [B,N,4] carrier")
        src = src.permute(0, 2, 1)
        batch_size, num_neurons = src.size(0), src.size(1)
        identity = self.identity_projection(calib_trialized_neural_features)
        if carrier.ndim != 3 or carrier.shape != (batch_size, num_neurons, 4):
            raise ValueError(f"carrier must be [B,N,4], got {tuple(carrier.shape)}")
        # V2 does not change topology: normalized source carrier enters at the
        # original single residual insertion point before source+identity sum.
        identity = identity + carrier.to(identity) @ self.eb_residual
        src = src + identity
        dropout_mask = torch.ones(batch_size, num_neurons).to(src)
        if self.dynamic_dropout:
            p = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training)
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training)
        src = src * dropout_mask.unsqueeze(-1)
        src = self.fc_in(src)
        rep = self.fc_in(self.rep).to(src)
        output, _ = self.transformer(rep.repeat(batch_size, 1, 1), src)
        return self.fc_out(output).permute(0, 2, 1)


# Explicit alias used by a few launch/config callers; both names retain the
# V2 module path and therefore never silently instantiate the V1 class.
H1M4EBNormalizedV2Spint = H1M4EBNormalizedV2ResidualSpint
