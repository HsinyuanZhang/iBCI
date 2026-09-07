"""CCE residual-SPINT topology.

The CCE adds no new decoder-side fusion path: its only trainable carrier
parameter is the same zero-initialized bias-free ``[4,700]`` residual used by
the normalized H1 pilot.  The base arm owns the identical parameter but keeps
it literal zero, allowing a fixed 0.5 base/joint prediction ensemble.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from src.models.components.spint import SpintModel


class H1M4CCEResidualSpint(SpintModel):
    """Matched base/joint SPINT with a zero-initialized 4x700 CCE residual."""

    def __init__(self, *, train_residual: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cce_residual = nn.Parameter(torch.zeros(4, self.window_size), requires_grad=bool(train_residual))

    def identity_projection(self, calib_trialized_neural_features: torch.Tensor) -> torch.Tensor:
        identity = calib_trialized_neural_features.permute(0, 1, 3, 2)
        identity = self.fc_id_in(identity).mean(dim=1, keepdim=False)
        return self.fc_id_out(identity)

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("H1 M4 CCE requires identity and carrier")
        src = src.permute(0, 2, 1)
        batch_size, num_neurons = src.size(0), src.size(1)
        if carrier.ndim != 3 or carrier.shape != (batch_size, num_neurons, 4):
            raise ValueError(f"CCE carrier must be [B,N,4], got {tuple(carrier.shape)}")
        identity = self.identity_projection(calib_trialized_neural_features)
        identity = identity + carrier.to(identity) @ self.cce_residual
        src = src + identity
        dropout_mask = torch.ones(batch_size, num_neurons, device=src.device, dtype=src.dtype)
        if self.dynamic_dropout:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high), training=self.training)
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training)
        src = self.fc_in(src * dropout_mask.unsqueeze(-1))
        representation = self.fc_in(self.rep).to(src)
        output, _ = self.transformer(representation.repeat(batch_size, 1, 1), src)
        return self.fc_out(output).permute(0, 2, 1)


H1M4CCESpint = H1M4CCEResidualSpint
