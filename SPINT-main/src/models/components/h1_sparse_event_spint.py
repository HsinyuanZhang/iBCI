"""Compact SPINT consumer for the five-dimensional H-SE5 carrier."""
from __future__ import annotations

import random

import torch
from torch import nn

from src.models.components.spint import SpintModel


HIDDEN_DIM = 32
CARRIER_DIM = 5
TRIAL_LENGTH = 1024
WINDOW_SIZE = 700
IDENTITY_PARAMETERS = 58_172


class H1SparseEventSpint(SpintModel):
    """Matched h=32 identity consumer with `[w1,w2,w3,w4,b]` input."""

    def __init__(
        self,
        *,
        carrier_hidden_dim: int = HIDDEN_DIM,
        carrier_dim: int = CARRIER_DIM,
        carrier_trial_length: int = TRIAL_LENGTH,
        zero_carrier: bool = False,
        **kwargs,
    ) -> None:
        if int(carrier_hidden_dim) != HIDDEN_DIM or int(carrier_dim) != CARRIER_DIM:
            raise ValueError("H-SE5 fixes hidden_dim=32 and carrier_dim=5")
        if int(carrier_trial_length) != TRIAL_LENGTH or int(kwargs.get("window_size", -1)) != WINDOW_SIZE:
            raise ValueError("H-SE5 fixes calibration T=1024 and window_size=700")
        super().__init__(**kwargs)
        del self.fc_id_in
        del self.fc_id_out
        self.carrier_hidden_dim = HIDDEN_DIM
        self.carrier_dim = CARRIER_DIM
        self.carrier_trial_length = TRIAL_LENGTH
        self.zero_carrier = bool(zero_carrier)
        self.carrier_pre_pool = nn.Sequential(nn.Linear(TRIAL_LENGTH, HIDDEN_DIM), nn.ReLU())
        self.carrier_post_pool = nn.Sequential(
            nn.Linear(HIDDEN_DIM + CARRIER_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WINDOW_SIZE),
        )
        with torch.no_grad():
            self.carrier_post_pool[0].weight[:, HIDDEN_DIM:].zero_()
        if self.carrier_parameter_count() != IDENTITY_PARAMETERS:
            raise RuntimeError(f"H-SE5 parameter accounting drift: {self.carrier_parameter_count()}")

    def carrier_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.carrier_pre_pool.parameters()) + sum(
            parameter.numel() for parameter in self.carrier_post_pool.parameters()
        )

    def carrierid_identity_projection(self, identity: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        if identity.ndim != 4:
            raise ValueError(f"H-SE5 identity must be [B,M,T,N], got {tuple(identity.shape)}")
        batch, _trials, trial_length, neurons = identity.shape
        if int(trial_length) != TRIAL_LENGTH or carrier.shape != (batch, neurons, CARRIER_DIM):
            raise ValueError(f"H-SE5 carrier/identity shape mismatch: {tuple(identity.shape)}, {tuple(carrier.shape)}")
        temporal = identity.permute(0, 1, 3, 2)
        pooled = self.carrier_pre_pool(temporal).mean(dim=1)
        effective = torch.zeros_like(carrier) if self.zero_carrier else carrier
        return self.carrier_post_pool(torch.cat((pooled, effective.to(pooled)), dim=-1))

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("H-SE5 requires calibration identity and sparse endpoint carrier")
        src = src.permute(0, 2, 1)
        batch, neurons = src.shape[:2]
        src = src + self.carrierid_identity_projection(calib_trialized_neural_features, carrier)
        dropout_mask = torch.ones(batch, neurons).to(src)
        if self.dynamic_dropout:
            dropout_mask = torch.nn.functional.dropout(
                dropout_mask,
                p=random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high),
                training=self.training,
            )
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training)
        src = self.fc_in(src * dropout_mask.unsqueeze(-1))
        rep = self.fc_in(self.rep).to(src)
        output, _ = self.transformer(rep.repeat(batch, 1, 1), src)
        return self.fc_out(output).permute(0, 2, 1)

