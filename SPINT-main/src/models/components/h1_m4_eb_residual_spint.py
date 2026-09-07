"""The sole topology used by both matched H1 M=4 EB pilot arms."""
from __future__ import annotations
import random
import torch
from torch import nn
from src.models.components.spint import SpintModel


class H1M4EBResidualSpint(SpintModel):
    """Original topology with the sole residual inserted after ``fc_id_out``.

    Constructing the zero matrix does not consume RNG.  In base mode the exact
    same parameter exists but is frozen; in joint mode it is trainable.
    """
    def __init__(self, *, train_residual: bool, **kwargs):
        super().__init__(**kwargs)
        self.eb_residual = nn.Parameter(torch.zeros(4, self.window_size), requires_grad=train_residual)

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("H1 M=4 EB SPINT requires both four-trial identity and [N,4] carrier")
        src = src.permute(0, 2, 1)
        batch_size, num_neurons = src.size(0), src.size(1)
        identity = calib_trialized_neural_features.permute(0,1,3,2)
        identity = self.fc_id_in(identity).mean(dim=1, keepdim=False)
        identity = self.fc_id_out(identity)
        if carrier.shape != (batch_size, num_neurons, 4):
            raise ValueError(f"carrier must be [B,N,4], got {tuple(carrier.shape)}")
        # This is the one permitted insertion point: immediately after the
        # original fc_id_out, before the original source-plus-identity sum.
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
