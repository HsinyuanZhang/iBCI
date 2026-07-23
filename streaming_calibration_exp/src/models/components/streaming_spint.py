"""SPINT decoder wrapper with pluggable streaming calibration encoder."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import CalibrationEncoder


class StreamingSpintModel(nn.Module):
    """Frozen decoder + trainable/streaming calibration encoder."""

    def __init__(self, decoder: SpintModel, id_encoder: CalibrationEncoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.id_encoder = id_encoder
        self.window_size = decoder.window_size
        self._decoder_frozen = False

    def compute_identity(self, calib_trials: torch.Tensor) -> torch.Tensor:
        return self.id_encoder.forward_batch(calib_trials)

    def decode_with_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        neuron_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """neural: [B,W,N], identity: [B,N,W] -> behavior [B,W,C].

        Frozen decoder weights stay in eval() with requires_grad=False, but the
        forward path must remain differentiable w.r.t. identity E.
        """
        src = neural.permute(0, 2, 1)
        if neuron_gate is not None:
            src = src * neuron_gate
        src = src + identity

        if self._decoder_frozen:
            self.decoder.eval()

        batch_size = src.size(0)
        num_neurons = src.size(1)
        dropout_mask = torch.ones(batch_size, num_neurons, device=src.device, dtype=src.dtype)
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                p = random.uniform(self.decoder.dynamic_dropout_low, self.decoder.dynamic_dropout_high)
                dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=True)
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=self.decoder.dropout_rate, training=True
                )
        src = src * dropout_mask.unsqueeze(-1)

        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def forward(
        self,
        neural: torch.Tensor,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError("Either calib_trials or identity must be provided")
            if hasattr(self.id_encoder, "forward_batch_with_gate"):
                identity, neuron_gate = self.id_encoder.forward_batch_with_gate(calib_trials)
            else:
                identity = self.compute_identity(calib_trials)
        behavior = self.decode_with_identity(neural, identity, neuron_gate=neuron_gate)
        return behavior, identity

    def freeze_decoder(self) -> int:
        frozen = 0
        for param in self.decoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
        self._decoder_frozen = True
        self.decoder.eval()
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._decoder_frozen:
            self.decoder.eval()
        return self

    def trainable_encoder_parameters(self):
        return (parameter for parameter in self.id_encoder.parameters() if parameter.requires_grad)
