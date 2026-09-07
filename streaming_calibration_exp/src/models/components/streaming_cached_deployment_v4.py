"""Strict one-shot T4 calibration adapter for Phase-C deployment."""
from __future__ import annotations

import torch


class T4CachedDeploymentAdapterV4:
    def __init__(self, student: torch.nn.Module) -> None:
        decoder = getattr(student, "decoder", None)
        encoder = getattr(student, "id_encoder", None)
        if (
            decoder is None
            or getattr(decoder, "window_size", None) != 50
            or getattr(decoder, "model_dim", None) != 512
            or getattr(encoder, "side_dim", None) != 4
        ):
            raise ValueError("cached T4 adapter requires exact Phase-C architecture")
        self.student = student

    def compute_identity(
        self, support: torch.Tensor, side_features: torch.Tensor
    ) -> torch.Tensor:
        if support.shape != (1, 33, 100, 96):
            raise ValueError(
                f"cached T4 support must be [1,33,100,96], got {tuple(support.shape)}"
            )
        if side_features.shape != (1, 96, 4):
            raise ValueError(
                f"cached T4 descriptor must be [1,96,4], got {tuple(side_features.shape)}"
            )
        identity = self.student.compute_identity(
            support, side_features=side_features
        )
        if identity.shape != (1, 96, 50):
            raise ValueError("cached T4 encoder returned wrong identity shape")
        return identity

    def decode_with_identity(
        self, neural: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        if neural.ndim != 3 or neural.shape[1:] != (50, 96):
            raise ValueError(f"cached T4 query must be [B,50,96], got {tuple(neural.shape)}")
        if identity.shape != (1, 96, 50):
            raise ValueError(
                f"cached T4 identity must be [1,96,50], got {tuple(identity.shape)}"
            )
        return self.student.decode_with_identity(neural, identity)
