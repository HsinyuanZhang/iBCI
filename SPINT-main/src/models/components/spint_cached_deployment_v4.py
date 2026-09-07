"""Strict cached-identity adapter for the historical SPINT decoder."""
from __future__ import annotations

import hashlib

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def require_repeated_support_batch(
    support: torch.Tensor, *, expected_sha256: str | None = None
) -> tuple[torch.Tensor, str]:
    if support.ndim != 4 or support.shape[1:] != (33, 100, 96):
        raise ValueError(f"support must have shape [B,33,100,96], got {tuple(support.shape)}")
    canonical = support[:1]
    if not torch.equal(support, canonical.expand_as(support)):
        raise ValueError("query batch contains non-identical repeated support tensors")
    digest = tensor_sha256(canonical)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("support tensor changed across query batches")
    return canonical, digest


class SpintCachedDeploymentAdapterV4:
    def __init__(self, network: torch.nn.Module) -> None:
        if (
            getattr(network, "window_size", None) != 50
            or getattr(network, "model_dim", None) != 512
            or getattr(network, "num_id_layers", None) != 3
            or getattr(network, "readin_layer_type", None) != "mlp"
        ):
            raise ValueError("cached adapter requires the exact Phase-C SPINT architecture")
        self.network = network

    def compute_identity(self, support: torch.Tensor) -> torch.Tensor:
        if support.shape != (1, 33, 100, 96):
            raise ValueError("cached SPINT identity requires exact one-session support")
        trials = support.permute(0, 1, 3, 2)
        return self.network.fc_id_out(self.network.fc_id_in(trials).mean(dim=1))

    def decode_with_identity(self, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        if neural.ndim != 3 or neural.shape[1:] != (50, 96):
            raise ValueError(f"cached SPINT query must be [B,50,96], got {tuple(neural.shape)}")
        if identity.shape != (1, 96, 50):
            raise ValueError(f"cached SPINT identity must be [1,96,50], got {tuple(identity.shape)}")
        network = self.network
        src = neural.permute(0, 2, 1) + identity
        # Exact test/deployment semantics: the model is in eval mode, hence the
        # historical dynamic-dropout branch is mathematically the identity.
        if network.training:
            raise RuntimeError("cached deployment decode requires eval mode")
        src = network.fc_in(src)
        rep = network.fc_in(network.rep).to(src)
        transformed, _ = network.transformer(rep.repeat(src.shape[0], 1, 1), src)
        return network.fc_out(transformed).permute(0, 2, 1)

