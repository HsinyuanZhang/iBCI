"""Source-frozen output-manifold primitives.

Nothing in this module loads a recording.  The behavior autoencoder is fitted
on source sessions only.  A new session may encode its labelled support and
fit one closed-form neural-to-latent ridge, but it never updates the manifold.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class AutoencoderError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AutoencoderError(message)


def canonical_sha256(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class AutoencoderSpec:
    kind: str
    latent_dim: int
    hidden_dim: int = 0
    raw_loss_fraction: float = 0.5
    activation: str = "gelu"

    def validate(self, output_dim: int) -> None:
        need(self.kind in {"pca", "mlp"}, "kind must be pca or mlp")
        need(1 <= self.latent_dim < output_dim, "latent dimension must be a strict bottleneck")
        if self.kind == "pca":
            need(self.hidden_dim == 0 and self.raw_loss_fraction == 0.5 and self.activation == "gelu", "PCA spec has MLP-only fields")
        else:
            need(self.hidden_dim >= self.latent_dim and self.activation in {"gelu", "tanh"}, "invalid MLP architecture")
            need(0.0 <= self.raw_loss_fraction <= 1.0, "raw loss fraction outside [0,1]")

    @property
    def name(self) -> str:
        if self.kind == "pca":
            return f"pca_q{self.latent_dim}"
        alpha = str(self.raw_loss_fraction).replace(".", "p")
        return f"mlp_q{self.latent_dim}_h{self.hidden_dim}_{self.activation}_raw{alpha}"


class BehaviorAutoencoder(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        need(output_dim > latent_dim >= 1 and hidden_dim >= latent_dim, "invalid autoencoder widths")
        activation_module: type[nn.Module] = nn.GELU if activation == "gelu" else nn.Tanh
        self.encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            activation_module(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_module(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))


@dataclass(frozen=True)
class PcaManifold:
    components: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.components, dtype=np.float64)
        need(array.ndim == 2 and array.shape[0] < array.shape[1], "PCA components require [q,C]")
        need(np.allclose(array @ array.T, np.eye(array.shape[0]), atol=1e-8), "PCA rows are not orthonormal")

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[0])

    def encode_numpy(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) @ np.asarray(self.components, dtype=np.float64).T

    def decode_numpy(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(latent, dtype=np.float64) @ np.asarray(self.components, dtype=np.float64)


def fit_pca_manifold(values: np.ndarray, latent_dim: int) -> PcaManifold:
    array = np.asarray(values, dtype=np.float64)
    need(array.ndim == 2 and 1 <= latent_dim < array.shape[1] and np.isfinite(array).all(), "invalid PCA training values")
    covariance = array.T @ array / float(array.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:latent_dim]
    components = eigenvectors[:, order].T
    # Canonical sign: largest-absolute loading is positive.
    for row in components:
        index = int(np.argmax(np.abs(row)))
        if row[index] < 0.0:
            row *= -1.0
    return PcaManifold(np.asarray(components, dtype=np.float64))


def fit_affine_ridge(
    x: np.ndarray,
    z: np.ndarray,
    *,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
    need(x.ndim == z.ndim == 2 and x.shape[0] == z.shape[0] and x.shape[0] > x.shape[1], "ridge shape drift")
    need(ridge_lambda >= 0.0 and np.isfinite(ridge_lambda), "invalid ridge lambda")
    x_mean = x.mean(axis=0); z_mean = z.mean(axis=0)
    xc = x - x_mean; zc = z - z_mean
    matrix = xc.T @ xc + float(ridge_lambda) * float(x.shape[0]) * np.eye(x.shape[1])
    try:
        weight = np.linalg.solve(matrix, xc.T @ zc)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(matrix, xc.T @ zc, rcond=None)[0]
    intercept = z_mean - x_mean @ weight
    need(np.isfinite(weight).all() and np.isfinite(intercept).all(), "ridge became nonfinite")
    return weight, intercept


def predict_affine_ridge(x: np.ndarray, weight: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64) + np.asarray(intercept, dtype=np.float64)
    need(np.isfinite(value).all(), "ridge prediction became nonfinite")
    return value


def weighted_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    source_scale: torch.Tensor,
    raw_loss_fraction: float,
) -> torch.Tensor:
    need(prediction.shape == target.shape and prediction.ndim == 2, "reconstruction shape drift")
    need(0.0 <= raw_loss_fraction <= 1.0, "raw loss fraction drift")
    raw_weight = source_scale.square()
    raw_weight = raw_weight / raw_weight.mean()
    weight = raw_loss_fraction * raw_weight + (1.0 - raw_loss_fraction) * torch.ones_like(raw_weight)
    return ((prediction - target).square() * weight).mean()


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
