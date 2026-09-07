"""Independent post-fc_in carrier projection. Torch is imported only on demand."""
from __future__ import annotations

import numpy as np

from . import plan


class InjectionError(RuntimeError):
    """Fail closed for carrier injection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InjectionError(message)


def zero_linear(in_features: int, out_features: int) -> np.ndarray:
    _require(in_features == plan.CARRIER_DIM and out_features >= 1, "P shape")
    return np.zeros((out_features, in_features), dtype=np.float64)


def apply_injection(
    hidden: np.ndarray, carrier: np.ndarray, projection: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    h = np.asarray(hidden, dtype=np.float64)
    c = np.asarray(carrier, dtype=np.float64)
    p = np.asarray(projection, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64).reshape(-1)
    _require(h.ndim == 2 and c.shape[0] == h.shape[0] == m.shape[0], "unit alignment")
    _require(c.shape[1] == plan.CARRIER_DIM and p.shape == (h.shape[1], plan.CARRIER_DIM), "P")
    unit_mask = m[:, None]
    hidden_masked = h * unit_mask
    carrier_masked = c * unit_mask
    return hidden_masked + carrier_masked @ p.T


class TorchCarrierProjection:
    """Lazy torch Linear(4, D, bias=False) with exact-zero init."""

    def __init__(self, model_dim: int) -> None:
        import torch
        from torch import nn
        self.torch = torch
        self.P = nn.Linear(plan.CARRIER_DIM, model_dim, bias=False)
        self.P.weight.data.zero_()


def torch_apply(hidden, carrier, module: TorchCarrierProjection, mask):
    torch = module.torch
    unit_mask = mask.reshape(-1, 1)
    return hidden * unit_mask + module.P(carrier * unit_mask)
