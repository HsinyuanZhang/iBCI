"""E-static: cache frozen E0/H-C first-layer terms across predict calls.

Does not edit sealed submission trees. Train-time must not reuse this cache
across optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import FactoredTokenMLP, H1Bank


@dataclass
class StaticTokenCache:
    e0_term: torch.Tensor
    hc_term: torch.Tensor
    bias: torch.Tensor


def compile_static(mlp: FactoredTokenMLP, bank: H1Bank) -> StaticTokenCache:
    cfg = mlp.cfg
    w_loc, w_e0, w_hc = mlp.fc1.weight.split([cfg.local_dim, cfg.e0_dim, cfg.hc_dim], dim=1)
    with torch.no_grad():
        e0_term = F.linear(bank.E0, w_e0, None)
        hc_term = F.linear(bank.T, w_hc, None)
    return StaticTokenCache(e0_term=e0_term.detach(), hc_term=hc_term.detach(), bias=mlp.fc1.bias.detach())


def forward_with_static(mlp: FactoredTokenMLP, local: torch.Tensor, cache: StaticTokenCache) -> torch.Tensor:
    w_loc, _, _ = mlp.fc1.weight.split([mlp.cfg.local_dim, mlp.cfg.e0_dim, mlp.cfg.hc_dim], dim=1)
    hidden = F.linear(local, w_loc, None)
    hidden = hidden + cache.e0_term.view(1, 1, *cache.e0_term.shape) + cache.hc_term.view(1, 1, *cache.hc_term.shape) + cache.bias
    return mlp.fc2(F.gelu(hidden))


def parity_max_abs(mlp: FactoredTokenMLP, local: torch.Tensor, bank: H1Bank) -> float:
    cache = compile_static(mlp, bank)
    with torch.no_grad():
        a = mlp.forward(local, bank.E0, bank.T)
        b = forward_with_static(mlp, local, cache)
    return float((a - b).abs().max().item())
