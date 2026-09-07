"""H1 finite-history current-query reader.

The unit-set frontend is deliberately reused from the audited H1 implementation.
Unlike its temporal self-attention, each reader layer projects K/V directly from
the immutable frontend sequence, so cached history never becomes stale.
"""
from __future__ import annotations

import torch
from torch import nn

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig, H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank, SharedSetFrontend, CausalTransformerStack, _reinit_param
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack


class _H1ScaledFrontendBase(nn.Module):
    training_target_space = "runtime_scaled_velocity__target_is_20x_native"
    prediction_divisor = 20.0

    def __init__(self, cfg: H1TemporalConfig | None = None, seed: int = 42, activity_scale: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg or H1_TEMPORAL
        if activity_scale <= 0:
            raise ValueError("activity_scale must be positive")
        self.activity_scale = float(activity_scale)
        self.frontend = SharedSetFrontend(self.cfg, routed=False)
        self.final_norm = nn.LayerNorm(self.cfg.temporal_width)
        self.readout = nn.Sequential(nn.Linear(self.cfg.temporal_width, self.cfg.readout_hidden), nn.GELU(), nn.Linear(self.cfg.readout_hidden, self.cfg.out_dim))
        g = torch.Generator(device="cpu").manual_seed(seed + 901_117)
        for name, p in sorted(self.named_parameters()):
            _reinit_param(name, p, g)

    def _z(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        keep = dropout_keep
        if keep is None:
            keep = bank.unit_mask
            if keep.ndim == 1:
                keep = keep.unsqueeze(0).expand(x.size(0), -1)
        # A single shared scalar preserves unit-permutation invariance and is causal.
        # activity_scale=4 is a distinct, explicitly named activity-balanced frontend
        # candidate; it is never silently applied to legacy weights.
        z = self.frontend(x * self.activity_scale, bank.E0.to(x), bank.T.to(x), keep.to(x.device), tile_size=None, use_checkpoint=False)
        return z

    # Stable adapter hook for the shared frontend/cache runtime.  It returns
    # exactly the causal, W=700 frontend tokens used by training; cache policy
    # belongs to the shared runtime and is intentionally not duplicated here.
    def encode_frontend(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        return self._z(x, bank, dropout_keep)

    def forward_last(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        return self.readout(self.final_norm(self.forward_hidden(x, bank, dropout_keep)))


class H1CurrentQueryDecoder(_H1ScaledFrontendBase):
    """T arm: shared finite-history current-query temporal reader."""
    def __init__(self, cfg: H1TemporalConfig | None = None, seed: int = 42, activity_scale: float = 1.0) -> None:
        super().__init__(cfg, seed, activity_scale)
        self.temporal = QueryTemporalStack(width=self.cfg.temporal_width, heads=self.cfg.heads, layers=self.cfg.layers, ffn=self.cfg.ffn, window=self.cfg.window, age_buckets=16, seed=seed)
        for block in self.temporal.blocks:
            block.age_bias.data.zero_()
    def forward_hidden(self, x, bank, dropout_keep=None):
        return self.forward_hidden_from_frontend(self._z(x, bank, dropout_keep))
    def forward_hidden_from_frontend(self, z: torch.Tensor) -> torch.Tensor:
        return self.temporal(z, use_checkpoint=False).squeeze(1)


class H1FullWindowControl(_H1ScaledFrontendBase):
    """B arm: original full-window temporal operator, matched to T everywhere else."""
    def __init__(self, cfg: H1TemporalConfig | None = None, seed: int = 42, activity_scale: float = 1.0) -> None:
        super().__init__(cfg, seed, activity_scale)
        self.temporal = CausalTransformerStack(self.cfg)
        g = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
        for name, p in sorted(self.temporal.named_parameters()):
            _reinit_param("temporal." + name, p, g)
    def forward_hidden(self, x, bank, dropout_keep=None):
        return self.forward_hidden_from_frontend(self._z(x, bank, dropout_keep))
    def forward_hidden_from_frontend(self, z: torch.Tensor) -> torch.Tensor:
        return self.temporal(z)[:, -1]


def make_matched_pair(*, cfg: H1TemporalConfig | None = None, seed: int = 42, activity_scale: float = 32.0):
    """Two independent arms with common frontend/readout and mapped temporal init."""
    full = H1FullWindowControl(cfg=cfg, seed=seed, activity_scale=activity_scale)
    query = H1CurrentQueryDecoder(cfg=cfg, seed=seed, activity_scale=activity_scale)
    query.frontend.load_state_dict(full.frontend.state_dict())
    query.final_norm.load_state_dict(full.final_norm.state_dict())
    query.readout.load_state_dict(full.readout.state_dict())
    query.temporal.initialize_from_full_window(full.temporal)
    return full, query
