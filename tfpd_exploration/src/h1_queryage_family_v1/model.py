"""Fresh H1 set-v2-localbalanced FLAT/ROUTE pair with exact W700 QueryAge.

This is an additive factory only.  It begins with the audited H1 v2
unscaled-dot/localbalanced initializer, then discards both freshly initialized
causal temporal modules and installs fresh, identically seeded current-query
stacks.  No trained temporal state, causal PE state, M1 shell, or shorter/log
age operator is imported.
"""

from __future__ import annotations

import torch

from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL, H1TemporalConfig, require
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import shared_parameter_max_abs_diff


FAMILY_FLAT = "CRST-B4[temporal=FW-QueryAge16,routing=FLAT,W700,spatial=set-v2-localbalanced]"
FAMILY_ROUTE = "CRST-B4[temporal=FW-QueryAge16,routing=ROUTE,W700,spatial=set-v2-localbalanced]"


def _stack(cfg: H1TemporalConfig, seed: int) -> QueryTemporalStack:
    if (cfg.temporal_width, cfg.heads, cfg.layers, cfg.ffn, cfg.window) != (256, 8, 4, 512, 700):
        raise ValueError("exact H1 FW-QueryAge16 requires W700/256/8/4/512")
    return QueryTemporalStack(width=256, heads=8, layers=4, ffn=512, window=700, age_buckets=16, seed=seed)


def make_queryage_localbalanced_pair(*, seed: int = 42, cfg: H1TemporalConfig | None = None):
    """Return fresh H1 FLAT/ROUTE with common frontend/readout and fresh QueryAge.

    Existing H1 objects retain their stable ``forward_last`` interface: both
    forward-hidden adapters index the QueryAge [B,1,256] result to [B,256],
    followed by the original final norm/readout in decoder-raw (20x native)
    domain.
    """
    active = cfg or H1_TEMPORAL
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=seed, cfg=active)
    flat.temporal = _stack(active, seed)
    route.temporal = _stack(active, seed)
    flat.name, route.name = FAMILY_FLAT, FAMILY_ROUTE
    if shared_parameter_max_abs_diff(flat, route) != 0.0:
        raise RuntimeError("paired H1 QueryAge non-routing initialization drift")
    if float(route.frontend.attn.g.detach().abs().max()) != 0.0:
        raise RuntimeError("H1 ROUTE gate must initialize at zero")
    return flat, route


__all__ = ["FAMILY_FLAT", "FAMILY_ROUTE", "make_queryage_localbalanced_pair"]
