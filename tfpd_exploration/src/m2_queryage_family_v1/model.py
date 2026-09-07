"""Fresh CRST-B4 FLAT/ROUTE pair with M1's finite current-query reader.

Only the temporal member differs from the existing M2 CRST-B4 family.  The
local k=5 / eight-slot frontend, calibration carrier, ROUTE logit bias,
readout, target domain, and inherited ``forward_last`` API remain M2's.
No M1 wrapper or M1 state is imported: ``QueryTemporalStack`` is a freshly
initialized operator with the M2 finite W=50 contract.
"""

from __future__ import annotations

import torch

from tfpd_exploration.src.m2_b_small_stability_v1.config import SMALL, SmallConfig, require
from tfpd_exploration.src.m2_family_v1.decoder import M2FamilyDecoder
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack


FAMILY_NAME_FLAT = "CRST-B4[temporal=FW-QueryAge16,routing=FLAT,W50]"
FAMILY_NAME_ROUTE = "CRST-B4[temporal=FW-QueryAge16,routing=ROUTE,W50]"


def _queryage_stack(*, cfg: SmallConfig, seed: int) -> QueryTemporalStack:
    """Bind the exact named M1 operator at M2's immutable dimensions."""
    require(cfg.temporal_width == 256, "FW-QueryAge16 requires width 256")
    require(cfg.heads == 8, "FW-QueryAge16 requires 8 heads")
    require(cfg.layers == 4, "FW-QueryAge16 requires depth 4")
    require(cfg.ffn == 512, "FW-QueryAge16 requires FFN 512")
    require(cfg.window == 50, "this finite family is explicitly W50")
    return QueryTemporalStack(
        width=256,
        heads=8,
        layers=4,
        ffn=512,
        window=50,
        age_buckets=16,
        seed=seed,
    )


class M2QueryAgeFamilyDecoder(M2FamilyDecoder):
    """M2's existing set frontend/readout plus a fresh FW-QueryAge16 reader.

    ``M2FamilyDecoder.forward_last`` needs no override: it feeds frontend
    tokens [B,W,256] to ``temporal``, then indexes the returned [B,1,256] at
    its last (and only) position.  Consequently its M2 decoder-raw/native-5x
    output API and checkpoint/EMA state-dict API are unchanged.
    """

    training_target_space = "decoder_raw"

    def __init__(self, *, routed: bool, seed: int = 42, cfg: SmallConfig = SMALL) -> None:
        super().__init__(routed=routed, seed=seed, cfg=cfg)
        # Deliberately discard the newly initialized M2 full-causal temporal
        # module.  No tensor from it—or from M1—is copied into this operator.
        self.temporal = _queryage_stack(cfg=cfg, seed=seed)
        self.name = FAMILY_NAME_ROUTE if routed else FAMILY_NAME_FLAT


def make_paired_queryage_decoders(
    seed: int = 42,
    cfg: SmallConfig = SMALL,
) -> tuple[M2QueryAgeFamilyDecoder, M2QueryAgeFamilyDecoder]:
    """Fresh FLAT/ROUTE pair with byte-identical non-routing initialization."""
    flat = M2QueryAgeFamilyDecoder(routed=False, seed=seed, cfg=cfg)
    route = M2QueryAgeFamilyDecoder(routed=True, seed=seed, cfg=cfg)
    flat_state, route_state = flat.state_dict(), route.state_dict()
    route_only = {name for name in route_state if ".routing." in name}
    require(route_only, "ROUTE state must contain its sole legal delta")
    require(set(flat_state).isdisjoint(route_only), "FLAT unexpectedly has routing state")
    with torch.no_grad():
        for name, value in route_state.items():
            if name not in route_only:
                require(name in flat_state, f"shared ROUTE key absent in FLAT: {name}")
                value.copy_(flat_state[name])
        route.frontend.attn.routing.g.zero_()
    return flat, route


def shared_parameter_max_abs_diff(
    flat: M2QueryAgeFamilyDecoder,
    route: M2QueryAgeFamilyDecoder,
) -> float:
    """Maximum difference over the paired, non-routing state tensors."""
    left, right = flat.state_dict(), route.state_dict()
    shared = [name for name in left if name in right and ".routing." not in name]
    require(bool(shared), "missing paired shared state")
    return max(float((left[name] - right[name]).abs().max()) for name in shared)
