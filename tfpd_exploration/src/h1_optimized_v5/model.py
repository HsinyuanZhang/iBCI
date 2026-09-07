"""V5 changes only the query-time temporal age lookup from V4."""
from __future__ import annotations

import torch

from tfpd_exploration.src.h1_optimized_v4.model import H1SignedFull, H1SignedQuery
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v3 import LogAgeQueryTemporalStack


FRONTEND_CONTRACT_VERSION = 4
TEMPORAL_CONTRACT_VERSION = 3


class H1LogAgeQuery(H1SignedQuery):
    """V4 reader architecture with log-age bias applied only at query read time."""
    def __init__(self) -> None:
        super().__init__()
        cfg = self.cfg
        self.temporal = LogAgeQueryTemporalStack(
            width=cfg.temporal_width, heads=cfg.heads, layers=cfg.layers,
            ffn=cfg.ffn, window=cfg.window, age_buckets=16, seed=42,
        )


def make_matched_pair(*, activity_scale: float = 1.0) -> tuple[H1SignedFull, H1LogAgeQuery]:
    """Fresh matched V4 FULL vs V5 log-age query pair; never accepts warmstarts."""
    if activity_scale != 1.0:
        raise ValueError("V5 retains V4's raw-count signed frontend contract (activity_scale=1)")
    # V4 performs deterministic fresh initialization (seed 42), copies the
    # matched frontend/readout, then initializes V2 query temporal weights from
    # FULL.  Replace only that temporal reader with the V3 log-age equivalent.
    from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair as v4_pair
    full, _discarded_v4_query = v4_pair(activity_scale=activity_scale)
    query = H1LogAgeQuery()
    query.frontend.load_state_dict(full.frontend.state_dict(), strict=True)
    query.final_norm.load_state_dict(full.final_norm.state_dict(), strict=True)
    query.readout.load_state_dict(full.readout.state_dict(), strict=True)
    query.temporal.initialize_from_full_window(full.temporal)
    if int(full.frontend_contract_version.item()) != FRONTEND_CONTRACT_VERSION:
        raise RuntimeError("unexpected V4 frontend contract")
    if int(query.frontend_contract_version.item()) != FRONTEND_CONTRACT_VERSION:
        raise RuntimeError("frontend contract drift")
    if int(query.temporal.temporal_contract_version.item()) != TEMPORAL_CONTRACT_VERSION:
        raise RuntimeError("V3 temporal contract drift")
    return full, query
