"""V5R2 post-authorization bridge entrypoint.

V5R2 deliberately does not duplicate or alter V5's corrected schema bridge.
After the V5R2 core has validated an explicit pin for every V5 source and
claimed its own nonce, this narrow wrapper imports and delegates to V5's
already-pinned concrete parity implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def concrete_parity_after_future_authorization_v5r2(root: Path) -> dict[str, Any]:
    """Run the exact V5 bridge only after the V5R2 authorization gate."""
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5 import (
        concrete_parity_after_future_authorization_v5,
    )

    result = concrete_parity_after_future_authorization_v5(root)
    if result.get("external_subm_scoring_performed") is not False:
        raise RuntimeError("V5 delegated result enabled forbidden external scoring")
    return {
        **result,
        "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_ROOT_REVIEW_V5R2",
        "v5r2_explicit_v5_source_closure": True,
    }
