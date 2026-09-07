"""P-FIX caches z/carrier from frozen D. P-CA recomputes from current D every step."""
from __future__ import annotations

from typing import Any

import torch

from .factory import FreshPArm, normalize_carrier, solve_carrier_float64


def build_pfix_cache(arm: FreshPArm) -> dict[str, dict[str, torch.Tensor]]:
    if arm.train_basis:
        raise RuntimeError("P-FIX cache requires a frozen dictionary")
    cache: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for name, support in arm.frozen.source_supports.items():
            emg = torch.as_tensor(support["emg"], dtype=torch.float64, device=arm.device)
            rates = torch.as_tensor(support["rates"], dtype=torch.float64, device=arm.device)
            z = arm.basis.encode(emg)
            raw = solve_carrier_float64(z, rates)
            carrier = normalize_carrier(raw, arm.frozen)
            cache[name] = {
                "z": z.detach(),
                "raw": raw.detach(),
                "carrier": carrier.detach(),
            }
    return cache


def carrier_for_step(arm: FreshPArm, session_name: str, cache: dict[str, dict[str, torch.Tensor]] | None) -> torch.Tensor:
    if not arm.train_basis:
        if cache is None or session_name not in cache:
            raise RuntimeError(f"P-FIX missing cached carrier for {session_name}")
        return cache[session_name]["carrier"]
    support = arm.frozen.source_supports[session_name]
    emg = torch.as_tensor(support["emg"], dtype=torch.float64, device=arm.device)
    rates = torch.as_tensor(support["rates"], dtype=torch.float64, device=arm.device)
    z = arm.basis.encode(emg)
    raw = solve_carrier_float64(z, rates)
    return normalize_carrier(raw, arm.frozen)


def cache_receipt(cache: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        "sessions": sorted(cache),
        "n_sessions": len(cache),
        "cached_fields": ["z", "raw", "carrier"],
    }
