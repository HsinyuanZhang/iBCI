"""Deterministic session/epoch-consistent misleading-identity swap primitives.

The mapping is a partial involution: selected units are paired with nearby
donors in a frozen standardized T4/rate descriptor space, while query activity
and the carrier row stay in their original order.  The caller must provide the
same unmasked matching features to T4 and Z4 siblings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


class MisleadingIdentitySwapError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchedSwap:
    permutation: torch.Tensor
    selected: torch.Tensor
    selected_count: int
    mean_pair_distance: float
    max_pair_distance: float
    seed: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MisleadingIdentitySwapError(message)


def session_epoch_seed(session: str, epoch: int, base_seed: int = 42) -> int:
    _require(bool(session), "session name is required")
    _require(epoch >= 0, "epoch must be nonnegative")
    raw = hashlib.sha256(
        f"misleading_identity_swap_v2::{base_seed}::{session}::{epoch}".encode("utf-8")
    ).digest()
    return int.from_bytes(raw[:8], "big", signed=False)


def build_partial_matched_involution(
    matching_features: torch.Tensor,
    *,
    fraction: float,
    seed: int,
) -> MatchedSwap:
    """Pair a frozen subset of units with nearest available feature donors."""
    _require(matching_features.ndim == 2, "matching features must have shape [N,D]")
    units, width = matching_features.shape
    _require(units >= 4 and width > 0, "matched swap requires at least four units and one feature")
    _require(torch.isfinite(matching_features).all().item(), "matching features must be finite")
    _require(0.0 < float(fraction) <= 1.0, "swap fraction must be in (0,1]")
    selected_count = min(units if units % 2 == 0 else units - 1, int(units * float(fraction)))
    selected_count -= selected_count % 2
    _require(selected_count >= 2, "swap fraction selects fewer than two units")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    priority = torch.randperm(units, generator=generator).tolist()
    chosen = priority[:selected_count]
    # Greedily pair each seeded anchor with its nearest still-available donor.
    # Ties are resolved by the seeded priority order, not device-dependent sort.
    rank = {unit: index for index, unit in enumerate(priority)}
    remaining = set(chosen)
    permutation = torch.arange(units, dtype=torch.long)
    selected = torch.zeros(units, dtype=torch.bool)
    distances: list[float] = []
    features = matching_features.detach().to(device="cpu", dtype=torch.float64)
    while remaining:
        anchor = min(remaining, key=lambda unit: rank[unit])
        remaining.remove(anchor)
        _require(bool(remaining), "unpaired matched-swap anchor")
        donor = min(
            remaining,
            key=lambda unit: (
                float(torch.sum((features[anchor] - features[unit]) ** 2).item()),
                rank[unit],
            ),
        )
        remaining.remove(donor)
        permutation[anchor] = donor
        permutation[donor] = anchor
        selected[anchor] = True
        selected[donor] = True
        distances.append(float(torch.linalg.vector_norm(features[anchor] - features[donor]).item()))
    _require(torch.equal(permutation.index_select(0, permutation), torch.arange(units)),
             "matched swap is not an involution")
    _require(torch.equal(permutation == torch.arange(units), ~selected),
             "selected/fixed-point contract drift")
    return MatchedSwap(
        permutation=permutation,
        selected=selected,
        selected_count=selected_count,
        mean_pair_distance=sum(distances) / len(distances),
        max_pair_distance=max(distances),
        seed=int(seed),
    )


def apply_activity_identity_swap(mean_feature: torch.Tensor, swap: MatchedSwap) -> torch.Tensor:
    """Swap activity-derived rows only; carrier/query tensors are untouched."""
    _require(mean_feature.ndim == 3, "mean feature must have shape [B,N,H]")
    _require(mean_feature.shape[1] == swap.permutation.numel(), "swap/unit count drift")
    permutation = swap.permutation.to(device=mean_feature.device)
    return mean_feature.index_select(1, permutation)
