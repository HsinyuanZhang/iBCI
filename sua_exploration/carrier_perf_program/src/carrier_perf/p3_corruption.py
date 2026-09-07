"""P3 Stage A: training-time carrier corruption sampler.

Motivation (handoff): RS4/LS4 land *below* Z4, so the decoder over-trusts a bad
carrier. Corruption augmentation should teach the decoder to fall back toward
the activity path when the carrier is unreliable.

This module only defines the *sampler*. It does not train. Probabilities and M
ranges must be frozen in a prelaunch receipt before any GPU cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

CorruptionKind = Literal["identity", "row_permute", "gaussian_noise", "zero"]


@dataclass(frozen=True)
class CorruptionConfig:
    """Pre-registered sampler parameters. Do not tune after seeing R²."""

    p_identity: float = 0.5
    p_row_permute: float = 0.2
    p_gaussian_noise: float = 0.2
    p_zero: float = 0.1
    noise_std: float = 1.0  # in the *already standardized* side-feature space
    # Optional M-sampling is expressed as a discrete set; runner samples uniformly.
    m_grid: tuple[int, ...] = (10, 15, 20, 30, 50)

    def validate(self) -> None:
        probs = (self.p_identity, self.p_row_permute, self.p_gaussian_noise, self.p_zero)
        if any(p < 0 for p in probs):
            raise ValueError(f"negative probability in {probs}")
        total = sum(probs)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"corruption probabilities must sum to 1, got {total}")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if not self.m_grid or any(m < 1 for m in self.m_grid):
            raise ValueError(f"invalid m_grid={self.m_grid}")


def sample_kind(config: CorruptionConfig, rng: np.random.RandomState) -> CorruptionKind:
    config.validate()
    kinds: tuple[CorruptionKind, ...] = (
        "identity",
        "row_permute",
        "gaussian_noise",
        "zero",
    )
    probs = np.asarray(
        [config.p_identity, config.p_row_permute, config.p_gaussian_noise, config.p_zero],
        dtype=np.float64,
    )
    idx = int(rng.choice(len(kinds), p=probs))
    return kinds[idx]


def apply_corruption(
    features: np.ndarray,
    kind: CorruptionKind,
    *,
    rng: np.random.RandomState,
    noise_std: float,
) -> np.ndarray:
    """Corrupt a ``[N, D]`` side-feature matrix. Never mutates the input."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [N,D] features, got {x.shape}")
    if kind == "identity":
        return x.copy()
    if kind == "zero":
        return np.zeros_like(x)
    if kind == "gaussian_noise":
        return (x + rng.normal(0.0, noise_std, size=x.shape)).astype(np.float32)
    if kind == "row_permute":
        if x.shape[0] < 2:
            # Degenerate session: permutation is identity; caller should treat as no-op.
            return x.copy()
        perm = rng.permutation(x.shape[0])
        # Reject accidental identity for N>=2 by reshuffling once (best effort).
        if np.array_equal(perm, np.arange(x.shape[0])):
            perm = np.roll(perm, 1)
        return x[perm].astype(np.float32)
    raise ValueError(f"unknown kind {kind!r}")


def sample_m(config: CorruptionConfig, rng: np.random.RandomState) -> int:
    config.validate()
    return int(rng.choice(np.asarray(config.m_grid, dtype=np.int64)))


def corrupt_batch(
    features: np.ndarray,
    config: CorruptionConfig,
    *,
    seed: int,
) -> dict[str, object]:
    """One draw: returns corrupted features plus a receipt of the draw."""
    config.validate()
    rng = np.random.RandomState(seed)
    kind = sample_kind(config, rng)
    out = apply_corruption(features, kind, rng=rng, noise_std=config.noise_std)
    m = sample_m(config, rng)
    return {
        "kind": kind,
        "m_sampled": m,
        "features": out,
        "config": asdict(config),
        "seed": seed,
    }


# Pre-registered falsifiable predictions (handoff P3). Recorded here so tests
# and future GPU receipts can cite a single source.
P3_PREDICTIONS = {
    "primary": "held-out cross-session R2 variance decreases more than the mean increases",
    "mechanism_readout": "post-aug RS4/LS4 scores move toward Z4 (no longer below Z4)",
    "falsification": (
        "if mean rises but variance does not fall, reject the trust-calibration "
        "hypothesis and report ordinary regularization instead"
    ),
}
