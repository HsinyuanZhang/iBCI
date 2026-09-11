"""16-direction carrier ablation math (user directive 2026-09-10, plan C1).

Doubles the angular resolution of the t4-family carrier design matrix: the
one-hot direction weights and the first-harmonic readout table go from
K_DIR=8 (45-degree bins, btransform_unified_v2.carrier_profile_v3) to 16
(22.5-degree bins).  Everything else in the recipe is untouched -- same
R700/H300 phase windows, same Poisson standardization, same n0 shrinkage,
same closed-form first-harmonic readout law (missing bins stay 0, normalization
2/K over the fixed table), same fourth column b = mean_k R - mean_hold.

DEGENERACY AUDIT (recorded at cache-build time): every target_dir in the 688
dataset sits EXACTLY on one of the 8 canonical angles (verified over all 68
sessions / 20089 trials, 2026-09-10), and each canonical angle lands on an
EVEN 16-bin center.  The occupied 16-bins therefore replicate the 8-bin
conditional responses exactly (same per-bin denominators n_k + n0), the empty
odd bins read 0, and the 2/16 table mean halves a/c/m/b relative to the 8-bin
2/8 mean.  Because the halving is an exact power-of-two scale, the TRAIN-FIT
column normalizer (mean/std over the same population) cancels it bit-for-bit:
the normalized dir16 carrier is expected to equal the normalized 8-dir carrier
to float32 rounding.  The builder records the observed max-abs deviation per
session so the reading is honest evidence, not an assumption.
"""
from __future__ import annotations

import math

import numpy as np

N_DIRS_16 = 16


def canonical_directions16_rad() -> np.ndarray:
    """The 16-bin table: -3*pi/4 + k*(pi/8), k = 0..15 (22.5-degree spacing).

    The stock 8-bin canonical angles all land on EVEN entries of this table
    (angle_8[k] == angle_16[2k], with 180 degrees == angle_16[14])."""
    return np.asarray(
        [-3.0 * math.pi / 4.0 + k * (math.pi / 8.0) for k in range(N_DIRS_16)],
        dtype=np.float64,
    )


def nearest_direction16_index(angle: float) -> int:
    """Nearest 16-bin canonical index of a raw angle (ties -> smaller index)."""
    directions = canonical_directions16_rad()
    wrapped = (directions - float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def direction16_indices(target_dirs: np.ndarray) -> np.ndarray:
    """Vectorized nearest-16 mapping of raw target angles (radians)."""
    target_dirs = np.asarray(target_dirs, dtype=np.float64)
    if target_dirs.ndim != 1 or target_dirs.size == 0:
        raise ValueError("target_dirs must be a nonempty 1-D angle array")
    if not np.isfinite(target_dirs).all():
        raise ValueError("target_dirs must be finite")
    return np.asarray(
        [nearest_direction16_index(float(a)) for a in target_dirs],
        dtype=np.int64,
    )


def one_hot_directions16(indices: np.ndarray) -> np.ndarray:
    """[T] indices -> [T, 16] one-hot weights (v3.one_hot_directions law)."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= N_DIRS_16):
        raise ValueError(f"direction16 indices must be in [0, {N_DIRS_16})")
    weights = np.zeros((indices.size, N_DIRS_16), dtype=np.float64)
    weights[np.arange(indices.size), indices] = 1.0
    return weights


def harmonic_readout16(
    response_dir: np.ndarray, theta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-harmonic readout over the fixed 16-direction table.

    Identical closed form to v3.harmonic_readout with K=16: missing bins stay
    0, a = 2*mean_k(R cos(theta_k)), c = 2*mean_k(R sin(theta_k)), m = hypot.
    """
    response_dir = np.asarray(response_dir, dtype=np.float64)
    if response_dir.ndim != 2 or response_dir.shape[1] != N_DIRS_16:
        raise ValueError(f"response_dir must be [units, {N_DIRS_16}]")
    if theta is None:
        theta = canonical_directions16_rad()
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (N_DIRS_16,):
        raise ValueError("theta must be length 16")
    a = 2.0 * (response_dir * np.cos(theta)[None, :]).mean(axis=1)
    c = 2.0 * (response_dir * np.sin(theta)[None, :]).mean(axis=1)
    return a, c, np.hypot(a, c)


__all__ = [
    "N_DIRS_16",
    "canonical_directions16_rad",
    "nearest_direction16_index",
    "direction16_indices",
    "one_hot_directions16",
    "harmonic_readout16",
]
