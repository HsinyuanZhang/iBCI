#!/usr/bin/env python3
"""CPU-only identity smoke for carrier_profile_v3 (no NWB, no GPU)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "carrier_profile_v3", ROOT / "src/btransform_unified_v2/carrier_profile_v3.py"
)
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v3)


def main() -> None:
    theta = v3.canonical_directions_rad()
    rng = np.random.default_rng(0)
    a_true, c_true, b_true = 1.2, -0.7, 3.0
    rates = np.stack(
        [b_true + a_true * np.cos(theta) + c_true * np.sin(theta) + 0.0 * rng.normal(size=8)
         for _ in range(16)],
        axis=1,
    ).T
    # rates: [8 dirs, 16 units] but we want [trials, units] with one trial per dir
    # rebuild as 8 trials × 16 units all sharing the same tuning
    trials = np.stack([b_true + a_true * np.cos(th) + c_true * np.sin(th) + np.zeros(16) for th in theta])
    dirs = np.arange(8, dtype=np.int64)
    got = v3.harmonic_t4_from_rows(trials, dirs, duration_s=None, n0=0.0)
    assert np.allclose(got[:, 0], a_true, atol=1e-6)
    assert np.allclose(got[:, 1], c_true, atol=1e-6)
    assert np.allclose(got[:, 2], math.hypot(a_true, c_true), atol=1e-6)
    assert np.allclose(got[:, 3], b_true, atol=1e-6)
    z, mean, noise = v3.poisson_standardize(np.ones((5, 3)), 0.1)
    assert z.shape == (5, 3) and np.isfinite(z).all()
    print("carrier_profile_v3 smoke OK")


if __name__ == "__main__":
    main()
