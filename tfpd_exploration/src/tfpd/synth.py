"""Synthetic task-frame session generator for Stage-0 gates.

Each session draws a variable unit count, per-unit tuning direction theta_i, gain a_i,
and baseline b_i.  A smooth 2-D trajectory (sum of low-frequency sinusoids) drives
Poisson rates lambda_i(t) = softplus(b_i + a_i cos(phi_t - theta_i)).  The carrier is a
noisy closed-form T4-style estimate fitted from *support* trials only; query bins never
influence it.  A carrier-free model can at best use activity statistics, which carry no
session-stable direction information by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticSession:
    counts: torch.Tensor  # [T, N] integer spike counts
    behaviour: torch.Tensor  # [T, 2] trajectory
    carrier: torch.Tensor  # [N, 4] T4-style estimate from support trials
    support_mask: torch.Tensor  # [T] True on support bins

    @property
    def num_units(self) -> int:
        return int(self.counts.shape[1])


def _smooth_trajectory(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    parts = []
    for channel in range(2):
        phase = rng.uniform(0, 2 * np.pi, size=3)
        amp = rng.uniform(0.4, 1.0, size=3)
        freq = np.array([0.02, 0.05, 0.11]) * (2 * np.pi)
        parts.append(np.sum(amp * np.sin(freq[None, :] * t[:, None] + phase[None, :]), axis=1))
    traj = np.stack(parts, axis=1)
    return traj - traj.mean(axis=0, keepdims=True)


def fit_carrier_support(
    counts: np.ndarray, phi: np.ndarray, support_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form T4-style fit on support bins only: (a, c, m, b).

    Deterministic given (counts, phi, support_mask); query bins cannot enter.
    """
    phi_s = phi[support_mask]
    design = np.stack([np.ones_like(phi_s), np.cos(phi_s), np.sin(phi_s)], axis=1)
    counts_s = counts[support_mask]
    coef, *_ = np.linalg.lstsq(design, counts_s, rcond=None)
    b_hat, a_hat, c_hat = coef
    m_hat = np.sqrt(a_hat**2 + c_hat**2)
    return a_hat, c_hat, m_hat, b_hat


def generate_session(
    seed: int,
    length: int = 256,
    num_units: int | None = None,
    support_fraction: float = 0.3,
    carrier_noise: float = 0.10,
    query_perturb: float = 0.0,
) -> SyntheticSession:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(24, 193)) if num_units is None else num_units

    theta = rng.uniform(0, 2 * np.pi, size=n)
    gain = rng.uniform(0.5, 2.0, size=n)
    baseline = rng.uniform(0.2, 1.0, size=n)

    traj = _smooth_trajectory(length, rng)  # [T, 2]
    direction = traj / np.clip(np.linalg.norm(traj, axis=1, keepdims=True), 1e-8, None)
    phi = np.arctan2(direction[:, 1], direction[:, 0])  # [T]

    modulation = baseline[None, :] + gain[None, :] * np.cos(phi[:, None] - theta[None, :])
    rate = np.logaddexp(0.0, modulation) + 0.05  # softplus floor, strictly positive
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
        raise AssertionError("synthetic lambda positivity violated")
    counts = rng.poisson(rate).astype(np.float32)

    # Optional query-bin perturbation, applied BEFORE the carrier fit.  Because
    # the closed-form fit reads support bins only and the rng draw order is
    # unchanged, the emitted carrier must stay bit-identical (gate G4b).
    if query_perturb:
        counts[int(length * support_fraction) :] += query_perturb

    support_end = int(length * support_fraction)
    support_mask = np.zeros(length, dtype=bool)
    support_mask[:support_end] = True

    # Closed-form T4-style fit on support bins only, then noise (estimation error).
    a_hat, c_hat, m_hat, b_hat = fit_carrier_support(counts, phi, support_mask)
    noise = rng.normal(0.0, carrier_noise, size=(3, n)) * np.stack(
        [a_hat + 0.5, c_hat + 0.5, b_hat + 0.5]
    )
    a_noisy = a_hat + noise[0]
    c_noisy = c_hat + noise[1]
    b_noisy = b_hat + noise[2]
    m_noisy = np.sqrt(a_noisy**2 + c_noisy**2)
    carrier = np.stack([a_noisy, c_noisy, m_noisy, b_noisy], axis=1)  # [N, 4]

    return SyntheticSession(
        counts=torch.from_numpy(counts),
        behaviour=torch.from_numpy(traj.astype(np.float32)),
        carrier=torch.from_numpy(carrier.astype(np.float32)),
        support_mask=torch.from_numpy(support_mask),
    )


def generate_cohort(seed: int, num_sessions: int) -> list[SyntheticSession]:
    return [generate_session(seed=seed * 1000 + k) for k in range(num_sessions)]


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Variance-weighted pooled R^2, matching the house metric family."""
    residual = ((predictions - targets) ** 2).sum().item()
    total = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - residual / max(total, 1e-12)
