"""P1 Stage A: two T4 estimators on the same (angles, rates) data.

Estimator A — SUA-style: snap to K canonical bins, fit OLS on per-direction means
(equal weight per observed direction; no trial-count reweighting).

Estimator B — FALCON-style: continuous angles, fit OLS on per-trial rates
(fail closed on rank ≠ 3 when ``strict_rank=True``).

Both estimators share one finite-angle mask. Non-finite angles are dropped with a
count in the receipt (production SUA uses a ``-1`` sentinel; FALCON emits NaN for
centre/rest targets). Never snap NaN into bin 0.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

CANONICAL_DIRECTIONS_RAD = tuple(
    -3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(8)
)


@dataclass(frozen=True)
class TuningFit:
    a: float
    c: float
    m: float
    b: float
    estimator: str
    n_rows: int
    design_rank: int
    present_directions: int | None = None
    n_dropped_nonfinite: int = 0
    status: str = "ok"  # ok | degenerate_zeros | b_undefined


def nearest_canonical_direction_index(target_dir_rad: float) -> int:
    if not math.isfinite(target_dir_rad):
        raise ValueError("nearest_canonical_direction_index refuses non-finite angles")
    directions = np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
    wrapped = (directions - target_dir_rad + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def shared_finite_mask(angles_rad: np.ndarray, rates: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    if angles.shape != rates.shape:
        raise ValueError(f"shape mismatch angles={angles.shape} rates={rates.shape}")
    return np.isfinite(angles) & np.isfinite(rates)


def _ols_abc(design: np.ndarray, response: np.ndarray) -> tuple[float, float, float, int]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    b, a, c = (float(v) for v in coefficients)
    return a, c, b, int(rank)


def fit_estimator_a_bin_means(
    angles_rad: np.ndarray,
    rates: np.ndarray,
    *,
    min_directions: int = 2,
    mask: np.ndarray | None = None,
) -> TuningFit:
    """SUA-style: snap finite angles, average rates per bin, then OLS on means."""
    angles = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    if mask is None:
        mask = shared_finite_mask(angles, rates)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != angles.shape:
            raise ValueError("mask shape mismatch")
    n_dropped = int((~mask).sum())
    angles = angles[mask]
    rates = rates[mask]
    if angles.size == 0:
        return TuningFit(
            a=0.0,
            c=0.0,
            m=0.0,
            b=0.0,
            estimator="A_bin_means",
            n_rows=0,
            design_rank=0,
            present_directions=0,
            n_dropped_nonfinite=n_dropped,
            status="degenerate_zeros",
        )

    indices = np.asarray([nearest_canonical_direction_index(float(a)) for a in angles])
    present = sorted({int(i) for i in indices})
    if len(present) < min_directions:
        return TuningFit(
            a=0.0,
            c=0.0,
            m=0.0,
            b=0.0,
            estimator="A_bin_means",
            n_rows=0,
            design_rank=0,
            present_directions=len(present),
            n_dropped_nonfinite=n_dropped,
            status="degenerate_zeros",
        )

    mean_dirs = []
    mean_rates = []
    for idx in present:
        unit_mask = indices == idx
        mean_dirs.append(CANONICAL_DIRECTIONS_RAD[idx])
        mean_rates.append(float(rates[unit_mask].mean()))
    theta = np.asarray(mean_dirs, dtype=np.float64)
    y = np.asarray(mean_rates, dtype=np.float64)
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    a, c, b, rank = _ols_abc(design, y)
    m = float(math.hypot(a, c))
    return TuningFit(
        a=a,
        c=c,
        m=m,
        b=b,
        estimator="A_bin_means",
        n_rows=len(present),
        design_rank=rank,
        present_directions=len(present),
        n_dropped_nonfinite=n_dropped,
        status="ok",
    )


def fit_estimator_b_per_trial(
    angles_rad: np.ndarray,
    rates: np.ndarray,
    *,
    strict_rank: bool = True,
    mask: np.ndarray | None = None,
) -> TuningFit:
    """FALCON-style: continuous angles, per-trial rates, optional rank hard-fail."""
    angles = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    if mask is None:
        mask = shared_finite_mask(angles, rates)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != angles.shape:
            raise ValueError("mask shape mismatch")
    n_dropped = int((~mask).sum())
    theta = angles[mask]
    y = rates[mask]
    if theta.size < 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="B_per_trial",
            n_rows=int(theta.size),
            design_rank=0,
            present_directions=None,
            n_dropped_nonfinite=n_dropped,
            status="b_undefined",
        )
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if strict_rank and rank != 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="B_per_trial",
            n_rows=int(theta.size),
            design_rank=rank,
            present_directions=None,
            n_dropped_nonfinite=n_dropped,
            status="b_undefined",
        )
    a, c, b, fitted_rank = _ols_abc(design, y)
    if strict_rank and fitted_rank != 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="B_per_trial",
            n_rows=int(theta.size),
            design_rank=fitted_rank,
            present_directions=None,
            n_dropped_nonfinite=n_dropped,
            status="b_undefined",
        )
    m = float(math.hypot(a, c))
    return TuningFit(
        a=a,
        c=c,
        m=m,
        b=b,
        estimator="B_per_trial",
        n_rows=int(theta.size),
        design_rank=fitted_rank,
        present_directions=None,
        n_dropped_nonfinite=n_dropped,
        status="ok",
    )


def vector_delta(a: TuningFit, b: TuningFit) -> dict[str, float] | None:
    if a.status != "ok" or b.status != "ok":
        return None
    return {
        "da": a.a - b.a,
        "dc": a.c - b.c,
        "dm": a.m - b.m,
        "db": a.b - b.b,
        "d_ac_l2": float(math.hypot(a.a - b.a, a.c - b.c)),
    }


def synthesize_cosine_unit(
    *,
    n_trials: int,
    a: float,
    c: float,
    b: float,
    seed: int,
    angle_mode: str = "canonical_balanced",
    rate_noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic labelled rates for estimator recovery tests."""
    rng = np.random.RandomState(seed)
    if angle_mode == "canonical_balanced":
        indices = np.arange(n_trials) % 8
        angles = np.asarray([CANONICAL_DIRECTIONS_RAD[int(i)] for i in indices], dtype=np.float64)
    elif angle_mode == "continuous_uniform":
        angles = rng.uniform(-math.pi, math.pi, size=n_trials)
    elif angle_mode == "continuous_imbalanced":
        # Off-canonical continuous angles with unequal density near three modes.
        modes = np.asarray([0.15, 1.40, -2.10], dtype=np.float64)  # not bin centers
        weights = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
        choice = rng.choice(3, size=n_trials, p=weights)
        angles = modes[choice] + rng.normal(0.0, 0.08, size=n_trials)
    else:
        raise ValueError(f"unknown angle_mode={angle_mode!r}")
    rates = b + a * np.cos(angles) + c * np.sin(angles)
    if rate_noise_std > 0:
        rates = rates + rng.normal(0.0, rate_noise_std, size=n_trials)
    return angles.astype(np.float64), rates.astype(np.float64)


def compare_estimators_on_arrays(
    angles_rad: np.ndarray,
    rates: np.ndarray,
) -> dict[str, object]:
    mask = shared_finite_mask(angles_rad, rates)
    fit_a = fit_estimator_a_bin_means(angles_rad, rates, mask=mask)
    fit_b = fit_estimator_b_per_trial(angles_rad, rates, strict_rank=True, mask=mask)
    return {
        "shared_n_kept": int(mask.sum()),
        "shared_n_dropped": int((~mask).sum()),
        "A": asdict(fit_a),
        "B": asdict(fit_b),
        "delta_A_minus_B": vector_delta(fit_a, fit_b),
        "comparable": fit_a.status == "ok" and fit_b.status == "ok",
    }


def sweep_m_synthetic(
    *,
    true_a: float = 1.2,
    true_c: float = -0.7,
    true_b: float = 3.0,
    ms: Sequence[int] = (10, 20, 30, 40, 50),
    noise_std: float = 0.5,
    seed: int = 0,
    angle_mode: str = "continuous_imbalanced",
) -> list[dict[str, object]]:
    """Noise-perturbed recovery sweep on off-canonical imbalanced angles."""
    rows: list[dict[str, object]] = []
    for m in ms:
        angles, rates = synthesize_cosine_unit(
            n_trials=m,
            a=true_a,
            c=true_c,
            b=true_b,
            seed=seed + m,
            angle_mode=angle_mode,
            rate_noise_std=noise_std,
        )
        cmp = compare_estimators_on_arrays(angles, rates)
        rows.append(
            {
                "M": m,
                "noise_std": noise_std,
                "angle_mode": angle_mode,
                "true": {"a": true_a, "c": true_c, "b": true_b},
                **cmp,
            }
        )
    return rows
