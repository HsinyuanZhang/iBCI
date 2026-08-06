"""Analytic functional-carrier primitives with a fixed four-value interface.

The functions in this module are deliberately independent of the legacy T4
implementation.  They consume calibration sufficient statistics only: neural
response sums, exposure, and a source-declared two-dimensional task basis.
They have no query-data argument and perform no target-session optimisation by
backpropagation.

For a circular basis ``[cos(theta), sin(theta)]`` with zero ridge penalty, the
four-value descriptor is exactly the legacy T4 descriptor
``[a, c, sqrt(a**2 + c**2), baseline]``.  Keeping this reduction executable is
the first correctness gate for the broader T4G/AFC4 programme.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np


AFC4_DIM = 4
AFC4_CIRCULAR_FEATURE_NAMES = ("w_cos", "w_sin", "w_norm", "baseline_rate")


def afc4_from_response_sums(
    response_sums: np.ndarray,
    exposure: np.ndarray,
    task_basis: np.ndarray,
    *,
    source: str,
    valid_mask: np.ndarray | None = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Fit a two-basis analytic carrier and return ``[w0,w1,||w||,b]``.

    ``response_sums`` is ``[M,N]``, ``exposure`` is ``[M]``, and
    ``task_basis`` is ``[M,2]``.  The intercept is never penalised.  ``ridge=0``
    follows the same LAPACK least-squares path used by native T4; positive ridge
    is solved from the regularised normal equations.
    """
    sums = np.asarray(response_sums, dtype=np.float64)
    lengths = np.asarray(exposure, dtype=np.float64).reshape(-1)
    basis = np.asarray(task_basis, dtype=np.float64)
    if sums.ndim != 2:
        raise ValueError(f"AFC4 response_sums must be [M,N] for {source}, got {sums.shape}")
    if basis.ndim != 2 or basis.shape[1] != 2:
        raise ValueError(f"AFC4 task_basis must be [M,2] for {source}, got {basis.shape}")
    if sums.shape[0] != lengths.size or sums.shape[0] != basis.shape[0]:
        raise ValueError(
            f"AFC4 shape mismatch for {source}: sums={sums.shape}, "
            f"exposure={lengths.shape}, basis={basis.shape}"
        )
    if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise ValueError(f"AFC4 exposure must be finite and positive for {source}")
    if not np.all(np.isfinite(sums)):
        raise ValueError(f"AFC4 response sums must be finite for {source}")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError(f"AFC4 ridge must be finite and non-negative for {source}")

    if valid_mask is None:
        usable = np.all(np.isfinite(basis), axis=1)
    else:
        usable = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if usable.size != sums.shape[0]:
            raise ValueError(f"AFC4 valid_mask length mismatch for {source}")
        usable = usable & np.all(np.isfinite(basis), axis=1)
    if int(usable.sum()) < 3:
        raise ValueError(f"AFC4 needs >=3 valid samples for {source}, got {int(usable.sum())}")

    phi = basis[usable]
    design = np.concatenate([np.ones((phi.shape[0], 1), dtype=np.float64), phi], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"AFC4 design rank is {rank}, not 3, for {source}")
    rates = sums[usable] / lengths[usable, None]

    if ridge == 0.0:
        coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, rates, rcond=None)
        if int(fitted_rank) != 3:
            raise ValueError(f"AFC4 least-squares rank is {fitted_rank}, not 3, for {source}")
    else:
        penalty = np.diag(np.asarray([0.0, ridge, ridge], dtype=np.float64))
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ rates)

    baseline = coefficients[0]
    weights = coefficients[1:]
    features = np.stack(
        [weights[0], weights[1], np.linalg.norm(weights, axis=0), baseline], axis=-1
    ).astype(np.float32)
    if features.shape != (sums.shape[1], AFC4_DIM) or not np.all(np.isfinite(features)):
        raise ValueError(f"Invalid AFC4 output for {source}: {features.shape}")
    return features


def circular_afc4_from_trial_sums(
    trial_spike_sums: np.ndarray,
    trial_lengths: np.ndarray,
    target_angles: np.ndarray,
    *,
    source: str,
    ridge: float = 0.0,
) -> np.ndarray:
    """Circular AFC4 specialisation using calibration-only target angles."""
    angles = np.asarray(target_angles, dtype=np.float64).reshape(-1)
    basis = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return afc4_from_response_sums(
        trial_spike_sums,
        trial_lengths,
        basis,
        source=source,
        valid_mask=np.isfinite(angles),
        ridge=ridge,
    )


def fit_train_circular_afc4_stats(
    session_trial_sums: Mapping[str, np.ndarray],
    session_trial_lengths: Mapping[str, np.ndarray],
    session_target_angles: Mapping[str, np.ndarray],
    session_names: Sequence[str],
    calibration_n_trials: int,
    all_support_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit circular-AFC4 z-score statistics from source sessions only."""
    if calibration_n_trials < 1:
        raise ValueError("Circular AFC4 requires calibration_n_trials >= 1")
    chunks: list[np.ndarray] = []
    for name in session_names:
        sums = session_trial_sums[name]
        lengths = session_trial_lengths[name]
        angles = session_target_angles[name]
        max_start = sums.shape[0] - calibration_n_trials
        if max_start < 0:
            raise ValueError(f"Session {name} has fewer trials than calibration_n_trials")
        starts = range(max_start + 1) if all_support_windows else (0,)
        for start in starts:
            stop = start + calibration_n_trials
            chunks.append(
                circular_afc4_from_trial_sums(
                    sums[start:stop],
                    lengths[start:stop],
                    angles[start:stop],
                    source=f"{name}[{start}:{stop}]",
                    ridge=0.0,
                )
            )
    if not chunks:
        raise ValueError("No source calibration windows were available for AFC4 stats")
    values = np.concatenate(chunks, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std


def deterministic_afc4_row_permutation(
    num_channels: int, *, session_name: str, seed: int
) -> np.ndarray:
    """Use the frozen TS4 permutation contract for the circular reduction."""
    if num_channels < 2:
        raise ValueError("AFC4 row shuffle requires at least two channels")
    digest = hashlib.sha256(f"native-mua-ts4-v1:{seed}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(num_channels)
    if np.array_equal(permutation, np.arange(num_channels)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)


def deterministic_label_permutation(
    num_samples: int, *, context: str, seed: int
) -> np.ndarray:
    """Return a stable non-identity calibration-label permutation."""
    if num_samples < 2:
        raise ValueError("AFC4 label shuffle requires at least two samples")
    digest = hashlib.sha256(f"afc4-label-shuffle-v1:{seed}:{context}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(num_samples)
    if np.array_equal(permutation, np.arange(num_samples)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)
