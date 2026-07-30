"""Calibration-only target-tuning (T4) features for native FALCON M1/M2 MUA.

This module intentionally has no access to FALCON query/evaluation covariates.  It
reads target metadata only from the corresponding calibration NWB file: held-in
during fit/validation and held-out-calib only during the explicit test-only
protocol.  It combines those labels with calibration-trial neural counts supplied
by ``FalconDataModule`` and raises on missing/misaligned labels rather than
inferring them from behaviour or query covariates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO


T4_DIM = 4
T4_FEATURE_NAMES = ("m_cos_phi", "m_sin_phi", "m", "baseline_rate")


def calibration_target_angles(nwb_path: Path, task: str) -> np.ndarray:
    """Return one target angle per NWB trial, ``NaN`` when no direction is defined.

    M1's ``tgt_loc`` is a scalar target azimuth in degrees.  M2's ``tgt_loc`` is a
    target screen coordinate; angles are measured relative to the documented centre
    ``(0.5, 0.5)``.  Centre/rest targets are deliberately unlabeled: they cannot be
    converted into an arbitrary direction without changing the estimand.
    """
    task = getattr(task, "name", getattr(task, "value", task))
    task = str(task).split(".")[-1].lower()
    if task not in {"m1", "m2"}:
        raise ValueError(f"Native FALCON T4 supports m1/m2, got {task!r}")
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        trials = io.read().trials
        if trials is None:
            raise ValueError(f"Calibration file has no trials table: {nwb_path}")
        frame = trials.to_dataframe()
    if "tgt_loc" not in frame:
        raise ValueError(f"Calibration trials have no tgt_loc labels: {nwb_path}")

    if task == "m1":
        values = np.asarray(frame["tgt_loc"], dtype=np.float64)
        angles = np.deg2rad(values)
        angles[~np.isfinite(angles)] = np.nan
        return angles.astype(np.float32)

    angles = np.full(len(frame), np.nan, dtype=np.float32)
    centre = np.asarray([0.5, 0.5], dtype=np.float64)
    for index, value in enumerate(frame["tgt_loc"]):
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        delta = point - centre
        if np.linalg.norm(delta) <= 1.0e-8:
            continue
        angles[index] = np.arctan2(delta[1], delta[0])
    return angles


def validate_trial_label_alignment(
    trial_change: np.ndarray, target_angles: np.ndarray, *, source: str
) -> None:
    """Fail closed unless the feature labels match the trialization boundary."""
    starts = int(np.asarray(trial_change, dtype=bool).sum())
    if starts != int(target_angles.shape[0]):
        raise ValueError(
            "FALCON T4 trial labels cannot be aligned to calibration neural trials for "
            f"{source}: trial_change has {starts} starts but NWB trials has "
            f"{target_angles.shape[0]} labels"
        )


def _design(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    usable = np.isfinite(angles)
    theta = angles[usable].astype(np.float64, copy=False)
    return np.stack([np.ones(theta.shape[0]), np.cos(theta), np.sin(theta)], axis=1), usable


def t4_from_trial_sums(
    trial_spike_sums: np.ndarray,
    trial_lengths: np.ndarray,
    target_angles: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    """Fit channel-level cosine tuning from calibration-only trial spike sums.

    ``trial_spike_sums`` is ``[M,N]`` and uses the un-interpolated valid prefix of
    each calibration trial.  This avoids treating cubic interpolation or pad values
    as spikes.  A rank-deficient design is an explicit data/protocol failure.
    """
    sums = np.asarray(trial_spike_sums, dtype=np.float64)
    lengths = np.asarray(trial_lengths, dtype=np.float64).reshape(-1)
    angles = np.asarray(target_angles, dtype=np.float64).reshape(-1)
    if sums.ndim != 2 or sums.shape[0] != lengths.size or sums.shape[0] != angles.size:
        raise ValueError(
            f"T4 shape mismatch for {source}: sums={sums.shape}, lengths={lengths.shape}, "
            f"angles={angles.shape}"
        )
    if np.any(lengths <= 0) or not np.all(np.isfinite(sums)):
        raise ValueError(f"Invalid native-MUA calibration spike sums for {source}")
    design, usable = _design(angles)
    if design.shape[0] < 3:
        raise ValueError(f"T4 needs >=3 directional trials for {source}, got {design.shape[0]}")
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"T4 direction design is rank {rank}, not 3, for {source}")
    rates = sums[usable] / lengths[usable, None]
    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, rates, rcond=None)
    if int(fitted_rank) != 3:
        raise ValueError(f"T4 least-squares rank {fitted_rank}, not 3, for {source}")
    baseline, a, c = coefficients
    modulation = np.sqrt(a * a + c * c)
    features = np.stack([a, c, modulation, baseline], axis=-1).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Non-finite native-MUA T4 features for {source}")
    return features


def fit_train_t4_stats(
    session_trial_sums: Mapping[str, np.ndarray],
    session_trial_lengths: Mapping[str, np.ndarray],
    session_target_angles: Mapping[str, np.ndarray],
    session_names: Sequence[str],
    calibration_n_trials: int,
    all_support_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score statistics from train sessions and a predeclared support policy."""
    if calibration_n_trials < 1:
        raise ValueError("Native T4 requires an integer calibration_n_trials >= 1")
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
            chunks.append(
                t4_from_trial_sums(
                    sums[start : start + calibration_n_trials],
                    lengths[start : start + calibration_n_trials],
                    angles[start : start + calibration_n_trials],
                    source=f"{name}[{start}:{start + calibration_n_trials}]",
                )
            )
    if not chunks:
        raise ValueError("No train calibration windows were available for native T4 stats")
    values = np.concatenate(chunks, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std


def deterministic_row_permutation(num_channels: int, *, session_name: str, seed: int) -> np.ndarray:
    """Stable non-identity channel-row permutation for the TS4 content control."""
    if num_channels < 2:
        raise ValueError("TS4 requires at least two native-MUA channels")
    digest = hashlib.sha256(f"native-mua-ts4-v1:{seed}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(num_channels)
    if np.array_equal(permutation, np.arange(num_channels)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)
