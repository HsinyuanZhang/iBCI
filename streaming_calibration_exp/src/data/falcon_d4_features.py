"""Calibration-only categorical profiles (D4) for native FALCON M1 MUA.

This standalone helper deliberately has no query/evaluation labels.  Its only
label input is the ``obj_id`` vector read from the corresponding calibration
NWB, which is paired with already-valid per-trial spike sums and valid trial
lengths supplied by the data module.  Despite the historical field name,
these features are called a *categorical calibration profile*, never an object
descriptor: at deployment M1's M=10 labels are collinear with target location.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO


D4_DIM = 4
M1_D4_LEVELS = (1, 2, 3, 4)


def _canonical_task(task: object) -> str:
    value = getattr(task, "name", getattr(task, "value", task))
    return str(value).split(".")[-1].lower()


def _validated_obj_ids(values: object, *, source: str) -> np.ndarray:
    """Return exact integer M1 categories, rejecting aliases and unknowns."""
    raw = np.asarray(values).reshape(-1)
    if raw.size == 0:
        raise ValueError(f"D4 needs non-empty obj_id labels for {source}")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"D4 obj_id labels must be numeric for {source}") from error
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"D4 obj_id labels must be finite exact integers for {source}")
    labels = numeric.astype(np.int64)
    unknown = sorted(set(labels.tolist()) - set(M1_D4_LEVELS))
    if unknown:
        raise ValueError(f"D4 obj_id labels outside {M1_D4_LEVELS} for {source}: {unknown}")
    return labels


def calibration_obj_id_labels(nwb_path: Path, task: object) -> np.ndarray:
    """Read M1 calibration-trial ``obj_id`` labels, never query labels."""
    if _canonical_task(task) != "m1":
        raise ValueError(f"Native FALCON D4 supports M1 only, got {task!r}")
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        trials = io.read().trials
        if trials is None:
            raise ValueError(f"Calibration file has no trials table: {nwb_path}")
        frame = trials.to_dataframe()
    if "obj_id" not in frame:
        raise ValueError(f"Calibration trials have no obj_id labels: {nwb_path}")
    return _validated_obj_ids(frame["obj_id"].to_numpy(), source=str(nwb_path))


def validate_trial_label_alignment(
    trial_change: np.ndarray, obj_ids: np.ndarray, *, source: str
) -> None:
    """Fail closed unless calibration obj_id labels align with trial boundaries."""
    starts = int(np.asarray(trial_change, dtype=bool).sum())
    labels = _validated_obj_ids(obj_ids, source=source)
    if starts != labels.size:
        raise ValueError(
            "FALCON D4 calibration labels cannot be aligned to neural trials for "
            f"{source}: trial_change has {starts} starts but obj_id has {labels.size} labels"
        )


def d4_from_trial_sums(
    trial_spike_sums: np.ndarray,
    trial_lengths: np.ndarray,
    obj_ids: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    """Compute raw ``[mu_1, mu_2, mu_3, mu_4]`` calibration rates per channel.

    ``trial_spike_sums`` must be the prevalidated valid-prefix sums, not values
    reconstructed from spike times.  Each trial is first exposure-corrected by
    its own valid length, then rate vectors are averaged within category.  Thus
    different trial lengths cannot make a longer trial dominate a condition.
    All four M1 categories must be present in the supplied chronological support.
    """
    sums = np.asarray(trial_spike_sums, dtype=np.float64)
    lengths = np.asarray(trial_lengths, dtype=np.float64).reshape(-1)
    labels = _validated_obj_ids(obj_ids, source=source)
    if sums.ndim != 2 or sums.shape[0] != lengths.size or sums.shape[0] != labels.size:
        raise ValueError(
            f"D4 shape mismatch for {source}: sums={sums.shape}, lengths={lengths.shape}, "
            f"obj_ids={labels.shape}"
        )
    if not np.all(np.isfinite(sums)) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError(f"Invalid native-MUA calibration spike sums or lengths for {source}")
    present = tuple(sorted(set(labels.tolist())))
    if present != M1_D4_LEVELS:
        missing = sorted(set(M1_D4_LEVELS) - set(present))
        raise ValueError(
            f"D4 support must contain all levels {M1_D4_LEVELS} exactly as its label vocabulary "
            f"for {source}; present={present}, missing={missing}"
        )
    rates = sums / lengths[:, None]
    features = np.stack([rates[labels == level].mean(axis=0) for level in M1_D4_LEVELS], axis=1)
    features = features.astype(np.float32)
    if features.shape[1] != D4_DIM or not np.all(np.isfinite(features)):
        raise ValueError(f"Non-finite native-MUA D4 features for {source}")
    return features


def fit_train_d4_stats(
    session_trial_sums: Mapping[str, np.ndarray],
    session_trial_lengths: Mapping[str, np.ndarray],
    session_obj_ids: Mapping[str, np.ndarray],
    session_names: Sequence[str],
    calibration_n_trials: int,
    all_support_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit D4 z-score statistics from *training-session* first-M=10 supports only."""
    if int(calibration_n_trials) != 10:
        raise ValueError("Native M1 D4 is frozen to chronological calibration_n_trials=10")
    if all_support_windows:
        raise ValueError("Native M1 D4 forbids random/all support windows; use chronological first-10 only")
    chunks: list[np.ndarray] = []
    for name in session_names:
        if name not in session_trial_sums or name not in session_trial_lengths or name not in session_obj_ids:
            raise KeyError(f"Missing training-session D4 inputs for {name}")
        sums = np.asarray(session_trial_sums[name])
        lengths = np.asarray(session_trial_lengths[name])
        labels = np.asarray(session_obj_ids[name])
        if sums.shape[0] < 10 or lengths.reshape(-1).size < 10 or labels.reshape(-1).size < 10:
            raise ValueError(f"Session {name} has fewer than ten D4 calibration trials")
        chunks.append(d4_from_trial_sums(sums[:10], lengths[:10], labels[:10], source=f"{name}[0:10]"))
    if not chunks:
        raise ValueError("No train calibration sessions were available for native D4 stats")
    values = np.concatenate(chunks, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std


def normalize_d4(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply pre-fit training-only D4 statistics with strict dimensions."""
    values = np.asarray(features, dtype=np.float32)
    location = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != D4_DIM:
        raise ValueError(f"D4 features must have shape [channels,{D4_DIM}], got {values.shape}")
    if location.shape != (D4_DIM,) or scale.shape != (D4_DIM,):
        raise ValueError(f"D4 normalization statistics must have shape ({D4_DIM},)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(location)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("D4 normalization values must be finite with strictly positive std")
    return ((values - location) / scale).astype(np.float32)


def deterministic_d4_row_permutation(num_channels: int, *, session_name: str, seed: int) -> np.ndarray:
    """Stable non-identity complete-row permutation for the DS4 attachment control."""
    if int(num_channels) < 2:
        raise ValueError("DS4 requires at least two native-MUA channels")
    digest = hashlib.sha256(f"native-mua-ds4-v1:{int(seed)}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(int(num_channels))
    if np.array_equal(permutation, np.arange(int(num_channels))):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)
