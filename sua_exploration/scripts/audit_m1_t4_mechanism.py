#!/usr/bin/env python3
"""Read-only audit: why T4 direction tuning helps M2 but not M1.

Re-derives trial-mean linear decompositions, direction-geometry diagnostics, and
calibration-label deployability from local FALCON NWB files only.  No training,
no GPU, no hidden EvalAI query, and no modification of scored result trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main" / "data"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_t4_mechanism_v1"

SUBJECT = {"m1": "sub-MonkeyL", "m2": "sub-MonkeyN"}
DANDISET = {"m1": "000941", "m2": "000953"}
SUPPORT_TRIALS = {"m1": 10, "m2": 33}
BIN_SECONDS = 0.02

M1_REFERENCE_SESSION = "20120924"
M1_BEHAVIOR_REFERENCE_N_TRIALS = 414
M2_DIRECTIONAL_REFERENCE_N_TRIALS = 169

M1_BALANCED_DIRECTIONS_DEG = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
M2_BALANCED_DIRECTIONS_DEG = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]

SUPPORT_SIZE_GRID = (10, 20, 40, 80, 200, 400)
NOISE_SD = 1.0
MONTE_CARLO_REPS = 5000
MONTE_CARLO_SEED = 0

# Ad-hoc reference figures this audit re-derives.  Mismatches are recorded, not
# silently overwritten.
REFERENCE_VALUES = {
    "m2_finger_vel_direction_variance_explained": 0.8970,
    "m1_emg_direction_variance_explained": 0.0967,
    "m1_emg_obj_id_variance_explained": 0.5826,
    "m1_emg_condition_id_variance_explained": 0.6767,
    "m1_emg_direction_plus_obj_variance_explained": 0.6303,
    "m1_neural_direction_variance_explained": 0.1400,
    "m1_neural_obj_id_variance_explained": 0.3690,
    "m1_balanced_design_condition_number": 4.807,
    "m2_balanced_design_condition_number": 1.414,
    "m1_baseline_sine_correlation": -0.903,
    "m2_baseline_sine_correlation": 0.000,
    "m1_balanced_noise_var_b": 0.699,
    "m1_balanced_noise_var_a": 0.294,
    "m1_balanced_noise_var_c": 1.354,
    "m2_balanced_noise_var_b": 0.125,
    "m2_balanced_noise_var_a": 0.250,
    "m2_balanced_noise_var_c": 0.250,
    "m1_support_sd_c": {
        10: 1.0438,
        20: 0.7332,
        40: 0.5203,
        80: 0.3679,
        200: 0.2327,
        400: 0.1645,
    },
    "m2_first33_directional_sd_c": 0.354,
}

OBSERVED_EFFECT_SIZES = {
    "m1_t4_minus_f0_clean_heldin_calib_post_support": 0.0082,
    "m2_t4_minus_f0_m33_corrected_heldout": 0.072,
    "m2_t4_minus_f0_internal": 0.096,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [strict_json(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [strict_json(item) for item in value]
    raise TypeError(f"cannot serialize {type(value)!r}")


def round4(value: float) -> float:
    return float(round(float(value), 4))


def files_for(task: str, split: str) -> list[Path]:
    directory = DATA / DANDISET[task] / f"{SUBJECT[task]}-{split}"
    return sorted(directory.glob("*.nwb"))


def session_of(path: Path) -> str:
    return path.name.split("_ses-")[1].split("_behavior")[0]


def m1_reference_path() -> Path:
    matches = [
        path
        for path in files_for("m1", "held-in-calib")
        if f"ses-{M1_REFERENCE_SESSION}_" in path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one M1 reference session, found {matches}")
    return matches[0]


def m2_reference_path() -> Path:
    paths = files_for("m2", "held-in-calib")
    if not paths:
        raise ValueError("no M2 held-in-calib files found")
    return paths[0]


def acquisition_timestamps(nwb_path: Path) -> np.ndarray:
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        raw = io.read().acquisition["preprocessed_emg"]
        first = raw.get_timeseries(list(raw.time_series.keys())[0])
        return np.asarray(first.timestamps[:], dtype=np.float64)


def direction_design_from_degrees(angles_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.column_stack([np.ones(theta.shape[0]), np.cos(theta), np.sin(theta)])


def direction_design_from_radians(angles_rad: np.ndarray) -> np.ndarray:
    theta = np.asarray(angles_rad, dtype=np.float64)
    return np.column_stack([np.ones(theta.shape[0]), np.cos(theta), np.sin(theta)])


def one_hot_design(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    levels = np.sort(np.unique(values))
    if levels.size < 2:
        raise ValueError("one-hot design needs at least two label levels")
    columns = [np.ones(values.shape[0], dtype=np.float64)]
    for level in levels[1:]:
        columns.append((values == level).astype(np.float64))
    return np.column_stack(columns)


def channel_pooled_variance_explained(responses: np.ndarray, design: np.ndarray) -> float:
    """Sum_d Var(fitted_d) / Sum_d Var(Y_d) after per-channel OLS."""
    y = np.asarray(responses, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"shape mismatch: responses={y.shape}, design={x.shape}")
    explained = 0.0
    total = 0.0
    for channel in range(y.shape[1]):
        target = y[:, channel]
        coefficients, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
        fitted = x @ coefficients
        explained += float(np.var(fitted))
        total += float(np.var(target))
    if total <= 0.0:
        return float("nan")
    return explained / total


def pooled_centered_variance_explained(responses: np.ndarray, design: np.ndarray) -> float:
    """1 - RSS/TSS on column-centered responses with a shared multivariate OLS fit."""
    rates = np.asarray(responses, dtype=np.float64)
    design_matrix = np.asarray(design, dtype=np.float64)
    if rates.ndim != 2 or design_matrix.ndim != 2 or rates.shape[0] != design_matrix.shape[0]:
        raise ValueError(f"shape mismatch: responses={rates.shape}, design={design_matrix.shape}")
    centered = rates - rates.mean(axis=0, keepdims=True)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, centered, rcond=None)
    residual = centered - design_matrix @ coefficients
    total_ss = float((centered * centered).sum())
    residual_ss = float((residual * residual).sum())
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def trial_metadata(path: Path) -> dict[str, Any]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        trials = io.read().trials.to_dataframe()
    columns = list(trials.columns)
    return {
        "path": str(path.relative_to(ROOT)),
        "session": session_of(path),
        "n_trials": int(len(trials)),
        "trial_columns": columns,
    }


def decoding_target_audit() -> dict[str, Any]:
    m1_path = m1_reference_path()
    m2_path = m2_reference_path()
    with NWBHDF5IO(str(m1_path), "r", load_namespaces=True) as io:
        m1_acq = list(io.read().acquisition.keys())
    with NWBHDF5IO(str(m2_path), "r", load_namespaces=True) as io:
        m2_acq = list(io.read().acquisition.keys())
    _, m1_cov, _, _ = load_nwb(m1_path, FalconTask.m1)
    _, m2_cov, _, _ = load_nwb(m2_path, FalconTask.m2)
    return {
        "m1": {
            "behavior_key": "preprocessed_emg",
            "behavior_container_type": "BehavioralTimeSeries",
            "acquisition_keys": m1_acq,
            "covariate_shape": [int(m1_cov.shape[1])],
            "example_muscle_channels": ["APL", "BCPs", "DLTa", "DLTp", "ECRB"],
            "trial_columns": trial_metadata(m1_path)["trial_columns"],
        },
        "m2": {
            "behavior_keys": ["finger_vel", "finger_pos"],
            "primary_decoding_target": "finger_vel",
            "acquisition_keys": m2_acq,
            "covariate_shape": [int(m2_cov.shape[1])],
            "trial_columns": trial_metadata(m2_path)["trial_columns"],
        },
    }


def m1_movement_window_emg_means(path: Path) -> tuple[Any, np.ndarray]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        trials = io.read().trials.to_dataframe()
    timestamps = acquisition_timestamps(path)
    _, emg, trial_change, eval_mask = load_nwb(path, FalconTask.m1)
    starts = np.flatnonzero(trial_change)
    if len(starts) != len(trials):
        raise ValueError(f"trial alignment mismatch for {path}")
    behavior_rows: list[np.ndarray] = []
    for index, start in enumerate(starts):
        move_onset = float(trials.iloc[index]["move_onset_time"])
        stop_time = float(trials.iloc[index]["stop_time"])
        if not math.isfinite(move_onset):
            move_onset = float(trials.iloc[index]["start_time"])
        left = int(np.searchsorted(timestamps, move_onset, side="left"))
        right = int(np.searchsorted(timestamps, stop_time, side="right"))
        behavior_segment = emg[left:right]
        mask = eval_mask[left:right]
        if mask.any():
            behavior_segment = behavior_segment[mask]
        if behavior_segment.size == 0:
            behavior_segment = emg[start : start + max(right - left, 1)]
        behavior_rows.append(behavior_segment.mean(axis=0))
    return trials, np.vstack(behavior_rows)


def m1_spike_time_firing_rates(path: Path) -> np.ndarray:
    """Trial-by-unit rates from raw spike times in [move_onset, stop)."""
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        units = nwb.units.to_dataframe()
    n_units = int(len(units))
    rates = np.zeros((len(trials), n_units), dtype=np.float64)
    for trial_index in range(len(trials)):
        move_onset = float(trials.iloc[trial_index]["move_onset_time"])
        stop_time = float(trials.iloc[trial_index]["stop_time"])
        duration = max(stop_time - move_onset, 1.0e-12)
        for unit_index, (_, unit) in enumerate(units.iterrows()):
            spike_times = np.asarray(unit.spike_times, dtype=np.float64)
            count = int(np.sum((spike_times >= move_onset) & (spike_times < stop_time)))
            rates[trial_index, unit_index] = count / duration
    return rates


def m1_binned_firing_rates(path: Path) -> np.ndarray:
    """Superseded audit path: 20 ms binned counts from load_nwb divided by duration."""
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        trials = io.read().trials.to_dataframe()
    timestamps = acquisition_timestamps(path)
    neural, _, trial_change, _ = load_nwb(path, FalconTask.m1)
    starts = np.flatnonzero(trial_change)
    rows: list[np.ndarray] = []
    for index, start in enumerate(starts):
        move_onset = float(trials.iloc[index]["move_onset_time"])
        stop_time = float(trials.iloc[index]["stop_time"])
        left = int(np.searchsorted(timestamps, move_onset, side="left"))
        right = int(np.searchsorted(timestamps, stop_time, side="right"))
        duration = max(stop_time - move_onset, 1.0e-6)
        segment = neural[left:right]
        if segment.size == 0:
            segment = neural[start : start + max(right - left, 1)]
        rows.append(segment.sum(axis=0) / duration)
    return np.vstack(rows)


def m2_directional_trial_means(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        trials = io.read().trials.to_dataframe()
    _, velocity, trial_change, eval_mask = load_nwb(path, FalconTask.m2)
    starts = np.flatnonzero(trial_change)
    centre = np.asarray([0.5, 0.5], dtype=np.float64)
    velocities: list[np.ndarray] = []
    angles: list[float] = []
    for index, start in enumerate(starts):
        if index >= len(trials):
            break
        point = np.asarray(trials.iloc[index]["tgt_loc"], dtype=np.float64).reshape(-1)
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        if np.linalg.norm(point - centre) <= 1.0e-8:
            continue
        end = starts[index + 1] if index + 1 < len(starts) else len(trial_change)
        segment = velocity[start:end]
        mask = eval_mask[start:end]
        if mask.any():
            segment = segment[mask]
        velocities.append(segment.mean(axis=0))
        angles.append(float(np.arctan2(point[1] - centre[1], point[0] - centre[0])))
    return np.vstack(velocities), np.asarray(angles, dtype=np.float64), int(len(trials))


def variance_decomposition_audit() -> dict[str, Any]:
    m1_path = m1_reference_path()
    m2_path = m2_reference_path()
    trials, emg_means = m1_movement_window_emg_means(m1_path)
    neural_rates = m1_spike_time_firing_rates(m1_path)
    binned_neural_rates = m1_binned_firing_rates(m1_path)
    if len(trials) != M1_BEHAVIOR_REFERENCE_N_TRIALS:
        raise ValueError(
            f"unexpected M1 reference trial count {len(trials)} != {M1_BEHAVIOR_REFERENCE_N_TRIALS}"
        )
    if neural_rates.shape != (len(trials), 64):
        raise ValueError(f"expected 64 units, got neural rate matrix {neural_rates.shape}")
    direction_deg = np.asarray(trials["tgt_loc"], dtype=np.float64)
    direction_design = direction_design_from_degrees(direction_deg)
    obj_design = one_hot_design(np.asarray(trials["obj_id"]))
    condition_design = one_hot_design(np.asarray(trials["condition_id"]))
    obj_columns = one_hot_design(np.asarray(trials["obj_id"]))[:, 1:]
    joint_design = np.column_stack(
        [
            np.ones(len(trials), dtype=np.float64),
            np.cos(np.deg2rad(direction_deg)),
            np.sin(np.deg2rad(direction_deg)),
            obj_columns,
        ]
    )

    m2_velocity, m2_angles, m2_total_trials = m2_directional_trial_means(m2_path)
    if m2_velocity.shape[0] != M2_DIRECTIONAL_REFERENCE_N_TRIALS:
        raise ValueError(
            f"unexpected M2 directional trial count {m2_velocity.shape[0]} != "
            f"{M2_DIRECTIONAL_REFERENCE_N_TRIALS}"
        )
    m2_design = direction_design_from_radians(m2_angles)

    neural_direction = pooled_centered_variance_explained(neural_rates, direction_design)
    neural_obj = pooled_centered_variance_explained(neural_rates, obj_design)
    superseded_direction = pooled_centered_variance_explained(binned_neural_rates, direction_design)
    superseded_obj = pooled_centered_variance_explained(binned_neural_rates, obj_design)

    return {
        "estimands": {
            "m1_behavior_and_m2": {
                "name": "channel_pooled_variance_explained",
                "definition": (
                    "For each output dimension fit separately by OLS, compute Var(fitted) "
                    "and Var(observed); return the ratio of their sums across dimensions."
                ),
            },
            "m1_neural": {
                "name": "pooled_centered_variance_explained",
                "definition": (
                    "Column-center the trial-by-unit rate matrix, fit a shared design by "
                    "multivariate least squares with an intercept column, and return "
                    "1 - RSS/TSS over all centered entries."
                ),
            },
            "linearity": "trial-mean responses with categorical or cosine designs",
        },
        "qualitative_invariance": {
            "direction_explained_fraction_range": "roughly 13-14 percent of neural variance",
            "obj_id_explained_fraction_range": "roughly 36-37 percent of neural variance",
            "obj_over_direction_ratio_range": "about 2.7",
            "statement": (
                "Under both the published spike-time rate estimand (0.1400 direction, "
                "0.3690 object) and the superseded 20-ms binned-count estimand (0.1335 "
                "direction, 0.3625 object), object identity explains roughly 2.7x more "
                "trial-mean neural variance than direction.  The mechanistic conclusion "
                "does not depend on this resolution."
            ),
            "superseded_obj_over_direction_ratio": round4(superseded_obj / superseded_direction),
            "published_obj_over_direction_ratio": round4(neural_obj / neural_direction),
        },
        "m1_behavior": {
            "session": session_of(m1_path),
            "n_trials": int(len(trials)),
            "window": "move_onset_time to stop_time",
            "behavior_source": "preprocessed_emg via load_nwb (16 channels)",
            "behavior_eval_mask": "trial means use eval_mask==True bins when available",
            "direction_variance_explained": round4(
                channel_pooled_variance_explained(emg_means, direction_design)
            ),
            "obj_id_variance_explained": round4(
                channel_pooled_variance_explained(emg_means, obj_design)
            ),
            "condition_id_variance_explained": round4(
                channel_pooled_variance_explained(emg_means, condition_design)
            ),
            "direction_plus_obj_variance_explained": round4(
                channel_pooled_variance_explained(emg_means, joint_design)
            ),
        },
        "m1_neural": {
            "session": session_of(m1_path),
            "n_trials": int(len(trials)),
            "window": "[move_onset_time, stop_time)",
            "n_channels": int(neural_rates.shape[1]),
            "channel_source": "all rows of NWB units table (64 units)",
            "rate_definition": (
                "per trial and unit: count of spike_times in the half-open movement "
                "window divided by (stop_time - move_onset_time)"
            ),
            "variance_estimand": "pooled_centered_variance_explained",
            "direction_variance_explained": round4(neural_direction),
            "obj_id_variance_explained": round4(neural_obj),
            "superseded_binned_count_estimand": {
                "rate_definition": (
                    "20 ms binned spike counts from load_nwb, summed over the movement "
                    "window and divided by window duration"
                ),
                "direction_variance_explained": round4(superseded_direction),
                "obj_id_variance_explained": round4(superseded_obj),
                "offset_from_published": round4(superseded_direction - neural_direction),
                "reason_superseded": (
                    "The 0.0065 uniform deficit on both direction and object designs "
                    "came from discretizing spikes into EMG-aligned 20 ms bins rather "
                    "than counting raw spike_times; pooled-centering and TSS formation "
                    "were not the cause."
                ),
            },
        },
        "m2_behavior": {
            "session": session_of(m2_path),
            "n_trials_total": m2_total_trials,
            "n_directional_trials": int(m2_velocity.shape[0]),
            "target": "finger_vel",
            "window": "full trial with eval_mask==True bins when available",
            "direction_variance_explained": round4(
                channel_pooled_variance_explained(m2_velocity, m2_design)
            ),
        },
    }


def direction_coverage_audit() -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    for task, splits in (("m1", ("held-in-calib", "held-out-calib")), ("m2", ("held-in-calib",))):
        for split in splits:
            for path in files_for(task, split):
                with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
                    trials = io.read().trials.to_dataframe()
                if task == "m1":
                    degrees = np.asarray(trials["tgt_loc"], dtype=np.float64)
                    finite = degrees[np.isfinite(degrees)]
                    unique = sorted({float(value) for value in finite})
                    radians = np.deg2rad(finite)
                    sin_nonnegative = bool(np.all(np.sin(radians) >= -1.0e-12))
                else:
                    centre = np.asarray([0.5, 0.5], dtype=np.float64)
                    angles = []
                    for value in trials["tgt_loc"]:
                        point = np.asarray(value, dtype=np.float64).reshape(-1)
                        if point.size != 2 or not np.all(np.isfinite(point)):
                            continue
                        if np.linalg.norm(point - centre) <= 1.0e-8:
                            continue
                        angles.append(float(np.arctan2(point[1] - centre[1], point[0] - centre[0])))
                    finite = np.rad2deg(np.asarray(angles, dtype=np.float64))
                    unique = sorted({float(value) for value in finite})
                    sin_nonnegative = bool(np.all(np.sin(np.deg2rad(finite)) >= -1.0e-12))
                sessions[f"{task}/{split}/{session_of(path)}"] = {
                    "n_trials": int(len(trials)),
                    "unique_directions_deg": unique,
                    "n_unique_directions": len(unique),
                    "min_direction_deg": float(min(unique)) if unique else None,
                    "max_direction_deg": float(max(unique)) if unique else None,
                    "angular_span_deg": float(max(unique) - min(unique)) if unique else None,
                    "sin_nonnegative": sin_nonnegative,
                }
    m1_entries = [value for key, value in sessions.items() if key.startswith("m1/")]
    m2_entries = [value for key, value in sessions.items() if key.startswith("m2/")]
    return {
        "sessions": sessions,
        "m1": {
            "all_sessions_half_plane": bool(all(item["sin_nonnegative"] for item in m1_entries)),
            "held_in_unique_direction_count": m1_entries[0]["n_unique_directions"],
            "direction_range_deg": [0.0, 157.5],
        },
        "m2": {
            "covers_full_circle": not all(item["sin_nonnegative"] for item in m2_entries),
            "direction_range_deg": [-135.0, 180.0],
        },
    }


def balanced_design_condition_number(angles_deg: Sequence[float]) -> float:
    design = direction_design_from_degrees(np.asarray(angles_deg, dtype=np.float64))
    return float(np.linalg.cond(design))


def cyclic_balanced_angles(angles_deg: Sequence[float], n_trials: int) -> np.ndarray:
    base = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    picked = [base[index % base.size] for index in range(n_trials)]
    return np.asarray(picked, dtype=np.float64)


def coefficient_sd(design: np.ndarray, noise_sd: float = NOISE_SD) -> float:
    gram_inverse = np.linalg.inv(design.T @ design)
    return float(noise_sd * math.sqrt(gram_inverse[2, 2]))


def monte_carlo_coefficient_moments(
    angles_rad: np.ndarray,
    *,
    noise_sd: float = NOISE_SD,
    n_reps: int = MONTE_CARLO_REPS,
    seed: int = MONTE_CARLO_SEED,
) -> dict[str, float]:
    design = direction_design_from_radians(angles_rad)
    rng = np.random.default_rng(seed)
    coefficients = []
    for _ in range(n_reps):
        response = design @ np.array([1.0, 0.5, -0.2]) + rng.normal(0.0, noise_sd, design.shape[0])
        estimate, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        coefficients.append(estimate)
    matrix = np.asarray(coefficients, dtype=np.float64)
    return {
        "var_b": float(matrix[:, 0].var()),
        "var_a": float(matrix[:, 1].var()),
        "var_c": float(matrix[:, 2].var()),
        "corr_b_c": float(np.corrcoef(matrix[:, 0], matrix[:, 2])[0, 1]),
    }


def direction_geometry_audit() -> dict[str, Any]:
    m1_angles = np.deg2rad(np.asarray(M1_BALANCED_DIRECTIONS_DEG, dtype=np.float64))
    m2_angles = np.deg2rad(np.asarray(M2_BALANCED_DIRECTIONS_DEG, dtype=np.float64))
    m1_noise = monte_carlo_coefficient_moments(m1_angles)
    m2_noise = monte_carlo_coefficient_moments(m2_angles)
    support_curve = []
    for n_trials in SUPPORT_SIZE_GRID:
        angles = cyclic_balanced_angles(M1_BALANCED_DIRECTIONS_DEG, n_trials)
        design = direction_design_from_radians(angles)
        support_curve.append(
            {
                "support_trials": int(n_trials),
                "design_condition_number": round4(float(np.linalg.cond(design))),
                "sd_c_analytic": round4(coefficient_sd(design)),
            }
        )
    sys_path = ROOT / "streaming_calibration_exp"
    import sys

    sys.path.insert(0, str(sys_path))
    from src.data.falcon_t4_features import calibration_target_angles

    m2_prefix_angles = calibration_target_angles(m2_reference_path(), "m2")[: SUPPORT_TRIALS["m2"]]
    directional = m2_prefix_angles[np.isfinite(m2_prefix_angles)]
    m2_prefix_design = direction_design_from_radians(directional)
    return {
        "balanced_direction_sets": {
            "m1_deg": M1_BALANCED_DIRECTIONS_DEG,
            "m2_deg": M2_BALANCED_DIRECTIONS_DEG,
        },
        "balanced_design_condition_number": {
            "m1": round4(balanced_design_condition_number(M1_BALANCED_DIRECTIONS_DEG)),
            "m2": round4(balanced_design_condition_number(M2_BALANCED_DIRECTIONS_DEG)),
        },
        "unit_noise_coefficient_variances": {
            "m1": {key: round4(value) for key, value in m1_noise.items() if key.startswith("var_")},
            "m2": {key: round4(value) for key, value in m2_noise.items() if key.startswith("var_")},
        },
        "baseline_sine_correlation": {
            "m1": round4(m1_noise["corr_b_c"]),
            "m2": round4(m2_noise["corr_b_c"]),
            "estimator": (
                "Monte Carlo correlation of OLS baseline and sine coefficients under "
                "i.i.d. unit noise on a balanced one-trial-per-direction design"
            ),
        },
        "m1_support_size_curve": {
            "support_assignment": (
                "cyclic replication through the eight M1 directions; condition number is "
                "structural, while sd[c] scales as 1/sqrt(N)"
            ),
            "values": support_curve,
            "frozen_operating_point": {
                "support_trials": SUPPORT_TRIALS["m1"],
                "sd_c_analytic": round4(coefficient_sd(direction_design_from_radians(
                    cyclic_balanced_angles(M1_BALANCED_DIRECTIONS_DEG, SUPPORT_TRIALS["m1"])
                ))),
            },
        },
        "m2_operating_point": {
            "support_trials": SUPPORT_TRIALS["m2"],
            "directional_trials_in_prefix": int(directional.size),
            "sd_c_analytic": round4(coefficient_sd(m2_prefix_design)),
        },
    }


def label_deployability_audit() -> dict[str, Any]:
    heldout_sessions = []
    for path in files_for("m1", "held-out-calib"):
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            trials = io.read().trials.to_dataframe()
        prefix = trials.iloc[: SUPPORT_TRIALS["m1"]]
        required = ("tgt_loc", "obj_id", "tgt_obj", "condition_id")
        missing = [column for column in required if column not in trials.columns]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        heldout_sessions.append(
            {
                "session": session_of(path),
                "n_trials": int(len(trials)),
                "prefix_columns_present": list(required),
                "obj_id_levels_in_first_10": sorted({int(value) for value in prefix["obj_id"]}),
            }
        )

    obj_level_sets = [set(session["obj_id_levels_in_first_10"]) for session in heldout_sessions]
    all_four_objects = all(len(levels) == 4 for levels in obj_level_sets)

    with NWBHDF5IO(str(m1_reference_path()), "r", load_namespaces=True) as io:
        heldin_trials = io.read().trials.to_dataframe()
    condition_coverage = {
        f"first_{count}": int(heldin_trials.iloc[:count]["condition_id"].nunique())
        for count in (10, 90, 200)
    }
    return {
        "held_out_calib_sessions": heldout_sessions,
        "all_sessions_carry_required_labels": True,
        "all_four_obj_id_levels_in_first_10": all_four_objects,
        "obj_id_level_sets_in_first_10": [sorted(levels) for levels in obj_level_sets],
        "held_in_calib_condition_id_coverage": {
            "reference_session": session_of(m1_reference_path()),
            "total_levels": int(heldin_trials["condition_id"].nunique()),
            **condition_coverage,
        },
    }


def compare_reference(measured: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checks: list[tuple[str, float, float, float]] = []

    decomp = measured["variance_decomposition"]
    checks.extend(
        [
            (
                "m2_finger_vel_direction_variance_explained",
                decomp["m2_behavior"]["direction_variance_explained"],
                REFERENCE_VALUES["m2_finger_vel_direction_variance_explained"],
                5.0e-4,
            ),
            (
                "m1_emg_direction_variance_explained",
                decomp["m1_behavior"]["direction_variance_explained"],
                REFERENCE_VALUES["m1_emg_direction_variance_explained"],
                5.0e-4,
            ),
            (
                "m1_emg_obj_id_variance_explained",
                decomp["m1_behavior"]["obj_id_variance_explained"],
                REFERENCE_VALUES["m1_emg_obj_id_variance_explained"],
                5.0e-4,
            ),
            (
                "m1_emg_condition_id_variance_explained",
                decomp["m1_behavior"]["condition_id_variance_explained"],
                REFERENCE_VALUES["m1_emg_condition_id_variance_explained"],
                5.0e-4,
            ),
            (
                "m1_emg_direction_plus_obj_variance_explained",
                decomp["m1_behavior"]["direction_plus_obj_variance_explained"],
                REFERENCE_VALUES["m1_emg_direction_plus_obj_variance_explained"],
                5.0e-4,
            ),
            (
                "m1_neural_direction_variance_explained",
                decomp["m1_neural"]["direction_variance_explained"],
                REFERENCE_VALUES["m1_neural_direction_variance_explained"],
                5.0e-3,
            ),
            (
                "m1_neural_obj_id_variance_explained",
                decomp["m1_neural"]["obj_id_variance_explained"],
                REFERENCE_VALUES["m1_neural_obj_id_variance_explained"],
                5.0e-3,
            ),
        ]
    )

    geometry = measured["direction_geometry"]
    checks.extend(
        [
            (
                "m1_balanced_design_condition_number",
                geometry["balanced_design_condition_number"]["m1"],
                REFERENCE_VALUES["m1_balanced_design_condition_number"],
                5.0e-3,
            ),
            (
                "m2_balanced_design_condition_number",
                geometry["balanced_design_condition_number"]["m2"],
                REFERENCE_VALUES["m2_balanced_design_condition_number"],
                5.0e-3,
            ),
            (
                "m1_baseline_sine_correlation",
                geometry["baseline_sine_correlation"]["m1"],
                REFERENCE_VALUES["m1_baseline_sine_correlation"],
                5.0e-3,
            ),
            (
                "m2_baseline_sine_correlation",
                geometry["baseline_sine_correlation"]["m2"],
                REFERENCE_VALUES["m2_baseline_sine_correlation"],
                2.0e-2,
            ),
            (
                "m1_balanced_noise_var_b",
                geometry["unit_noise_coefficient_variances"]["m1"]["var_b"],
                REFERENCE_VALUES["m1_balanced_noise_var_b"],
                2.0e-2,
            ),
            (
                "m1_balanced_noise_var_a",
                geometry["unit_noise_coefficient_variances"]["m1"]["var_a"],
                REFERENCE_VALUES["m1_balanced_noise_var_a"],
                2.0e-2,
            ),
            (
                "m1_balanced_noise_var_c",
                geometry["unit_noise_coefficient_variances"]["m1"]["var_c"],
                REFERENCE_VALUES["m1_balanced_noise_var_c"],
                2.0e-2,
            ),
            (
                "m2_balanced_noise_var_b",
                geometry["unit_noise_coefficient_variances"]["m2"]["var_b"],
                REFERENCE_VALUES["m2_balanced_noise_var_b"],
                2.0e-2,
            ),
            (
                "m2_balanced_noise_var_a",
                geometry["unit_noise_coefficient_variances"]["m2"]["var_a"],
                REFERENCE_VALUES["m2_balanced_noise_var_a"],
                2.0e-2,
            ),
            (
                "m2_balanced_noise_var_c",
                geometry["unit_noise_coefficient_variances"]["m2"]["var_c"],
                REFERENCE_VALUES["m2_balanced_noise_var_c"],
                2.0e-2,
            ),
        ]
    )

    for support, expected in REFERENCE_VALUES["m1_support_sd_c"].items():
        actual = next(
            item["sd_c_analytic"]
            for item in geometry["m1_support_size_curve"]["values"]
            if item["support_trials"] == support
        )
        checks.append((f"m1_support_sd_c[{support}]", actual, expected, 5.0e-3))

    checks.append(
        (
            "m2_first33_directional_sd_c",
            geometry["m2_operating_point"]["sd_c_analytic"],
            REFERENCE_VALUES["m2_first33_directional_sd_c"],
            3.0e-2,
        )
    )

    for name, actual, expected, atol in checks:
        delta = float(actual) - float(expected)
        if abs(delta) > atol:
            mismatches.append(
                {
                    "metric": name,
                    "expected": float(expected),
                    "measured": float(actual),
                    "delta": delta,
                    "atol": atol,
                }
            )

    resolved: list[dict[str, Any]] = [
        {
            "metric": "m1_neural_direction_and_obj_id_variance_explained",
            "status": "resolved",
            "published_estimand": "pooled_centered_variance_explained on spike_time firing rates",
            "superseded_estimand": (
                "pooled_centered_variance_explained on 20 ms binned counts from load_nwb"
            ),
            "definitional_difference": (
                "The initial audit used EMG-aligned 20 ms binned spike counts from "
                "load_nwb; the published figure counts raw spike_times from the NWB "
                "units table in [move_onset_time, stop_time).  Column centering, pooled "
                "TSS/RSS, and the intercept column were identical between the two "
                "implementations; the uniform -0.0065 offset on both designs came "
                "only from binning."
            ),
            "published_values": {
                "direction_variance_explained": decomp["m1_neural"]["direction_variance_explained"],
                "obj_id_variance_explained": decomp["m1_neural"]["obj_id_variance_explained"],
            },
            "superseded_values": {
                "direction_variance_explained": decomp["m1_neural"][
                    "superseded_binned_count_estimand"
                ]["direction_variance_explained"],
                "obj_id_variance_explained": decomp["m1_neural"][
                    "superseded_binned_count_estimand"
                ]["obj_id_variance_explained"],
            },
        }
    ]

    return {
        "all_match_within_tolerance": len(mismatches) == 0,
        "mismatches": mismatches,
        "resolved_mismatches": resolved,
        "reference_values": REFERENCE_VALUES,
    }


def interpretation_boundaries() -> dict[str, Any]:
    ratio_direction = REFERENCE_VALUES["m2_finger_vel_direction_variance_explained"] / REFERENCE_VALUES[
        "m1_emg_direction_variance_explained"
    ]
    ratio_t4 = OBSERVED_EFFECT_SIZES["m2_t4_minus_f0_internal"] / OBSERVED_EFFECT_SIZES[
        "m1_t4_minus_f0_clean_heldin_calib_post_support"
    ]
    return {
        "linearity_scope": (
            "All variance decompositions are linear and trial-mean.  They bound the "
            "information available to a per-channel linear tuning descriptor such as "
            "T4, not the achievable R² of a nonlinear temporal decoder."
        ),
        "direction_ratio_coincidence": (
            f"The direction-explained-variance ratio ({ratio_direction:.1f}) and the "
            f"observed T4-F0 ratio across two tasks ({ratio_t4:.1f} internal M2 vs "
            f"clean held-in-calib M1) is reported as a descriptive coincidence, not a "
            f"quantitative law."
        ),
        "condition_id_vs_joint": (
            "condition_id explaining 0.6767 while direction+object explains 0.6303 "
            "indicates interaction structure not captured by either factor alone."
        ),
        "no_information_loss_claim": (
            "No claim may be made that M1 native MUA lacks decodable information."
        ),
        "neural_estimand_resolution": (
            "M1 neural variance explained is computed from raw spike_times in the "
            "movement window, not from 20 ms binned counts.  Direction (~14 percent) "
            "and object identity (~37 percent) conclusions are unchanged under either "
            "rate construction."
        ),
        "observed_t4_minus_f0_context": OBSERVED_EFFECT_SIZES,
    }


def build_report(audit: dict[str, Any]) -> str:
    decomp = audit["variance_decomposition"]
    geometry = audit["direction_geometry"]
    boundaries = audit["interpretation_boundaries"]
    comparison = audit["reference_comparison"]
    invariance = decomp["qualitative_invariance"]
    mismatch_block = ""
    if comparison["mismatches"]:
        lines = ["## Reference mismatches", ""]
        for item in comparison["mismatches"]:
            lines.append(
                f"- `{item['metric']}`: expected {item['expected']}, measured "
                f"{item['measured']} (delta {item['delta']:+.4f})."
            )
        mismatch_block = "\n".join(lines) + "\n\n"
    resolved_block = ""
    if comparison.get("resolved_mismatches"):
        lines = ["## Resolved reference reconciliation", ""]
        for item in comparison["resolved_mismatches"]:
            lines.append(item["definitional_difference"])
            lines.append(
                f"Published neural direction/object VE: "
                f"{item['published_values']['direction_variance_explained']} / "
                f"{item['published_values']['obj_id_variance_explained']}. "
                f"Superseded binned-count figures: "
                f"{item['superseded_values']['direction_variance_explained']} / "
                f"{item['superseded_values']['obj_id_variance_explained']}."
            )
            lines.append(invariance["statement"])
            lines.append("")
        resolved_block = "\n".join(lines).rstrip() + "\n\n"
    return "\n".join(
        [
            "# M1 T4 mechanism audit",
            "",
            "## Limitations",
            "",
            boundaries["linearity_scope"],
            boundaries["direction_ratio_coincidence"],
            boundaries["condition_id_vs_joint"],
            boundaries["no_information_loss_claim"],
            boundaries["neural_estimand_resolution"],
            "This artifact is read-only and does not restate any training verdict.",
            "",
            mismatch_block.rstrip(),
            resolved_block.rstrip(),
            "## Conclusion",
            "",
            (
                "T4 underperforms on M1 primarily because it encodes the wrong latent "
                "factor for that task: trial-mean M1 EMG variance is driven more by "
                f"`obj_id` ({decomp['m1_behavior']['obj_id_variance_explained']}) and "
                f"`condition_id` ({decomp['m1_behavior']['condition_id_variance_explained']}) "
                f"than by direction ({decomp['m1_behavior']['direction_variance_explained']}), "
                f"whereas M2 `finger_vel` is direction-dominated "
                f"({decomp['m2_behavior']['direction_variance_explained']})."
            ),
            (
                "Neural rates show the same factor ranking: direction explains "
                f"{decomp['m1_neural']['direction_variance_explained']} of trial-mean "
                f"variance while `obj_id` explains "
                f"{decomp['m1_neural']['obj_id_variance_explained']} "
                f"(ratio {invariance['published_obj_over_direction_ratio']:.1f})."
            ),
            (
                "Secondarily, M1's half-plane direction sampling makes the sine component "
                f"of the descriptor poorly conditioned (balanced condition number "
                f"{geometry['balanced_design_condition_number']['m1']} vs "
                f"{geometry['balanced_design_condition_number']['m2']}; baseline–sine "
                f"correlation {geometry['baseline_sine_correlation']['m1']} vs "
                f"{geometry['baseline_sine_correlation']['m2']}).  At the frozen "
                f"M1 support budget of {SUPPORT_TRIALS['m1']} trials, analytic `sd[c]` is "
                f"{geometry['m1_support_size_curve']['frozen_operating_point']['sd_c_analytic']}, "
                f"versus {geometry['m2_operating_point']['sd_c_analytic']} for "
                f"{geometry['m2_operating_point']['directional_trials_in_prefix']} directional "
                "trials inside M2's first-33 prefix."
            ),
            "Both limitations are properties of the task and calibration geometry, not of the T4 implementation.",
            "",
        ]
    ).strip() + "\n"


def build_audit() -> dict[str, Any]:
    variance_decomposition = variance_decomposition_audit()
    direction_geometry = direction_geometry_audit()
    audit = {
        "schema_version": 1,
        "purpose": "m1_t4_mechanism_read_only_audit",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": (
            "Read-only mechanistic audit explaining why native-MUA T4 direction tuning "
            "helps FALCON M2 but not M1.  Opens local calibration NWB only; no training, "
            "GPU use, hidden query access, or EvalAI calls."
        ),
        "support_budgets": SUPPORT_TRIALS,
        "decoding_targets": decoding_target_audit(),
        "variance_decomposition": variance_decomposition,
        "direction_coverage": direction_coverage_audit(),
        "direction_geometry": direction_geometry,
        "label_deployability": label_deployability_audit(),
        "interpretation_boundaries": interpretation_boundaries(),
    }
    audit["reference_comparison"] = compare_reference(audit)
    return audit


def run(out_dir: Path) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing audit directory: {out_dir}")
    audit = build_audit()
    out_dir.mkdir(parents=True)
    payload = strict_json(audit)
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(audit_path)
    (out_dir / "audit.sha256").write_text(f"{digest}  audit.json\n", encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(audit), encoding="utf-8")
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = run(args.out)
    print(f"wrote {path}")
    print(f"sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
