"""Population-vector applicability audit and M2 scoring skeleton.

Part A audits native discrete direction metadata before any fitting.  Part B
implements the M2 M24/q24 arm by reusing the sealed subject-M PV estimator and
the Native-M2 chronological split contract.  Part C records an evidenced H1
inapplicability statement when no native direction field exists.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import native_m2_m24_ridge_w50 as m2_ridge
from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import (
    BIN_SIZE_S,
    HISTORY_BINS,
    fit_population_vector_gain,
    population_vectors,
    predict_population_vector,
    preferred_directions_from_cosine,
    prefix_sums,
    window_rates_from_prefix,
)


M2_DATASET = "m2"
H1_DATASET = "h1"
M2_CALIBRATION_TRIALS = m2_ridge.CALIBRATION_TRIALS
M2_WINDOW_BINS = m2_ridge.WINDOW_BINS
M2_OUTPUT_DIM = m2_ridge.OUTPUT_DIM
M2_CHANNELS = m2_ridge.CHANNELS
M2_CENTRE = np.asarray([0.5, 0.5], dtype=np.float64)
M2_DIRECTION_FIELD_PATH = "trials.tgt_loc"
M2_DERIVED_ANGLE_PATH = "arctan2(tgt_loc[1]-0.5, tgt_loc[0]-0.5); centre/rest at (0.5,0.5) excluded"
H1_VELOCITY_FIELD_PATH = "acquisition.OpenLoopKinematicsVelocity"
H1_OUTPUT_DIM = 7
MIN_UNIQUE_DIRECTIONS = 2
SEED = 42


class PopulationVectorComparatorError(RuntimeError):
    """Raised when a population-vector comparator contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PopulationVectorComparatorError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def digest_query_target_bins(values: np.ndarray) -> str:
    return m2_ridge.sha256_array_int64(np.ascontiguousarray(values, dtype=np.int64))


@dataclass(frozen=True)
class SessionDirectionAudit:
    session_name: str
    native_field_exists: bool
    native_field_path: str | None
    derived_field_path: str | None
    trial_count: int
    finite_direction_count: int
    unique_direction_count: int
    unique_direction_values: tuple[float, ...]
    direction_value_distribution: dict[str, int]
    calibration_trial_count: int
    calibration_finite_direction_count: int
    calibration_unique_direction_count: int
    calibration_unique_direction_values: tuple[float, ...]
    calibration_direction_distribution: dict[str, int]
    pv_definable_on_native_field: bool
    projection_assessment: str | None = None


@dataclass(frozen=True)
class M2PopulationVectorResult:
    session_name: str
    r2_variance_weighted: float
    query_identity_hash: str
    query_identity_bound: bool
    calibration_rows: int
    query_rows: int
    target_direction_labels_used: int
    supervision_coordinates_consumed: int


def _distribution(values: np.ndarray) -> dict[str, int]:
    rounded = [float(value) for value in np.round(np.asarray(values, dtype=np.float64), decimals=6)]
    return {str(key): int(count) for key, count in sorted(Counter(rounded).items())}


def m2_target_angles_from_nwb(path: Path) -> np.ndarray:
    """Return one derived target angle per trial; NaN for centre/rest or invalid rows."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        require(nwb.trials is not None, f"{path}: trials table missing")
        require(M2_DIRECTION_FIELD_PATH.split(".")[-1] in nwb.trials.colnames, f"{path}: {M2_DIRECTION_FIELD_PATH} missing")
        tgt_loc = nwb.trials["tgt_loc"][:]
    angles = np.full(len(tgt_loc), np.nan, dtype=np.float64)
    for index, value in enumerate(tgt_loc):
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        delta = point - M2_CENTRE
        if np.linalg.norm(delta) <= 1.0e-8:
            continue
        angles[index] = np.arctan2(delta[1], delta[0])
    return angles


def audit_m2_native_direction(
    path: Path,
    *,
    calibration_trials: int = M2_CALIBRATION_TRIALS,
) -> SessionDirectionAudit:
    angles = m2_target_angles_from_nwb(path)
    finite = angles[np.isfinite(angles)]
    unique = np.unique(np.round(finite, decimals=12))
    calib = angles[:calibration_trials]
    calib_finite = calib[np.isfinite(calib)]
    calib_unique = np.unique(np.round(calib_finite, decimals=12))
    session_name = path.name.split("_ses-")[1].split("_behavior")[0]
    return SessionDirectionAudit(
        session_name=session_name,
        native_field_exists=True,
        native_field_path=M2_DIRECTION_FIELD_PATH,
        derived_field_path=M2_DERIVED_ANGLE_PATH,
        trial_count=int(angles.size),
        finite_direction_count=int(finite.size),
        unique_direction_count=int(unique.size),
        unique_direction_values=tuple(float(value) for value in unique.tolist()),
        direction_value_distribution=_distribution(finite),
        calibration_trial_count=int(calibration_trials),
        calibration_finite_direction_count=int(calib_finite.size),
        calibration_unique_direction_count=int(calib_unique.size),
        calibration_unique_direction_values=tuple(float(value) for value in calib_unique.tolist()),
        calibration_direction_distribution=_distribution(calib_finite),
        pv_definable_on_native_field=int(calib_unique.size) >= MIN_UNIQUE_DIRECTIONS,
    )


def audit_h1_native_direction(path: Path) -> SessionDirectionAudit:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials
        velocity_present = H1_VELOCITY_FIELD_PATH.split(".")[-1] in nwb.acquisition
        session_name = path.name.split("_ses-")[1].split(".nwb")[0]
        if trials is None:
            projection = (
                "No native discrete direction field is present. H1 exposes continuous "
                f"{H1_VELOCITY_FIELD_PATH} with output dimension {H1_OUTPUT_DIM}. A 2-D "
                "directional projection such as arctan2(vy, vx) would discard five task "
                "coordinates, is not task-native, and is not defensible as a population-vector "
                "direction label without manufacturing an estimand."
            )
            return SessionDirectionAudit(
                session_name=session_name,
                native_field_exists=False,
                native_field_path=None,
                derived_field_path=None,
                trial_count=0,
                finite_direction_count=0,
                unique_direction_count=0,
                unique_direction_values=(),
                direction_value_distribution={},
                calibration_trial_count=0,
                calibration_finite_direction_count=0,
                calibration_unique_direction_count=0,
                calibration_unique_direction_values=(),
                calibration_direction_distribution={},
                pv_definable_on_native_field=False,
                projection_assessment=projection,
            )
        direction_columns = [
            column
            for column in trials.colnames
            if any(token in column.lower() for token in ("dir", "theta", "target", "direction"))
        ]
        projection = (
            f"Trials table exists but exposes no native discrete direction column "
            f"({direction_columns or 'none matched'}). Continuous {H1_VELOCITY_FIELD_PATH} "
            f"remains {H1_OUTPUT_DIM}-D; no defensible 2-D directional projection was found."
        )
        return SessionDirectionAudit(
            session_name=session_name,
            native_field_exists=False,
            native_field_path=None,
            derived_field_path=None,
            trial_count=int(len(trials)),
            finite_direction_count=0,
            unique_direction_count=0,
            unique_direction_values=(),
            direction_value_distribution={},
            calibration_trial_count=0,
            calibration_finite_direction_count=0,
            calibration_unique_direction_count=0,
            calibration_unique_direction_values=(),
            calibration_direction_distribution={},
            pv_definable_on_native_field=False,
            projection_assessment=projection,
        )


def applicability_verdict(
    dataset: str,
    audits: Sequence[SessionDirectionAudit],
) -> dict[str, Any]:
    per_session = {
        audit.session_name: {
            "native_field_exists": audit.native_field_exists,
            "native_field_path": audit.native_field_path,
            "derived_field_path": audit.derived_field_path,
            "trial_count": audit.trial_count,
            "finite_direction_count": audit.finite_direction_count,
            "unique_direction_count": audit.unique_direction_count,
            "unique_direction_values": list(audit.unique_direction_values),
            "direction_value_distribution": dict(audit.direction_value_distribution),
            "calibration_trial_count": audit.calibration_trial_count,
            "calibration_finite_direction_count": audit.calibration_finite_direction_count,
            "calibration_unique_direction_count": audit.calibration_unique_direction_count,
            "calibration_unique_direction_values": list(audit.calibration_unique_direction_values),
            "calibration_direction_distribution": dict(audit.calibration_direction_distribution),
            "pv_definable_on_native_field": audit.pv_definable_on_native_field,
            "projection_assessment": audit.projection_assessment,
        }
        for audit in audits
    }
    definable = [audit.pv_definable_on_native_field for audit in audits]
    if dataset == M2_DATASET:
        if all(definable):
            verdict = "PV_DEFINABLE"
        elif any(definable):
            verdict = "PV_PARTIALLY_DEFINABLE_ON_NATIVE_FIELD"
        else:
            verdict = "PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE"
    else:
        if any(definable):
            verdict = "PV_DEFINABLE"
        else:
            verdict = "PV_INAPPLICABLE_NO_NATIVE_DISCRETE_DIRECTION_FIELD"
    return {
        "dataset": dataset,
        "verdict": verdict,
        "sessions_with_nondegenerate_native_field": int(sum(definable)),
        "sessions_total": int(len(audits)),
        "per_session": per_session,
    }


def discover_m2_sessions(data_dir: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {session: [] for session in m2_ridge.EXPECTED_HELDOUT_SESSIONS}
    for path in sorted(data_dir.rglob("*held-out-calib*.nwb")):
        pieces = path.name.split("_")
        if len(pieces) < 2:
            continue
        session = pieces[1].split(".")[0]
        if session in candidates:
            candidates[session].append(path.resolve())
    result: dict[str, Path] = {}
    for session, paths in candidates.items():
        require(len(paths) == 1, f"{session}: expected exactly one held-out-calib NWB, found {paths}")
        result[session] = paths[0]
    return result


def discover_h1_sessions(data_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(data_dir.rglob("*held-out-calib*.nwb")):
        session = path.name.split("_ses-")[1].split(".nwb")[0]
        require(session not in indexed, f"duplicate H1 session path for {session}")
        indexed[session] = path.resolve()
    require(indexed, f"no H1 held-out-calib NWB files under {data_dir}")
    return indexed


def audit_dataset_applicability(dataset: str, data_dir: Path) -> dict[str, Any]:
    if dataset == M2_DATASET:
        sessions = discover_m2_sessions(data_dir)
        audits = [audit_m2_native_direction(path) for path in sessions.values()]
    elif dataset == H1_DATASET:
        sessions = discover_h1_sessions(data_dir)
        audits = [audit_h1_native_direction(path) for path in sessions.values()]
    else:
        raise PopulationVectorComparatorError(f"unsupported dataset {dataset!r}")
    verdict = applicability_verdict(dataset, audits)
    verdict["input_paths"] = {name: str(path) for name, path in sorted(sessions.items())}
    return verdict


def _trial_bounds_from_change(trial_change: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.asarray(trial_change, dtype=bool))
    bounds: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        stop = int(starts[index + 1]) if index + 1 < starts.size else int(trial_change.shape[0])
        bounds.append((int(start), stop))
    return bounds


def fit_cosine_tuning_hand(
    thetas: np.ndarray,
    mean_rates: np.ndarray,
) -> tuple[float, float, float, float]:
    design = np.stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, mean_rates, rcond=None)
    b, a, c = (float(value) for value in coefficients)
    return a, c, float(math.hypot(a, c)), b


def build_pv_readout_from_trial_data(
    *,
    neural: np.ndarray,
    behavior: np.ndarray,
    support_trial_bounds: Sequence[tuple[int, int]],
    support_direction_angles: Sequence[float],
    support_window_target_bins: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Reuse the sealed subject-M PV estimator on explicit calibration inputs."""
    from sua_exploration.mc_maze.unit_side_features import _nearest_canonical_direction_index, _unit_tuning_features

    require(len(support_trial_bounds) == len(support_direction_angles), "trial/direction count mismatch")
    durations = np.asarray([(stop - start) * BIN_SIZE_S for start, stop in support_trial_bounds], dtype=np.float64)
    require(np.all(durations > 0), "nonpositive trial exposure")
    counts = np.stack([neural[start:stop].sum(axis=0, dtype=np.float64) for start, stop in support_trial_bounds], axis=1)
    trial_rates = np.ascontiguousarray(counts / durations[None, :], dtype=np.float64)
    direction_indices = np.asarray(
        [_nearest_canonical_direction_index(float(theta)) for theta in support_direction_angles],
        dtype=np.int64,
    )
    present = sorted({int(value) for value in direction_indices})
    require(len(present) >= MIN_UNIQUE_DIRECTIONS, "PV cosine fit is direction-degenerate")
    n_channels = int(neural.shape[1])
    a = np.empty(n_channels, dtype=np.float64)
    c = np.empty(n_channels, dtype=np.float64)
    m = np.empty(n_channels, dtype=np.float64)
    b = np.empty(n_channels, dtype=np.float64)
    for channel in range(n_channels):
        feature, _t8, _zero_spike, _zero_modulation = _unit_tuning_features(
            trial_rates[channel], direction_indices, present
        )
        a[channel], c[channel], m[channel], b[channel] = (float(value) for value in feature)
    preferred, zero_from_pd = preferred_directions_from_cosine(a, c, m)
    prefix = prefix_sums(neural)
    support_starts = np.ascontiguousarray(support_window_target_bins - (M2_WINDOW_BINS - 1), dtype=np.int64)
    calibration_rates = window_rates_from_prefix(prefix, support_starts)
    calibration_vectors = population_vectors(calibration_rates, preferred, b)
    calibration_targets = behavior[support_window_target_bins]
    gain, intercept, rank = fit_population_vector_gain(calibration_vectors, calibration_targets)
    readout = {
        "preferred_direction": preferred,
        "baseline_rate": np.ascontiguousarray(b, dtype=np.float64),
        "gain": gain,
        "intercept": intercept,
        "calibration_rank": rank,
        "zero_modulation_channels": zero_from_pd,
    }
    detail = {
        "target_direction_labels_used": int(len(support_direction_angles)),
        "dense_behavior_windows_used": int(support_window_target_bins.size),
        "present_direction_count": len(present),
        "affine_gain_rank": int(rank),
        "channel_count": n_channels,
    }
    return readout, detail


def predict_pv_from_readout(
    neural: np.ndarray,
    readout: Mapping[str, Any],
    query_target_bins: np.ndarray,
) -> np.ndarray:
    prefix = prefix_sums(neural)
    query_starts = np.ascontiguousarray(query_target_bins - (M2_WINDOW_BINS - 1), dtype=np.int64)
    query_rates = window_rates_from_prefix(prefix, query_starts)
    query_vectors = population_vectors(
        query_rates,
        np.asarray(readout["preferred_direction"], dtype=np.float64),
        np.asarray(readout["baseline_rate"], dtype=np.float64),
    )
    return predict_population_vector(
        query_vectors,
        np.asarray(readout["gain"], dtype=np.float64),
        np.asarray(readout["intercept"], dtype=np.float64),
    )


def load_raw_m2(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    neural, covariates, trial_change, eval_mask = load_nwb(path, FalconTask.m2)
    return (
        np.asarray(neural, dtype=np.float32),
        np.asarray(covariates, dtype=np.float32),
        np.asarray(trial_change, dtype=bool),
        np.asarray(eval_mask, dtype=bool),
    )


def evaluate_m2_session_pv(
    *,
    nwb_path: Path,
    reference_manifest: Mapping[str, Any],
) -> M2PopulationVectorResult:
    session_name = nwb_path.name.split("_ses-")[1].split("_behavior")[0]
    neural, covariates, trial_change, eval_mask = load_raw_m2(nwb_path)
    layout = m2_ridge.chronological_m24_layout(session_name, neural, covariates, trial_change, eval_mask)
    m2_ridge.validate_selected_layouts_against_reference_manifest([layout], reference_manifest)
    query_hash = digest_query_target_bins(layout.query_target_bins)
    angles = m2_target_angles_from_nwb(nwb_path)
    bounds = _trial_bounds_from_change(trial_change)
    require(len(bounds) >= M2_CALIBRATION_TRIALS, f"{session_name}: fewer than {M2_CALIBRATION_TRIALS} trials")
    support_bounds = bounds[:M2_CALIBRATION_TRIALS]
    support_angles = [float(value) for value in angles[:M2_CALIBRATION_TRIALS] if np.isfinite(value)]
    labeled_bounds: list[tuple[int, int]] = []
    labeled_angles: list[float] = []
    for (start, stop), angle in zip(support_bounds, angles[:M2_CALIBRATION_TRIALS]):
        if not np.isfinite(angle):
            continue
        labeled_bounds.append((start, stop))
        labeled_angles.append(float(angle))
    require(len(labeled_angles) >= MIN_UNIQUE_DIRECTIONS, f"{session_name}: directional calibration degenerate")
    readout, detail = build_pv_readout_from_trial_data(
        neural=neural,
        behavior=covariates,
        support_trial_bounds=labeled_bounds,
        support_direction_angles=labeled_angles,
        support_window_target_bins=layout.support_target_bins,
    )
    predictions = predict_pv_from_readout(neural, readout, layout.query_target_bins)
    targets = m2_ridge.velocity_targets_at_bins(covariates, layout.query_target_bins)
    score = m2_ridge.score_r2_variance_weighted_torchmetrics151(predictions, targets)
    return M2PopulationVectorResult(
        session_name=session_name,
        r2_variance_weighted=float(score),
        query_identity_hash=query_hash,
        query_identity_bound=True,
        calibration_rows=int(layout.support_target_bins.size),
        query_rows=int(layout.query_target_bins.size),
        target_direction_labels_used=int(detail["target_direction_labels_used"]),
        supervision_coordinates_consumed=int(detail["target_direction_labels_used"] + layout.support_target_bins.size * M2_OUTPUT_DIM),
    )


def evaluate_m2_population_vector(
    *,
    data_dir: Path,
    reference_manifest_path: Path,
) -> dict[str, Any]:
    audit = audit_dataset_applicability(M2_DATASET, data_dir)
    require(audit["verdict"] == "PV_DEFINABLE", f"M2 PV scoring refused: {audit['verdict']}")
    manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    m2_ridge.validate_reference_split_manifest(manifest)
    sessions = discover_m2_sessions(data_dir)
    results = [
        evaluate_m2_session_pv(nwb_path=path, reference_manifest=manifest)
        for path in (sessions[name] for name in m2_ridge.EXPECTED_HELDOUT_SESSIONS)
    ]
    scores = np.asarray([result.r2_variance_weighted for result in results], dtype=np.float64)
    return {
        "verdict": audit["verdict"],
        "applicability": audit,
        "query_identity_hashes": {result.session_name: result.query_identity_hash for result in results},
        "all_sessions_query_identity_bound": all(result.query_identity_bound for result in results),
        "per_session": {
            result.session_name: {
                "r2_variance_weighted": result.r2_variance_weighted,
                "query_identity_hash": result.query_identity_hash,
                "query_identity_bound": result.query_identity_bound,
                "calibration_rows": result.calibration_rows,
                "query_rows": result.query_rows,
                "target_direction_labels_used": result.target_direction_labels_used,
                "supervision_coordinates_consumed": result.supervision_coordinates_consumed,
            }
            for result in results
        },
        "aggregate": {
            "mean_r2_variance_weighted": float(scores.mean()),
            "median_r2_variance_weighted": float(np.median(scores)),
            "ordered_r2_variance_weighted": [float(value) for value in scores.tolist()],
        },
        "reference_split_manifest": {
            "path": str(reference_manifest_path.resolve()),
            "sha256": sha256_file(reference_manifest_path),
        },
    }


def refuse_h1_scoring_if_inapplicable(applicability: Mapping[str, Any]) -> None:
    require(
        applicability.get("dataset") == H1_DATASET,
        "H1 scoring guard requires an H1 applicability payload",
    )
    if applicability.get("verdict") != "PV_DEFINABLE":
        raise PopulationVectorComparatorError(
            "H1 population-vector scoring refused: "
            f"{applicability.get('verdict')} with supporting counts "
            f"{applicability.get('sessions_with_nondegenerate_native_field')}/"
            f"{applicability.get('sessions_total')}"
        )
