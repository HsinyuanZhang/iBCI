"""CPU classical comparators for the sealed RT Stage-2 outer-LOSO matrix.

Ridge readouts map a causal ``W50 x N`` spike history to 2-D cursor velocity on
the outer target session only.  Population-vector applicability is audited from
native trial metadata; a forced score is not manufactured when the native
direction field is degenerate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import (
    FEATURE_STD_EPS,
    HISTORY_BINS,
    fit_population_vector_gain,
    fit_ridge,
    population_vectors,
    predict_population_vector,
    predict_ridge,
    prefix_sums,
    raw_window_features,
    targets_at_window_end,
    window_rates_from_prefix,
)


CALIBRATION_TRIALS = 24
QUERY_START_TRIAL = 24
WINDOW_BINS = HISTORY_BINS
OUTPUT_DIM = 2
NORMALIZED_LAMBDA_FIXED = 1.0
LAMBDA_CV_GRID = (1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0)
LAMBDA_CV_FOLDS = 5
SEED = 42
EXPECTED_FOLDS = 15
STAGE2_CELL_GLOB = "f{fold:02d}_rt_sparse_endpoint_t4d.json"

SEALED_RT_REFERENCES = {
    "t4d_sparse_endpoint": 0.448176,
    "dense_full_velocity": 0.445189,
    "zero4": 0.179272,
    "matched_b2_d1024": 0.145148,
}


class RtClassicalComparatorError(RuntimeError):
    """Raised when an RT classical-comparator contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RtClassicalComparatorError(message)


@dataclass(frozen=True)
class RtWindowLayout:
    """Support/query causal window starts on the FalconDataset-padded time axis."""

    session_name: str
    n_channels: int
    feature_dim: int
    support_window_starts: np.ndarray
    query_window_starts: np.ndarray
    query_window_audit: dict[str, Any]
    support_boundary_padded_bin: int


@dataclass(frozen=True)
class RidgeArmResult:
    arm: str
    normalized_lambda: float
    r2: float
    calibration_rows: int
    query_rows: int
    feature_dim: int
    parameters_fitted: int
    observations_per_parameter: float
    supervision_coordinates_consumed: int
    lambda_selection_detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class NativeDirectionAudit:
    session_name: str
    trial_count: int
    finite_target_dir_count: int
    unique_target_dir_count: int
    unique_target_dir_values_rad: tuple[float, ...]
    m24_finite_target_dir_count: int
    m24_unique_target_dir_count: int
    pv_definable_on_native_field: bool


@dataclass(frozen=True)
class FoldEvaluation:
    fold: int
    session_name: str
    t4d_reference_r2: float
    sealed_query_identity: dict[str, str]
    query_identity_bound: bool
    ridge_fixed_lambda: RidgeArmResult
    ridge_cv_lambda: RidgeArmResult
    native_direction_audit: NativeDirectionAudit
    endpoint_pv_diagnostic: dict[str, Any] | None


def session_name_from_nwb_path(path: Path) -> str:
    name = path.name
    suffix = "_behavior+ecephys.nwb"
    if not name.startswith("sub-C_ses-RT-") or not name.endswith(suffix):
        raise RtClassicalComparatorError(f"Not an RT sub-C session NWB: {path}")
    return name.removesuffix(suffix).removeprefix("sub-C_")


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


def digest_falcon_query_values(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pad_rt_like_falcon(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    eval_mask: np.ndarray,
    *,
    window_size: int = WINDOW_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre_history = window_size - 1
    neural_p = np.pad(np.asarray(neural, dtype=np.float32), ((pre_history, 0), (0, 0)))
    cov_p = np.pad(np.asarray(covariates, dtype=np.float32), ((pre_history, 0), (0, 0)))
    eval_p = np.pad(np.asarray(eval_mask, dtype=bool), (pre_history, 0), constant_values=False)
    trial_p = np.pad(np.asarray(trial_change, dtype=bool), (pre_history, 0), constant_values=False)
    starts = np.where(trial_p)[0]
    require(starts.size > QUERY_START_TRIAL, "session lacks query trials after M24 support")
    return neural_p, cov_p, trial_p, eval_p, starts


def rt_outer_window_layout(
    session_name: str,
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    eval_mask: np.ndarray,
    *,
    calibration_trials: int = CALIBRATION_TRIALS,
    query_start_trial: int = QUERY_START_TRIAL,
    window_size: int = WINDOW_BINS,
) -> RtWindowLayout:
    neural_p, cov_p, _trial_p, eval_p, starts = pad_rt_like_falcon(
        neural, covariates, trial_change, eval_mask, window_size=window_size
    )
    require(query_start_trial == calibration_trials, "RT comparator is frozen to M24/q24")
    boundary_padded = int(starts[query_start_trial])
    n_time = int(neural_p.shape[0])
    n_channels = int(neural_p.shape[1])
    feature_dim = window_size * n_channels

    query_starts: list[int] = []
    for start in range(boundary_padded, n_time - window_size + 1):
        target = start + window_size - 1
        if eval_p[target]:
            query_starts.append(start)
    query_window_starts = np.ascontiguousarray(query_starts, dtype=np.int64)
    require(query_window_starts.size > 0, f"{session_name}: zero eligible query windows")

    support_starts: list[int] = []
    for start in range(0, boundary_padded):
        target = start + window_size - 1
        if target < boundary_padded + window_size - 1 and eval_p[target]:
            support_starts.append(start)
    support_window_starts = np.ascontiguousarray(support_starts, dtype=np.int64)
    require(support_window_starts.size >= 3, f"{session_name}: insufficient support windows for ridge")

    selected = query_window_starts
    target_indices = selected + window_size - 1
    target_rows = np.asarray(cov_p[target_indices], dtype=np.float32)
    target_mask = np.asarray(eval_p[target_indices], dtype=bool)
    audit = {
        "total_trials": int(starts.size),
        "support_trials": calibration_trials,
        "query_start_trial": query_start_trial,
        "query_end_trial": None,
        "query_trials": int(starts.size - query_start_trial),
        "raw_query_start_bin": int(boundary_padded - (window_size - 1)),
        "minimum_window_start_padded_bin": boundary_padded,
        "maximum_window_start_padded_bin": int(n_time - window_size),
        "window_size": window_size,
        "eligible_windows": int(query_window_starts.size),
        "full_window_disjoint": True,
        "ineligible_reason": None,
        "ordered_window_start_sha256": digest_falcon_query_values((selected,)),
        "ordered_target_covariate_evalmask_sha256": digest_falcon_query_values(
            (target_indices, target_rows, target_mask)
        ),
        "ordered_query_identity_sha256": digest_falcon_query_values(
            (selected, target_indices, target_rows, target_mask)
        ),
    }
    return RtWindowLayout(
        session_name=session_name,
        n_channels=n_channels,
        feature_dim=feature_dim,
        support_window_starts=support_window_starts,
        query_window_starts=query_window_starts,
        query_window_audit=audit,
        support_boundary_padded_bin=boundary_padded,
    )


def r2_variance_weighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    require(pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == OUTPUT_DIM, "invalid R2 arrays")
    require(pred.shape[0] >= 3, "R2 requires at least three query rows")
    residual = np.square(target - pred).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    denominator = float(total.sum())
    require(math.isfinite(denominator) and denominator > 0.0, "R2 denominator is not positive")
    score = 1.0 - float(residual.sum()) / denominator
    require(math.isfinite(score), "R2 is non-finite")
    return float(score)


def hand_solve_normalized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    normalized_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < FEATURE_STD_EPS] = 1.0
    z = (x - mean) / scale
    ymean = y.mean(axis=0)
    yc = y - ymean
    n_rows = int(z.shape[0])
    sqrt_weight = np.sqrt(np.full(n_rows, 1.0 / n_rows))
    design = z * sqrt_weight[:, None]
    weighted_target = yc * sqrt_weight[:, None]
    if design.shape[0] < design.shape[1]:
        gram = design @ design.T
        gram.flat[:: gram.shape[0] + 1] += normalized_lambda
        coefficients = design.T @ np.linalg.solve(gram, weighted_target)
    else:
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += normalized_lambda
        coefficients = np.linalg.solve(gram, design.T @ weighted_target)
    return mean, scale, coefficients


def select_lambda_on_support_only(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    *,
    grid: Sequence[float] = LAMBDA_CV_GRID,
    n_folds: int = LAMBDA_CV_FOLDS,
) -> tuple[float, dict[str, Any]]:
    x = np.asarray(support_features, dtype=np.float64)
    y = np.asarray(support_targets, dtype=np.float64)
    require(x.shape[0] == y.shape[0] and x.shape[0] >= n_folds * 3, "insufficient support rows for lambda CV")
    order = np.arange(x.shape[0], dtype=np.int64)
    folds = np.array_split(order, n_folds)
    scores: dict[float, list[float]] = {float(value): [] for value in grid}
    for lam in grid:
        for fold_rows in folds:
            val_rows = fold_rows
            train_mask = np.ones(x.shape[0], dtype=bool)
            train_mask[val_rows] = False
            readout = fit_ridge(x[train_mask], y[train_mask], normalized_lambda=float(lam), device="cpu")
            pred = predict_ridge(x[val_rows], readout, device="cpu")
            scores[float(lam)].append(r2_variance_weighted(pred, y[val_rows]))
    mean_scores = {lam: float(np.mean(vals)) for lam, vals in scores.items()}
    best_lambda = max(mean_scores, key=lambda key: (mean_scores[key], -math.log10(key)))
    return float(best_lambda), {
        "grid": [float(value) for value in grid],
        "fold_count": int(n_folds),
        "mean_validation_r2_by_lambda": {str(lam): mean_scores[lam] for lam in sorted(mean_scores)},
        "selected_lambda": float(best_lambda),
        "query_rows_used": 0,
    }


def evaluate_ridge_arm(
    *,
    arm: str,
    neural: np.ndarray,
    covariates: np.ndarray,
    layout: RtWindowLayout,
    normalized_lambda: float,
    lambda_selection_detail: dict[str, Any] | None = None,
) -> RidgeArmResult:
    support_features = raw_window_features(neural, layout.support_window_starts, window_size=WINDOW_BINS)
    support_targets = targets_at_window_end(covariates, layout.support_window_starts, window_size=WINDOW_BINS)
    query_features = raw_window_features(neural, layout.query_window_starts, window_size=WINDOW_BINS)
    query_targets = targets_at_window_end(covariates, layout.query_window_starts, window_size=WINDOW_BINS)
    readout = fit_ridge(support_features, support_targets, normalized_lambda=float(normalized_lambda), device="cpu")
    predictions = predict_ridge(query_features, readout, device="cpu")
    rows = int(support_features.shape[0])
    params = int(layout.feature_dim * OUTPUT_DIM)
    return RidgeArmResult(
        arm=arm,
        normalized_lambda=float(normalized_lambda),
        r2=r2_variance_weighted(predictions, query_targets),
        calibration_rows=rows,
        query_rows=int(query_features.shape[0]),
        feature_dim=int(layout.feature_dim),
        parameters_fitted=params,
        observations_per_parameter=float(rows / params) if params else float("nan"),
        supervision_coordinates_consumed=int(rows * OUTPUT_DIM),
        lambda_selection_detail=lambda_selection_detail,
    )


def audit_native_target_direction(path: Path, *, calibration_trials: int = CALIBRATION_TRIALS) -> NativeDirectionAudit:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        require(nwb.trials is not None and "target_dir" in nwb.trials.colnames, "native target_dir column missing")
        target_dir = np.asarray(nwb.trials["target_dir"][:], dtype=np.float64).reshape(-1)
    finite = target_dir[np.isfinite(target_dir)]
    unique = np.unique(np.round(finite, decimals=12))
    m24 = target_dir[:calibration_trials]
    m24_finite = m24[np.isfinite(m24)]
    m24_unique = np.unique(np.round(m24_finite, decimals=12))
    unique_count = int(unique.size)
    return NativeDirectionAudit(
        session_name=path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb"),
        trial_count=int(target_dir.size),
        finite_target_dir_count=int(finite.size),
        unique_target_dir_count=unique_count,
        unique_target_dir_values_rad=tuple(float(value) for value in unique.tolist()),
        m24_finite_target_dir_count=int(m24_finite.size),
        m24_unique_target_dir_count=int(m24_unique.size),
        pv_definable_on_native_field=unique_count >= 2,
    )


def extract_endpoint_reach_directions(path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return M24 endpoint-derived reach directions and raw bin bounds."""
    from pynwb import NWBHDF5IO
    from streaming_calibration_exp.src.data.rt_sparse_endpoint_loader import (
        BIN_SIZE_S,
        MIN_DISPLACEMENT_CM,
        _complete_reaches,
        _endpoint_from_sparse,
        _go_cues,
        _sparse_coordinate_map,
    )

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials
        starts = np.asarray(trials["start_time"][:], dtype=np.float64)
        stops = np.asarray(trials["stop_time"][:], dtype=np.float64)
        cues = _go_cues(trials["go_cue_time_array"][:], starts.size)
        target_count = np.asarray(trials["num_targets"][:])
        pos = nwb.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        times = np.asarray(pos.timestamps[:], dtype=np.float64)
        reaches = _complete_reaches(starts, stops, cues, target_count)
        sparse_coordinates, _audit = _sparse_coordinate_map(
            pos, times, [time for reach in reaches for time in reach]
        )
    thetas: list[float] = []
    bounds: list[tuple[int, int]] = []
    for left_t, right_t in reaches:
        left = int(np.ceil(left_t / BIN_SIZE_S - 1.0e-10))
        right = int(np.floor(right_t / BIN_SIZE_S + 1.0e-10))
        first, first_reason = _endpoint_from_sparse(times, sparse_coordinates, left_t)
        last, last_reason = _endpoint_from_sparse(times, sparse_coordinates, right_t)
        if first_reason or last_reason or first is None or last is None or right <= left:
            continue
        displacement = last - first
        if not np.isfinite(displacement).all() or np.linalg.norm(displacement) < MIN_DISPLACEMENT_CM:
            continue
        thetas.append(float(np.arctan2(float(displacement[1]), float(displacement[0]))))
        bounds.append((left, right))
    return np.asarray(thetas, dtype=np.float64), bounds


def evaluate_endpoint_pv_diagnostic(
    *,
    neural: np.ndarray,
    covariates: np.ndarray,
    eval_mask: np.ndarray,
    reach_thetas: np.ndarray,
    reach_bounds: Sequence[tuple[int, int]],
    support_window_starts: np.ndarray,
    query_window_starts: np.ndarray,
) -> dict[str, Any]:
    from sua_exploration.mc_maze.unit_side_features import _nearest_canonical_direction_index, _unit_tuning_features

    require(reach_thetas.size >= 3, "endpoint PV diagnostic needs at least three reach directions")
    direction_indices = np.asarray(
        [_nearest_canonical_direction_index(float(theta)) for theta in reach_thetas], dtype=np.int64
    )
    present = sorted({int(value) for value in direction_indices})
    require(len(present) >= 2, "endpoint PV diagnostic directions are degenerate")
    rates_by_reach: list[np.ndarray] = []
    for left, right in reach_bounds:
        duration = max((right - left) * 0.02, 1.0e-8)
        rates_by_reach.append(neural[left:right].sum(axis=0) / duration)
    trial_rates = np.stack(rates_by_reach, axis=1)
    n_channels = int(neural.shape[1])
    a = np.empty(n_channels, dtype=np.float64)
    c = np.empty(n_channels, dtype=np.float64)
    m = np.empty(n_channels, dtype=np.float64)
    b = np.empty(n_channels, dtype=np.float64)
    for channel in range(n_channels):
        feature, _t8, _zero_spike, _zero_mod = _unit_tuning_features(
            trial_rates[channel], direction_indices, present
        )
        a[channel], c[channel], m[channel], b[channel] = (float(v) for v in feature)
    preferred = np.zeros((n_channels, 2), dtype=np.float64)
    active = m > 1.0e-6
    preferred[active, 0] = a[active] / m[active]
    preferred[active, 1] = c[active] / m[active]
    prefix = prefix_sums(neural)
    calib_starts = np.asarray(support_window_starts, dtype=np.int64)
    require(calib_starts.size >= 3, "endpoint PV diagnostic lacks calibration windows")
    calib_rates = window_rates_from_prefix(prefix, calib_starts)
    calib_vectors = population_vectors(calib_rates, preferred, b)
    calib_targets = targets_at_window_end(covariates, calib_starts, window_size=WINDOW_BINS)
    gain, intercept, rank = fit_population_vector_gain(calib_vectors, calib_targets)
    query_rates = window_rates_from_prefix(prefix, query_window_starts)
    query_vectors = population_vectors(query_rates, preferred, b)
    query_pred = predict_population_vector(query_vectors, gain, intercept)
    query_targets = targets_at_window_end(covariates, query_window_starts, window_size=WINDOW_BINS)
    return {
        "label": "ENDPOINT_DERIVED_DIRECTION_DIAGNOSTIC_NOT_NATIVE_BASELINE",
        "reach_direction_labels_used": int(reach_thetas.size),
        "unique_reach_direction_count": int(len(present)),
        "calibration_windows": int(calib_starts.size),
        "query_windows": int(query_window_starts.size),
        "affine_gain_rank": int(rank),
        "r2_variance_weighted": r2_variance_weighted(query_pred, query_targets),
        "supervision_coordinates_consumed": int(reach_thetas.size + calib_starts.size * OUTPUT_DIM),
    }


def load_sealed_stage2_cell(stage2_cell_root: Path, fold: int) -> dict[str, Any]:
    path = stage2_cell_root / STAGE2_CELL_GLOB.format(fold=fold)
    require(path.is_file(), f"missing sealed Stage-2 cell receipt for fold {fold}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    outer = payload["outer_receipt"]
    return {
        "fold": int(fold),
        "session_name": str(outer["outer_target_session"]),
        "t4d_r2": float(outer["r2_variance_weighted"]),
        "query_identity": outer["matched_query_window_identity"][outer["outer_target_session"]],
        "cell_path": str(path),
        "cell_sha256": sha256_file(path),
    }


def evaluate_fold(
    *,
    fold: int,
    nwb_path: Path,
    stage2_cell_root: Path,
    raw_loader: Any,
) -> FoldEvaluation:
    sealed = load_sealed_stage2_cell(stage2_cell_root, fold)
    raw = raw_loader(nwb_path)
    session_name = str(raw["session_name"])
    require(session_name == sealed["session_name"], f"fold {fold}: session name drift")
    neural, cov, trial_change, eval_mask = (
        raw["neural"],
        raw["covariates"],
        raw["trial_change"],
        raw["eval_mask"],
    )
    layout = rt_outer_window_layout(session_name, neural, cov, trial_change, eval_mask)
    neural_p, cov_p, _trial_p, eval_p, _starts = pad_rt_like_falcon(neural, cov, trial_change, eval_mask)
    sealed_q = sealed["query_identity"]
    bound = (
        layout.query_window_audit["ordered_window_start_sha256"] == sealed_q["ordered_window_start_sha256"]
        and layout.query_window_audit["ordered_target_covariate_evalmask_sha256"]
        == sealed_q["ordered_target_covariate_evalmask_sha256"]
        and layout.query_window_audit["ordered_query_identity_sha256"] == sealed_q["ordered_query_identity_sha256"]
    )
    selected_lambda, lambda_detail = select_lambda_on_support_only(
        raw_window_features(neural_p, layout.support_window_starts),
        targets_at_window_end(cov_p, layout.support_window_starts),
    )
    ridge_fixed = evaluate_ridge_arm(
        arm="ridge_w50_lambda1",
        neural=neural_p,
        covariates=cov_p,
        layout=layout,
        normalized_lambda=NORMALIZED_LAMBDA_FIXED,
    )
    ridge_cv = evaluate_ridge_arm(
        arm="ridge_w50_lambda_cv",
        neural=neural_p,
        covariates=cov_p,
        layout=layout,
        normalized_lambda=selected_lambda,
        lambda_selection_detail=lambda_detail,
    )
    native_audit = audit_native_target_direction(nwb_path)
    endpoint_diag = None
    try:
        thetas, bounds = extract_endpoint_reach_directions(nwb_path)
        unpadded_support = layout.support_window_starts - (WINDOW_BINS - 1)
        unpadded_query = layout.query_window_starts - (WINDOW_BINS - 1)
        endpoint_diag = evaluate_endpoint_pv_diagnostic(
            neural=neural,
            covariates=cov,
            eval_mask=eval_mask,
            reach_thetas=thetas,
            reach_bounds=bounds,
            support_window_starts=unpadded_support,
            query_window_starts=unpadded_query,
        )
    except Exception as exc:
        endpoint_diag = {"label": "ENDPOINT_DERIVED_DIRECTION_DIAGNOSTIC_FAILED", "error": str(exc)}
    return FoldEvaluation(
        fold=fold,
        session_name=session_name,
        t4d_reference_r2=float(sealed["t4d_r2"]),
        sealed_query_identity={
            "ordered_window_start_sha256": sealed_q["ordered_window_start_sha256"],
            "ordered_target_covariate_evalmask_sha256": sealed_q["ordered_target_covariate_evalmask_sha256"],
            "ordered_query_identity_sha256": sealed_q["ordered_query_identity_sha256"],
        },
        query_identity_bound=bool(bound),
        ridge_fixed_lambda=ridge_fixed,
        ridge_cv_lambda=ridge_cv,
        native_direction_audit=native_audit,
        endpoint_pv_diagnostic=endpoint_diag,
    )


def summarize_arm(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ordered": [float(value) for value in array.tolist()],
    }


def paired_contrast(arm_values: Sequence[float], reference_values: Sequence[float]) -> dict[str, Any]:
    arm = np.asarray(arm_values, dtype=np.float64)
    ref = np.asarray(reference_values, dtype=np.float64)
    require(arm.shape == ref.shape and arm.size > 0, "paired contrast shape mismatch")
    delta = arm - ref
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "positive": int((delta > 0).sum()),
        "zero": int((delta == 0).sum()),
        "negative": int((delta < 0).sum()),
        "per_fold_delta": [float(value) for value in delta.tolist()],
    }


def direction_degeneracy_verdict(audits: Sequence[NativeDirectionAudit]) -> dict[str, Any]:
    per_session = {
        audit.session_name: {
            "finite_target_dir_count": audit.finite_target_dir_count,
            "unique_target_dir_count": audit.unique_target_dir_count,
            "unique_target_dir_values_rad": list(audit.unique_target_dir_values_rad),
            "m24_unique_target_dir_count": audit.m24_unique_target_dir_count,
            "pv_definable_on_native_field": audit.pv_definable_on_native_field,
        }
        for audit in audits
    }
    definable = [audit.pv_definable_on_native_field for audit in audits]
    if any(definable):
        verdict = "PV_PARTIALLY_DEFINABLE_ON_NATIVE_FIELD"
    else:
        verdict = "PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE"
    return {
        "verdict": verdict,
        "sessions_with_nondegenerate_native_field": int(sum(definable)),
        "sessions_total": int(len(audits)),
        "per_session": per_session,
    }


def aggregate_fold_results(folds: Sequence[FoldEvaluation]) -> dict[str, Any]:
    require(len(folds) == EXPECTED_FOLDS, f"expected {EXPECTED_FOLDS} folds, found {len(folds)}")
    sessions = [fold.session_name for fold in folds]
    t4d = [fold.t4d_reference_r2 for fold in folds]
    ridge_fixed = [fold.ridge_fixed_lambda.r2 for fold in folds]
    ridge_cv = [fold.ridge_cv_lambda.r2 for fold in folds]
    all_bound = all(fold.query_identity_bound for fold in folds)
    return {
        "query_identity": {
            "all_folds_bound_to_sealed_stage2": bool(all_bound),
            "per_fold_bound": {fold.session_name: fold.query_identity_bound for fold in folds},
            "per_fold_hashes": {fold.session_name: fold.sealed_query_identity for fold in folds},
        },
        "arms": {
            "t4d_reference": {"per_session": dict(zip(sessions, t4d)), **summarize_arm(t4d)},
            "ridge_w50_lambda1": {
                "per_session": {fold.session_name: fold.ridge_fixed_lambda.r2 for fold in folds},
                **summarize_arm(ridge_fixed),
                "paired_vs_t4d": paired_contrast(ridge_fixed, t4d),
                "supervision_coordinates_consumed_per_session": {
                    fold.session_name: fold.ridge_fixed_lambda.supervision_coordinates_consumed for fold in folds
                },
                "observations_per_parameter_per_session": {
                    fold.session_name: fold.ridge_fixed_lambda.observations_per_parameter for fold in folds
                },
                "parameters_fitted_per_session": {
                    fold.session_name: fold.ridge_fixed_lambda.parameters_fitted for fold in folds
                },
            },
            "ridge_w50_lambda_cv": {
                "per_session": {fold.session_name: fold.ridge_cv_lambda.r2 for fold in folds},
                **summarize_arm(ridge_cv),
                "paired_vs_t4d": paired_contrast(ridge_cv, t4d),
                "selected_lambda_per_session": {
                    fold.session_name: fold.ridge_cv_lambda.normalized_lambda for fold in folds
                },
                "supervision_coordinates_consumed_per_session": {
                    fold.session_name: fold.ridge_cv_lambda.supervision_coordinates_consumed for fold in folds
                },
                "observations_per_parameter_per_session": {
                    fold.session_name: fold.ridge_cv_lambda.observations_per_parameter for fold in folds
                },
            },
        },
        "native_population_vector": direction_degeneracy_verdict([fold.native_direction_audit for fold in folds]),
        "endpoint_pv_diagnostic": {
            fold.session_name: fold.endpoint_pv_diagnostic for fold in folds if fold.endpoint_pv_diagnostic is not None
        },
    }
