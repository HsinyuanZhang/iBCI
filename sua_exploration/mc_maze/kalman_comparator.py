"""Dataset-agnostic velocity Kalman comparator with thin per-dataset adapters.

Closed-form least-squares fits from the calibration block (Section 8.2 of
``HANDOFF_COMPARATORS_20260812``), then a standard linear-Gaussian predict/update
recursion on the query block.  CPU-only; no experiment I/O in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


BIN_SIZE_S_DEFAULT = 0.020
DEFAULT_Q_FLOOR = 1.0e-6
DEFAULT_W_FLOOR = 1.0e-6
CONDITION_NUMBER_LIMIT = 1.0e12
CONSTANT_VALUE = 1.0
CONSTANT_TOLERANCE = 0.0
STATE_POSITION_VELOCITY = "position_velocity"
STATE_VELOCITY_ONLY = "velocity_only"
STATE_PARAMETERISATIONS = (STATE_POSITION_VELOCITY, STATE_VELOCITY_ONLY)

SUBM_FIT_TRIALS = 50
SUBM_HISTORY_BINS = 50
SUBM_OUTPUT_DIM = 2

RT_CALIBRATION_TRIALS = 24
RT_WINDOW_BINS = 50
RT_OUTPUT_DIM = 2
RT_EXPECTED_FOLDS = 15

M2_CALIBRATION_TRIALS = 24
M2_WINDOW_BINS = 50
M2_CHANNELS = 96
M2_OUTPUT_DIM = 2

H1_SUPPORT_TRIALS = 4
H1_WINDOW = 700
H1_HISTORY_BINS = 50
H1_VELOCITY_DIM = 7
H1_EXPECTED_QUERY_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"


class KalmanComparatorError(RuntimeError):
    """Raised when a Kalman comparator contract is violated."""


class KalmanFitError(KalmanComparatorError):
    """Raised when closed-form calibration fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise KalmanComparatorError(message)


def kinematic_dim_for_dataset(dataset: str) -> int:
    if dataset in {"subject_m", "falcon_m2", "rt"}:
        return 2
    if dataset == "falcon_h1":
        return H1_VELOCITY_DIM
    raise KalmanComparatorError(f"unknown dataset: {dataset}")


def require_state_parameterisation(state_parameterisation: str) -> str:
    require(
        state_parameterisation in STATE_PARAMETERISATIONS,
        f"unknown state parameterisation: {state_parameterisation}",
    )
    return str(state_parameterisation)


def state_dim_for_kinematic_dim(
    kinematic_dim: int,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> int:
    require(kinematic_dim > 0, "kinematic dimension must be positive")
    mode = require_state_parameterisation(state_parameterisation)
    if mode == STATE_VELOCITY_ONLY:
        return int(kinematic_dim + 1)
    return int(2 * kinematic_dim + 1)


def constant_index(state_dim: int) -> int:
    return int(state_dim - 1)


def velocity_slice(
    kinematic_dim: int,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> slice:
    mode = require_state_parameterisation(state_parameterisation)
    if mode == STATE_VELOCITY_ONLY:
        return slice(0, int(kinematic_dim))
    return slice(int(kinematic_dim), int(2 * kinematic_dim))


def matrix_condition_number(matrix: np.ndarray) -> float:
    square = np.asarray(matrix, dtype=np.float64)
    require(square.ndim == 2 and square.shape[0] == square.shape[1], "condition number requires a square matrix")
    try:
        cond = float(np.linalg.cond(square))
    except np.linalg.LinAlgError:
        return float("inf")
    if not math.isfinite(cond):
        return float("inf")
    return cond


def json_safe_condition_number(value: float | None) -> dict[str, Any]:
    if value is None:
        return {"value": None, "finite": False, "infinite": False}
    finite = bool(math.isfinite(float(value)))
    return {
        "value": float(value) if finite else None,
        "finite": finite,
        "infinite": bool(not finite and float(value) > 0.0),
    }


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


def window_manifest_sha256(rows: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session_name, start in rows:
        digest.update(session_name.encode("ascii"))
        digest.update(np.int64(start).tobytes())
    return digest.hexdigest()


def integrate_position(velocity: np.ndarray, *, bin_size_s: float) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    require(vel.ndim == 2 and vel.shape[1] >= 1, "velocity must be [time, kinematic_dim]")
    position = np.cumsum(vel, axis=0, dtype=np.float64) * float(bin_size_s)
    return np.ascontiguousarray(position, dtype=np.float64)


def build_state_matrix(
    velocity: np.ndarray,
    *,
    kinematic_dim: int,
    bin_size_s: float = BIN_SIZE_S_DEFAULT,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    require(vel.ndim == 2 and vel.shape[1] == kinematic_dim, "velocity shape mismatch")
    mode = require_state_parameterisation(state_parameterisation)
    state_dim = state_dim_for_kinematic_dim(kinematic_dim, mode)
    states = np.empty((vel.shape[0], state_dim), dtype=np.float64)
    if mode == STATE_VELOCITY_ONLY:
        states[:, velocity_slice(kinematic_dim, mode)] = vel
    else:
        position = integrate_position(vel, bin_size_s=bin_size_s)
        states[:, :kinematic_dim] = position
        states[:, velocity_slice(kinematic_dim, mode)] = vel
    states[:, constant_index(state_dim)] = CONSTANT_VALUE
    return np.ascontiguousarray(states, dtype=np.float64)


def bin_rates(neural: np.ndarray, *, bin_size_s: float = BIN_SIZE_S_DEFAULT) -> np.ndarray:
    counts = np.asarray(neural, dtype=np.float64)
    require(counts.ndim == 2 and bin_size_s > 0.0, "invalid neural/bin_size for rates")
    rates = counts / float(bin_size_s)
    require(np.isfinite(rates).all(), "non-finite bin rates")
    return np.ascontiguousarray(rates, dtype=np.float64)


def assert_constant_term(
    state: np.ndarray,
    *,
    kinematic_dim: int,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> None:
    array = np.asarray(state, dtype=np.float64)
    idx = constant_index(state_dim_for_kinematic_dim(kinematic_dim, state_parameterisation))
    value = float(array[..., idx])
    require(value == CONSTANT_VALUE, f"constant term drifted to {value}")


def fix_constant_dynamics(
    A: np.ndarray,
    *,
    kinematic_dim: int,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> np.ndarray:
    out = np.array(A, dtype=np.float64, copy=True)
    state_dim = state_dim_for_kinematic_dim(kinematic_dim, state_parameterisation)
    idx = constant_index(state_dim)
    out[idx, :] = 0.0
    out[idx, idx] = CONSTANT_VALUE
    return out


def apply_noise_floor(matrix: np.ndarray, floor: float) -> np.ndarray:
    require(floor >= 0.0 and math.isfinite(floor), "noise floor must be finite and non-negative")
    out = np.array(matrix, dtype=np.float64, copy=True)
    diag = np.diag(out).copy()
    diag = np.maximum(diag, float(floor))
    np.fill_diagonal(out, diag)
    return out


def invert_fail_closed(gram: np.ndarray, label: str) -> np.ndarray:
    square = np.asarray(gram, dtype=np.float64)
    require(square.ndim == 2 and square.shape[0] == square.shape[1], f"{label}: gram must be square")
    require(np.isfinite(square).all(), f"{label}: non-finite gram")
    try:
        cond = float(np.linalg.cond(square))
    except np.linalg.LinAlgError as exc:
        raise KalmanFitError(f"{label}: singular gram") from exc
    if not math.isfinite(cond) or cond > CONDITION_NUMBER_LIMIT:
        raise KalmanFitError(f"{label}: ill-conditioned gram (cond={cond})")
    try:
        return np.linalg.solve(square, np.eye(square.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise KalmanFitError(f"{label}: singular gram") from exc


@dataclass(frozen=True)
class KalmanParameters:
    A: np.ndarray
    W: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    kinematic_dim: int
    q_floor: float
    w_floor: float
    state_parameterisation: str = STATE_POSITION_VELOCITY

    @property
    def state_dim(self) -> int:
        return state_dim_for_kinematic_dim(self.kinematic_dim, self.state_parameterisation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kinematic_dim": int(self.kinematic_dim),
            "state_dim": int(self.state_dim),
            "state_parameterisation": self.state_parameterisation,
            "q_floor": float(self.q_floor),
            "w_floor": float(self.w_floor),
            "W_cond": json_safe_condition_number(matrix_condition_number(self.W)),
            "Q_cond": json_safe_condition_number(matrix_condition_number(self.Q)),
            "A_sha256": sha256_bytes(self.A.tobytes()),
            "W_sha256": sha256_bytes(self.W.tobytes()),
            "H_sha256": sha256_bytes(self.H.tobytes()),
            "Q_sha256": sha256_bytes(self.Q.tobytes()),
        }


@dataclass(frozen=True)
class SessionBlocks:
    """Calibration and query tensors in a common layout."""

    session_name: str
    kinematic_dim: int
    calibration_rates: np.ndarray
    calibration_states: np.ndarray
    query_rates: np.ndarray
    query_states: np.ndarray
    query_velocity_truth: np.ndarray
    metric_name: str
    query_identity: dict[str, Any]
    calibration_rows: int
    query_rows: int
    n_channels: int
    state_parameterisation: str = STATE_POSITION_VELOCITY

    @property
    def state_dim(self) -> int:
        return state_dim_for_kinematic_dim(self.kinematic_dim, self.state_parameterisation)


@dataclass(frozen=True)
class KalmanFitReceipt:
    success: bool
    failure_reason: str | None
    parameters: KalmanParameters | None
    q_floor_applied: bool
    w_floor_applied: bool
    w_cond: float | None = None
    q_cond: float | None = None
    w_cond_raw: float | None = None
    q_cond_raw: float | None = None


@dataclass(frozen=True)
class SessionEvaluation:
    session_name: str
    success: bool
    failure_reason: str | None
    r2: float | None
    fit: KalmanFitReceipt
    calibration_rows: int
    query_rows: int
    metric_name: str
    query_identity: dict[str, Any]
    parameters: KalmanParameters | None


def fit_kalman_parameters(
    calibration_states: np.ndarray,
    calibration_rates: np.ndarray,
    *,
    kinematic_dim: int,
    q_floor: float = DEFAULT_Q_FLOOR,
    w_floor: float = DEFAULT_W_FLOOR,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> KalmanFitReceipt:
    states = np.asarray(calibration_states, dtype=np.float64)
    rates = np.asarray(calibration_rates, dtype=np.float64)
    mode = require_state_parameterisation(state_parameterisation)
    state_dim = state_dim_for_kinematic_dim(kinematic_dim, mode)
    require(
        states.ndim == 2 and states.shape[1] == state_dim and rates.ndim == 2 and states.shape[0] == rates.shape[0],
        "calibration state/rate shape mismatch",
    )
    require(states.shape[0] >= 3, "calibration block too short for Kalman fit")
    for row in states:
        assert_constant_term(row, kinematic_dim=kinematic_dim, state_parameterisation=mode)

    X0 = states[:-1].T
    X1 = states[1:].T
    X = states.T
    Z = rates.T
    transitions = int(X0.shape[1])
    observations = int(Z.shape[1])
    require(transitions >= 2 and observations >= 2, "insufficient calibration transitions")

    try:
        inv_x0 = invert_fail_closed(X0 @ X0.T, "X0X0T")
        A = fix_constant_dynamics(X1 @ X0.T @ inv_x0, kinematic_dim=kinematic_dim, state_parameterisation=mode)
        residual_dyn = X1 - A @ X0
        W_raw = (residual_dyn @ residual_dyn.T) / float(transitions)
        W = apply_noise_floor(W_raw, w_floor)
        w_floor_applied = bool(np.any(np.diag(W_raw) < w_floor))

        inv_xx = invert_fail_closed(X @ X.T, "XXT")
        H = Z @ X.T @ inv_xx
        residual_obs = Z - H @ X
        Q_raw = (residual_obs @ residual_obs.T) / float(observations)
        Q = apply_noise_floor(Q_raw, q_floor)
        q_floor_applied = bool(np.any(np.diag(Q_raw) < q_floor))

        for matrix, label in ((A, "A"), (W, "W"), (H, "H"), (Q, "Q")):
            require(np.isfinite(matrix).all(), f"{label} contains non-finite values")

        params = KalmanParameters(
            A=np.ascontiguousarray(A, dtype=np.float64),
            W=np.ascontiguousarray(W, dtype=np.float64),
            H=np.ascontiguousarray(H, dtype=np.float64),
            Q=np.ascontiguousarray(Q, dtype=np.float64),
            kinematic_dim=int(kinematic_dim),
            q_floor=float(q_floor),
            w_floor=float(w_floor),
            state_parameterisation=mode,
        )
        return KalmanFitReceipt(
            success=True,
            failure_reason=None,
            parameters=params,
            q_floor_applied=q_floor_applied,
            w_floor_applied=w_floor_applied,
            w_cond=matrix_condition_number(W),
            q_cond=matrix_condition_number(Q),
            w_cond_raw=matrix_condition_number(W_raw),
            q_cond_raw=matrix_condition_number(Q_raw),
        )
    except KalmanFitError as exc:
        return KalmanFitReceipt(
            success=False,
            failure_reason=str(exc),
            parameters=None,
            q_floor_applied=False,
            w_floor_applied=False,
        )


def kalman_filter_query(
    query_rates: np.ndarray,
    parameters: KalmanParameters,
    *,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray | None = None,
) -> np.ndarray:
    rates = np.asarray(query_rates, dtype=np.float64)
    require(rates.ndim == 2 and rates.shape[1] == parameters.H.shape[0], "query rate shape mismatch")
    state_dim = parameters.state_dim
    kinematic_dim = parameters.kinematic_dim
    const_idx = constant_index(state_dim)
    x = np.asarray(initial_state, dtype=np.float64).copy()
    require(x.shape == (state_dim,), "initial state shape mismatch")
    assert_constant_term(x, kinematic_dim=kinematic_dim, state_parameterisation=parameters.state_parameterisation)
    if initial_covariance is None:
        P = np.eye(state_dim, dtype=np.float64)
    else:
        P = np.asarray(initial_covariance, dtype=np.float64)
        require(P.shape == (state_dim, state_dim), "initial covariance shape mismatch")

    filtered = np.empty((rates.shape[0], state_dim), dtype=np.float64)
    identity = np.eye(state_dim, dtype=np.float64)
    for index in range(rates.shape[0]):
        x_pred = parameters.A @ x
        x_pred[const_idx] = CONSTANT_VALUE
        P_pred = parameters.A @ P @ parameters.A.T + parameters.W
        innovation = parameters.H @ P_pred @ parameters.H.T + parameters.Q
        gain = P_pred @ parameters.H.T @ invert_fail_closed(innovation, f"query_innovation_t{index}")
        residual = rates[index] - parameters.H @ x_pred
        x = x_pred + gain @ residual
        x[const_idx] = CONSTANT_VALUE
        assert_constant_term(x, kinematic_dim=kinematic_dim, state_parameterisation=parameters.state_parameterisation)
        P = (identity - gain @ parameters.H) @ P_pred
        filtered[index] = x
    return filtered


def r2_pooled(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    require(truth64.shape == estimate64.shape and truth64.ndim == 2, "invalid pooled R2 arrays")
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    require(math.isfinite(tss) and tss > 0.0, "pooled R2 denominator is not positive")
    score = 1.0 - sse / tss
    require(math.isfinite(score), "pooled R2 is non-finite")
    return float(score)


def r2_variance_weighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    require(pred.shape == target.shape and pred.ndim == 2 and pred.shape[0] >= 3, "invalid variance-weighted R2 arrays")
    residual = np.square(target - pred).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    denominator = float(total.sum())
    require(math.isfinite(denominator) and denominator > 0.0, "variance-weighted R2 denominator is not positive")
    score = 1.0 - float(residual.sum()) / denominator
    require(math.isfinite(score), "variance-weighted R2 is non-finite")
    return float(score)


def score_session(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    metric_name: str,
) -> float:
    if metric_name == "pooled_r2":
        return r2_pooled(truth, estimate)
    if metric_name == "variance_weighted_r2":
        return r2_variance_weighted(truth, estimate)
    raise KalmanComparatorError(f"unknown metric: {metric_name}")


def evaluate_session(
    blocks: SessionBlocks,
    *,
    q_floor: float = DEFAULT_Q_FLOOR,
    w_floor: float = DEFAULT_W_FLOOR,
) -> SessionEvaluation:
    fit = fit_kalman_parameters(
        blocks.calibration_states,
        blocks.calibration_rates,
        kinematic_dim=blocks.kinematic_dim,
        q_floor=q_floor,
        w_floor=w_floor,
        state_parameterisation=blocks.state_parameterisation,
    )
    if not fit.success or fit.parameters is None:
        return SessionEvaluation(
            session_name=blocks.session_name,
            success=False,
            failure_reason=fit.failure_reason,
            r2=None,
            fit=fit,
            calibration_rows=int(blocks.calibration_rows),
            query_rows=int(blocks.query_rows),
            metric_name=blocks.metric_name,
            query_identity=dict(blocks.query_identity),
            parameters=None,
        )

    initial_state = blocks.calibration_states[-1]
    try:
        filtered = kalman_filter_query(blocks.query_rates, fit.parameters, initial_state=initial_state)
    except KalmanFitError as exc:
        return SessionEvaluation(
            session_name=blocks.session_name,
            success=False,
            failure_reason=str(exc),
            r2=None,
            fit=fit,
            calibration_rows=int(blocks.calibration_rows),
            query_rows=int(blocks.query_rows),
            metric_name=blocks.metric_name,
            query_identity=dict(blocks.query_identity),
            parameters=fit.parameters,
        )
    vel_slice = velocity_slice(blocks.kinematic_dim, blocks.state_parameterisation)
    predictions = filtered[:, vel_slice]
    truth = np.asarray(blocks.query_velocity_truth, dtype=np.float64)
    require(predictions.shape == truth.shape, "velocity prediction/truth shape mismatch")
    r2 = score_session(truth, predictions, metric_name=blocks.metric_name)
    return SessionEvaluation(
        session_name=blocks.session_name,
        success=True,
        failure_reason=None,
        r2=float(r2),
        fit=fit,
        calibration_rows=int(blocks.calibration_rows),
        query_rows=int(blocks.query_rows),
        metric_name=blocks.metric_name,
        query_identity=dict(blocks.query_identity),
        parameters=fit.parameters,
    )


def summarize_sessions(results: Sequence[SessionEvaluation]) -> dict[str, Any]:
    scores = [float(item.r2) for item in results if item.success and item.r2 is not None]
    return {
        "sessions_total": int(len(results)),
        "sessions_succeeded": int(sum(item.success for item in results)),
        "sessions_failed": int(sum(not item.success for item in results)),
        "mean_r2": float(np.mean(scores)) if scores else None,
        "per_session_r2": {item.session_name: item.r2 for item in results},
        "per_session_success": {item.session_name: item.success for item in results},
        "per_session_failure_reason": {
            item.session_name: item.failure_reason for item in results if not item.success
        },
    }


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _forbidden_path(path: Path) -> bool:
    lowered = str(path).lower()
    forbidden = ("held-out", "heldout", "minival", "formal", "private", "evalai", "test_ecephys")
    return any(token in lowered for token in forbidden)


def _trial_bounds_from_change(trial_change: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.asarray(trial_change, dtype=bool))
    ends = np.r_[starts[1:], len(trial_change)]
    return [(int(left), int(right)) for left, right in zip(starts, ends)]


def _bins_in_trials(
    trial_bounds: Sequence[tuple[int, int]],
    trial_indices: Sequence[int],
    *,
    eval_mask: np.ndarray | None = None,
) -> np.ndarray:
    selected: list[int] = []
    for trial_index in trial_indices:
        left, right = trial_bounds[int(trial_index)]
        for bin_index in range(left, right):
            if eval_mask is None or bool(eval_mask[bin_index]):
                selected.append(bin_index)
    return np.ascontiguousarray(selected, dtype=np.int64)


def adapter_dry_run_subject_m(
    *,
    repo_root: Path,
    view: str,
    budget: int,
    data_dir: Path,
    v9_run_root: Path,
) -> dict[str, Any]:
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        FIT_TRIALS,
        VIEWS,
        _build_view_base,
        _nwb_path_and_pin,
        _runtime_owners,
        calibration_starts_first50,
        load_mean_std,
        load_v9_inputs,
        load_v9_target,
    )

    require(view in VIEWS, f"unsupported subject-M view: {view}")
    require(budget == FIT_TRIALS, "subject-M Kalman binds the sealed M50 calibration budget")
    v9 = load_v9_inputs(v9_run_root, repo_root=repo_root)
    cohort_row = v9.cohort[0]
    nwb_path = _nwb_path_and_pin(data_dir, cohort_row)
    require(not _forbidden_path(nwb_path), f"refusing forbidden path: {nwb_path}")
    mean, std = load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior")
    owners = _runtime_owners(repo_root)
    record, _rebuilt, builder_trials, bridge = _build_view_base(
        repo_root=repo_root,
        nwb_path=nwb_path,
        view=view,
        mean=mean,
        std=std,
        owners=owners,
    )
    _support_starts, support_trials = calibration_starts_first50(builder_trials, record)
    query_target, query_target_sha, _ = load_v9_target(v9, cohort_row, view)
    support_bounds = [(int(trial["start"]), int(trial["stop"])) for trial in support_trials]
    calibration_bin_list: list[int] = []
    for left, right in support_bounds:
        calibration_bin_list.extend(range(left, right))
    calibration_bins = np.ascontiguousarray(calibration_bin_list, dtype=np.int64)
    query_bins = record.valid_starts + SUBM_HISTORY_BINS - 1
    query_identity = {
        "query_starts_sha256": digest_falcon_query_values((record.valid_starts,)),
        "query_target_sha256": query_target_sha,
        "ordered_query_identity_sha256": digest_falcon_query_values((record.valid_starts, query_bins, query_target)),
    }
    return {
        "dataset": "subject_m",
        "view": view,
        "status": "ready",
        "session_example": cohort_row.session_id,
        "calibration_budget_trials": int(budget),
        "calibration_rows": int(calibration_bins.size),
        "query_rows": int(query_bins.size),
        "state_dim": state_dim_for_kinematic_dim(2),
        "kinematic_dim": 2,
        "n_channels": int(record.neural.shape[1]),
        "metric": "variance_weighted_r2",
        "query_identity": query_identity,
        "field_shapes": {
            "neural": list(record.neural.shape),
            "behavior": list(record.behavior.shape),
            "valid_starts": [int(record.valid_starts.size)],
        },
    }


def build_subject_m_session_blocks(
    *,
    repo_root: Path,
    data_dir: Path,
    v9_run_root: Path,
    view: str,
    budget: int,
    cohort_row: Any,
) -> SessionBlocks:
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        FIT_TRIALS,
        VIEWS,
        _build_view_base,
        _nwb_path_and_pin,
        _runtime_owners,
        calibration_starts_first50,
        load_mean_std,
        load_v9_inputs,
        load_v9_target,
    )

    require(view in VIEWS, f"unsupported subject-M view: {view}")
    require(budget == FIT_TRIALS, "subject-M Kalman binds the sealed M50 calibration budget")
    v9 = load_v9_inputs(v9_run_root, repo_root=repo_root)
    nwb_path = _nwb_path_and_pin(data_dir, cohort_row)
    require(not _forbidden_path(nwb_path), f"refusing forbidden path: {nwb_path}")
    mean, std = load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior")
    owners = _runtime_owners(repo_root)
    record, _rebuilt, builder_trials, _bridge = _build_view_base(
        repo_root=repo_root,
        nwb_path=nwb_path,
        view=view,
        mean=mean,
        std=std,
        owners=owners,
    )
    _support_starts, support_trials = calibration_starts_first50(builder_trials, record)
    query_target, query_target_sha, _ = load_v9_target(v9, cohort_row, view)
    support_bounds = [(int(trial["start"]), int(trial["stop"])) for trial in support_trials]
    calibration_bins = np.concatenate(
        [np.arange(left, right, dtype=np.int64) for left, right in support_bounds]
    )
    query_bins = np.ascontiguousarray(record.valid_starts + SUBM_HISTORY_BINS - 1, dtype=np.int64)
    kinematic_dim = 2
    calibration_states = build_state_matrix(record.behavior[calibration_bins], kinematic_dim=kinematic_dim)
    query_states = build_state_matrix(record.behavior[query_bins], kinematic_dim=kinematic_dim)
    return SessionBlocks(
        session_name=str(cohort_row.session_id),
        kinematic_dim=kinematic_dim,
        calibration_rates=bin_rates(record.neural[calibration_bins]),
        calibration_states=calibration_states,
        query_rates=bin_rates(record.neural[query_bins]),
        query_states=query_states,
        query_velocity_truth=np.ascontiguousarray(record.behavior[query_bins], dtype=np.float64),
        metric_name="variance_weighted_r2",
        query_identity={
            "query_starts_sha256": digest_falcon_query_values((record.valid_starts,)),
            "query_target_sha256": query_target_sha,
            "ordered_query_identity_sha256": digest_falcon_query_values(
                (record.valid_starts, query_bins, query_target)
            ),
        },
        calibration_rows=int(calibration_bins.size),
        query_rows=int(query_bins.size),
        n_channels=int(record.neural.shape[1]),
    )


def load_all_subject_m_blocks(
    *,
    repo_root: Path,
    data_dir: Path,
    v9_run_root: Path,
    view: str,
    budget: int,
) -> list[SessionBlocks]:
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import load_v9_inputs

    v9 = load_v9_inputs(v9_run_root, repo_root=repo_root)
    return [
        build_subject_m_session_blocks(
            repo_root=repo_root,
            data_dir=data_dir,
            v9_run_root=v9_run_root,
            view=view,
            budget=budget,
            cohort_row=row,
        )
        for row in v9.cohort
    ]


def adapter_dry_run_falcon_m2(*, repo_root: Path) -> dict[str, Any]:
    from sua_exploration.mc_maze import native_m2_m24_ridge_w50 as m2

    rows = {
        session: dict(m2.EXPECTED_HELDOUT_LAYOUT[session]) for session in m2.EXPECTED_HELDOUT_SESSIONS
    }
    return {
        "dataset": "falcon_m2",
        "status": "blocked_data_access",
        "reason": (
            "M24 sealed comparator sources live under *held-out-calib* NWBs; this agent session "
            "forbids opening those paths. Adapter logic is implemented against "
            "native_m2_m24_ridge_w50.EXPECTED_HELDOUT_LAYOUT but real-data binding is deferred."
        ),
        "calibration_trials": M2_CALIBRATION_TRIALS,
        "state_dim": state_dim_for_kinematic_dim(2),
        "kinematic_dim": 2,
        "n_channels": M2_CHANNELS,
        "metric": "variance_weighted_r2",
        "sessions": rows,
        "query_identity_binding": "per-session query_target_bins_sha256 from chronological_m24_layout",
        "repo_root": str(repo_root),
    }


def build_falcon_m2_session_blocks(
    *,
    session_id: str,
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    eval_mask: np.ndarray,
) -> SessionBlocks:
    from sua_exploration.mc_maze import native_m2_m24_ridge_w50 as m2

    layout = m2.chronological_m24_layout(session_id, neural, covariates, trial_change, eval_mask)
    calibration_bins = layout.support_target_bins
    query_bins = layout.query_target_bins
    kinematic_dim = 2
    calibration_states = build_state_matrix(covariates[calibration_bins], kinematic_dim=kinematic_dim)
    query_states = build_state_matrix(covariates[query_bins], kinematic_dim=kinematic_dim)
    return SessionBlocks(
        session_name=session_id,
        kinematic_dim=kinematic_dim,
        calibration_rates=bin_rates(neural[calibration_bins]),
        calibration_states=calibration_states,
        query_rates=bin_rates(neural[query_bins]),
        query_states=query_states,
        query_velocity_truth=np.ascontiguousarray(covariates[query_bins], dtype=np.float64),
        metric_name="variance_weighted_r2",
        query_identity={
            "support_target_bins_sha256": layout.as_audit_dict()["support_target_bins_sha256"],
            "query_target_bins_sha256": layout.as_audit_dict()["query_target_bins_sha256"],
        },
        calibration_rows=int(calibration_bins.size),
        query_rows=int(query_bins.size),
        n_channels=int(neural.shape[1]),
    )


def adapter_dry_run_rt(
    *,
    repo_root: Path,
    data_dir: Path,
    stage2_cell_root: Path,
) -> dict[str, Any]:
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions, load_rt_session
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    sessions = {
        rt.session_name_from_nwb_path(Path(path)): Path(path) for path in find_rt_sessions(data_dir)
    }
    require(len(sessions) == RT_EXPECTED_FOLDS, f"expected {RT_EXPECTED_FOLDS} RT sessions")
    sealed = rt.load_sealed_stage2_cell(stage2_cell_root, 0)
    nwb_path = sessions[sealed["session_name"]]
    require(not _forbidden_path(nwb_path), f"refusing forbidden path: {nwb_path}")
    raw = load_rt_session(nwb_path)
    layout = rt.rt_outer_window_layout(str(raw["session_name"]), raw["neural"], raw["covariates"], raw["trial_change"], raw["eval_mask"])
    neural_p, cov_p, _trial_p, eval_p, _starts = rt.pad_rt_like_falcon(
        raw["neural"], raw["covariates"], raw["trial_change"], raw["eval_mask"]
    )
    support_bins = layout.support_window_starts + RT_WINDOW_BINS - 1
    query_bins = layout.query_window_starts + RT_WINDOW_BINS - 1
    return {
        "dataset": "rt",
        "status": "ready",
        "session_example": sealed["session_name"],
        "calibration_trials": RT_CALIBRATION_TRIALS,
        "calibration_rows": int(support_bins.size),
        "query_rows": int(query_bins.size),
        "state_dim": state_dim_for_kinematic_dim(2),
        "kinematic_dim": 2,
        "n_channels": int(neural_p.shape[1]),
        "metric": "variance_weighted_r2",
        "query_identity": {
            "sealed_stage2": sealed["query_identity"],
            "computed": layout.query_window_audit,
            "query_identity_bound": bool(
                layout.query_window_audit["ordered_window_start_sha256"]
                == sealed["query_identity"]["ordered_window_start_sha256"]
                and layout.query_window_audit["ordered_target_covariate_evalmask_sha256"]
                == sealed["query_identity"]["ordered_target_covariate_evalmask_sha256"]
                and layout.query_window_audit["ordered_query_identity_sha256"]
                == sealed["query_identity"]["ordered_query_identity_sha256"]
            ),
        },
        "field_shapes": {
            "neural_padded": list(neural_p.shape),
            "covariates_padded": list(cov_p.shape),
            "eval_mask_padded": [int(eval_p.size)],
        },
    }


def build_rt_fold_blocks(
    *,
    fold: int,
    nwb_path: Path,
    stage2_cell_root: Path,
    raw_loader: Callable[[Path], Mapping[str, Any]],
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> SessionBlocks:
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    sealed = rt.load_sealed_stage2_cell(stage2_cell_root, fold)
    raw = raw_loader(nwb_path)
    session_name = str(raw["session_name"])
    require(session_name == sealed["session_name"], f"fold {fold}: session name drift")
    layout = rt.rt_outer_window_layout(session_name, raw["neural"], raw["covariates"], raw["trial_change"], raw["eval_mask"])
    neural_p, cov_p, _trial_p, _eval_p, _starts = rt.pad_rt_like_falcon(
        raw["neural"], raw["covariates"], raw["trial_change"], raw["eval_mask"]
    )
    calibration_bins = np.ascontiguousarray(layout.support_window_starts + RT_WINDOW_BINS - 1, dtype=np.int64)
    query_bins = np.ascontiguousarray(layout.query_window_starts + RT_WINDOW_BINS - 1, dtype=np.int64)
    kinematic_dim = 2
    mode = require_state_parameterisation(state_parameterisation)
    calibration_states = build_state_matrix(
        cov_p[calibration_bins], kinematic_dim=kinematic_dim, state_parameterisation=mode
    )
    query_states = build_state_matrix(
        cov_p[query_bins], kinematic_dim=kinematic_dim, state_parameterisation=mode
    )
    return SessionBlocks(
        session_name=session_name,
        kinematic_dim=kinematic_dim,
        calibration_rates=bin_rates(neural_p[calibration_bins]),
        calibration_states=calibration_states,
        query_rates=bin_rates(neural_p[query_bins]),
        query_states=query_states,
        query_velocity_truth=np.ascontiguousarray(cov_p[query_bins], dtype=np.float64),
        metric_name="variance_weighted_r2",
        query_identity={
            "fold": int(fold),
            "sealed_stage2": sealed["query_identity"],
            "computed": layout.query_window_audit,
        },
        calibration_rows=int(calibration_bins.size),
        query_rows=int(query_bins.size),
        n_channels=int(neural_p.shape[1]),
        state_parameterisation=mode,
    )


def _import_h1_pilot_module(repo_root: Path):
    import importlib.util
    import sys

    module_path = repo_root / "SPINT-main/src/data/h1_m4_eb_pilot.py"
    require(module_path.is_file(), f"missing H1 pilot loader: {module_path}")
    module_name = "kalman_h1_m4_eb_pilot"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    require(spec is not None and spec.loader is not None, "cannot load H1 pilot module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def adapter_dry_run_falcon_h1(*, repo_root: Path, data_dir: Path) -> dict[str, Any]:
    pilot = _import_h1_pilot_module(repo_root)
    H1_M4_FOLD0_TARGET = pilot.H1_M4_FOLD0_TARGET
    PILOT_WINDOW = pilot.WINDOW
    load_target_records = pilot.load_target_records
    from sua_exploration.mc_maze import h1_ridge_family as h1

    require(PILOT_WINDOW == h1.WINDOW, "H1 query-window length drift")
    records = load_target_records(data_dir)
    target_names = tuple(H1_M4_FOLD0_TARGET)
    split = h1.build_session_split(records[target_names[0]])
    scored_windows: list[tuple[str, int]] = []
    for name in target_names:
        session_split = h1.build_session_split(records[name])
        for output_bin in session_split.query_output_bins:
            scored_windows.append((name, int(output_bin - h1.WINDOW + 1)))
    query_sha = h1.window_manifest_sha256(scored_windows)
    require(query_sha == H1_EXPECTED_QUERY_SHA256, "H1 scored-query manifest SHA drift")
    return {
        "dataset": "falcon_h1",
        "status": "ready",
        "session_example": target_names[0],
        "calibration_trials": H1_SUPPORT_TRIALS,
        "calibration_rows": int(split.calibration_target_bins.size),
        "query_rows": int(len(scored_windows)),
        "state_dim": state_dim_for_kinematic_dim(H1_VELOCITY_DIM),
        "kinematic_dim": H1_VELOCITY_DIM,
        "n_channels": int(split.neural.shape[1]),
        "metric": "pooled_r2",
        "query_identity": {
            "window_manifest_sha256": query_sha,
            "expected_window_manifest_sha256": H1_EXPECTED_QUERY_SHA256,
            "query_windows": int(len(scored_windows)),
        },
        "field_shapes": {
            "neural": list(split.neural.shape),
            "velocity": list(split.velocity.shape),
            "trial_num": [int(split.trial_num.size)],
        },
        "repo_root": str(repo_root),
    }


def build_h1_session_blocks(
    record: Any,
    *,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> SessionBlocks:
    from sua_exploration.mc_maze import h1_ridge_family as h1

    split = h1.build_session_split(record)
    calibration_bins = split.calibration_target_bins
    query_bins = split.query_output_bins
    kinematic_dim = H1_VELOCITY_DIM
    mode = require_state_parameterisation(state_parameterisation)
    calibration_states = build_state_matrix(
        split.velocity[calibration_bins], kinematic_dim=kinematic_dim, state_parameterisation=mode
    )
    query_states = build_state_matrix(
        split.velocity[query_bins], kinematic_dim=kinematic_dim, state_parameterisation=mode
    )
    scored_windows = [(split.session_name, int(bin_index - h1.WINDOW + 1)) for bin_index in query_bins]
    return SessionBlocks(
        session_name=str(split.session_name),
        kinematic_dim=kinematic_dim,
        calibration_rates=bin_rates(split.neural[calibration_bins]),
        calibration_states=calibration_states,
        query_rates=bin_rates(split.neural[query_bins]),
        query_states=query_states,
        query_velocity_truth=np.ascontiguousarray(split.query_truth, dtype=np.float64),
        metric_name="pooled_r2",
        query_identity={
            "window_manifest_sha256": window_manifest_sha256(scored_windows),
            "query_output_bins_sha256": digest_falcon_query_values((query_bins,)),
        },
        calibration_rows=int(calibration_bins.size),
        query_rows=int(query_bins.size),
        n_channels=int(split.neural.shape[1]),
        state_parameterisation=mode,
    )


def load_all_h1_blocks(
    *,
    data_dir: Path,
    repo_root: Path,
    state_parameterisation: str = STATE_POSITION_VELOCITY,
) -> list[SessionBlocks]:
    pilot = _import_h1_pilot_module(repo_root)
    H1_M4_FOLD0_TARGET = pilot.H1_M4_FOLD0_TARGET
    load_target_records = pilot.load_target_records

    records = load_target_records(data_dir)
    return [
        build_h1_session_blocks(records[name], state_parameterisation=state_parameterisation)
        for name in H1_M4_FOLD0_TARGET
    ]


def bind_h1_query_manifest(*, data_dir: Path, repo_root: Path) -> dict[str, Any]:
    from sua_exploration.mc_maze import h1_ridge_family as h1

    pilot = _import_h1_pilot_module(repo_root)
    records = pilot.load_target_records(data_dir)
    scored_windows: list[tuple[str, int]] = []
    for name in pilot.H1_M4_FOLD0_TARGET:
        split = h1.build_session_split(records[name])
        for output_bin in split.query_output_bins:
            scored_windows.append((name, int(output_bin - h1.WINDOW + 1)))
    query_sha = window_manifest_sha256(scored_windows)
    expected = H1_EXPECTED_QUERY_SHA256
    return {
        "window_manifest_sha256": query_sha,
        "expected_window_manifest_sha256": expected,
        "matched": query_sha == expected,
        "query_windows": int(len(scored_windows)),
        "sessions": list(pilot.H1_M4_FOLD0_TARGET),
    }


def dry_run_report(
    *,
    dataset: str,
    repo_root: Path,
    view: str | None = None,
    budget: int = SUBM_FIT_TRIALS,
    data_dir: Path | None = None,
    v9_run_root: Path | None = None,
    stage2_cell_root: Path | None = None,
) -> dict[str, Any]:
    if dataset == "subject_m":
        require(view is not None and data_dir is not None and v9_run_root is not None, "subject-M dry-run needs view/data/v9 root")
        return adapter_dry_run_subject_m(
            repo_root=repo_root,
            view=view,
            budget=budget,
            data_dir=data_dir,
            v9_run_root=v9_run_root,
        )
    if dataset == "falcon_m2":
        return adapter_dry_run_falcon_m2(repo_root=repo_root)
    if dataset == "rt":
        require(data_dir is not None and stage2_cell_root is not None, "RT dry-run needs data/stage2 roots")
        return adapter_dry_run_rt(repo_root=repo_root, data_dir=data_dir, stage2_cell_root=stage2_cell_root)
    if dataset == "falcon_h1":
        require(data_dir is not None, "H1 dry-run needs data_dir")
        return adapter_dry_run_falcon_h1(repo_root=repo_root, data_dir=data_dir)
    raise KalmanComparatorError(f"unknown dataset for dry-run: {dataset}")


def dry_run_all(
    *,
    repo_root: Path,
    data_dir_subject_m: Path,
    data_dir_rt: Path,
    data_dir_h1: Path,
    v9_run_root: Path,
    stage2_cell_root: Path,
) -> dict[str, Any]:
    reports = {
        "subject_m_sua": dry_run_report(
            dataset="subject_m",
            repo_root=repo_root,
            view="sua",
            data_dir=data_dir_subject_m,
            v9_run_root=v9_run_root,
        ),
        "subject_m_pseudo_mua": dry_run_report(
            dataset="subject_m",
            repo_root=repo_root,
            view="pseudo_mua",
            data_dir=data_dir_subject_m,
            v9_run_root=v9_run_root,
        ),
        "falcon_m2": dry_run_report(dataset="falcon_m2", repo_root=repo_root),
        "rt": dry_run_report(
            dataset="rt",
            repo_root=repo_root,
            data_dir=data_dir_rt,
            stage2_cell_root=stage2_cell_root,
        ),
        "falcon_h1": dry_run_report(dataset="falcon_h1", repo_root=repo_root, data_dir=data_dir_h1),
    }
    return {
        "schema": "kalman_comparator_dry_run_v1",
        "adapters": reports,
        "regularisation_floors_default": {"q_floor": DEFAULT_Q_FLOOR, "w_floor": DEFAULT_W_FLOOR},
        "state_parameterisation": {
            "available": list(STATE_PARAMETERISATIONS),
            "position_velocity": {
                "layout": "[position, velocity, constant]",
                "position_source": "cumulative_sum(velocity) * bin_size_s with zero initial offset",
                "state_dim": "2 * kinematic_dim + 1",
            },
            "velocity_only": {
                "layout": "[velocity, constant]",
                "position_source": "none; position is not in the state",
                "state_dim": "kinematic_dim + 1",
            },
            "constant_handling": "last state element fixed to 1.0; A last row [0..0,1]",
        },
    }
