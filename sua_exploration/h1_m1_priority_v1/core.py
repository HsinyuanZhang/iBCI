"""Pure numerical contracts for the H1/M1 priority-1--3 execution.

The module has no Torch, NWB, or repository imports.  It contains the parts
that need especially sharp leakage tests: strict same-trial lag alignment,
nested leave-one-session-out DirectRidge selection, matched regression
metrics, and covariance effective-rank accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


LAG_GRID: tuple[int, ...] = tuple(range(0, 11))
LAMBDA_GRID: tuple[float, ...] = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
)
NORMALIZER_FLOOR = 1.0e-8


class PriorityAnalysisError(ValueError):
    """A leakage, shape, numerical, or protocol invariant failed."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PriorityAnalysisError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DirectRidgeSession:
    """One M1 session with an explicit M10 support/query boundary."""

    name: str
    neural: np.ndarray
    target: np.ndarray
    eval_mask: np.ndarray
    trial_change: np.ndarray
    output_names: tuple[str, ...]
    input_path: str = ""
    input_sha256: str = ""

    def validate(self) -> None:
        x = np.asarray(self.neural)
        y = np.asarray(self.target)
        mask = np.asarray(self.eval_mask)
        change = np.asarray(self.trial_change)
        need(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0], f"{self.name}: aligned [T,D] arrays required")
        need(x.shape[1] > 0 and y.shape[1] == len(self.output_names), f"{self.name}: output-name/shape drift")
        need(mask.shape == change.shape == (x.shape[0],), f"{self.name}: mask/change length drift")
        need(np.isfinite(x).all() and np.isfinite(y).all(), f"{self.name}: nonfinite data")
        starts = np.flatnonzero(change)
        need(starts.size >= 11 and starts[0] > 0 and np.all(np.diff(starts) > 0), f"{self.name}: fewer than 11 chronological trials")
        need(np.asarray(mask, dtype=bool).any(), f"{self.name}: empty evaluation mask")

    @property
    def boundary(self) -> int:
        self.validate()
        return int(np.flatnonzero(np.asarray(self.trial_change, dtype=bool))[10])

    @property
    def trial_id(self) -> np.ndarray:
        return np.cumsum(np.asarray(self.trial_change, dtype=np.int64), dtype=np.int64)

    def aligned(self, lag: int, *, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X[t-lag] -> Y[t] without crossing a trial or M10 boundary."""

        self.validate()
        need(int(lag) == lag and lag >= 0, "lag must be a nonnegative integer")
        need(split in {"support", "query"}, "split must be support or query")
        lag = int(lag)
        t = np.arange(self.neural.shape[0], dtype=np.int64)
        source = t - lag
        boundary = self.boundary
        trial_id = self.trial_id
        legal = np.asarray(self.eval_mask, dtype=bool) & (source >= 0) & (trial_id > 0)
        safe_source = np.maximum(source, 0)
        legal &= trial_id[safe_source] == trial_id
        if split == "support":
            legal &= (t < boundary) & (source < boundary) & (trial_id >= 1) & (trial_id <= 10)
        else:
            legal &= (t >= boundary) & (source >= boundary) & (trial_id >= 11)
        indices = t[legal]
        need(indices.size > self.target.shape[1] + 2, f"{self.name}/{split}/lag{lag}: insufficient aligned bins")
        x = np.asarray(self.neural[source[legal]], dtype=np.float64)
        y = np.asarray(self.target[indices], dtype=np.float64)
        need(x.shape[0] == y.shape[0] == indices.size, "aligned rows drift")
        need(np.all(trial_id[source[legal]] == trial_id[indices]), "lag crossed a trial boundary")
        return x, y, indices


def _moments(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    need(bool(arrays), "normalizer needs at least one array")
    stacked = np.concatenate([np.asarray(value, dtype=np.float64) for value in arrays], axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64)
    scale = stacked.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, NORMALIZER_FLOOR)
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0.0), "invalid source moments")
    return mean, scale


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine ridge readout; only the slope is penalized."""

    need(ridge_lambda >= 0.0 and np.isfinite(ridge_lambda), "invalid ridge lambda")
    zx = (np.asarray(x, dtype=np.float64) - x_mean) / x_scale
    zy = (np.asarray(y, dtype=np.float64) - y_mean) / y_scale
    local_x = zx.mean(axis=0, dtype=np.float64)
    local_y = zy.mean(axis=0, dtype=np.float64)
    xc = zx - local_x
    yc = zy - local_y
    gram = xc.T @ xc
    penalty = float(ridge_lambda) * float(xc.shape[0])
    matrix = gram + penalty * np.eye(gram.shape[0], dtype=np.float64)
    rhs = xc.T @ yc
    try:
        weight = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    intercept = local_y - local_x @ weight
    need(np.isfinite(weight).all() and np.isfinite(intercept).all(), "ridge fit became nonfinite")
    return weight, intercept


def _predict_ridge(
    x: np.ndarray,
    *,
    weight: np.ndarray,
    intercept: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> np.ndarray:
    zx = (np.asarray(x, dtype=np.float64) - x_mean) / x_scale
    prediction = (zx @ weight + intercept) * y_scale + y_mean
    need(np.isfinite(prediction).all(), "ridge prediction became nonfinite")
    return prediction


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Matched per-output and variance-weighted R2 with float64 accumulation."""

    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    need(truth.shape == estimate.shape and truth.ndim == 2 and truth.shape[0] > 1, "metric shape drift")
    need(np.isfinite(truth).all() and np.isfinite(estimate).all(), "metric received nonfinite data")
    residual = truth - estimate
    centered = truth - truth.mean(axis=0, keepdims=True)
    sse = np.square(residual).sum(axis=0, dtype=np.float64)
    tss = np.square(centered).sum(axis=0, dtype=np.float64)
    need(np.all(tss > 0.0), "metric target contains a zero-variance output")
    per_output = 1.0 - sse / tss
    pooled = 1.0 - float(sse.sum(dtype=np.float64) / tss.sum(dtype=np.float64))
    equal_output = float(per_output.mean(dtype=np.float64))
    return {
        "definition": "per-output R2; pooled=1-sum_output(SSE)/sum_output(TSS); float64",
        "samples": int(truth.shape[0]),
        "outputs": int(truth.shape[1]),
        "sse_per_output": sse.tolist(),
        "tss_per_output": tss.tolist(),
        "variance_weight": (tss / tss.sum()).tolist(),
        "r2_per_output": per_output.tolist(),
        "pooled_variance_weighted_r2": pooled,
        "equal_output_mean_r2": equal_output,
        "target_sha256": array_sha256(truth),
        "prediction_sha256": array_sha256(estimate),
    }


def covariance_spectrum(target: np.ndarray) -> dict[str, Any]:
    values = np.asarray(target, dtype=np.float64)
    need(values.ndim == 2 and values.shape[0] > values.shape[1] and np.isfinite(values).all(), "spectrum needs finite [N,D]")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum(dtype=np.float64))
    need(total > 0.0 and float(np.square(eigenvalues).sum(dtype=np.float64)) > 0.0, "zero covariance spectrum")
    ratio = eigenvalues / total
    cumulative = np.cumsum(ratio)
    pr = total * total / float(np.square(eigenvalues).sum(dtype=np.float64))
    return {
        "samples": int(values.shape[0]),
        "outputs": int(values.shape[1]),
        "covariance_eigenvalues_desc": eigenvalues.tolist(),
        "variance_fraction_desc": ratio.tolist(),
        "participation_ratio": float(pr),
        "components_for_90pct": int(np.searchsorted(cumulative, 0.9, side="left") + 1),
        "per_output_variance": np.diag(covariance).tolist(),
        "covariance_sha256": array_sha256(covariance),
    }


def shared_axis_semantic_gate(
    left_names: Sequence[str],
    left_units: Sequence[str],
    right_names: Sequence[str],
    right_units: Sequence[str],
    *,
    minimum_axes: int = 2,
) -> dict[str, Any]:
    """Metadata-only physical-axis gate; numerical alignment cannot override it."""

    need(minimum_axes >= 1, "minimum shared axes must be positive")
    need(len(left_names) == len(left_units) and len(right_names) == len(right_units), "axis/unit cardinality drift")
    left = {str(name): str(unit) for name, unit in zip(left_names, left_units)}
    right = {str(name): str(unit) for name, unit in zip(right_names, right_units)}
    matches = tuple(sorted(name for name in set(left).intersection(right) if left[name] == right[name]))
    return {
        "minimum_axes": int(minimum_axes),
        "exact_name_and_unit_matches": list(matches),
        "matched_axis_count": len(matches),
        "passed": len(matches) >= minimum_axes,
    }


def _source_normalizer(sessions: Sequence[DirectRidgeSession], lag: int) -> dict[str, np.ndarray]:
    support = [session.aligned(lag, split="support") for session in sessions]
    x_mean, x_scale = _moments([row[0] for row in support])
    y_mean, y_scale = _moments([row[1] for row in support])
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}


def _fit_score_session(
    session: DirectRidgeSession,
    *,
    lag: int,
    ridge_lambda: float,
    normalizer: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y, support_indices = session.aligned(lag, split="support")
    query_x, query_y, query_indices = session.aligned(lag, split="query")
    weight, intercept = _fit_ridge(train_x, train_y, ridge_lambda=ridge_lambda, **normalizer)
    prediction = _predict_ridge(query_x, weight=weight, intercept=intercept, **normalizer)
    metrics = regression_metrics(query_y, prediction)
    metrics.update({
        "support_bins": int(train_x.shape[0]),
        "query_bins": int(query_x.shape[0]),
        "support_indices_sha256": array_sha256(support_indices),
        "query_indices_sha256": array_sha256(query_indices),
        "weight_sha256": array_sha256(weight),
        "intercept_sha256": array_sha256(intercept),
    })
    return metrics, query_y, prediction, query_indices


def _prepare_selection_statistics(
    session: DirectRidgeSession,
    *,
    lag: int,
    normalizer: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Cache exact query SSE sufficient statistics for a lag/normalizer."""

    train_x, train_y, _ = session.aligned(lag, split="support")
    query_x, query_y, _ = session.aligned(lag, split="query")
    z = (query_x - normalizer["x_mean"]) / normalizer["x_scale"]
    ones_sum = float(z.shape[0])
    sum_z = z.sum(axis=0, dtype=np.float64)
    qtq = np.empty((z.shape[1] + 1, z.shape[1] + 1), dtype=np.float64)
    qtq[:-1, :-1] = z.T @ z
    qtq[:-1, -1] = sum_z
    qtq[-1, :-1] = sum_z
    qtq[-1, -1] = ones_sum
    qty = np.concatenate((z.T @ query_y, query_y.sum(axis=0, dtype=np.float64)[None, :]), axis=0)
    sum_y2 = np.square(query_y).sum(axis=0, dtype=np.float64)
    centered = query_y - query_y.mean(axis=0, keepdims=True)
    tss = np.square(centered).sum(axis=0, dtype=np.float64)
    need(np.all(tss > 0.0), "selection query contains zero-variance output")
    return {"train_x": train_x, "train_y": train_y, "qtq": qtq, "qty": qty, "sum_y2": sum_y2, "tss": tss}


def _selection_r2(
    prepared: Mapping[str, np.ndarray],
    *,
    ridge_lambda: float,
    normalizer: Mapping[str, np.ndarray],
) -> float:
    weight, intercept = _fit_ridge(
        prepared["train_x"], prepared["train_y"], ridge_lambda=ridge_lambda, **normalizer,
    )
    slope = weight * normalizer["y_scale"][None, :]
    bias = intercept * normalizer["y_scale"] + normalizer["y_mean"]
    beta = np.concatenate((slope, bias[None, :]), axis=0)
    cross = np.sum(beta * prepared["qty"], axis=0, dtype=np.float64)
    quadratic = np.einsum("io,ij,jo->o", beta, prepared["qtq"], beta, optimize=True)
    sse = prepared["sum_y2"] - 2.0 * cross + quadratic
    # Roundoff may produce a tiny negative SSE only for a perfect synthetic fit.
    sse = np.maximum(sse, 0.0)
    value = 1.0 - float(sse.sum(dtype=np.float64) / prepared["tss"].sum(dtype=np.float64))
    need(np.isfinite(value), "selection R2 became nonfinite")
    return value


def direct_ridge_loso(
    sessions: Mapping[str, DirectRidgeSession],
    *,
    lag_grid: Sequence[int] = LAG_GRID,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> dict[str, Any]:
    """Nested four-session M1 DirectRidge.

    For each target, every candidate is scored only on the other sessions.
    Each source validation session is fitted on its own first ten trials and
    scored after its own M10 boundary.  The selected candidate is then fitted
    once to the target's first ten trials and scored on its post-M10 query.
    """

    names = tuple(sorted(sessions))
    need(len(names) >= 3, "nested DirectRidge requires at least three sessions")
    for session in sessions.values():
        session.validate()
    output_names = {sessions[name].output_names for name in names}
    need(len(output_names) == 1, "M1 sessions disagree on output order")
    candidates = tuple((int(lag), float(value)) for lag in lag_grid for value in lambda_grid)
    need(candidates and len(candidates) == len(set(candidates)), "candidate grid is empty or duplicated")
    need(all(0 <= lag <= 10 and value >= 0.0 for lag, value in candidates), "candidate grid outside frozen bounds")
    folds: dict[str, Any] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_prediction: list[np.ndarray] = []
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name)
        source_sessions = [sessions[name] for name in source_names]
        candidate_rows: list[dict[str, Any]] = []
        normalizers: dict[int, dict[str, np.ndarray]] = {}
        prepared_by_lag: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for lag, ridge_lambda in candidates:
            if lag not in normalizers:
                normalizers[lag] = _source_normalizer(source_sessions, lag)
                prepared_by_lag[lag] = {
                    name: _prepare_selection_statistics(sessions[name], lag=lag, normalizer=normalizers[lag])
                    for name in source_names
                }
            normalizer = normalizers[lag]
            source_scores = [
                _selection_r2(prepared_by_lag[lag][source_name], ridge_lambda=ridge_lambda, normalizer=normalizer)
                for source_name in source_names
            ]
            candidate_rows.append({
                "lag_bins": lag,
                "lag_ms": 20 * lag,
                "ridge_lambda_per_sample": ridge_lambda,
                "source_sessions": list(source_names),
                "source_validation_r2": source_scores,
                "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
            })
        # Candidate order is lag-major/lambda-minor.  max() therefore keeps
        # the first (smallest lag, then smallest lambda) on an exact tie.
        best_index = max(range(len(candidate_rows)), key=lambda i: (candidate_rows[i]["equal_source_session_mean_r2"], -i))
        best = candidate_rows[best_index]
        lag = int(best["lag_bins"])
        ridge_lambda = float(best["ridge_lambda_per_sample"])
        metrics, truth, prediction, _ = _fit_score_session(
            sessions[target_name], lag=lag, ridge_lambda=ridge_lambda, normalizer=normalizers[lag],
        )
        folds[target_name] = {
            "target_session": target_name,
            "source_sessions": list(source_names),
            "target_excluded_from_hyperparameter_selection": target_name not in source_names,
            "selected": best,
            "candidate_count": len(candidate_rows),
            "candidate_table_sha256": canonical_sha256(candidate_rows),
            "target_metrics": metrics,
            "target_boundary_bin": sessions[target_name].boundary,
            "input_path": sessions[target_name].input_path,
            "input_sha256": sessions[target_name].input_sha256,
        }
        pooled_truth.append(truth)
        pooled_prediction.append(prediction)
    session_scores = [float(folds[name]["target_metrics"]["pooled_variance_weighted_r2"]) for name in names]
    pooled = regression_metrics(np.concatenate(pooled_truth), np.concatenate(pooled_prediction))
    return {
        "schema": "m1_m10_directridge_nested_loso_v1",
        "protocol": {
            "support": "first 10 chronological trials; eval-valid dense 20-ms neural/EMG bins",
            "query": "strict trial 11 onward; eval-valid; same-trial causal lag only",
            "lag_convention": "X[t-lag] predicts Y[t]",
            "lag_grid_bins": list(lag_grid),
            "lambda_grid_per_sample": [float(value) for value in lambda_grid],
            "normalization": "other-session support-only moments; affine target support fit; slope-only ridge lambda*N",
            "hyperparameter_selection": "nested leave-one-session-out; target query and support excluded",
            "label_cost_disclosure": "M10 contains dense per-bin 16-D EMG labels, not ten scalar labels",
        },
        "sessions": list(names),
        "output_names": list(next(iter(output_names))),
        "folds": folds,
        "equal_session": {
            "mean_r2": float(np.mean(session_scores, dtype=np.float64)),
            "median_r2": float(np.median(session_scores)),
            "per_session_r2": dict(zip(names, session_scores)),
        },
        "pooled": pooled,
        "target_support_or_query_used_for_candidate_selection": False,
    }
