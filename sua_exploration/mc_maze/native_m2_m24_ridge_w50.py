"""Strict numerical and split contracts for ``native_m2_m24_ridge_w50_v1``.

This module is intentionally small and does *not* know about checkpoints,
training runs, P2/P3, or an R² endpoint.  It defines an independently
inspectable conventional baseline for the already-defined Native-M2 M24
chronological held-out replay:

* each held-out calibration file provides its own first 24 trials only;
* a row is a raw, causal 50-bin x 96-channel history (4,800 values);
* the two-dimensional velocity at the history's final bin is the target;
* support and query windows are separated by a full 50-bin history; and
* a fixed ``lambda=1`` ridge objective is solved in the *dual*.

The runner may materialize query predictions into an immutable, metrics-free
artifact after review.  R² is deliberately not calculated in the runner.  A
small TorchMetrics-1.5.1 helper lives here solely so a later, separately
reviewed scorer has a pinned implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


PROGRAM_ID = "native_m2_m24_ridge_w50_v1"
CALIBRATION_TRIALS = 24
WINDOW_BINS = 50
CHANNELS = 96
OUTPUT_DIM = 2
FEATURE_DIM = WINDOW_BINS * CHANNELS
RIDGE_NORMALIZED_LAMBDA = 1.0
FEATURE_STD_EPS = 1.0e-8

# This is the exact six-session M24 comparator.  The support/query row counts
# are not fitted or selected values: they are raw-layout facts that allow the
# preflight to detect an accidental M33, padding, or endpoint drift before a
# solver can see any velocity value.
EXPECTED_HELDOUT_LAYOUT: dict[str, dict[str, int]] = {
    "ses-2020-10-30-Run1": {
        "raw_time_bins": 2727,
        "total_trials": 43,
        "support_boundary_raw_bin": 1509,
        "support_rows": 1460,
        "query_rows": 1169,
    },
    "ses-2020-10-30-Run2": {
        "raw_time_bins": 2550,
        "total_trials": 41,
        "support_boundary_raw_bin": 1531,
        "support_rows": 1482,
        "query_rows": 970,
    },
    "ses-2020-11-18-Run1": {
        "raw_time_bins": 2517,
        "total_trials": 41,
        "support_boundary_raw_bin": 1500,
        "support_rows": 1451,
        "query_rows": 968,
    },
    "ses-2020-11-19-Run1": {
        "raw_time_bins": 3045,
        "total_trials": 43,
        "support_boundary_raw_bin": 1747,
        "support_rows": 1698,
        "query_rows": 1249,
    },
    "ses-2020-11-24-Run1": {
        "raw_time_bins": 2214,
        "total_trials": 33,
        "support_boundary_raw_bin": 1671,
        "support_rows": 1622,
        "query_rows": 494,
    },
    "ses-2020-11-24-Run2": {
        "raw_time_bins": 2350,
        "total_trials": 33,
        "support_boundary_raw_bin": 1696,
        "support_rows": 1647,
        "query_rows": 605,
    },
}
EXPECTED_HELDOUT_SESSIONS = tuple(sorted(EXPECTED_HELDOUT_LAYOUT))


class NativeM2RidgeError(RuntimeError):
    """Raised when an immutable Native-M2 M24 contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeM2RidgeError(message)


def sha256_array_int64(values: np.ndarray) -> str:
    """Hash a vector of raw target-bin indices in a platform-stable form."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _as_raw_arrays(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    eval_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate raw unpadded M2 arrays without changing their time axis."""
    neural = np.asarray(neural)
    covariates = np.asarray(covariates)
    trial_change = np.asarray(trial_change, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    require(neural.ndim == 2 and neural.shape[1] == CHANNELS, f"raw neural must be [time,{CHANNELS}]")
    require(covariates.ndim == 2 and covariates.shape[1] == OUTPUT_DIM, "raw M2 velocity must be [time,2]")
    require(
        neural.shape[0] == covariates.shape[0] == trial_change.shape[0] == eval_mask.shape[0],
        "raw neural/covariates/trial_change/eval_mask time axes disagree",
    )
    require(neural.shape[0] >= WINDOW_BINS, "raw source is shorter than W50")
    require(np.isfinite(neural).all(), "raw neural contains non-finite values")
    # ``covariates`` is deliberately not inspected beyond shape here.  The
    # preflight is a source/split audit, not a target-value or metric audit.
    return neural, covariates, trial_change, eval_mask


@dataclass(frozen=True)
class ChronologicalWindowLayout:
    """Raw target-bin indices for one immutable M24 support/query split."""

    session_id: str
    raw_time_bins: int
    total_trials: int
    support_boundary_raw_bin: int
    support_target_bins: np.ndarray
    query_target_bins: np.ndarray

    @property
    def query_minimum_window_start_raw_bin(self) -> int:
        return self.support_boundary_raw_bin

    @property
    def query_minimum_window_start_padded_bin(self) -> int:
        return self.support_boundary_raw_bin + WINDOW_BINS - 1

    def as_audit_dict(self) -> dict[str, Any]:
        support = self.support_target_bins
        query = self.query_target_bins
        return {
            "session": self.session_id,
            "raw_time_bins": int(self.raw_time_bins),
            "total_trials": int(self.total_trials),
            "channels": CHANNELS,
            "output_dim": OUTPUT_DIM,
            "calibration_trials": CALIBRATION_TRIALS,
            "window_size": WINDOW_BINS,
            "raw_history_only": True,
            "support_boundary_raw_bin": int(self.support_boundary_raw_bin),
            "support_target_bin_min": int(support.min()),
            "support_target_bin_max": int(support.max()),
            "support_rows": int(support.size),
            "support_target_bins_sha256": sha256_array_int64(support),
            "query_target_bin_min": int(query.min()),
            "query_target_bin_max": int(query.max()),
            "query_rows": int(query.size),
            "query_target_bins_sha256": sha256_array_int64(query),
            "raw_query_start_bin": int(self.support_boundary_raw_bin),
            "minimum_window_start_padded_bin": self.query_minimum_window_start_padded_bin,
            "full_window_disjoint": True,
        }


def chronological_m24_layout(
    session_id: str,
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    eval_mask: np.ndarray,
) -> ChronologicalWindowLayout:
    """Build raw W50 support/query target indices for one held-out M2 file.

    The first raw W50 support target is bin 49, rather than a left-zero-padded
    bin 0.  Query target bin ``b+49`` is likewise the first legal row, where
    ``b`` is the start of trial 25.  Thus a query window begins exactly at the
    raw support boundary and cannot read any support neural sample.
    """
    neural, _covariates, trial_change, eval_mask = _as_raw_arrays(
        neural, covariates, trial_change, eval_mask
    )
    starts = np.flatnonzero(trial_change)
    require(starts.size > CALIBRATION_TRIALS, f"{session_id}: no query trial after first {CALIBRATION_TRIALS}")
    require(int(starts[0]) == 0, f"{session_id}: raw trial 0 must start at bin 0")
    require(np.all(np.diff(starts) > 0), f"{session_id}: trial boundaries are not strictly increasing")
    boundary = int(starts[CALIBRATION_TRIALS])
    raw_bins = np.arange(neural.shape[0], dtype=np.int64)
    support = raw_bins[
        (raw_bins >= WINDOW_BINS - 1)
        & (raw_bins < boundary)
        & eval_mask
    ]
    query = raw_bins[(raw_bins >= boundary + WINDOW_BINS - 1) & eval_mask]
    require(support.size >= 3, f"{session_id}: fewer than three raw W50 support rows")
    require(query.size > 0, f"{session_id}: zero full-history query rows after chronological support")
    require(int(support.max()) < boundary, f"{session_id}: support target leaks past trial 24")
    require(int(query.min()) - WINDOW_BINS + 1 == boundary, f"{session_id}: query history boundary drift")
    require(int(query.min()) > int(support.max()), f"{session_id}: query target is not post-support")
    return ChronologicalWindowLayout(
        session_id=str(session_id),
        raw_time_bins=int(neural.shape[0]),
        total_trials=int(starts.size),
        support_boundary_raw_bin=boundary,
        support_target_bins=np.ascontiguousarray(support, dtype=np.int64),
        query_target_bins=np.ascontiguousarray(query, dtype=np.int64),
    )


def validate_expected_native_m2_layout(layout: ChronologicalWindowLayout) -> None:
    """Fail closed unless a source file is one exact member of the six-session grid."""
    expected = EXPECTED_HELDOUT_LAYOUT.get(layout.session_id)
    require(expected is not None, f"unexpected Native-M2 held-out session: {layout.session_id}")
    observed = {
        "raw_time_bins": layout.raw_time_bins,
        "total_trials": layout.total_trials,
        "support_boundary_raw_bin": layout.support_boundary_raw_bin,
        "support_rows": int(layout.support_target_bins.size),
        "query_rows": int(layout.query_target_bins.size),
    }
    mismatch = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    require(not mismatch, f"{layout.session_id}: strict M24 source layout drift: {mismatch}")


def validate_all_expected_layouts(layouts: Sequence[ChronologicalWindowLayout]) -> None:
    observed = {layout.session_id for layout in layouts}
    require(observed == set(EXPECTED_HELDOUT_SESSIONS), f"expected exact six held-out sessions, found {sorted(observed)}")
    require(len(layouts) == len(observed), "duplicate held-out session layout")
    for layout in layouts:
        validate_expected_native_m2_layout(layout)


def validate_reference_split_manifest(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Validate the existing strict M24 K4/KS4/F0/T4 split/query manifest.

    The method accepts any completed strict M24 arm's split manifest, because
    all four arms must share the same support/query layout.  It does not read
    metrics, checkpoints, or scores.
    """
    require(manifest.get("heldout_evaluated_in_fit") is False, "reference manifest evaluated held-out data during fit")
    require(manifest.get("heldout_evaluated_in_test") is True, "reference manifest is not a held-out test replay")
    require(manifest.get("query_start_trial") == CALIBRATION_TRIALS, "reference manifest is not M24")
    audit = manifest.get("heldout_query_window_audit")
    require(isinstance(audit, Mapping), "reference manifest lacks heldout_query_window_audit")
    require(set(audit) == set(EXPECTED_HELDOUT_SESSIONS), "reference manifest does not cover exact six held-out sessions")
    for session in EXPECTED_HELDOUT_SESSIONS:
        row = audit[session]
        require(isinstance(row, Mapping), f"{session}: malformed query audit")
        expected = EXPECTED_HELDOUT_LAYOUT[session]
        expected_values = {
            "support_trials": CALIBRATION_TRIALS,
            "query_start_trial": CALIBRATION_TRIALS,
            "window_size": WINDOW_BINS,
            "full_window_disjoint": True,
            "raw_query_start_bin": expected["support_boundary_raw_bin"],
            "minimum_window_start_padded_bin": expected["support_boundary_raw_bin"] + WINDOW_BINS - 1,
            "eligible_windows": expected["query_rows"],
        }
        mismatch = {
            key: {"expected": value, "observed": row.get(key)}
            for key, value in expected_values.items()
            if row.get(key) != value
        }
        require(not mismatch, f"{session}: strict M24 reference query layout drift: {mismatch}")
        require(int(row.get("query_trials", 0)) > 0, f"{session}: reference manifest marks a six-session M24 query ineligible")

    # If the supplied comparator is a K4/KS4 arm, bind its raw estimator too.
    # F0/T4 manifests legitimately lack this field and still provide the same
    # query-boundary provenance.
    estimator = manifest.get("k4_estimator")
    if estimator is not None:
        require(isinstance(estimator, Mapping), "reference k4_estimator is malformed")
        require(estimator.get("calibration_trials") == CALIBRATION_TRIALS, "reference K4 estimator is not M24")
        k4_audit = manifest.get("heldout_k4_calibration_audit")
        require(isinstance(k4_audit, Mapping) and set(k4_audit) == set(EXPECTED_HELDOUT_SESSIONS), "reference K4 manifest lacks exact six-session calibration audit")
        for session, row in k4_audit.items():
            require(isinstance(row, Mapping), f"{session}: malformed K4 audit")
            require(row.get("calibration_trials") == CALIBRATION_TRIALS, f"{session}: K4 calibration count drift")
            require(row.get("design_rank") == 3, f"{session}: K4 rank drift")
    return {str(session): audit[session] for session in EXPECTED_HELDOUT_SESSIONS}


def validate_layouts_against_reference_manifest(
    layouts: Sequence[ChronologicalWindowLayout],
    manifest: Mapping[str, Any],
) -> None:
    """Bind raw source records to an independently completed M24 manifest."""
    validate_all_expected_layouts(layouts)
    validate_selected_layouts_against_reference_manifest(layouts, manifest)


def validate_selected_layouts_against_reference_manifest(
    layouts: Sequence[ChronologicalWindowLayout],
    manifest: Mapping[str, Any],
) -> None:
    """Validate one nonempty session shard against the globally six-way split."""
    audit = validate_reference_split_manifest(manifest)
    require(len(layouts) > 0, "empty M24 session shard")
    observed = [layout.session_id for layout in layouts]
    require(len(set(observed)) == len(observed), "duplicate session in M24 session shard")
    for layout in layouts:
        validate_expected_native_m2_layout(layout)
        row = audit[layout.session_id]
        comparisons = {
            "raw_query_start_bin": layout.support_boundary_raw_bin,
            "minimum_window_start_padded_bin": layout.query_minimum_window_start_padded_bin,
            "eligible_windows": int(layout.query_target_bins.size),
        }
        mismatch = {
            key: {"reference": row.get(key), "source": value}
            for key, value in comparisons.items()
            if row.get(key) != value
        }
        require(not mismatch, f"{layout.session_id}: source/reference split layout disagreement: {mismatch}")


def materialize_raw_w50_features(neural: np.ndarray, target_bins: np.ndarray) -> np.ndarray:
    """Materialize chronological raw [rows, 50*96] histories without padding."""
    neural = np.asarray(neural, dtype=np.float32)
    targets = np.ascontiguousarray(np.asarray(target_bins, dtype=np.int64))
    require(neural.ndim == 2 and neural.shape[1] == CHANNELS, f"neural must be [time,{CHANNELS}]")
    require(targets.ndim == 1 and targets.size > 0, "target bins must be a nonempty vector")
    require(int(targets.min()) >= WINDOW_BINS - 1, "raw W50 feature requires 49 prior raw bins")
    require(int(targets.max()) < neural.shape[0], "target bin reaches outside raw neural")
    offsets = np.arange(-(WINDOW_BINS - 1), 1, dtype=np.int64)
    features = neural[targets[:, None] + offsets[None, :]].reshape(targets.size, FEATURE_DIM)
    features = np.ascontiguousarray(features, dtype=np.float32)
    require(np.isfinite(features).all(), "raw W50 features contain non-finite values")
    return features


def velocity_targets_at_bins(covariates: np.ndarray, target_bins: np.ndarray) -> np.ndarray:
    """Return the raw 2-D velocity target at each causal history endpoint."""
    covariates = np.asarray(covariates, dtype=np.float32)
    targets = np.ascontiguousarray(np.asarray(target_bins, dtype=np.int64))
    require(covariates.ndim == 2 and covariates.shape[1] == OUTPUT_DIM, "M2 target must be [time,2]")
    require(targets.ndim == 1 and targets.size > 0 and int(targets.min()) >= 0, "invalid target bins")
    require(int(targets.max()) < covariates.shape[0], "target bin reaches outside raw velocity")
    result = np.ascontiguousarray(covariates[targets], dtype=np.float32)
    require(np.isfinite(result).all(), "velocity target contains non-finite values")
    return result


@dataclass(frozen=True)
class DualRidgeReadout:
    """Support-only normalized dual-ridge parameters before raw compilation."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    standardized_weights: np.ndarray
    normalized_lambda: float
    support_rows: int
    feature_dim: int


@dataclass(frozen=True)
class CompiledRawRidgeReadout:
    """Streaming form: raw W50 features times weight plus intercept."""

    raw_weights: np.ndarray
    intercept: np.ndarray


def fit_dual_ridge_w50(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    *,
    normalized_lambda: float = RIDGE_NORMALIZED_LAMBDA,
    device: str = "cpu",
) -> DualRidgeReadout:
    """Fit a fixed-lambda ridge readout from support rows only using the dual.

    Objective (after support-only feature standardization) is

    ``mean(||Y - Z W - b||²) + lambda ||W||²``.

    The target mean is the unpenalized intercept.  No query array, metric, or
    validation criterion is accepted by this function, which makes accidental
    query-dependent tuning structurally impossible at this layer.
    """
    x = np.asarray(support_features, dtype=np.float64)
    y = np.asarray(support_targets, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == FEATURE_DIM, f"support features must be [rows,{FEATURE_DIM}]")
    require(y.ndim == 2 and y.shape == (x.shape[0], OUTPUT_DIM), "support targets must be [same rows,2]")
    require(x.shape[0] >= 3, "dual ridge needs at least three support rows")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "dual ridge support input is non-finite")
    require(math.isfinite(normalized_lambda) and normalized_lambda > 0.0, "normalized lambda must be finite and positive")

    if device == "cpu":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < FEATURE_STD_EPS] = 1.0
        z = (x - mean) / scale
        target_mean = y.mean(axis=0)
        centered = y - target_mean
        n_rows = int(z.shape[0])

        # Dual system: (Z Zᵀ / n + λI) A = Y-b, W = Zᵀ A / n.  It avoids
        # constructing a 4,800 x 4,800 primal Gram matrix for each session.
        dual_gram = (z @ z.T) / float(n_rows)
        dual_gram.flat[:: n_rows + 1] += float(normalized_lambda)
        try:
            lower = np.linalg.cholesky(dual_gram)
            alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, centered))
        except np.linalg.LinAlgError as exc:
            raise NativeM2RidgeError("dual ridge Cholesky failed") from exc
        weights = (z.T @ alpha) / float(n_rows)
    else:
        require(device.startswith("cuda:"), "dual ridge device must be cpu or cuda:<index>")
        # The deployment representation is FP32.  GPU solving therefore uses
        # FP32 explicitly, avoids TF32, and never silently changes the frozen
        # lambda/objective.  CPU remains the numerically higher-precision
        # reference path for a later reproducibility spot-check.
        import torch

        require(torch.cuda.is_available(), f"CUDA dual ridge requested but unavailable: {device}")
        runtime_device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            tx = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=runtime_device)
            ty = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), device=runtime_device)
            tmean = tx.mean(dim=0)
            tscale = tx.std(dim=0, correction=0)
            tscale = torch.where(tscale < FEATURE_STD_EPS, torch.ones_like(tscale), tscale)
            tz = (tx - tmean) / tscale
            tymean = ty.mean(dim=0)
            tcentered = ty - tymean
            n_rows = int(tz.shape[0])
            tdual = (tz @ tz.transpose(0, 1)) / float(n_rows)
            tdual.diagonal().add_(float(normalized_lambda))
            lower, info = torch.linalg.cholesky_ex(tdual, check_errors=False)
            require(int(info.max().item()) == 0, "CUDA dual ridge Cholesky failed")
            alpha = torch.cholesky_solve(tcentered, lower)
            tweights = (tz.transpose(0, 1) @ alpha) / float(n_rows)
            require(bool(torch.isfinite(tweights).all().item()), "CUDA dual ridge produced non-finite weights")
            mean = tmean.detach().cpu().numpy().astype(np.float64, copy=False)
            scale = tscale.detach().cpu().numpy().astype(np.float64, copy=False)
            target_mean = tymean.detach().cpu().numpy().astype(np.float64, copy=False)
            weights = tweights.detach().cpu().numpy().astype(np.float64, copy=False)
        torch.cuda.synchronize(runtime_device)
    require(np.isfinite(weights).all(), "dual ridge produced non-finite weights")
    return DualRidgeReadout(
        feature_mean=np.ascontiguousarray(mean, dtype=np.float32),
        feature_scale=np.ascontiguousarray(scale, dtype=np.float32),
        target_mean=np.ascontiguousarray(target_mean, dtype=np.float32),
        standardized_weights=np.ascontiguousarray(weights, dtype=np.float32),
        normalized_lambda=float(normalized_lambda),
        support_rows=n_rows,
        feature_dim=FEATURE_DIM,
    )


def compile_raw_ridge(readout: DualRidgeReadout) -> CompiledRawRidgeReadout:
    """Fold support normalization into the linear weights for streaming use."""
    require(readout.feature_mean.shape == (FEATURE_DIM,), "ridge feature mean shape drift")
    require(readout.feature_scale.shape == (FEATURE_DIM,), "ridge feature scale shape drift")
    require(readout.target_mean.shape == (OUTPUT_DIM,), "ridge target mean shape drift")
    require(readout.standardized_weights.shape == (FEATURE_DIM, OUTPUT_DIM), "ridge weight shape drift")
    raw_weights = readout.standardized_weights / readout.feature_scale[:, None]
    intercept = readout.target_mean - readout.feature_mean @ raw_weights
    require(np.isfinite(raw_weights).all() and np.isfinite(intercept).all(), "raw ridge compilation is non-finite")
    return CompiledRawRidgeReadout(
        raw_weights=np.ascontiguousarray(raw_weights, dtype=np.float32),
        intercept=np.ascontiguousarray(intercept, dtype=np.float32),
    )


def predict_compiled_raw_ridge(features: np.ndarray, readout: CompiledRawRidgeReadout) -> np.ndarray:
    """Apply a frozen compiled readout; this function has no scoring logic."""
    x = np.asarray(features, dtype=np.float32)
    require(x.ndim == 2 and x.shape[1] == FEATURE_DIM, f"features must be [rows,{FEATURE_DIM}]")
    require(readout.raw_weights.shape == (FEATURE_DIM, OUTPUT_DIM), "compiled raw weight shape drift")
    require(readout.intercept.shape == (OUTPUT_DIM,), "compiled raw intercept shape drift")
    prediction = x @ readout.raw_weights + readout.intercept
    require(np.isfinite(prediction).all(), "compiled ridge prediction is non-finite")
    return np.ascontiguousarray(prediction, dtype=np.float32)


def compiled_streaming_profile() -> dict[str, int | str | float]:
    """Exact FP32 online parameter/state/MAC profile for the W50 reference."""
    weights_bytes = FEATURE_DIM * OUTPUT_DIM * np.dtype(np.float32).itemsize
    intercept_bytes = OUTPUT_DIM * np.dtype(np.float32).itemsize
    history_bytes = WINDOW_BINS * CHANNELS * np.dtype(np.float32).itemsize
    macs_per_output = FEATURE_DIM * OUTPUT_DIM
    return {
        "representation": "compiled_raw_fp32_w50x96_to_velocity2",
        "history_bins": WINDOW_BINS,
        "channels": CHANNELS,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
        "online_weight_bytes": weights_bytes + intercept_bytes,
        "online_history_bytes": history_bytes,
        "online_total_state_bytes": weights_bytes + intercept_bytes + history_bytes,
        "macs_per_output": macs_per_output,
        "macs_per_second_at_50hz": macs_per_output * 50,
    }


def score_r2_variance_weighted_torchmetrics151(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pinned future scorer primitive; not called by the preflight/forward CLI.

    The explicit version check prevents a later evaluator from silently
    changing the definition of multi-output R².  Calling this helper is a
    *separate scoring action* and must not be folded into fitting or artifact
    generation.
    """
    import torch
    import torchmetrics
    from torchmetrics.regression import R2Score

    require(torchmetrics.__version__ == "1.5.1", f"TorchMetrics 1.5.1 required, found {torchmetrics.__version__}")
    pred = torch.as_tensor(np.asarray(predictions, dtype=np.float32))
    target = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    require(pred.ndim == target.ndim == 2 and pred.shape == target.shape and pred.shape[1] == OUTPUT_DIM, "R² input shape drift")
    require(pred.shape[0] >= 2 and bool(torch.isfinite(pred).all()) and bool(torch.isfinite(target).all()), "R² input is invalid")
    metric = R2Score(multioutput="variance_weighted")
    value = float(metric(pred, target).item())
    require(math.isfinite(value), "TorchMetrics produced non-finite R²")
    return value
