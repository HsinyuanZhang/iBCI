"""Trial-level rate ridge: shared numerics for the equal-information comparator.

This module provides:
  - conditioning_report: design rank, condition number, trace(H) for [M,N] ridge
  - trial_rates_and_velocity_from_bounds: [M,N] rates + [M,2] velocity from trial bounds
  - segment_rates_and_velocity: same for reach segments (RT)
  - block_query_rates_rt: 5-bin causal window rate at eval bins
  - bootstrap_sessions: session-resampling bootstrap (copy of existing)
  - immutable write helpers (copy of existing stdlib pattern)

The ridge solver itself is NOT reimplemented — numerical.fit_ridge is reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Constants mirrored from subm_v9_f0_pv_ridge to guarantee identical standardization
FEATURE_STD_EPS = 1.0e-8
RIDGE_NORMALIZED_LAMBDA = 1.0
HISTORY_BINS = 50
BIN_SIZE_S = 0.020

EXPECTED_SESSIONS_SUA = 15
EXPECTED_SESSIONS_RT = 15
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 68_820_260_805


def conditioning_report(
    features: np.ndarray,
    targets: np.ndarray,
    normalized_lambda: float = RIDGE_NORMALIZED_LAMBDA,
) -> dict[str, Any]:
    """Report conditioning of the [M, N] standardized ridge design."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    n, p = x.shape

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < FEATURE_STD_EPS] = 1.0
    Z = (x - mean) / scale

    design_rank = int(np.linalg.matrix_rank(Z))
    sv = np.linalg.svd(Z, compute_uv=False)

    gram = np.ascontiguousarray((Z.T @ Z) / n)
    eigvals = np.linalg.eigvalsh(gram)
    reg_eigvals = eigvals + normalized_lambda
    gram_cond = float(reg_eigvals.max() / reg_eigvals.min()) if reg_eigvals.min() > 0 else float("inf")

    # Effective degrees of freedom: trace(H) = sum_j sigma_j^2 / (sigma_j^2 + n * lambda)
    sigma_sq = sv ** 2
    trace_hat = float(np.sum(sigma_sq / (sigma_sq + n * normalized_lambda)))

    return {
        "rows": int(n),
        "features": int(p),
        "ratio_rows_over_features": float(n / p) if p > 0 else float("inf"),
        "design_rank": design_rank,
        "underdetermined": bool(design_rank < p),
        "singular_values_z": [float(s) for s in sv],
        "gram_regularized_condition": gram_cond,
        "trace_hat": trace_hat,
        "lambda": float(normalized_lambda),
        "intercept_unpenalized": True,
        "n_target_dims": int(y.shape[1]),
    }


def trial_rates_and_velocity_from_bounds(
    neural: np.ndarray,
    behavior: np.ndarray,
    bounds: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute [M, N] trial-mean firing rates and [M, 2] trial-mean velocity."""
    neural = np.asarray(neural, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64)
    assert neural.ndim == 2 and behavior.ndim == 2 and behavior.shape[1] == 2
    assert neural.shape[0] == behavior.shape[0]

    rates = np.empty((len(bounds), neural.shape[1]), dtype=np.float64)
    velocity = np.empty((len(bounds), 2), dtype=np.float64)
    prev_stop = -1
    for i, (start, stop) in enumerate(bounds):
        assert int(start) >= 0 and int(stop) > int(start) and int(stop) <= neural.shape[0]
        assert int(start) >= prev_stop, f"trial bounds not chronological at index {i}"
        duration = (int(stop) - int(start)) * BIN_SIZE_S
        assert duration > 0
        rates[i] = neural[int(start):int(stop)].mean(axis=0)
        velocity[i] = behavior[int(start):int(stop)].mean(axis=0)
        prev_stop = int(stop)

    assert np.isfinite(rates).all() and np.isfinite(velocity).all()
    return np.ascontiguousarray(rates), np.ascontiguousarray(velocity)


def segment_rates_and_velocity(
    neural: np.ndarray,
    covariates: np.ndarray,
    segment_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-segment mean rates and velocity. Returns ([Mseg,N], [Mseg,2], seg_ids)."""
    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    assert neural.ndim == 2 and covariates.ndim == 2 and segment_id.ndim == 1
    assert neural.shape[0] == covariates.shape[0] == segment_id.shape[0]

    unique_ids = sorted(int(s) for s in np.unique(segment_id) if int(s) >= 0)
    rates = np.empty((len(unique_ids), neural.shape[1]), dtype=np.float64)
    velocity = np.empty((len(unique_ids), 2), dtype=np.float64)

    for i, sid in enumerate(unique_ids):
        mask = segment_id == sid
        assert mask.sum() > 0
        rates[i] = neural[mask].mean(axis=0)
        velocity[i] = covariates[mask].mean(axis=0)

    assert np.isfinite(rates).all() and np.isfinite(velocity).all()
    return np.ascontiguousarray(rates), np.ascontiguousarray(velocity), np.array(unique_ids, dtype=np.int64)


def block_query_rates_rt(
    neural: np.ndarray,
    eval_mask: np.ndarray,
    segment_id: np.ndarray,
    block_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 5-bin causal window rate at each eval bin. Returns ([Q,N], target_bins[Q])."""
    neural = np.asarray(neural, dtype=np.float64)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    assert neural.ndim == 2 and eval_mask.ndim == 1 and segment_id.ndim == 1

    T = neural.shape[0]
    eval_bins = np.flatnonzero(eval_mask)
    # Keep bins where a full causal block fits and stays within one segment
    valid = []
    for b in eval_bins:
        if b < block_bins - 1:
            continue
        seg = segment_id[b]
        if seg < 0:
            continue
        if not np.all(segment_id[b - block_bins + 1 : b + 1] == seg):
            continue
        valid.append(b)

    valid = np.array(valid, dtype=np.int64)
    assert valid.size > 0, "no valid query blocks"

    rates = np.empty((valid.size, neural.shape[1]), dtype=np.float64)
    for i, b in enumerate(valid):
        rates[i] = neural[int(b) - block_bins + 1 : int(b) + 1].mean(axis=0)

    assert np.isfinite(rates).all()
    return np.ascontiguousarray(rates), valid


def window_query_rates(
    neural: np.ndarray,
    target_bins: np.ndarray,
    window_size: int = HISTORY_BINS,
) -> np.ndarray:
    """Compute window-size causal window-mean rate ending at each target bin. Returns [Q,N]."""
    neural = np.asarray(neural, dtype=np.float64)
    target_bins = np.asarray(target_bins, dtype=np.int64)
    T, N = neural.shape

    prefix = np.empty((T + 1, N), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(neural, axis=0, dtype=np.float64, out=prefix[1:])

    rates = np.empty((target_bins.size, N), dtype=np.float64)
    for i, b in enumerate(target_bins):
        assert int(b) >= window_size - 1 and int(b) < T
        rates[i] = (prefix[int(b) + 1] - prefix[int(b) - window_size + 1]) / (window_size * BIN_SIZE_S)

    assert np.isfinite(rates).all()
    return np.ascontiguousarray(rates)


def bootstrap_sessions(
    delta: np.ndarray,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float | int]:
    n = delta.size
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(draws, dtype=np.float64)
    chunk = 10_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = delta[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {"seed": seed, "draws": draws, "lower_95": float(lower), "upper_95": float(upper)}


# ---- Immutable write helpers (copied from run_subm_v9_f0_pv_ridge_controls.py) ----

def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_regular(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
    except OSError:
        return False


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(dict(payload))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return sha256_bytes(raw)
