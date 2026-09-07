"""Budget-generalized M1 support/query protocol and receipt conventions.

The frozen protocol fixes support to the first ten chronological trials and the
query to strict post-M10 bins.  E2 sweeps the support budget M over {4, 10, 30}
and E1 needs an oracle support, so this module defines the unique generalization
that reproduces the frozen M10 split exactly at M=10:

    support(M) = eval-valid bins with 1 <= trial_id <= M
    query(M)   = eval-valid bins with trial_id >= M + 1

Both remain lag-0 same-trial.  At M=10 these index sets are identical to
DirectRidgeSession.aligned(0, split=...), which the tests and the runner assert
bit-exactly (index SHA and prediction SHA) before any budget cell is trusted.

DirectRidge at budget M reuses the frozen nested selection rule of
h1_m1_priority_v1.core.direct_ridge_loso verbatim (source-support normalizers,
slope-only lambda*N penalty, source validation on the other sessions, equal
source session mean, first-grid-order tie break); only the aligned arrays are
produced by the budget split.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.behavior_autoencoder_v1.core import need
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    _fit_ridge,
    _moments,
    _predict_ridge,
    _selection_r2,
    array_sha256,
    regression_metrics,
)

RESULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "behavior_manifold_v2"
BASELINE_LAMBDA_GRID: tuple[float, ...] = (1.0e-1, 3.0e-1)


def _trial_id(session: DirectRidgeSession) -> np.ndarray:
    return session.trial_id


def aligned_budget(
    session: DirectRidgeSession, budget: int, *, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag-0 same-trial bins for a chronological first-M-trial support budget."""

    need(split in {"support", "query"}, "split must be support or query")
    need(int(budget) == budget and budget >= 1, "budget must be a positive integer")
    budget = int(budget)
    trial_id = _trial_id(session)
    legal = np.asarray(session.eval_mask, dtype=bool) & (trial_id > 0)
    if split == "support":
        legal &= trial_id <= budget
    else:
        legal &= trial_id >= budget + 1
    indices = np.flatnonzero(legal).astype(np.int64)
    need(
        indices.size > session.target.shape[1] + 2,
        f"{session.name}/{split}/M{budget}: insufficient aligned bins ({indices.size})",
    )
    x = np.asarray(session.neural[indices], dtype=np.float64)
    y = np.asarray(session.target[indices], dtype=np.float64)
    need(x.shape[0] == y.shape[0] == indices.size, "budget alignment row drift")
    return x, y, indices


def full_session_arrays(session: DirectRidgeSession) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every eval-valid labelled bin (all trials); at M=10 this is exactly the
    union of the frozen M10 support and the strict post-M10 query."""

    x_support, y_support, i_support = aligned_budget(session, 10, split="support")
    x_query, y_query, i_query = aligned_budget(session, 10, split="query")
    trial_id = _trial_id(session)
    legal = np.asarray(session.eval_mask, dtype=bool) & (trial_id > 0)
    indices = np.flatnonzero(legal).astype(np.int64)
    need(
        np.array_equal(np.sort(np.concatenate((i_support, i_query))), indices),
        f"{session.name}: full session is not the disjoint union of M10 support and query",
    )
    x = np.asarray(session.neural[indices], dtype=np.float64)
    y = np.asarray(session.target[indices], dtype=np.float64)
    return x, y, indices


def budget_normalizers(
    source_sessions: Sequence[DirectRidgeSession], budget: int
) -> dict[str, np.ndarray]:
    """Source-session budget-support moments; identical to _source_normalizer at M=10."""

    support = [aligned_budget(session, budget, split="support") for session in source_sessions]
    x_mean, x_scale = _moments([row[0] for row in support])
    y_mean, y_scale = _moments([row[1] for row in support])
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}


def budget_selection_statistics(
    normalizer: Mapping[str, np.ndarray],
    support: tuple[np.ndarray, np.ndarray],
    query: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Query SSE sufficient statistics, mirroring _prepare_selection_statistics."""

    train_x, train_y = support
    query_x, query_y = query
    z = (query_x - normalizer["x_mean"]) / normalizer["x_scale"]
    ones_sum = float(z.shape[0])
    sum_z = z.sum(axis=0, dtype=np.float64)
    qtq = np.empty((z.shape[1] + 1, z.shape[1] + 1), dtype=np.float64)
    qtq[:-1, :-1] = z.T @ z
    qtq[:-1, -1] = sum_z
    qtq[-1, :-1] = sum_z
    qtq[-1, -1] = ones_sum
    qty = np.concatenate(
        (z.T @ query_y, query_y.sum(axis=0, dtype=np.float64)[None, :]), axis=0
    )
    sum_y2 = np.square(query_y).sum(axis=0, dtype=np.float64)
    centered = query_y - query_y.mean(axis=0, keepdims=True)
    tss = np.square(centered).sum(axis=0, dtype=np.float64)
    need(np.all(tss > 0.0), "selection query contains zero-variance output")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "qtq": qtq,
        "qty": qty,
        "sum_y2": sum_y2,
        "tss": tss,
    }


def fit_score_budget_ridge(
    session: DirectRidgeSession,
    normalizer: Mapping[str, np.ndarray],
    *,
    budget: int,
    ridge_lambda: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit the affine DirectRidge on the first M trials; score strict post-M."""

    train_x, train_y, support_indices = aligned_budget(session, budget, split="support")
    query_x, query_y, query_indices = aligned_budget(session, budget, split="query")
    weight, intercept = _fit_ridge(train_x, train_y, ridge_lambda=ridge_lambda, **normalizer)
    prediction = _predict_ridge(
        query_x, weight=weight, intercept=intercept, **normalizer
    )
    metrics = regression_metrics(query_y, prediction)
    metrics.update(
        {
            "support_bins": int(train_x.shape[0]),
            "query_bins": int(query_x.shape[0]),
            "support_indices_sha256": array_sha256(support_indices),
            "query_indices_sha256": array_sha256(query_indices),
            "weight_sha256": array_sha256(weight),
            "intercept_sha256": array_sha256(intercept),
        }
    )
    return metrics, query_y, prediction


def direct_ridge_budget(
    sessions: Mapping[str, DirectRidgeSession],
    budget: int,
    *,
    lambda_grid: Sequence[float] = BASELINE_LAMBDA_GRID,
) -> dict[str, Any]:
    """Nested DirectRidge at a support budget, mirroring direct_ridge_loso."""

    names = tuple(sorted(sessions))
    need(len(names) >= 3, "nested budget DirectRidge needs at least three sessions")
    for session in sessions.values():
        session.validate()
    output_names = {sessions[name].output_names for name in names}
    need(len(output_names) == 1, "sessions disagree on output order")
    lambdas = tuple(float(value) for value in lambda_grid)
    need(lambdas and len(set(lambdas)) == len(lambdas), "budget lambda grid drift")
    folds: dict[str, Any] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_prediction: list[np.ndarray] = []
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name)
        source_sessions = [sessions[name] for name in source_names]
        normalizer = budget_normalizers(source_sessions, budget)
        prepared = {
            name: budget_selection_statistics(
                normalizer,
                aligned_budget(sessions[name], budget, split="support")[:2],
                aligned_budget(sessions[name], budget, split="query")[:2],
            )
            for name in source_names
        }
        candidate_rows: list[dict[str, Any]] = []
        for ridge_lambda in lambdas:
            source_scores = [
                _selection_r2(
                    prepared[name], ridge_lambda=ridge_lambda, normalizer=normalizer
                )
                for name in source_names
            ]
            candidate_rows.append(
                {
                    "ridge_lambda_per_sample": ridge_lambda,
                    "source_sessions": list(source_names),
                    "source_validation_r2": source_scores,
                    "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
                }
            )
        best_index = max(
            range(len(candidate_rows)),
            key=lambda index: (candidate_rows[index]["equal_source_session_mean_r2"], -index),
        )
        best = candidate_rows[best_index]
        metrics, truth, prediction = fit_score_budget_ridge(
            sessions[target_name],
            normalizer,
            budget=budget,
            ridge_lambda=float(best["ridge_lambda_per_sample"]),
        )
        folds[target_name] = {
            "target_session": target_name,
            "source_sessions": list(source_names),
            "target_excluded_from_hyperparameter_selection": True,
            "selected": best,
            "candidate_count": len(candidate_rows),
            "candidate_table_sha256": json_sha256(candidate_rows),
            "target_metrics": metrics,
            "support_budget_trials": int(budget),
        }
        pooled_truth.append(truth)
        pooled_prediction.append(prediction)
    session_scores = [
        float(folds[name]["target_metrics"]["pooled_variance_weighted_r2"]) for name in names
    ]
    return {
        "schema": "m1_budget_directridge_nested_loso_v1",
        "budget_trials": int(budget),
        "query_rule": f"strict post-M{int(budget)}; support trials 1..{int(budget)}; lag 0",
        "lambda_grid_per_sample": list(lambdas),
        "sessions": list(names),
        "output_names": list(next(iter(output_names))),
        "folds": folds,
        "equal_session": {
            "mean_r2": float(np.mean(session_scores, dtype=np.float64)),
            "median_r2": float(np.median(session_scores)),
            "per_session_r2": dict(zip(names, session_scores)),
        },
        "pooled": regression_metrics(np.concatenate(pooled_truth), np.concatenate(pooled_prediction)),
        "target_support_or_query_used_for_candidate_selection": False,
    }


def latent_route_prediction(
    session: DirectRidgeSession,
    manifold: Any,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    *,
    budget: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The deployment rule once: encode support, one closed-form ridge, decode query."""

    from sua_exploration.behavior_autoencoder_v1.core import fit_affine_ridge, predict_affine_ridge

    support_x, support_y, support_indices = aligned_budget(session, budget, split="support")
    query_x, query_y, query_indices = aligned_budget(session, budget, split="query")
    latent_support = manifold.encode(support_y)
    weight, intercept = fit_affine_ridge(
        (support_x - x_mean) / x_scale, latent_support, ridge_lambda=ridge_lambda
    )
    latent_query = predict_affine_ridge((query_x - x_mean) / x_scale, weight, intercept)
    return manifold.decode(latent_query), query_y, query_indices, (weight, intercept)


def json_sha256(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(name: str, body: Mapping[str, Any]) -> str:
    """Publish one immutable JSON receipt plus canonical sha256 sidecar.

    Mirrors the lane's transactional 0444 convention (h1_m1_priority_v1
    _write_pair): temp file, fsync, chmod 0444, atomic rename, verify; any
    existing file is refused rather than overwritten.
    """

    path = RESULT_ROOT / name
    need(path.parent == RESULT_ROOT, "receipt must stay in the canonical result root")
    sidecar = path.with_name(path.name + ".sha256")
    need(
        not path.exists() and not sidecar.exists() and not path.is_symlink() and not sidecar.is_symlink(),
        f"refusing overwrite: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    for destination, contents in (
        (path, payload),
        (sidecar, f"{digest}  {path.name}\n".encode("ascii")),
    ):
        handle_fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(path.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle_fd, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    need(
        stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444,
        "receipt mode drift",
    )
    need(
        _sha_file(path) == digest and sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n",
        "receipt pair verification failed",
    )
    return digest
