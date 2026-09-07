"""Core affine-diagnostic math (EXECUTION_GUIDANCE §8.1).  All fits are
closed-form float64 least squares; every function is deterministic.

Conventions
-----------
* ``pred``/``target`` are ``[n, 2]`` float64 arrays of governing-bin rows.
* The correction acts as ``pred_new = pred @ A.T + b`` with
  ``A = [[A00, A01], [A10, A11]]``, ``b = [b0, b1]``; the six-vector is the
  flat ``[A00, A01, A10, A11, b0, b1]``.
* R² is the house variance-weighted law (float64 path), identical to the
  governing deployment metric.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.tfpd_lane import matched_scorer


class AffineDiagnosticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AffineDiagnosticError(message)


def fit_affine(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares six-vector of the affine mapping pred -> target."""
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1] == 2 and x.shape == y.shape and x.shape[0] >= 4,
             "affine fit needs [n>=4, 2] paired rows")
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = weights[:2, :].T
    b = weights[2, :]
    return np.concatenate([a.reshape(-1), b])


def apply_affine(pred: np.ndarray, six: Sequence[float]) -> np.ndarray:
    vector = np.asarray(six, dtype=np.float64)
    _require(vector.shape == (6,), "six-vector shape drift")
    a = vector[:4].reshape(2, 2)
    b = vector[4:]
    return np.asarray(pred, dtype=np.float64) @ a.T + b


def r2_float64(pred: np.ndarray, target: np.ndarray) -> float:
    """House variance-weighted R² on the float64 path."""
    import torch

    return matched_scorer.session_r2(
        torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(target, dtype=np.float64)),
    )


def six_vector_payload(six: Sequence[float]) -> Mapping[str, float]:
    vector = np.asarray(six, dtype=np.float64)
    keys = ("A00", "A01", "A10", "A11", "b0", "b1")
    return {key: float(value) for key, value in zip(keys, vector, strict=True)}


def dispersion_stats(six_vectors: Sequence[np.ndarray]) -> Mapping[str, float]:
    """Dispersion of session-specific corrections across one cell (§8.1)."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in six_vectors], axis=0)
    _require(stack.ndim == 2 and stack.shape[1] == 6 and stack.shape[0] >= 2,
             "dispersion needs >= 2 six-vectors")
    centered = stack - stack.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(stack.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    leading = float(eigenvalues[-1])
    trace = float(eigenvalues.sum())
    distances = np.linalg.norm(centered, axis=1)
    return {
        "n_sessions": int(stack.shape[0]),
        "covariance_trace": trace,
        "leading_eigenvalue": leading,
        "first_pc_explained_fraction": float(leading / trace) if trace > 0 else 0.0,
        "median_distance_to_mean_six_vector": float(np.median(distances)),
        "mean_six_vector": six_vector_payload(stack.mean(axis=0)),
    }


def full_opportunity_rows(
    rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """Per-session own-affine gain.  Target-label leakage: oracle rows only."""
    out: list[dict[str, object]] = []
    for session, (pred, target) in rows.items():
        raw = r2_float64(pred, target)
        six = fit_affine(pred, target)
        corrected = r2_float64(apply_affine(pred, six), target)
        out.append({
            "session": session,
            "raw_r2": raw,
            "own_affine_r2": corrected,
            "gain": corrected - raw,
            "six_vector": six_vector_payload(six),
            "target_label_leakage": True,
            "checkpoint_selection_eligible": False,
            "deployment_eligible": False,
        })
    return out


def loo_r0_rows(
    rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Leave-one-out source-mean correction (r=0): zero target-label fitting
    on the held session; the applied correction is the mean six-vector of the
    OTHER sessions of the same cell."""
    sessions = list(rows)
    _require(len(sessions) >= 3, "loo r0 needs >= 3 sessions in a cell")
    per_session: list[dict[str, object]] = []
    deltas: list[float] = []
    for held in sessions:
        pred, target = rows[held]
        others = [fit_affine(*rows[name]) for name in sessions if name != held]
        mean_six = np.mean(np.stack(others, axis=0), axis=0)
        raw = r2_float64(pred, target)
        corrected = r2_float64(apply_affine(pred, mean_six), target)
        deltas.append(corrected - raw)
        per_session.append({
            "session": held,
            "raw_r2": raw,
            "source_mean_affine_r2": corrected,
            "r0_gain": corrected - raw,
            "applied_six_vector": six_vector_payload(mean_six),
            "target_label_leakage": False,
            "checkpoint_selection_eligible": False,
            "deployment_eligible": False,
        })
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "sessions": per_session,
        "equal_session_mean_r0_gain": float(deltas_arr.mean()),
        "n_positive": int((deltas_arr > 0).sum()),
        "n_sessions": int(deltas_arr.size),
        "note": (
            "r=0: the mean correction of the other sessions of the cell is "
            "applied unchanged; the held session's own labels are never read "
            "by the fit"
        ),
    }


def cross_surface_r0_rows(
    source_rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
    held_rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> Mapping[str, object]:
    """The deployable r0 diagnostic (§8.1): the mean correction of the SOURCE
    sessions applied unchanged to every held-surface session.

    No held-surface label is ever read by the fit; the applied six-vector is
    one fixed vector per cell.
    """
    _require(len(source_rows) >= 2 and len(held_rows) >= 1,
             "cross-surface r0 needs >= 2 source sessions and >= 1 held session")
    mean_six = np.mean(
        np.stack([fit_affine(*rows) for rows in source_rows.values()], axis=0),
        axis=0,
    )
    per_session: list[dict[str, object]] = []
    deltas: list[float] = []
    for session, (pred, target) in held_rows.items():
        raw = r2_float64(pred, target)
        corrected = r2_float64(apply_affine(pred, mean_six), target)
        deltas.append(corrected - raw)
        per_session.append({
            "session": session,
            "raw_r2": raw,
            "source_mean_affine_r2": corrected,
            "r0_gain": corrected - raw,
            "target_label_leakage": False,
            "checkpoint_selection_eligible": False,
            "deployment_eligible": False,
        })
    return {
        "applied_six_vector": six_vector_payload(mean_six),
        "source_sessions": list(source_rows),
        "held_sessions": per_session,
        "equal_session_mean_r0_gain": float(np.mean(deltas)),
        "n_positive": int((np.asarray(deltas) > 0).sum()),
        "n_sessions": int(len(deltas)),
        "note": (
            "one fixed source-mean correction per cell; held-surface labels "
            "are read only by the scoring metric, never by the fit"
        ),
    }


def paired_session_stats(deltas: Sequence[float], *, seed: int = 42,
                         n_boot: int = 10000) -> Mapping[str, object]:
    """Fixed-seed session bootstrap over paired deltas (house law)."""
    array = np.asarray(deltas, dtype=np.float64)
    _require(array.size >= 2, "paired stats need >= 2 sessions")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    boots = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "n_positive": int((array > 0).sum()),
        "n_total": int(array.size),
        "bootstrap95_lower": float(np.quantile(boots, 0.025)),
        "bootstrap95_upper": float(np.quantile(boots, 0.975)),
        "bootstrap_seed": seed,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None
