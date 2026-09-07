"""Data plane for FABLE TKD M2 v1 (CPU-only mirror of the frozen M2 loader).

Mirrors ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py``
loading law: the frozen exporter
``sua_exploration.evalai_t4_m2.export_t4_payload.load_frozen_model_and_data``
rebuilds the pinned champion checkpoint and hands back the SAME preprocessed
dataset objects; its Lightning model is ignored (kept in eval / no_grad).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4
from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core

from . import plan


@dataclasses.dataclass
class SessionFit:
    """M30 closed-form T4 + reliability for one session."""

    session: str
    surface: str
    raw: np.ndarray  # [96, 4] float32 = [a, c, m, b]
    rho: np.ndarray  # [96] float32 in [0, 1]
    t: np.ndarray | None = None  # [96, 4] float32, filled after authority
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)


def require(condition: bool, message: str) -> None:
    plan.require(condition, message)


def load_frozen_bundle() -> dict[str, Any]:
    """Load the pinned champion datasets; the model object is ignored.

    Returns ``{"metadata", "within", "external"}`` where within/external are
    the frozen datamodule dataset objects used verbatim by the budget screen.
    The rebuild is CPU-side by construction (checkpoint loaded with
    map_location="cpu"); device discipline (CPU-only Stage 0 vs GPU1 Stage 1
    with TF32 off) is owned by the callers/runners.
    """
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == plan.CHAMPION_CKPT_SHA256,
            "champion checkpoint drift")
    require(metadata["normalization_sha256"] == plan.CHAMPION_NORMALIZATION_SHA256,
            "champion normalization drift")
    require(model is not None and model.student is not None, "frozen model missing")
    model.student.eval()  # kept for D14 teacher targets; frozen below
    for parameter in model.student.parameters():
        parameter.requires_grad_(False)
    within = data_module.train_dataset
    external = data_module.val_heldout_dataset
    require(within is not None, "within dataset missing")
    require(external is not None, "external dataset missing")
    within_sessions = sorted(within.calib_trialized_neural_features)
    external_sessions = sorted(external.calib_trialized_neural_features)
    require(within_sessions == list(plan.HELDIN_SESSIONS),
            f"held-in roster drift: {within_sessions}")
    require(external_sessions == list(plan.EXTERNAL_SESSIONS),
            f"external roster drift: {external_sessions}")
    return {
        "metadata": metadata,
        "within": within,
        "external": external,
        "champion_model": model,  # D14 teacher (frozen eval); CPU-resident
    }


def m30_t4(dataset: Any, session: str) -> SessionFit:
    """First-30 chronological support law + ridge T4 + per-unit rho.

    Mirrors ``physical.py::_support_indices`` (budget 30: arange(30), >= 3
    finite-direction trials) and ``physical.py::_ridge_side`` (trial mean
    rates from spike sums / lengths, ``fit_ridge_t4`` with normalized lambda
    0.1).  rho is the closed-form per-unit R^2 of that same cosine fit
    (float64), clipped to [0, 1].
    """
    surface = "within" if session in plan.HELDIN_SESSIONS else "external"
    angles_all = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    require(angles_all.size >= plan.ACTIVITY_HORIZON,
            f"{session} lacks first-30 target metadata")
    selected = np.arange(plan.ACTIVITY_HORIZON, dtype=np.int64)
    usable = np.isfinite(angles_all[selected])
    require(int(usable.sum()) >= 3,
            f"{session} M30 has fewer than three directional trials")
    sums = np.asarray(dataset.calib_trial_spike_sums[session][selected], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][selected], dtype=np.float64)
    rates = sums[usable] / lengths[usable, None]
    angles = angles_all[selected][usable]
    raw, evidence = fit_ridge_t4(rates, angles, normalized_lambda=plan.RIDGE_LAMBDA)

    # rho: identical design/coefficients as fit_ridge_t4 (same float64 normal
    # equations, deterministic), y_hat strictly float64.
    design = np.column_stack((np.cos(angles), np.sin(angles), np.ones(angles.size)))
    penalty = np.diag((angles.size * plan.RIDGE_LAMBDA,
                       angles.size * plan.RIDGE_LAMBDA, 0.0))
    system = design.T @ design + penalty
    coefficients = np.linalg.solve(system, design.T @ rates)
    predicted = design @ coefficients
    residual = rates - predicted
    ss_res = np.square(residual).sum(axis=0, dtype=np.float64)
    centered = rates - rates.mean(axis=0, keepdims=True)
    ss_tot = np.square(centered).sum(axis=0, dtype=np.float64)
    safe_tot = np.where(ss_tot > 0.0, ss_tot, 1.0)
    rho64 = np.clip(1.0 - ss_res / safe_tot, 0.0, 1.0)
    rho = np.ascontiguousarray(rho64, dtype=np.float32)
    require(raw.shape == (plan.CHANNELS, 4) and rho.shape == (plan.CHANNELS,)
            and np.isfinite(raw).all() and np.isfinite(rho).all(), "M30 T4 shape drift")
    fit = SessionFit(
        session=session,
        surface=surface,
        raw=np.ascontiguousarray(raw),
        rho=rho,
        evidence={
            **evidence,
            "budget": int(selected.size),
            "selection": "chronological_first_30",
            "selected_indices": selected.tolist(),
            "selected_indices_sha256": screen_core.array_sha256(selected),
            "usable_directional_trials": int(usable.sum()),
            "rho_sha256": screen_core.array_sha256(rho),
            "raw_t4_sha256": screen_core.array_sha256(raw),
            "zero_variance_units": int((ss_tot <= 0.0).sum()),
        },
    )
    return fit


def normalize_t4(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(raw - mean) / std with the authority float32 statistics."""
    mean32 = np.asarray(mean, dtype=np.float32).reshape(1, 4)
    std32 = np.asarray(std, dtype=np.float32).reshape(1, 4)
    t = np.ascontiguousarray((np.asarray(raw, dtype=np.float32) - mean32) / std32)
    require(t.shape == (plan.CHANNELS, 4) and np.isfinite(t).all(), "normalized T4 drift")
    return t


def session_window_starts(dataset: Any, session: str) -> np.ndarray:
    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )
    require(starts.size > 0 and np.all(np.diff(starts) > 0), "query window order drift")
    return starts


def _stack_windows(
    dataset: Any,
    session: str,
    starts: np.ndarray,
    *,
    scale_targets: bool,
) -> tuple[np.ndarray, np.ndarray]:
    covariate = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    require(covariate.ndim == 2 and covariate.shape[1] == plan.OUT_DIM,
            "covariate shape drift")
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    require(neural.ndim == 2 and neural.shape[1] == plan.CHANNELS,
            "neural shape drift")
    windows = np.stack([neural[start : start + plan.WINDOW] for start in starts], axis=0)
    require(windows.shape == (starts.size, plan.WINDOW, plan.CHANNELS),
            "window stack drift")
    targets = np.stack(
        [covariate[start + plan.WINDOW - 1] for start in starts], axis=0
    )
    if scale_targets:
        targets = targets * np.float32(plan.BEHAVIOR_SCALE)
    return (
        np.ascontiguousarray(windows, dtype=np.float32),
        np.ascontiguousarray(targets, dtype=np.float32),
    )


def heldin_windows(
    dataset: Any, session: str, *, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-30 windows of a held-in session (training face).

    Window starts mirror the screen law ``select_common_post30_window_starts``
    (starts >= the session's trial-30 start); targets are
    ``covariate[start + 49] * 5.0`` (scaled training targets).
    """
    require(session in plan.HELDIN_SESSIONS, "held-in face misuse")
    starts = session_window_starts(dataset, session)
    post = screen_core.select_common_post30_window_starts(
        starts, dataset.trial_start_indices[session]
    )
    if limit is not None:
        require(int(limit) > 0, "limit must be positive")
        post = post[: int(limit)]
    windows, targets = _stack_windows(dataset, session, post, scale_targets=True)
    return post, windows, targets


def external_windows(
    dataset: Any, session: str, *, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official external query windows (all windows, raw targets).

    Targets stored UNSCALED; predictions are divided by BEHAVIOR_SCALE before
    comparison (the screen ``_score_session`` convention).
    """
    require(session in plan.EXTERNAL_SESSIONS, "external face misuse")
    starts = session_window_starts(dataset, session)
    if limit is not None:
        require(int(limit) > 0, "limit must be positive")
        starts = starts[: int(limit)]
    windows, targets = _stack_windows(dataset, session, starts, scale_targets=False)
    return starts, windows, targets


# ---------------------------------------------------------------------------
# T4 normalization authority (spec section 1).
# ---------------------------------------------------------------------------


def _authority_payload(fits: dict[str, SessionFit]) -> dict[str, Any]:
    pooled = np.concatenate(
        [fits[name].raw.astype(np.float64) for name in plan.HELDIN_SESSIONS], axis=0
    )
    require(pooled.shape == (plan.CHANNELS * len(plan.HELDIN_SESSIONS), 4),
            "pooled T4 shape drift")
    mean64 = pooled.mean(axis=0)
    std64 = pooled.std(axis=0)  # ddof = 0
    require(bool((std64 > 0).all()), "authority std degenerate")
    phi = np.arctan2(pooled[:, 1], pooled[:, 0])
    psi = np.asarray(plan.QUERY_DIRECTIONS, dtype=np.float64)
    bins = np.argmax(np.cos(phi[:, None] - psi[None, :]), axis=1)
    per_session_counts = []
    for index in range(0, pooled.shape[0], plan.CHANNELS):
        counts = np.bincount(bins[index : index + plan.CHANNELS], minlength=plan.N_QUERIES)
        per_session_counts.append(counts.astype(np.float64))
    masses = np.mean(np.stack(per_session_counts, axis=0), axis=0)
    return {
        "schema": f"{plan.SCHEMA}:t4_authority",
        "source": (
            "held-in 7 session M30 raw T4 (chronological first-30 support, "
            "fit_ridge_t4 normalized_lambda=0.1, the screen physical law); "
            "pooled per-column mean/std ddof=0 frozen for all sessions"
        ),
        "heldin_sessions": list(plan.HELDIN_SESSIONS),
        "external_sessions": list(plan.EXTERNAL_SESSIONS),
        "mean": mean64.astype(np.float32).tolist(),
        "std": std64.astype(np.float32).tolist(),
        "mean_float64": mean64.tolist(),
        "std_float64": std64.tolist(),
        "std_ddof": 0,
        "query_directions": plan.QUERY_DIRECTIONS,
        "bin_masses": masses.tolist(),
        "per_session_bin_counts": {
            name: counts.tolist()
            for name, counts in zip(plan.HELDIN_SESSIONS, per_session_counts)
        },
        "per_session_raw_t4_sha256": {
            name: fits[name].evidence["raw_t4_sha256"] for name in plan.HELDIN_SESSIONS
        },
        "per_session_rho_sha256": {
            name: fits[name].evidence["rho_sha256"] for name in plan.HELDIN_SESSIONS
        },
    }


def build_or_verify_authority(
    fits: dict[str, SessionFit], stage0_dir: Path
) -> dict[str, Any]:
    """First run: write the frozen authority receipt.  Later runs: recompute
    and require bit-equality (data or fit-law drift fails closed)."""
    require(set(fits) == set(plan.ALL_SESSIONS), "fit roster drift")
    payload = _authority_payload(fits)
    path = Path(stage0_dir) / "t4_authority.json"
    if path.exists():
        plan.verify_sidecar(path)
        import json

        existing = json.loads(path.read_text(encoding="utf-8"))
        require(existing == payload, "T4 authority drift (recomputed != sealed)")
        return existing
    plan.atomic_receipt(path, payload)
    return payload
