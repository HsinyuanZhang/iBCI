"""P1 Stage A: source-only estimator A vs B with rate-MSE and linear (PV) R².

Uses only chronological train sessions from the frozen 27/6/6 manifest. Never opens
val/test NWBs for spike/behavior/trial content. Within each train session:

  support = first M rewarded trials
  eval    = next EVAL_TRIALS rewarded trials (chronologically after support)

Metrics (both estimators share the same finite-angle mask on support):
  - held-out rate-MSE of the cosine model on eval trials (continuous θ)
  - population-vector decode R² on eval trials (preferred direction from fit;
    2D gain+bias fit on support only)

CPU-only. No gradients. No GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO

from carrier_perf.p1_estimators import (
    TuningFit,
    compare_estimators_on_arrays,
    fit_estimator_a_bin_means,
    fit_estimator_b_per_trial,
    shared_finite_mask,
)
from carrier_perf.protocol import mean, sample_std

SCHEMA = "carrier_perf_p1_stage_a_source_audit_v1"
DEFAULT_MS: tuple[int, ...] = (10, 20, 30, 40, 50)
EVAL_TRIALS = 30
BIN_SIZE_MS = 20
WINDOW_SIZE = 50
TRIAL_RESULT_FILTER = "R"
RIDGE_ALPHAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_sessions(manifest_path: Path, data_root: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest["session_splits"]["train"])
    # Fail closed if caller accidentally passes a manifest that also wants test opened.
    forbidden = set(manifest["session_splits"].get("test", []))
    out: dict[str, Path] = {}
    base = (data_root / "sub-C").resolve()
    for name in names:
        if name in forbidden:
            raise RuntimeError(f"refusing train list entry that is also a test session: {name}")
        path = (base / f"{name}_behavior+ecephys.nwb").resolve()
        if path.parent != base or not path.is_file():
            raise FileNotFoundError(path)
        out[name] = path
    return out


def _trial_mean_velocity(nwb_path: Path, trials: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Mean cursor_vel in each trial window; shape [T, 2]."""
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        times = np.asarray(vel_series.timestamps[:], dtype=np.float64)
        vel = np.asarray(vel_series.data[:], dtype=np.float64)
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"unexpected cursor_vel shape {vel.shape} in {nwb_path}")
    out = np.zeros((len(trials), 2), dtype=np.float64)
    for i, trial in enumerate(trials):
        start = float(trial["start_time"])
        stop = float(trial["stop_time"])
        mask = (times >= start) & (times < stop)
        if not np.any(mask):
            out[i] = np.nan
        else:
            out[i] = vel[mask, :2].mean(axis=0)
    return out


def _predict_rates(fit: TuningFit, angles: np.ndarray) -> np.ndarray:
    """Broadcast scalar unit fit to trial angles -> rates [T]. Used per unit."""
    theta = np.asarray(angles, dtype=np.float64)
    return fit.b + fit.a * np.cos(theta) + fit.c * np.sin(theta)


def _fit_units(
    estimator: str,
    angles: np.ndarray,
    rates: np.ndarray,
) -> list[TuningFit]:
    """Fit one TuningFit per unit. rates: [N, M]."""
    mask = shared_finite_mask(angles, rates[0])  # angle mask shared; rates may still be finite
    # Angle finiteness only — rates can be zero.
    angle_mask = np.isfinite(angles)
    fits: list[TuningFit] = []
    for unit_idx in range(rates.shape[0]):
        unit_rates = rates[unit_idx]
        unit_mask = angle_mask & np.isfinite(unit_rates)
        if estimator == "A":
            fits.append(fit_estimator_a_bin_means(angles, unit_rates, mask=unit_mask))
        elif estimator == "B":
            fits.append(fit_estimator_b_per_trial(angles, unit_rates, strict_rank=True, mask=unit_mask))
        else:
            raise ValueError(estimator)
    return fits


def _rate_mse(fits: Sequence[TuningFit], angles: np.ndarray, rates: np.ndarray) -> float:
    """Mean squared error over finite eval angles and all units with ok fits."""
    angle_ok = np.isfinite(angles)
    if not np.any(angle_ok):
        return float("nan")
    errors = []
    for unit_idx, fit in enumerate(fits):
        if fit.status != "ok":
            continue
        pred = _predict_rates(fit, angles[angle_ok])
        actual = rates[unit_idx, angle_ok]
        errors.append(np.mean((pred - actual) ** 2))
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def _preferred_dirs(fits: Sequence[TuningFit]) -> np.ndarray:
    """Unit preferred-direction unit vectors [N, 2]; zeros for undefined fits."""
    out = np.zeros((len(fits), 2), dtype=np.float64)
    for i, fit in enumerate(fits):
        if fit.status != "ok" or fit.m <= 0:
            continue
        out[i, 0] = fit.a / fit.m
        out[i, 1] = fit.c / fit.m
    return out


def _population_vector(rates: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """rates [N,T], dirs [N,2] -> PV [T,2]."""
    return (rates.T @ dirs) / max(1, dirs.shape[0])


def _fit_affine_2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit y ~= x @ A + b with least squares; x,y [T,2]."""
    usable = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x_u = x[usable]
    y_u = y[usable]
    if x_u.shape[0] < 3:
        raise ValueError("need >=3 finite trials for affine fit")
    design = np.concatenate([x_u, np.ones((x_u.shape[0], 1))], axis=1)  # [T,3]
    # Solve for each output dim.
    coef, _, _, _ = np.linalg.lstsq(design, y_u, rcond=None)  # [3,2]
    A = coef[:2, :]  # [2,2]
    b = coef[2, :]  # [2]
    return A, b


def _apply_affine(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ A + b


def _r2_variance_weighted(pred: np.ndarray, target: np.ndarray) -> float:
    usable = np.isfinite(pred).all(axis=1) & np.isfinite(target).all(axis=1)
    p = pred[usable]
    t = target[usable]
    if t.shape[0] < 2:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    t_mean = t.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum((t - t_mean) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _pv_decode_r2(
    fits: Sequence[TuningFit],
    support_rates: np.ndarray,
    support_vel: np.ndarray,
    eval_rates: np.ndarray,
    eval_vel: np.ndarray,
) -> float:
    dirs = _preferred_dirs(fits)
    if not np.any(np.linalg.norm(dirs, axis=1) > 0):
        return float("nan")
    pv_s = _population_vector(support_rates, dirs)
    pv_e = _population_vector(eval_rates, dirs)
    try:
        A, b = _fit_affine_2d(pv_s, support_vel)
    except ValueError:
        return float("nan")
    pred = _apply_affine(pv_e, A, b)
    return _r2_variance_weighted(pred, eval_vel)


def audit_session(
    nwb_path: Path,
    *,
    session_name: str,
    ms: Sequence[int],
    eval_trials: int = EVAL_TRIALS,
) -> dict[str, Any]:
    # Local imports keep library importable without sua path until execute.
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix

    trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE,
        trial_result_filter=TRIAL_RESULT_FILTER,
    )
    max_m = max(ms)
    need = max_m + eval_trials
    if len(trials) < need:
        raise ValueError(
            f"{session_name}: only {len(trials)} rewarded trials, need {need} for max M+eval"
        )
    # Load rates once for the prefix we need.
    prefix = trials[:need]
    rates, n_units = _pool_trial_rate_matrix(nwb_path, prefix)  # [N, need]
    angles = np.asarray(
        [t["target_dir"] if t["target_dir"] is not None else np.nan for t in prefix],
        dtype=np.float64,
    )
    vel = _trial_mean_velocity(nwb_path, prefix)  # [need, 2]

    per_m: dict[str, Any] = {}
    for m in ms:
        support_sl = slice(0, m)
        eval_sl = slice(m, m + eval_trials)
        support_angles = angles[support_sl]
        support_rates = rates[:, support_sl]
        eval_angles = angles[eval_sl]
        eval_rates = rates[:, eval_sl]
        support_vel = vel[support_sl]
        eval_vel = vel[eval_sl]

        fits_a = _fit_units("A", support_angles, support_rates)
        fits_b = _fit_units("B", support_angles, support_rates)
        n_ok_a = sum(1 for f in fits_a if f.status == "ok")
        n_ok_b = sum(1 for f in fits_b if f.status == "ok")
        n_undef_b = sum(1 for f in fits_b if f.status == "b_undefined")

        # Coefficient distance on units where both ok.
        ac_deltas = []
        for fa, fb in zip(fits_a, fits_b):
            if fa.status == "ok" and fb.status == "ok":
                ac_deltas.append(math.hypot(fa.a - fb.a, fa.c - fb.c))
        mean_ac_l2 = float(np.mean(ac_deltas)) if ac_deltas else float("nan")

        mse_a = _rate_mse(fits_a, eval_angles, eval_rates)
        mse_b = _rate_mse(fits_b, eval_angles, eval_rates)
        r2_a = _pv_decode_r2(fits_a, support_rates, support_vel, eval_rates, eval_vel)
        r2_b = _pv_decode_r2(fits_b, support_rates, support_vel, eval_rates, eval_vel)

        per_m[str(m)] = {
            "n_units": int(n_units),
            "n_ok_A": n_ok_a,
            "n_ok_B": n_ok_b,
            "n_b_undefined": n_undef_b,
            "mean_ac_l2_A_minus_B": mean_ac_l2,
            "rate_mse_A": mse_a,
            "rate_mse_B": mse_b,
            "rate_mse_A_minus_B": (
                mse_a - mse_b if math.isfinite(mse_a) and math.isfinite(mse_b) else float("nan")
            ),
            "pv_r2_A": r2_a,
            "pv_r2_B": r2_b,
            "pv_r2_A_minus_B": (
                r2_a - r2_b if math.isfinite(r2_a) and math.isfinite(r2_b) else float("nan")
            ),
            "support_n_finite_angles": int(np.isfinite(support_angles).sum()),
            "eval_n_finite_angles": int(np.isfinite(eval_angles).sum()),
        }
    return {
        "session": session_name,
        "nwb_sha256": sha256_file(nwb_path),
        "n_rewarded_trials": len(trials),
        "per_m": per_m,
    }


def _aggregate_m(rows: Sequence[Mapping[str, Any]], m: int) -> dict[str, Any]:
    key = str(m)
    mse_delta = [r["per_m"][key]["rate_mse_A_minus_B"] for r in rows]
    r2_delta = [r["per_m"][key]["pv_r2_A_minus_B"] for r in rows]
    ac_l2 = [r["per_m"][key]["mean_ac_l2_A_minus_B"] for r in rows]
    mse_delta_f = [v for v in mse_delta if math.isfinite(v)]
    r2_delta_f = [v for v in r2_delta if math.isfinite(v)]
    ac_l2_f = [v for v in ac_l2 if math.isfinite(v)]

    def _stats(vals: list[float]) -> dict[str, Any]:
        if len(vals) < 2:
            return {"n": len(vals), "mean": mean(vals) if vals else float("nan"), "std": float("nan")}
        return {"n": len(vals), "mean": mean(vals), "std": sample_std(vals)}

    mse_stats = _stats(mse_delta_f)
    r2_stats = _stats(r2_delta_f)
    # Transduction: A better than B on rate-MSE means lower MSE (negative delta good for A).
    # For R2, A better means positive A-B.
    # "Same direction improvement of B over A" for deciding whether B's variance-reduction
    # gap matters: look at whether B beats A on both metrics.
    # B better on MSE: mse_A - mse_B > 0; B better on R2: r2_A - r2_B < 0.
    b_better_mse = mse_stats["mean"] > 0 if math.isfinite(mse_stats["mean"]) else False
    b_better_r2 = r2_stats["mean"] < 0 if math.isfinite(r2_stats["mean"]) else False
    a_better_mse = mse_stats["mean"] < 0 if math.isfinite(mse_stats["mean"]) else False
    a_better_r2 = r2_stats["mean"] > 0 if math.isfinite(r2_stats["mean"]) else False
    transduction_consistent = (b_better_mse and b_better_r2) or (a_better_mse and a_better_r2)
    winner = (
        "B_per_trial"
        if b_better_mse and b_better_r2
        else "A_bin_means"
        if a_better_mse and a_better_r2
        else "inconsistent_or_tie"
    )
    return {
        "M": m,
        "mean_ac_l2": _stats(ac_l2_f),
        "rate_mse_A_minus_B": mse_stats,
        "pv_r2_A_minus_B": r2_stats,
        "transduction_consistent": transduction_consistent,
        "preferred_estimator_if_consistent": winner,
        "sessions_positive_mse_delta": sum(1 for v in mse_delta_f if v > 0),
        "sessions_positive_r2_delta": sum(1 for v in r2_delta_f if v > 0),
        "n_sessions": len(rows),
    }


def build_audit(
    *,
    manifest_path: Path,
    data_root: Path,
    ms: Sequence[int] = DEFAULT_MS,
    eval_trials: int = EVAL_TRIALS,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    sessions = load_train_sessions(manifest_path, data_root)
    names = list(sessions.keys())
    if max_sessions is not None:
        names = names[: max_sessions]
    rows = []
    for name in names:
        rows.append(
            audit_session(
                sessions[name],
                session_name=name,
                ms=ms,
                eval_trials=eval_trials,
            )
        )
    aggregates = [_aggregate_m(rows, m) for m in ms]
    # Overall transduction: consistent on a majority of M grid points.
    consistent_ms = [a for a in aggregates if a["transduction_consistent"]]
    overall = {
        "n_m_consistent": len(consistent_ms),
        "n_m_total": len(aggregates),
        "transduction_consistent_majority": len(consistent_ms) > len(aggregates) / 2,
        "per_m_winners": {str(a["M"]): a["preferred_estimator_if_consistent"] for a in aggregates},
    }
    # Pre-declared outlet (handoff): differences ignorable if mean |ac_l2| small AND
    # transduction never consistently prefers one estimator.
    mean_ac = [
        a["mean_ac_l2"]["mean"]
        for a in aggregates
        if math.isfinite(a["mean_ac_l2"]["mean"])
    ]
    ignorable = (not overall["transduction_consistent_majority"]) and (
        (mean(mean_ac) if mean_ac else 0.0) < 0.05
    )
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "no_val_opened": True,
        "no_test_opened": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "data_root": str(data_root),
        "sessions": names,
        "n_sessions": len(names),
        "ms": list(ms),
        "eval_trials": eval_trials,
        "rows": rows,
        "aggregate_per_m": aggregates,
        "overall": overall,
        "predeclared_outlet": {
            "differences_ignorable": ignorable,
            "action_if_ignorable": "cross-setting T4 table may compare operationally; still disclose estimator",
            "action_if_not_ignorable": "unify implementation before publishing cross-setting quantitative law",
        },
    }


def write_audit(audit: Mapping[str, Any], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "audit.json"
    tmp = output_dir / "audit.json.tmp"

    def _strict(obj: Any) -> Any:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): _strict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strict(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return _strict(float(obj))
        return obj

    tmp.write_text(json.dumps(_strict(dict(audit)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
