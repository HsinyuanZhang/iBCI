"""P1 Stage B: FALCON continuous per-trial (B) vs kernel-smoothed (C).

Estimator B: continuous angles, per-trial OLS (FALCON semantics).
Estimator C: continuous angles with angular kernel smoothing before OLS —
variance reduction **without** 8-bin snapping.

Held-in calibration NWBs only. Never opens held-out-calib / EvalAI.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from carrier_perf.p1_estimators import TuningFit, _ols_abc, fit_estimator_b_per_trial
from carrier_perf.protocol import mean

SCHEMA = "carrier_perf_p1_stage_b_falcon_cpu_v1"
DEFAULT_MS: tuple[int, ...] = (10, 20, 30)
EVAL_TRIALS = 16
KERNEL_BANDWIDTH_RAD = math.pi / 6.0
RAW_BIN_MS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spint_root() -> Path:
    # .../sua_exploration/carrier_perf_program/src/carrier_perf/this.py
    return Path(__file__).resolve().parents[4]


def heldin_m2_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("**/*held-in-calib*.nwb"))
    bad = [p for p in paths if "held-out" in str(p).lower()]
    if bad:
        raise ValueError(f"held-out path leaked into Stage B list: {bad}")
    return paths


def load_m2_trial_arrays(nwb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """angles [T], rates [N,T] Hz, trial_vel [T,2] from held-in calib NWB."""
    if "held-out" in str(nwb_path).lower():
        raise ValueError(f"refusing held-out NWB: {nwb_path}")
    sce = _spint_root() / "streaming_calibration_exp"
    sys.path.insert(0, str(sce))
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from src.data.falcon_t4_features import calibration_target_angles, validate_trial_label_alignment

    neural, covariates, trial_change, _eval_mask = load_nwb(nwb_path, FalconTask.m2)
    neural = np.asarray(neural, dtype=np.float64)  # [time, N]
    covariates = np.asarray(covariates, dtype=np.float64)  # [time, 2]
    trial_change = np.asarray(trial_change, dtype=bool)
    angles = np.asarray(calibration_target_angles(nwb_path, "m2"), dtype=np.float64)
    validate_trial_label_alignment(trial_change, angles, source=str(nwb_path))
    starts = np.flatnonzero(trial_change)
    ends = np.r_[starts[1:], len(trial_change)]
    n_channels = neural.shape[1]
    rates = np.zeros((n_channels, starts.size), dtype=np.float64)
    vel = np.zeros((starts.size, 2), dtype=np.float64)
    bin_s = RAW_BIN_MS / 1000.0
    for i, (start, end) in enumerate(zip(starts, ends)):
        duration = max(1, int(end) - int(start)) * bin_s
        rates[:, i] = neural[int(start) : int(end)].sum(axis=0) / duration
        vel[i] = covariates[int(start) : int(end)].mean(axis=0)
    return angles, rates, vel


def fit_estimator_b_matrix(angles: np.ndarray, rates: np.ndarray) -> list[TuningFit]:
    return [fit_estimator_b_per_trial(angles, rates[i], strict_rank=True) for i in range(rates.shape[0])]


def _circular_kernel_weights(query: float, angles: np.ndarray, bandwidth: float) -> np.ndarray:
    d = np.arctan2(np.sin(angles - query), np.cos(angles - query))
    return np.exp(-0.5 * (d / bandwidth) ** 2)


def fit_estimator_c_kernel_smoothed(
    angles: np.ndarray,
    rates: np.ndarray,
    *,
    bandwidth: float = KERNEL_BANDWIDTH_RAD,
    n_grid: int = 16,
) -> TuningFit:
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    usable = np.isfinite(angles) & np.isfinite(rates)
    theta = angles[usable]
    y = rates[usable]
    n_dropped = int((~usable).sum())
    if theta.size < 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="C_kernel_smooth",
            n_rows=int(theta.size),
            design_rank=0,
            status="b_undefined",
            n_dropped_nonfinite=n_dropped,
        )
    grid = np.linspace(-math.pi, math.pi, n_grid, endpoint=False)
    smoothed = np.empty(n_grid, dtype=np.float64)
    for i, g in enumerate(grid):
        w = _circular_kernel_weights(float(g), theta, bandwidth)
        w_sum = float(w.sum())
        smoothed[i] = float(np.dot(w, y) / w_sum) if w_sum > 0 else float("nan")
    ok = np.isfinite(smoothed)
    if int(ok.sum()) < 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="C_kernel_smooth",
            n_rows=int(ok.sum()),
            design_rank=0,
            status="b_undefined",
            n_dropped_nonfinite=n_dropped,
        )
    design = np.stack([np.ones(int(ok.sum())), np.cos(grid[ok]), np.sin(grid[ok])], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        return TuningFit(
            a=float("nan"),
            c=float("nan"),
            m=float("nan"),
            b=float("nan"),
            estimator="C_kernel_smooth",
            n_rows=int(ok.sum()),
            design_rank=rank,
            status="b_undefined",
            n_dropped_nonfinite=n_dropped,
        )
    a, c, b, fitted_rank = _ols_abc(design, smoothed[ok])
    return TuningFit(
        a=a,
        c=c,
        m=float(math.hypot(a, c)),
        b=b,
        estimator="C_kernel_smooth",
        n_rows=int(ok.sum()),
        design_rank=fitted_rank,
        status="ok",
        n_dropped_nonfinite=n_dropped,
    )


def fit_estimator_c_matrix(angles: np.ndarray, rates: np.ndarray) -> list[TuningFit]:
    return [fit_estimator_c_kernel_smoothed(angles, rates[i]) for i in range(rates.shape[0])]


def _predict(fit: TuningFit, angles: np.ndarray) -> np.ndarray:
    return fit.b + fit.a * np.cos(angles) + fit.c * np.sin(angles)


def rate_mse(fits: Sequence[TuningFit], angles: np.ndarray, rates: np.ndarray) -> float:
    ok_ang = np.isfinite(angles)
    if not np.any(ok_ang):
        return float("nan")
    errs = []
    for i, fit in enumerate(fits):
        if fit.status != "ok":
            continue
        pred = _predict(fit, angles[ok_ang])
        errs.append(float(np.mean((pred - rates[i, ok_ang]) ** 2)))
    return float(np.mean(errs)) if errs else float("nan")


def audit_m2_file(nwb_path: Path, ms: Sequence[int], eval_trials: int) -> dict[str, Any]:
    from carrier_perf.p1_stage_a_audit import _pv_decode_r2

    angles, rates, vel = load_m2_trial_arrays(nwb_path)
    need = max(ms) + eval_trials
    if angles.shape[0] < need:
        raise ValueError(f"{nwb_path.name}: only {angles.shape[0]} trials, need {need}")
    per_m: dict[str, Any] = {}
    for m in ms:
        s = slice(0, m)
        e = slice(m, m + eval_trials)
        fits_b = fit_estimator_b_matrix(angles[s], rates[:, s])
        fits_c = fit_estimator_c_matrix(angles[s], rates[:, s])
        mse_b = rate_mse(fits_b, angles[e], rates[:, e])
        mse_c = rate_mse(fits_c, angles[e], rates[:, e])
        ac = [
            math.hypot(fb.a - fc.a, fb.c - fc.c)
            for fb, fc in zip(fits_b, fits_c)
            if fb.status == "ok" and fc.status == "ok"
        ]
        row: dict[str, Any] = {
            "rate_mse_B": mse_b,
            "rate_mse_C": mse_c,
            "rate_mse_B_minus_C": (
                mse_b - mse_c if math.isfinite(mse_b) and math.isfinite(mse_c) else float("nan")
            ),
            "mean_ac_l2_B_minus_C": float(np.mean(ac)) if ac else float("nan"),
            "n_ok_B": sum(1 for f in fits_b if f.status == "ok"),
            "n_ok_C": sum(1 for f in fits_c if f.status == "ok"),
            "n_channels": int(rates.shape[0]),
        }
        r2_b = _pv_decode_r2(fits_b, rates[:, s], vel[s], rates[:, e], vel[e])
        r2_c = _pv_decode_r2(fits_c, rates[:, s], vel[s], rates[:, e], vel[e])
        row["pv_r2_B"] = r2_b
        row["pv_r2_C"] = r2_c
        row["pv_r2_B_minus_C"] = (
            r2_b - r2_c if math.isfinite(r2_b) and math.isfinite(r2_c) else float("nan")
        )
        per_m[str(m)] = row
    return {
        "nwb": str(nwb_path),
        "nwb_sha256": sha256_file(nwb_path),
        "n_trials": int(angles.shape[0]),
        "per_m": per_m,
    }


def build_stage_b_audit(
    *,
    data_dir: Path,
    ms: Sequence[int] = DEFAULT_MS,
    eval_trials: int = EVAL_TRIALS,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    paths = heldin_m2_paths(data_dir)
    if max_sessions is not None:
        paths = paths[:max_sessions]
    if not paths:
        return {
            "schema": SCHEMA,
            "status": "blocked_no_heldin_nwbs_found",
            "no_gpu": True,
            "data_dir": str(data_dir),
        }
    rows = []
    errors = []
    for path in paths:
        try:
            rows.append(audit_m2_file(path, ms, eval_trials))
        except Exception as exc:  # noqa: BLE001
            errors.append({"nwb": str(path), "error": str(exc)})
    aggregates = []
    for m in ms:
        key = str(m)
        mse_f = [
            r["per_m"][key]["rate_mse_B_minus_C"]
            for r in rows
            if math.isfinite(r["per_m"][key]["rate_mse_B_minus_C"])
        ]
        r2_f = [
            r["per_m"][key]["pv_r2_B_minus_C"]
            for r in rows
            if math.isfinite(r["per_m"][key]["pv_r2_B_minus_C"])
        ]
        mse_mean = mean(mse_f) if mse_f else float("nan")
        r2_mean = mean(r2_f) if r2_f else float("nan")
        c_better_mse = mse_mean > 0 if math.isfinite(mse_mean) else False
        c_better_r2 = r2_mean < 0 if math.isfinite(r2_mean) else False
        consistent = bool(c_better_mse and c_better_r2)
        aggregates.append(
            {
                "M": m,
                "rate_mse_B_minus_C_mean": mse_mean,
                "rate_mse_n": len(mse_f),
                "pv_r2_B_minus_C_mean": r2_mean,
                "pv_r2_n": len(r2_f),
                "transduction_consistent_C_over_B": consistent,
                "gate_stage_c_gpu": consistent,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only" if rows else "failed_all_sessions",
        "no_gpu": True,
        "no_heldout_opened": True,
        "kernel_bandwidth_rad": KERNEL_BANDWIDTH_RAD,
        "data_dir": str(data_dir),
        "n_sessions": len(rows),
        "rows": rows,
        "errors": errors,
        "aggregate_per_m": aggregates,
        "stage_c_gpu_authorized_by_this_audit": any(a["gate_stage_c_gpu"] for a in aggregates),
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
