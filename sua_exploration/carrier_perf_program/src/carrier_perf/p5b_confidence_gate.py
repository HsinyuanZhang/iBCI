"""P5b Stage A: session-specific fit-confidence gating (train-free).

Distinct from the judged-negative static electrode ``t4gate``. Here confidence is
per-unit residual variance of the continuous cosine fit on the current support.

Arms (same support / eval split):
  baseline   — estimator B
  gated      — zero units with residual_var above the support median
  shrunk     — Wiener-like shrink of (a,c) using residual_var (b unchanged)

Proxy gate: rate-MSE and PV-R² must move together vs baseline before any GPU FiLM.
Runs on CO train (SUA) and optionally M2 held-in.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from carrier_perf.p1_estimators import TuningFit, fit_estimator_b_per_trial
from carrier_perf.p1_stage_a_audit import (
    EVAL_TRIALS,
    _pv_decode_r2,
    _rate_mse,
    _trial_mean_velocity,
    load_train_sessions,
    sha256_file,
)
from carrier_perf.p1_stage_b_audit import heldin_m2_paths, load_m2_trial_arrays
from carrier_perf.protocol import mean

SCHEMA = "carrier_perf_p5b_confidence_gate_cpu_v1"
SUPPORT_M = 30
SHRINK_STRENGTH = 3.0


def _residual_var(fit: TuningFit, angles: np.ndarray, rates: np.ndarray) -> float:
    if fit.status != "ok":
        return float("nan")
    pred = fit.b + fit.a * np.cos(angles) + fit.c * np.sin(angles)
    ok = np.isfinite(angles) & np.isfinite(rates)
    if int(ok.sum()) < 4:
        return float("nan")
    resid = rates[ok] - pred[ok]
    return float(np.dot(resid, resid) / max(1, int(ok.sum()) - 3))


def fit_units_with_confidence(angles: np.ndarray, rates: np.ndarray) -> tuple[list[TuningFit], np.ndarray]:
    fits: list[TuningFit] = []
    vars_ = np.full(rates.shape[0], np.nan, dtype=np.float64)
    for i in range(rates.shape[0]):
        fit = fit_estimator_b_per_trial(angles, rates[i], strict_rank=True)
        fits.append(fit)
        vars_[i] = _residual_var(fit, angles, rates[i])
    return fits, vars_


def gate_fits(fits: Sequence[TuningFit], vars_: np.ndarray) -> list[TuningFit]:
    finite = vars_[np.isfinite(vars_)]
    if finite.size == 0:
        return list(fits)
    thresh = float(np.median(finite))
    out = []
    for fit, v in zip(fits, vars_):
        if fit.status != "ok" or not math.isfinite(v) or v > thresh:
            out.append(
                TuningFit(
                    a=0.0,
                    c=0.0,
                    m=0.0,
                    b=0.0,
                    estimator=fit.estimator,
                    n_rows=fit.n_rows,
                    design_rank=fit.design_rank,
                    status="degenerate_zeros",
                    n_dropped_nonfinite=fit.n_dropped_nonfinite,
                )
            )
        else:
            out.append(fit)
    return out


def shrink_fits(fits: Sequence[TuningFit], vars_: np.ndarray, *, strength: float = SHRINK_STRENGTH) -> list[TuningFit]:
    out = []
    for fit, v in zip(fits, vars_):
        if fit.status != "ok" or not math.isfinite(v):
            out.append(fit)
            continue
        signal = fit.a * fit.a + fit.c * fit.c
        factor = signal / (signal + strength * max(v, 0.0)) if (signal + strength * max(v, 0.0)) > 0 else 0.0
        a = fit.a * factor
        c = fit.c * factor
        out.append(
            TuningFit(
                a=a,
                c=c,
                m=float(math.hypot(a, c)),
                b=fit.b,
                estimator=fit.estimator + "_shrink",
                n_rows=fit.n_rows,
                design_rank=fit.design_rank,
                status="ok",
                n_dropped_nonfinite=fit.n_dropped_nonfinite,
            )
        )
    return out


def _arm_metrics(
    fits: Sequence[TuningFit],
    support_angles: np.ndarray,
    support_rates: np.ndarray,
    support_vel: np.ndarray,
    eval_angles: np.ndarray,
    eval_rates: np.ndarray,
    eval_vel: np.ndarray,
) -> dict[str, float]:
    return {
        "rate_mse": _rate_mse(fits, eval_angles, eval_rates),
        "pv_r2": _pv_decode_r2(fits, support_rates, support_vel, eval_rates, eval_vel),
        "n_ok": float(sum(1 for f in fits if f.status == "ok")),
    }


def audit_arrays(
    angles: np.ndarray,
    rates: np.ndarray,
    vel: np.ndarray,
    *,
    support_m: int = SUPPORT_M,
    eval_trials: int = EVAL_TRIALS,
) -> dict[str, Any]:
    need = support_m + eval_trials
    if angles.shape[0] < need or rates.shape[1] < need:
        raise ValueError(f"need {need} trials, got {angles.shape[0]}")
    s = slice(0, support_m)
    e = slice(support_m, support_m + eval_trials)
    base, vars_ = fit_units_with_confidence(angles[s], rates[:, s])
    gated = gate_fits(base, vars_)
    shrunk = shrink_fits(base, vars_)
    arms = {
        "baseline": _arm_metrics(base, angles[s], rates[:, s], vel[s], angles[e], rates[:, e], vel[e]),
        "gated": _arm_metrics(gated, angles[s], rates[:, s], vel[s], angles[e], rates[:, e], vel[e]),
        "shrunk": _arm_metrics(shrunk, angles[s], rates[:, s], vel[s], angles[e], rates[:, e], vel[e]),
    }
    for name in ("gated", "shrunk"):
        arms[name]["rate_mse_delta_vs_baseline"] = (
            arms[name]["rate_mse"] - arms["baseline"]["rate_mse"]
            if math.isfinite(arms[name]["rate_mse"]) and math.isfinite(arms["baseline"]["rate_mse"])
            else float("nan")
        )
        arms[name]["pv_r2_delta_vs_baseline"] = (
            arms[name]["pv_r2"] - arms["baseline"]["pv_r2"]
            if math.isfinite(arms[name]["pv_r2"]) and math.isfinite(arms["baseline"]["pv_r2"])
            else float("nan")
        )
    return {
        "mean_log_residual_var": float(np.nanmean(np.log(vars_ + 1e-12))),
        "median_residual_var": float(np.nanmedian(vars_)),
        "arms": arms,
    }


def audit_sua_session(nwb_path: Path, session_name: str) -> dict[str, Any]:
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix

    need = SUPPORT_M + EVAL_TRIALS
    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    if len(trials) < need:
        raise ValueError(f"{session_name}: need {need} trials")
    prefix = trials[:need]
    rates, _ = _pool_trial_rate_matrix(nwb_path, prefix)
    angles = np.asarray(
        [t["target_dir"] if t["target_dir"] is not None else np.nan for t in prefix],
        dtype=np.float64,
    )
    vel = _trial_mean_velocity(nwb_path, prefix)
    payload = audit_arrays(angles, rates, vel)
    payload.update({"session": session_name, "nwb_sha256": sha256_file(nwb_path)})
    return payload


def audit_m2_session(nwb_path: Path) -> dict[str, Any]:
    angles, rates, vel = load_m2_trial_arrays(nwb_path)
    payload = audit_arrays(angles, rates, vel, eval_trials=16)
    payload.update({"session": nwb_path.name, "nwb_sha256": sha256_file(nwb_path)})
    return payload


def _aggregate_arm(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    mse_d = [r["arms"][arm]["rate_mse_delta_vs_baseline"] for r in rows]
    r2_d = [r["arms"][arm]["pv_r2_delta_vs_baseline"] for r in rows]
    mse_f = [v for v in mse_d if math.isfinite(v)]
    r2_f = [v for v in r2_d if math.isfinite(v)]
    mse_mean = mean(mse_f) if mse_f else float("nan")
    r2_mean = mean(r2_f) if r2_f else float("nan")
    # Better than baseline: lower MSE (delta<0) and higher R2 (delta>0)
    better_mse = mse_mean < 0 if math.isfinite(mse_mean) else False
    better_r2 = r2_mean > 0 if math.isfinite(r2_mean) else False
    return {
        "arm": arm,
        "rate_mse_delta_mean": mse_mean,
        "pv_r2_delta_mean": r2_mean,
        "n": len(mse_f),
        "transduction_consistent_over_baseline": bool(better_mse and better_r2),
        "gate_gpu_film": bool(better_mse and better_r2),
    }


def build_p5b_audit(
    *,
    manifest_path: Path,
    data_root: Path,
    falcon_data_dir: Path,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    sessions = load_train_sessions(manifest_path, data_root)
    names = list(sessions.keys())
    if max_sessions is not None:
        names = names[:max_sessions]
    sua_rows = [audit_sua_session(sessions[n], n) for n in names]
    m2_paths = heldin_m2_paths(falcon_data_dir)
    if max_sessions is not None:
        m2_paths = m2_paths[:max_sessions]
    m2_rows = [audit_m2_session(p) for p in m2_paths]

    def _setting(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        aggs = [_aggregate_arm(rows, "gated"), _aggregate_arm(rows, "shrunk")]
        return {
            "setting": label,
            "n_sessions": len(rows),
            "aggregates": aggs,
            "any_arm_gates_gpu": any(a["gate_gpu_film"] for a in aggs),
            "rows": rows,
        }

    sua = _setting(sua_rows, "SUA_CO_train")
    m2 = _setting(m2_rows, "FALCON_M2_heldin")
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "shrink_strength": SHRINK_STRENGTH,
        "support_m": SUPPORT_M,
        "sua": sua,
        "falcon": m2,
        "gate_gpu_film_authorized_by_this_audit": bool(
            sua["any_arm_gates_gpu"] or m2["any_arm_gates_gpu"]
        ),
        "note": (
            "GPU FiLM/confidence cells require an arm with transduction_consistent_over_baseline "
            "on the target setting. Distinct from judged-negative static t4gate."
        ),
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
