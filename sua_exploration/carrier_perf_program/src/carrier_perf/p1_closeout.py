"""P1 closeout: bootstrap CIs on sealed Stage-A deltas + SUA vs FALCON fit variance.

1) Bootstrap (session resample) of P1A aggregate deltas — no NWB reopen.
2) Residual-variance pass: estimator-B mean residual variance on CO train vs M2
   held-in. W3 conditional reopen requires FALCON/SUA ratio >= 10 (handoff).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from carrier_perf.p1_estimators import fit_estimator_b_per_trial
from carrier_perf.p1_stage_a_audit import (
    load_train_sessions,
    sha256_file,
)
from carrier_perf.p1_stage_b_audit import heldin_m2_paths, load_m2_trial_arrays
from carrier_perf.protocol import mean, sample_std

SCHEMA = "carrier_perf_p1_closeout_v1"
W3_VARIANCE_RATIO_FLOOR = 10.0
SUPPORT_M = 30
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size < 2:
        return {
            "n": int(vals.size),
            "mean": float(vals[0]) if vals.size else float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": n_boot,
        }
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = vals[rng.randint(0, vals.size, size=vals.size)]
        boots[i] = float(sample.mean())
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(sample_std(vals.tolist())),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
    }


def bootstrap_from_p1a_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = audit["rows"]
    per_m = []
    for m in audit["ms"]:
        key = str(m)
        mse = [r["per_m"][key]["rate_mse_A_minus_B"] for r in rows]
        r2 = [r["per_m"][key]["pv_r2_A_minus_B"] for r in rows]
        ac = [r["per_m"][key]["mean_ac_l2_A_minus_B"] for r in rows]
        per_m.append(
            {
                "M": m,
                "rate_mse_A_minus_B": bootstrap_mean_ci(mse),
                "pv_r2_A_minus_B": bootstrap_mean_ci(r2),
                "mean_ac_l2_A_minus_B": bootstrap_mean_ci(ac),
            }
        )
    return {
        "source_schema": audit.get("schema"),
        "source_n_sessions": audit.get("n_sessions"),
        "per_m": per_m,
        "predeclared_outlet_echo": audit.get("predeclared_outlet"),
        "overall_echo": audit.get("overall"),
    }


def _unit_residual_vars(angles: np.ndarray, rates: np.ndarray) -> list[float]:
    """Per-unit residual variance of estimator B on the given support."""
    out = []
    for i in range(rates.shape[0]):
        fit = fit_estimator_b_per_trial(angles, rates[i], strict_rank=True)
        if fit.status != "ok":
            continue
        pred = fit.b + fit.a * np.cos(angles) + fit.c * np.sin(angles)
        ok = np.isfinite(angles) & np.isfinite(rates[i])
        if int(ok.sum()) < 4:
            continue
        resid = rates[i, ok] - pred[ok]
        out.append(float(np.dot(resid, resid) / max(1, int(ok.sum()) - 3)))
    return out


def sua_residual_variance_pass(
    *,
    manifest_path: Path,
    data_root: Path,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix

    sessions = load_train_sessions(manifest_path, data_root)
    names = list(sessions.keys())
    if max_sessions is not None:
        names = names[:max_sessions]
    session_means = []
    all_vars: list[float] = []
    for name in names:
        path = sessions[name]
        trials = list_datamodule_rewarded_trials(
            path, bin_size_ms=20, window_size=50, trial_result_filter="R"
        )
        if len(trials) < SUPPORT_M:
            raise ValueError(f"{name}: need {SUPPORT_M} trials")
        rates, _ = _pool_trial_rate_matrix(path, trials[:SUPPORT_M])
        angles = np.asarray(
            [t["target_dir"] if t["target_dir"] is not None else np.nan for t in trials[:SUPPORT_M]],
            dtype=np.float64,
        )
        vars_ = _unit_residual_vars(angles, rates)
        if not vars_:
            continue
        session_means.append(float(np.mean(vars_)))
        all_vars.extend(vars_)
    return {
        "setting": "SUA_CO_train",
        "n_sessions": len(session_means),
        "n_units_ok": len(all_vars),
        "mean_unit_residual_var": float(np.mean(all_vars)) if all_vars else float("nan"),
        "median_unit_residual_var": float(np.median(all_vars)) if all_vars else float("nan"),
        "mean_of_session_means": mean(session_means) if session_means else float("nan"),
        "session_means": session_means,
    }


def falcon_residual_variance_pass(
    *,
    data_dir: Path,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    paths = heldin_m2_paths(data_dir)
    if max_sessions is not None:
        paths = paths[:max_sessions]
    session_means = []
    all_vars: list[float] = []
    for path in paths:
        angles, rates, _vel = load_m2_trial_arrays(path)
        if angles.shape[0] < SUPPORT_M:
            raise ValueError(f"{path.name}: need {SUPPORT_M} trials")
        vars_ = _unit_residual_vars(angles[:SUPPORT_M], rates[:, :SUPPORT_M])
        if not vars_:
            continue
        session_means.append(float(np.mean(vars_)))
        all_vars.extend(vars_)
    return {
        "setting": "FALCON_M2_heldin",
        "n_sessions": len(session_means),
        "n_units_ok": len(all_vars),
        "mean_unit_residual_var": float(np.mean(all_vars)) if all_vars else float("nan"),
        "median_unit_residual_var": float(np.median(all_vars)) if all_vars else float("nan"),
        "mean_of_session_means": mean(session_means) if session_means else float("nan"),
        "session_means": session_means,
    }


def build_p1_closeout(
    *,
    p1a_audit_path: Path,
    manifest_path: Path,
    data_root: Path,
    falcon_data_dir: Path,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    audit = json.loads(p1a_audit_path.read_text(encoding="utf-8"))
    boot = bootstrap_from_p1a_audit(audit)
    sua = sua_residual_variance_pass(
        manifest_path=manifest_path, data_root=data_root, max_sessions=max_sessions
    )
    falcon = falcon_residual_variance_pass(
        data_dir=falcon_data_dir, max_sessions=max_sessions
    )
    sua_v = sua["mean_unit_residual_var"]
    fal_v = falcon["mean_unit_residual_var"]
    ratio = (
        fal_v / sua_v if math.isfinite(sua_v) and math.isfinite(fal_v) and sua_v > 0 else float("nan")
    )
    w3_reopen = bool(math.isfinite(ratio) and ratio >= W3_VARIANCE_RATIO_FLOOR)
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "p1a_audit_path": str(p1a_audit_path),
        "p1a_audit_sha256": sha256_file(p1a_audit_path),
        "bootstrap": boot,
        "fit_variance": {
            "support_m": SUPPORT_M,
            "sua": sua,
            "falcon": falcon,
            "falcon_over_sua_mean_residual_var": ratio,
            "w3_reopen_ratio_floor": W3_VARIANCE_RATIO_FLOOR,
            "w3_conditional_reopen_authorized": w3_reopen,
            "note": (
                "W3 may be re-pre-registered on MUA only if ratio >= floor AND an explicit "
                "receipt argues a new mechanism; this boolean alone is not a launch."
            ),
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
