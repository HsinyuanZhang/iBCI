"""P4: 2nd-harmonic gain stratified by electrode preferred-direction dispersion.

Hypothesis (handoff): pooling opposite preferred directions cancels the
fundamental but not the 2nd harmonic. GPU is authorized only if harmonic gain
rises monotonically across low → mid → high electrode-dispersion strata.

Design (continuous angles; no 8-bin snap):
  fund:  r ≈ b + a cosθ + c sinθ
  harm:  r ≈ b + a cosθ + c sinθ + a2 cos2θ + c2 sin2θ

Dispersion on electrode e with units u:
  1 - |Σ_u m_u exp(i φ_u)| / Σ_u m_u ,  φ=atan2(c,a), m=hypot(a,c)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO

from carrier_perf.p1_estimators import TuningFit, fit_estimator_b_per_trial
from carrier_perf.p1_stage_a_audit import (
    EVAL_TRIALS,
    _pv_decode_r2,
    _rate_mse,
    _trial_mean_velocity,
    load_train_sessions,
    sha256_file,
)
from carrier_perf.protocol import mean

SCHEMA = "carrier_perf_p4_harmonic_dispersion_strata_v1"
SUPPORT_M = 30
N_STRATA = 3


@dataclass(frozen=True)
class HarmonicFit:
    a: float
    c: float
    a2: float
    c2: float
    b: float
    m: float
    n_rows: int
    design_rank: int
    residual_var: float
    status: str


def circular_dispersion(phis: np.ndarray, weights: np.ndarray) -> float:
    phis = np.asarray(phis, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if phis.size != w.size:
        raise ValueError("phis/weights length mismatch")
    ok = np.isfinite(phis) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    phis = phis[ok]
    w = w[ok]
    num = abs(complex(np.dot(w, np.cos(phis)), np.dot(w, np.sin(phis))))
    den = float(w.sum())
    return float(1.0 - num / den)


def higher_harmonic_ratio(a: float, c: float, a2: float, c2: float) -> float:
    fund = a * a + c * c
    harm = a2 * a2 + c2 * c2
    tot = fund + harm
    if tot <= 0:
        return float("nan")
    return float(harm / tot)


def fit_harmonic(angles: np.ndarray, rates: np.ndarray) -> HarmonicFit:
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    mask = np.isfinite(angles) & np.isfinite(rates)
    theta = angles[mask]
    y = rates[mask]
    if theta.size < 5:
        return HarmonicFit(
            a=float("nan"),
            c=float("nan"),
            a2=float("nan"),
            c2=float("nan"),
            b=float("nan"),
            m=float("nan"),
            n_rows=int(theta.size),
            design_rank=0,
            residual_var=float("nan"),
            status="b_undefined",
        )
    design = np.stack(
        [
            np.ones(theta.size),
            np.cos(theta),
            np.sin(theta),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
        ],
        axis=1,
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank != 5:
        return HarmonicFit(
            a=float("nan"),
            c=float("nan"),
            a2=float("nan"),
            c2=float("nan"),
            b=float("nan"),
            m=float("nan"),
            n_rows=int(theta.size),
            design_rank=rank,
            residual_var=float("nan"),
            status="b_undefined",
        )
    coef, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
    b, a, c, a2, c2 = (float(v) for v in coef)
    if residuals.size:
        rss = float(residuals[0])
    else:
        rss = float(np.sum((y - design @ coef) ** 2))
    dof = max(1, int(theta.size) - 5)
    return HarmonicFit(
        a=a,
        c=c,
        a2=a2,
        c2=c2,
        b=b,
        m=float(math.hypot(a, c)),
        n_rows=int(theta.size),
        design_rank=rank,
        residual_var=rss / dof,
        status="ok",
    )


def _predict_fund(fit, angles: np.ndarray) -> np.ndarray:
    return fit.b + fit.a * np.cos(angles) + fit.c * np.sin(angles)


def _predict_harm(fit: HarmonicFit, angles: np.ndarray) -> np.ndarray:
    return (
        fit.b
        + fit.a * np.cos(angles)
        + fit.c * np.sin(angles)
        + fit.a2 * np.cos(2.0 * angles)
        + fit.c2 * np.sin(2.0 * angles)
    )


def _mse(pred: np.ndarray, actual: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(actual)
    if not np.any(ok):
        return float("nan")
    return float(np.mean((pred[ok] - actual[ok]) ** 2))


def _electrode_ids(nwb_path: Path) -> np.ndarray:
    from mc_maze.multisession_datamodule import electrode_ids_from_units

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"no units: {nwb_path}")
        return electrode_ids_from_units(nwb.units.to_dataframe())


def _assign_tertiles(values: np.ndarray) -> np.ndarray:
    """Return stratum index 0/1/2 for finite values; -1 for non-finite."""
    out = np.full(values.shape, -1, dtype=np.int64)
    ok = np.isfinite(values)
    if int(ok.sum()) < N_STRATA:
        return out
    # Equal-count tertiles via rank percentiles.
    ranks = np.empty(ok.sum(), dtype=np.float64)
    order = np.argsort(values[ok])
    ranks[order] = np.linspace(0.0, 1.0, int(ok.sum()), endpoint=False)
    strata = np.floor(ranks * N_STRATA).astype(np.int64)
    strata = np.clip(strata, 0, N_STRATA - 1)
    out[ok] = strata
    return out


def audit_session(nwb_path: Path, session_name: str) -> dict[str, Any]:
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix

    need = SUPPORT_M + EVAL_TRIALS
    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    if len(trials) < need:
        raise ValueError(f"{session_name}: need {need} trials, got {len(trials)}")
    prefix = trials[:need]
    rates, n_units = _pool_trial_rate_matrix(nwb_path, prefix)
    angles = np.asarray(
        [t["target_dir"] if t["target_dir"] is not None else np.nan for t in prefix],
        dtype=np.float64,
    )
    vel = _trial_mean_velocity(nwb_path, prefix)
    electrode_ids = _electrode_ids(nwb_path)
    if electrode_ids.shape[0] != n_units:
        raise ValueError("electrode_ids / unit count mismatch")

    s = slice(0, SUPPORT_M)
    e = slice(SUPPORT_M, SUPPORT_M + EVAL_TRIALS)
    fund_fits: list[TuningFit] = []
    harm_fits: list[HarmonicFit] = []
    unit_rows = []
    for i in range(n_units):
        fb = fit_estimator_b_per_trial(angles[s], rates[i, s], strict_rank=True)
        fh = fit_harmonic(angles[s], rates[i, s])
        fund_fits.append(fb)
        harm_fits.append(fh)
        mse_f = (
            _mse(_predict_fund(fb, angles[e]), rates[i, e])
            if fb.status == "ok"
            else float("nan")
        )
        mse_h = (
            _mse(_predict_harm(fh, angles[e]), rates[i, e])
            if fh.status == "ok"
            else float("nan")
        )
        phi = float(math.atan2(fb.c, fb.a)) if fb.status == "ok" and fb.m > 0 else float("nan")
        unit_rows.append(
            {
                "unit": i,
                "electrode": int(electrode_ids[i]),
                "phi": phi,
                "m": fb.m if fb.status == "ok" else float("nan"),
                "mse_fund": mse_f,
                "mse_harm": mse_h,
                "mse_fund_minus_harm": (
                    mse_f - mse_h if math.isfinite(mse_f) and math.isfinite(mse_h) else float("nan")
                ),
                "hh_ratio": (
                    higher_harmonic_ratio(fh.a, fh.c, fh.a2, fh.c2)
                    if fh.status == "ok"
                    else float("nan")
                ),
                "fund_ok": fb.status == "ok",
                "harm_ok": fh.status == "ok",
            }
        )

    # Electrode-level dispersion and mean harmonic gain.
    electrodes = sorted({int(e_id) for e_id in electrode_ids})
    electrode_rows = []
    for e_id in electrodes:
        members = [u for u in unit_rows if u["electrode"] == e_id and u["fund_ok"]]
        if len(members) < 2:
            continue
        disp = circular_dispersion(
            np.asarray([u["phi"] for u in members]),
            np.asarray([u["m"] for u in members]),
        )
        gains = [u["mse_fund_minus_harm"] for u in members if math.isfinite(u["mse_fund_minus_harm"])]
        electrode_rows.append(
            {
                "electrode": e_id,
                "n_units": len(members),
                "dispersion": disp,
                "mean_harmonic_gain": float(np.mean(gains)) if gains else float("nan"),
                "mean_hh_ratio": float(
                    np.mean([u["hh_ratio"] for u in members if math.isfinite(u["hh_ratio"])])
                )
                if any(math.isfinite(u["hh_ratio"]) for u in members)
                else float("nan"),
            }
        )

    # Session-level transduction: fund vs harm via rate-MSE / PV-R2.
    # For PV we only have fundamental preferred dirs from fund fits.
    mse_fund = _rate_mse(fund_fits, angles[e], rates[:, e])
    # Build pseudo TuningFit list from harmonic fundamental coeffs for PV dirs.
    harm_as_fund = [
        TuningFit(
            a=h.a,
            c=h.c,
            m=h.m,
            b=h.b,
            estimator="H",
            n_rows=h.n_rows,
            design_rank=h.design_rank,
            status=h.status,
        )
        for h in harm_fits
    ]
    # Rate-MSE for harmonic predictions (manual).
    harm_errs = []
    for i, h in enumerate(harm_fits):
        if h.status != "ok":
            continue
        harm_errs.append(_mse(_predict_harm(h, angles[e]), rates[i, e]))
    mse_harm = float(np.mean(harm_errs)) if harm_errs else float("nan")
    r2_fund = _pv_decode_r2(fund_fits, rates[:, s], vel[s], rates[:, e], vel[e])
    r2_harm = _pv_decode_r2(harm_as_fund, rates[:, s], vel[s], rates[:, e], vel[e])

    return {
        "session": session_name,
        "nwb_sha256": sha256_file(nwb_path),
        "n_units": int(n_units),
        "n_electrodes_with_ge2_units": len(electrode_rows),
        "rate_mse_fund": mse_fund,
        "rate_mse_harm": mse_harm,
        "rate_mse_fund_minus_harm": (
            mse_fund - mse_harm if math.isfinite(mse_fund) and math.isfinite(mse_harm) else float("nan")
        ),
        "pv_r2_fund": r2_fund,
        "pv_r2_harm": r2_harm,
        "pv_r2_fund_minus_harm": (
            r2_fund - r2_harm if math.isfinite(r2_fund) and math.isfinite(r2_harm) else float("nan")
        ),
        "electrodes": electrode_rows,
        "units": unit_rows,
    }


def _monotonic_nondecreasing(xs: Sequence[float]) -> bool:
    if any(not math.isfinite(x) for x in xs):
        return False
    return all(xs[i] <= xs[i + 1] + 1e-12 for i in range(len(xs) - 1))


def build_p4_audit(
    *,
    manifest_path: Path,
    data_root: Path,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    sessions = load_train_sessions(manifest_path, data_root)
    names = list(sessions.keys())
    if max_sessions is not None:
        names = names[:max_sessions]
    rows = [audit_session(sessions[n], n) for n in names]

    # Pool electrodes across sessions for global tertiles.
    pooled = []
    for r in rows:
        for e in r["electrodes"]:
            if math.isfinite(e["dispersion"]) and math.isfinite(e["mean_harmonic_gain"]):
                pooled.append(e)
    dispersions = np.asarray([e["dispersion"] for e in pooled], dtype=np.float64)
    gains = np.asarray([e["mean_harmonic_gain"] for e in pooled], dtype=np.float64)
    strata = _assign_tertiles(dispersions)
    stratum_stats = []
    for s_idx in range(N_STRATA):
        mask = strata == s_idx
        g = gains[mask]
        d = dispersions[mask]
        stratum_stats.append(
            {
                "stratum": s_idx,
                "label": ["low", "mid", "high"][s_idx],
                "n_electrodes": int(mask.sum()),
                "mean_dispersion": float(np.mean(d)) if mask.any() else float("nan"),
                "mean_harmonic_gain": float(np.mean(g)) if mask.any() else float("nan"),
                "mean_hh_ratio": float(
                    np.mean(
                        [
                            e["mean_hh_ratio"]
                            for e, keep in zip(pooled, mask)
                            if keep and math.isfinite(e["mean_hh_ratio"])
                        ]
                    )
                )
                if any(mask)
                else float("nan"),
            }
        )
    gain_curve = [s["mean_harmonic_gain"] for s in stratum_stats]
    mono = _monotonic_nondecreasing(gain_curve)

    mse_deltas = [r["rate_mse_fund_minus_harm"] for r in rows if math.isfinite(r["rate_mse_fund_minus_harm"])]
    r2_deltas = [r["pv_r2_fund_minus_harm"] for r in rows if math.isfinite(r["pv_r2_fund_minus_harm"])]
    mse_mean = mean(mse_deltas) if mse_deltas else float("nan")
    r2_mean = mean(r2_deltas) if r2_deltas else float("nan")
    # Harmonic better on MSE: fund-harm > 0; on R2: fund-harm < 0 (harm higher R2).
    harm_better_mse = mse_mean > 0 if math.isfinite(mse_mean) else False
    harm_better_r2 = r2_mean < 0 if math.isfinite(r2_mean) else False
    transduction = bool(harm_better_mse and harm_better_r2)
    gate_gpu = bool(mono and transduction)

    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "no_test_opened": True,
        "support_m": SUPPORT_M,
        "eval_trials": EVAL_TRIALS,
        "manifest_sha256": sha256_file(manifest_path),
        "n_sessions": len(rows),
        "sessions": names,
        "n_pooled_electrodes": len(pooled),
        "stratum_stats": stratum_stats,
        "monotonic_gain_vs_dispersion": mono,
        "overall_rate_mse_fund_minus_harm_mean": mse_mean,
        "overall_pv_r2_fund_minus_harm_mean": r2_mean,
        "transduction_consistent_harmonic_over_fund": transduction,
        "gate_harmonic_gpu": gate_gpu,
        "gate_reason": (
            "monotonic_and_transduction"
            if gate_gpu
            else (
                "monotonicity_failed"
                if not mono
                else "transduction_failed"
                if not transduction
                else "unknown"
            )
        ),
        "rows": [
            {
                "session": r["session"],
                "nwb_sha256": r["nwb_sha256"],
                "n_units": r["n_units"],
                "n_electrodes_with_ge2_units": r["n_electrodes_with_ge2_units"],
                "rate_mse_fund_minus_harm": r["rate_mse_fund_minus_harm"],
                "pv_r2_fund_minus_harm": r["pv_r2_fund_minus_harm"],
                "electrodes": r["electrodes"],
            }
            for r in rows
        ],
        "note": (
            "GPU harmonic cells require gate_harmonic_gpu=true. "
            "A bare positive delta without stratum monotonicity is not evidence."
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
