"""Z4 — U-stability diagnostic (IMPLEMENTATION; pure CPU linear algebra).

Question (handoff §2 Z4 / subspace_stability runbook): in the STRICT
total-calibration regime the only target-session activity available for
Path-1's neural-covariance subspace U is the M calibration trials — is U
estimated from M4/M10 trials stable enough to restrict the carrier's column
space?

Protocol (zero labels, zero training, numpy only):
  per external-15 session, on the SAME binned calibration activity the sealed
  model consumes (``calib_trials``: 30 interpolated 100-bin trials, the B3S
  stream):
    U_ref  = top-k principal subspace (k in {4, 6, 8, 10}) of the M30-window
             covariance (30*100 bins);
    U_M    = same estimator on the first-M-trial windows (M in {4, 10});
             200 bootstrap resamples of the M TRIALS (bins travel with their
             trial — within-trial bins are strongly correlated) give a CI;
    report principal angles between U_M and U_ref, effective rank of the
    M-window sample covariance vs N (units), raw + Ledoit-Wolf-shrunk variants.

Pre-registered reading (bound on Path 1 BEFORE any build):
  median top-subspace angle (k=6) < ~20 deg at M10 and < ~35 deg at M4
  -> U usable, P1 proceed.  Angles near random (~90 deg) at M4 -> U starves
  in the strict regime; P1 needs the Z6 contract answer (transductive
  evaluation-stream activity) or re-scopes to M10+.

Primary scalar (this implementation, declared before unblinding): the LARGEST
principal angle theta_max(U_M, U_ref) at k=6, summarized as the median across
the external-15 sessions.  The mean principal angle and the full spectrum are
reported alongside so the reading is auditable either way.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

KS = (4, 6, 8, 10)
MS = (4, 10)
BOOTSTRAPS = 200
BOOTSTRAP_SEED = 42
# Pre-registered thresholds (degrees), from the runbook.
THRESHOLD_M10_DEG = 20.0
THRESHOLD_M4_DEG = 35.0
RANDOM_ANGLE_DEG = 90.0


class Z4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Z4Error(message)


# ---------------------------------------------------------------------------
# pure linear algebra
# ---------------------------------------------------------------------------


def principal_angles_deg(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles between two orthonormal column-space bases, ascending.

    ``U``: [N, k1], ``V``: [N, k2] with orthonormal columns.  Returns
    ``min(k1, k2)`` angles in degrees in [0, 90]: ``arccos`` of the singular
    values of ``U^T V`` (with a clamped cosine to protect against float
    overshoot).
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    _require(U.ndim == 2 and V.ndim == 2 and U.shape[0] == V.shape[0], "basis shape mismatch")
    cosines = np.linalg.svd(U.T @ V, compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Centered sample covariance of windows ``X`` [n, N] (maximum likelihood)."""
    X = np.asarray(X, dtype=np.float64)
    _require(X.ndim == 2, "covariance input must be [windows, units]")
    centered = X - X.mean(axis=0, keepdims=True)
    return centered.T @ centered / float(X.shape[0])


def ledoit_wolf_covariance(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity (numpy implementation).

    Returns ``(Sigma, shrinkage)`` with ``Sigma = (1-s) S + s mu I`` where
    ``mu = tr(S)/N`` and ``s = min(1, b^2 / d^2)`` with the standard moment
    estimators (Ledoit & Wolf 2004, "A well-conditioned estimator for
    large-dimensional covariance matrices", eq. 15): with centered windows
    ``x_i`` and gram ``G = X_c X_c^T``,
        d^2 = || S - mu I ||_F^2
        b̄^2 = ( sum_i ||x_i||^4 - ||G||_F^2 / n ) / n^2
    (the optional per-dimension normalization cancels in the ratio b^2/d^2).
    """
    X = np.asarray(X, dtype=np.float64)
    _require(X.ndim == 2 and X.shape[0] >= 2, "Ledoit-Wolf needs [>=2 windows, units]")
    n, N = X.shape
    centered = X - X.mean(axis=0, keepdims=True)
    S = centered.T @ centered / float(n)
    mu = float(np.trace(S)) / float(N)
    delta2 = float(np.sum((S - mu * np.eye(N)) ** 2))
    row_norm2 = np.sum(centered ** 2, axis=1)
    norms4 = float(np.sum(row_norm2 ** 2))  # sum_i ||x_i||^4
    gram2 = float(np.sum((centered @ centered.T) ** 2))
    beta2bar = (norms4 - gram2 / float(n)) / float(n * n)
    beta2 = min(beta2bar, delta2)
    shrinkage = 0.0 if delta2 <= 0.0 else min(1.0, beta2 / delta2)
    Sigma = (1.0 - shrinkage) * S + shrinkage * mu * np.eye(N)
    return Sigma, float(shrinkage)


def topk_subspace(covariance: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvectors/eigenvalues of a symmetric covariance, descending."""
    covariance = np.asarray(covariance, dtype=np.float64)
    k = int(k)
    _require(covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1], "covariance shape drift")
    _require(1 <= k <= covariance.shape[0], "k exceeds the covariance dimension")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:k]
    return eigenvectors[:, order], eigenvalues[order]


def effective_rank_measures(eigenvalues: np.ndarray) -> dict[str, float]:
    """Participation-ratio and entropy effective ranks of an eigenspectrum."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    _require(bool(np.all(eigenvalues >= -1e-12)), "negative eigenvalue in effective rank")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return {"participation_ratio": 0.0, "entropy_effective_rank": 0.0}
    participation = total ** 2 / float(np.sum(eigenvalues ** 2)) if np.sum(eigenvalues ** 2) > 0 else 0.0
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0]
    entropy = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    return {"participation_ratio": float(participation), "entropy_effective_rank": entropy}


def trial_windows(calib_trials: np.ndarray, m_trials: int) -> np.ndarray:
    """Flatten the first ``m_trials`` trials of the B3S stream to [m*T, N]."""
    calib = np.asarray(calib_trials, dtype=np.float64)
    _require(calib.ndim == 3, "calib_trials must be [trials, bins, units]")
    _require(1 <= int(m_trials) <= calib.shape[0], "trial prefix out of range")
    return calib[: int(m_trials)].reshape(-1, calib.shape[2])


# ---------------------------------------------------------------------------
# per-session analysis
# ---------------------------------------------------------------------------


def _angles_summary(angles: np.ndarray) -> dict[str, float]:
    angles = np.asarray(angles, dtype=np.float64)
    return {
        "angles_deg_ascending": [float(value) for value in angles],
        "max_angle_deg": float(angles.max()),
        "mean_angle_deg": float(angles.mean()),
        "min_angle_deg": float(angles.min()),
    }


def subspace_cell(
    calib_trials: np.ndarray,
    *,
    ks: Sequence[int] = KS,
    ms: Sequence[int] = MS,
    n_bootstrap: int = BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """One session's U-stability cell: (k, M) angle table + effective ranks."""
    calib = np.asarray(calib_trials, dtype=np.float64)
    _require(calib.ndim == 3, "calib_trials must be [trials, bins, units]")
    n_trials, n_bins, n_units = calib.shape
    rng = np.random.default_rng(seed)
    X_ref = calib.reshape(-1, n_units)
    cov_ref = sample_covariance(X_ref)
    cov_ref_lw, shrink_ref = ledoit_wolf_covariance(X_ref)
    U_ref = {int(k): topk_subspace(cov_ref, int(k))[0] for k in ks}
    U_ref_lw = {int(k): topk_subspace(cov_ref_lw, int(k))[0] for k in ks}
    result: dict[str, Any] = {
        "n_units": int(n_units),
        "n_trials": int(n_trials),
        "n_bins_per_trial": int(n_bins),
        "reference": {
            "windows": int(X_ref.shape[0]),
            "ledoit_wolf_shrinkage": float(shrink_ref),
            "effective_rank": effective_rank_measures(np.linalg.eigvalsh(cov_ref)[::-1]),
            "effective_rank_ledoit_wolf": effective_rank_measures(np.linalg.eigvalsh(cov_ref_lw)[::-1]),
        },
        "by_m": {},
    }
    for m in ms:
        m = int(m)
        X_m = trial_windows(calib, m)
        cov_m = sample_covariance(X_m)
        cov_m_lw, shrink_m = ledoit_wolf_covariance(X_m)
        entry: dict[str, Any] = {
            "windows": int(X_m.shape[0]),
            "windows_per_unit": float(X_m.shape[0] / n_units),
            "ledoit_wolf_shrinkage": float(shrink_m),
            "effective_rank": effective_rank_measures(np.linalg.eigvalsh(cov_m)[::-1]),
            "effective_rank_ledoit_wolf": effective_rank_measures(np.linalg.eigvalsh(cov_m_lw)[::-1]),
            "by_k": {},
        }
        for k in ks:
            k = int(k)
            U_m = topk_subspace(cov_m, k)[0]
            U_m_lw = topk_subspace(cov_m_lw, k)[0]
            boot_max: list[float] = []
            boot_mean: list[float] = []
            for _ in range(int(n_bootstrap)):
                resampled = rng.integers(0, m, size=m)
                X_b = calib[resampled].reshape(-1, n_units)
                cov_b, _ = ledoit_wolf_covariance(X_b)
                U_b = topk_subspace(cov_b, k)[0]
                angles_b = principal_angles_deg(U_b, U_ref[k])
                boot_max.append(float(angles_b.max()))
                boot_mean.append(float(angles_b.mean()))
            entry["by_k"][str(k)] = {
                "raw": _angles_summary(principal_angles_deg(U_m, U_ref[k])),
                "ledoit_wolf": _angles_summary(principal_angles_deg(U_m_lw, U_ref_lw[k])),
                "bootstrap_n": int(n_bootstrap),
                "bootstrap_seed": int(seed),
                "bootstrap_max_angle_deg_percentiles": {
                    "p2_5": float(np.percentile(boot_max, 2.5)),
                    "p50": float(np.percentile(boot_max, 50.0)),
                    "p97_5": float(np.percentile(boot_max, 97.5)),
                },
                "bootstrap_mean_angle_deg_percentiles": {
                    "p2_5": float(np.percentile(boot_mean, 2.5)),
                    "p50": float(np.percentile(boot_mean, 50.0)),
                    "p97_5": float(np.percentile(boot_mean, 97.5)),
                },
            }
        result["by_m"][str(m)] = entry
    return result


def summarize_sessions(cells: Mapping[str, Mapping]) -> dict:
    """Medians across sessions per (M, k), plus the reference effective ranks."""
    sessions = sorted(cells)
    _require(bool(sessions), "no Z4 cells to summarize")
    summary: dict[str, Any] = {"session_count": len(sessions), "by_m": {}}
    ms = sorted({int(m) for cell in cells.values() for m in cell["by_m"]}, key=str)
    ks = sorted({int(k) for cell in cells.values() for entry in cell["by_m"].values() for k in entry["by_k"]})
    for m in ms:
        entry: dict[str, Any] = {
            "median_effective_rank_participation": float(np.median([
                cells[s]["by_m"][str(m)]["effective_rank"]["participation_ratio"] for s in sessions
            ])),
            "median_effective_rank_entropy": float(np.median([
                cells[s]["by_m"][str(m)]["effective_rank"]["entropy_effective_rank"] for s in sessions
            ])),
            "median_ledoit_wolf_shrinkage": float(np.median([
                cells[s]["by_m"][str(m)]["ledoit_wolf_shrinkage"] for s in sessions
            ])),
            "median_n_units": float(np.median([cells[s]["n_units"] for s in sessions])),
            "by_k": {},
        }
        for k in ks:
            key = str(k)
            if not all(key in cells[s]["by_m"][str(m)]["by_k"] for s in sessions):
                continue
            raw_max = [cells[s]["by_m"][str(m)]["by_k"][key]["raw"]["max_angle_deg"] for s in sessions]
            raw_mean = [cells[s]["by_m"][str(m)]["by_k"][key]["raw"]["mean_angle_deg"] for s in sessions]
            lw_max = [cells[s]["by_m"][str(m)]["by_k"][key]["ledoit_wolf"]["max_angle_deg"] for s in sessions]
            boot_p50 = [cells[s]["by_m"][str(m)]["by_k"][key]["bootstrap_max_angle_deg_percentiles"]["p50"] for s in sessions]
            boot_hi = [cells[s]["by_m"][str(m)]["by_k"][key]["bootstrap_max_angle_deg_percentiles"]["p97_5"] for s in sessions]
            entry["by_k"][key] = {
                "median_max_angle_deg": float(np.median(raw_max)),
                "median_mean_angle_deg": float(np.median(raw_mean)),
                "median_max_angle_deg_ledoit_wolf": float(np.median(lw_max)),
                "median_bootstrap_p50_max_angle_deg": float(np.median(boot_p50)),
                "median_bootstrap_p97_5_max_angle_deg": float(np.median(boot_hi)),
                "per_session_max_angle_deg": {s: float(v) for s, v in zip(sessions, raw_max)},
                "per_session_mean_angle_deg": {s: float(v) for s, v in zip(sessions, raw_mean)},
            }
        summary["by_m"][str(m)] = entry
    summary["reference"] = {
        "median_effective_rank_participation": float(np.median([
            cells[s]["reference"]["effective_rank"]["participation_ratio"] for s in sessions
        ])),
        "median_effective_rank_entropy": float(np.median([
            cells[s]["reference"]["effective_rank"]["entropy_effective_rank"] for s in sessions
        ])),
        "median_ledoit_wolf_shrinkage": float(np.median([
            cells[s]["reference"]["ledoit_wolf_shrinkage"] for s in sessions
        ])),
    }
    return summary


def verdict(summary: Mapping) -> dict:
    """The pre-registered Z4 reading from the summarized medians."""
    def median_max(m: int, k: int = 6, lw: bool = False) -> float | None:
        try:
            entry = summary["by_m"][str(m)]["by_k"][str(k)]
        except KeyError:
            return None
        return float(entry["median_max_angle_deg_ledoit_wolf" if lw else "median_max_angle_deg"])

    primary = {
        "primary_scalar": "median over external-15 sessions of theta_max(U_M, U_ref) at k=6 (raw sample covariance)",
        "k6_median_max_angle_deg": {"M4": median_max(4), "M10": median_max(10)},
        "k6_median_max_angle_deg_ledoit_wolf": {"M4": median_max(4, lw=True), "M10": median_max(10, lw=True)},
        "thresholds_deg": {"M10": THRESHOLD_M10_DEG, "M4": THRESHOLD_M4_DEG},
    }
    m4, m10 = median_max(4), median_max(10)
    if m4 is None or m10 is None:
        return {**primary, "reading": "INCOMPLETE (k=6 cells missing)"}
    if m10 < THRESHOLD_M10_DEG and m4 < THRESHOLD_M4_DEG:
        reading = (
            "U is usable: median k=6 theta_max below the pre-registered "
            f"thresholds (M10 {m10:.1f} deg < {THRESHOLD_M10_DEG:.0f}, M4 {m4:.1f} deg "
            f"< {THRESHOLD_M4_DEG:.0f}) -> Path 1 proceed"
        )
        u_starves = False
    elif m10 >= THRESHOLD_M10_DEG and m4 >= RANDOM_ANGLE_DEG * 0.8:
        reading = (
            "U starves in the strict regime (angles near random at both M4 and "
            f"M10: {m4:.1f}/{m10:.1f} deg) -> Path 1 requires the Z6 contract "
            "answer (transductive evaluation-stream activity) or re-scopes to M30"
        )
        u_starves = True
    elif m10 < THRESHOLD_M10_DEG:
        reading = (
            f"MIXED: U is usable at M10 ({m10:.1f} deg < {THRESHOLD_M10_DEG:.0f}) "
            f"but starves at M4 ({m4:.1f} deg >= {THRESHOLD_M4_DEG:.0f}) -> "
            "Path 1 re-scopes to M10+ unless Z6 permits transductive U"
        )
        u_starves = False
    else:
        reading = (
            f"U is unstable: M10 {m10:.1f} deg >= {THRESHOLD_M10_DEG:.0f} and "
            f"M4 {m4:.1f} deg -> Path 1 infeasible in the strict regime without "
            "the transductive contract route"
        )
        u_starves = True
    return {**primary, "reading": reading, "u_starves_at_m4": m4 >= THRESHOLD_M4_DEG, "u_starves": u_starves}


def run_z4(session_calibs: Mapping[str, Any], **cell_kwargs: Any) -> dict:
    """Run the Z4 cells for ``{session: calib_trials}`` and summarize."""
    cells = {
        str(session): subspace_cell(calib, **cell_kwargs)
        for session, calib in sorted(session_calibs.items())
    }
    summary = summarize_sessions(cells)
    return {
        "schema": "calibration_gap_z4_subspace_v1",
        "status": "COMPLETE",
        "ks": list(KS),
        "ms": list(MS),
        "bootstraps": int(cell_kwargs.get("n_bootstrap", BOOTSTRAPS)),
        "bootstrap_seed": int(cell_kwargs.get("seed", BOOTSTRAP_SEED)),
        "activity_source": "b3s calibration stream (30 interpolated 100-bin trials), read-only",
        "cells": cells,
        "summary": summary,
        "verdict": verdict(summary),
    }
