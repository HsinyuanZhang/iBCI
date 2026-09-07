#!/usr/bin/env python3
"""Support-only diagnostic for the already-frozen M2 M24 K4 held-out result.

No query covariate or query target is loaded.  The optional final association
reads only the immutable, already-written aggregate delta table; it never
selects a threshold, fit, or fallback rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
from falcon_challenge.config import FalconTask  # noqa: E402
from falcon_challenge.dataloaders import load_nwb  # noqa: E402
from src.data.falcon_k4_features import (  # noqa: E402
    K4_ACTIVE_EPSILON, K4_BEHAVIOR_LEAD_BINS, K4_BLOCK_WIDTH_BINS, K4_RAW_BIN_MS,
    fit_train_k4_stats,
)
from src.data.falcon_t4_features import calibration_target_angles, t4_from_trial_sums  # noqa: E402


M = 24
SCREEN = "m2_m24_disjoint_heldout_v1"
FAILURE_SESSION = "ses-2020-10-30-Run1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def session_name(path: Path) -> str:
    return path.name.split("_")[1].split(".")[0]


def blocks(neural: np.ndarray, velocity: np.ndarray, trial_change: np.ndarray, *, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw contiguous K4 blocks for a fixed trial range, with no query access."""
    starts = np.flatnonzero(np.asarray(trial_change, dtype=bool))
    if lo < 0 or hi > len(starts) or hi <= lo:
        raise ValueError("invalid support trial range")
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(velocity) < K4_ACTIVE_EPSILON, axis=1)
    rates, ys = [], []
    for start, end in zip(starts[lo:hi], ends[lo:hi]):
        for left in range(int(start), int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1, K4_BLOCK_WIDTH_BINS):
            right, yl, yr = left + K4_BLOCK_WIDTH_BINS, left + K4_BEHAVIOR_LEAD_BINS, left + K4_BEHAVIOR_LEAD_BINS + K4_BLOCK_WIDTH_BINS
            if active[left:right].all() and active[yl:yr].all():
                rates.append(neural[left:right].sum(axis=0) / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0))
                ys.append(velocity[yl:yr].mean(axis=0))
    rate, y = np.asarray(rates, dtype=float), np.asarray(ys, dtype=float)
    if len(rate) < 3:
        raise ValueError("fewer than three active raw K4 blocks")
    return rate, y


def fit(rate: np.ndarray, y: np.ndarray) -> dict:
    x = np.column_stack([np.ones(len(y)), y])
    coef, *_ = np.linalg.lstsq(x, rate, rcond=None)
    pred, residual = x @ coef, rate - x @ coef
    rank = int(np.linalg.matrix_rank(x))
    condition = float(np.linalg.cond(x)) if rank == 3 else math.inf
    if rank != 3 or not math.isfinite(condition):
        raise ValueError("K4 support design must have rank 3 and finite condition")
    residual_var = np.mean(residual ** 2, axis=0)
    inv = np.linalg.inv(x.T @ x)
    weight_se = np.sqrt(np.maximum(residual_var[:, None] * np.diag(inv)[1:], 0.0))
    explained = np.sqrt(np.mean((pred - pred.mean(axis=0)) ** 2, axis=0))
    resid_rms = np.sqrt(residual_var)
    weights, intercept = coef[1:].T, coef[0]
    feature = np.column_stack([weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), intercept])
    count = np.maximum(rate * (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0), 0.0)
    mu = np.maximum(pred * (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0), 1e-8)
    poisson = 2.0 * np.where(count > 0, count * np.log(np.maximum(count, 1e-8) / mu) - (count - mu), mu)
    return {"feature": feature, "weights": weights, "intercept": intercept, "active_blocks": len(rate), "condition": condition,
            "velocity_covariance_eigenvalues": np.linalg.eigvalsh(np.cov(y, rowvar=False))[::-1],
            "residual_mse_mean": float(np.mean(residual_var)), "poisson_deviance_mean": float(np.mean(poisson)),
            "modulation_to_residual_median": float(np.median(explained / np.maximum(resid_rms, 1e-8))),
            "weight_se_l2_median": float(np.median(np.linalg.norm(weight_se, axis=1))),
            "weight_snr_median": float(np.median(np.linalg.norm(weights, axis=1) / np.maximum(np.linalg.norm(weight_se, axis=1), 1e-8)))}


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12)


def corr(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-12 and np.std(right) > 1e-12 else float("nan")


def t4_from_raw(neural: np.ndarray, trial_change: np.ndarray, angles: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(trial_change)[:M]
    ends = np.r_[np.flatnonzero(trial_change)[1:], len(neural)][:M]
    sums, lengths = [], []
    for start, end in zip(starts, ends):
        valid = min(int(end - start), 100)  # frozen native T4 exposure rule
        sums.append(neural[start:start + valid].sum(axis=0)); lengths.append(valid)
    return t4_from_trial_sums(np.asarray(sums), np.asarray(lengths), np.asarray(angles[:M]), source="support-only diagnostic")


def details(neural: np.ndarray, velocity: np.ndarray, change: np.ndarray, angles: np.ndarray, *, sign: float, k_mean: np.ndarray, k_std: np.ndarray, t_mean: np.ndarray, t_std: np.ndarray) -> dict:
    whole = fit(*blocks(neural, velocity, change, lo=0, hi=M))
    first, second = fit(*blocks(neural, velocity, change, lo=0, hi=12)), fit(*blocks(neural, velocity, change, lo=12, hi=M))
    t4 = t4_from_raw(neural, change, angles)
    agreement = cosine_rows(sign * whole["weights"], t4[:, :2])
    weighted = float(np.sum(sign * whole["weights"] * t4[:, :2]) / np.maximum(np.sum(np.linalg.norm(whole["weights"], axis=1) * np.linalg.norm(t4[:, :2], axis=1)), 1e-12))
    stability = cosine_rows(first["weights"], second["weights"])
    z_k, z_t = (whole["feature"] - k_mean) / k_std, (t4 - t_mean) / t_std
    eig = whole["velocity_covariance_eigenvalues"]
    return {"active_blocks": whole.pop("active_blocks"), "design_condition": whole.pop("condition"),
            "velocity_covariance_eigenvalues": eig.tolist(), "velocity_balance_min_over_max": float(eig[-1] / max(eig[0], 1e-12)),
            **{key: value for key, value in whole.items() if key not in {"feature", "weights", "intercept", "velocity_covariance_eigenvalues"}},
            "first12_vs_second12_weight_cosine_median": float(np.median(stability)), "first12_vs_second12_intercept_correlation": corr(first["intercept"], second["intercept"]),
            "k4_t4_signed_weight_cosine_median": float(np.median(agreement)), "k4_t4_signed_weighted_cosine": weighted,
            "k4_t4_wx_a_correlation": corr(sign * whole["weights"][:, 0], t4[:, 0]), "k4_t4_wy_c_correlation": corr(sign * whole["weights"][:, 1], t4[:, 1]),
            "normalized_k4_t4_disagreement_median_l2": float(np.median(np.linalg.norm(z_k - z_t, axis=1))),
            "k4_ood_z_l2_median": float(np.median(np.linalg.norm(z_k, axis=1))),
            "k4_feature": whole["feature"], "t4_feature": t4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953")
    parser.add_argument("--out", type=Path, default=ROOT / "sua_exploration/results/m2_m24_k4_support_failure_v1/diagnostic_v2.json")
    args = parser.parse_args(); data_dir = args.data_dir.resolve()
    heldin = sorted(data_dir.rglob("*held-in-calib*.nwb")); heldout = sorted(data_dir.rglob("*held-out-calib*.nwb"))
    if len(heldin) != 7 or len(heldout) != 6: raise ValueError("expected M2's seven held-in and six held-out calibration files")
    internal_path = ROOT / "sua_exploration/results/m2_m24_disjoint_source_v1/aggregate_internal.json"
    heldout_path = ROOT / f"sua_exploration/results/{SCREEN}/aggregate_heldout.json"
    internal = json.loads(internal_path.read_text())
    train_names = set(json.loads((Path(internal["arms"]["k4"]["path"]) / "split_manifest.json").read_text())["train_sessions"])
    raw = {}
    for path in heldin + heldout:
        neural, velocity, change, _ = load_nwb(path, FalconTask.m2)
        raw[session_name(path)] = (np.asarray(neural), np.asarray(velocity), np.asarray(change, bool), calibration_target_angles(path, "m2"))
    train_fit = {name: fit(*blocks(*raw[name][:3], lo=0, hi=M))["feature"] for name in train_names}
    k_mean, k_std = fit_train_k4_stats(train_fit, sorted(train_names)); t_train = np.concatenate([t4_from_raw(raw[n][0], raw[n][2], raw[n][3]) for n in sorted(train_names)])
    t_mean, t_std = t_train.mean(0), t_train.std(0); t_std[t_std <= 1e-6] = 1.0
    raw_agreement = []
    for name in sorted(train_names):
        x = fit(*blocks(*raw[name][:3], lo=0, hi=M)); t = t4_from_raw(raw[name][0], raw[name][2], raw[name][3]); raw_agreement.extend(cosine_rows(x["weights"], t[:, :2]))
    sign = 1.0 if np.nanmedian(raw_agreement) >= 0 else -1.0
    records = {name: details(*values, sign=sign, k_mean=k_mean, k_std=k_std, t_mean=t_mean, t_std=t_std) for name, values in raw.items()}
    frozen = json.loads(heldout_path.read_text())["paired_deltas_r2"]
    heldout_names = {session_name(path) for path in heldout}
    for name in sorted(heldout_names): records[name]["frozen_query_delta_k4_minus_t4"] = frozen["K4_minus_T4"]["per_session"][name]
    for value in records.values(): value.pop("k4_feature"); value.pop("t4_feature")
    key = "k4_t4_signed_weighted_cosine"
    heldout_order = sorted(heldout_names, key=lambda n: records[n][key])
    payload = {"schema_version": 1, "scope": "support-only descriptors; frozen query deltas are appended after fitting only", "M": M,
               "hidden_evalai_evaluated": False, "heldout_calibration_files_previously_opened_in_frozen_replay": True,
               "query_rows_used_in_descriptors": False, "descriptor_trial_range": [0, M], "train_normalization_sessions": sorted(train_names),
               "fixed_coordinate_sign_from_train_median": sign, "coordinate_sign_fit_sessions": sorted(train_names), "records": records,
               "input_provenance": {"internal_aggregate": {"path": str(internal_path.resolve()), "sha256": sha256(internal_path)},
                                    "frozen_heldout_aggregate": {"path": str(heldout_path.resolve()), "sha256": sha256(heldout_path)},
                                    "heldin_calibration_files": [{"session": session_name(p), "path": str(p.resolve()), "sha256": sha256(p)} for p in heldin],
                                    "heldout_calibration_files": [{"session": session_name(p), "path": str(p.resolve()), "sha256": sha256(p)} for p in heldout]},
               "predeclared_mechanistic_rank_metric": key, "heldout_rank_low_to_high": heldout_order,
               "failure_session": FAILURE_SESSION, "failure_session_is_lowest_agreement": heldout_order[0] == FAILURE_SESSION,
               "fallback_status": "no threshold or fallback formula fitted; any later fallback based on this result is post-hoc exploratory"}
    if args.out.exists(): raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__": main()
