"""Explicit standard / historical R² aggregation for the shared local face."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import r2_score


def array_sha(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def score_arrays(target: np.ndarray, prediction: np.ndarray) -> dict:
    y, p = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    if y.shape != p.shape or y.ndim != 2 or len(y) < 2:
        raise ValueError("target and prediction must have matching [N,O] shapes, N >= 2")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite target or prediction")
    sse = float(np.square(y - p).sum())
    standard_sst = float(np.square(y - y.mean(axis=0)).sum())
    legacy_sst = float(np.square(y - y.mean()).sum())
    if min(standard_sst, legacy_sst) <= 0:
        raise ValueError("constant target has undefined aggregate R²")
    return {
        "standard_variance_weighted_r2": float(r2_score(y, p, multioutput="variance_weighted")),
        "legacy_flattened_r2": 1.0 - sse / legacy_sst,
        "sse": sse,
        "standard_sst": standard_sst,
        "legacy_sst": legacy_sst,
        "n_windows": len(y),
    }


def prediction_report(task: str, items: Mapping[str, dict], predictions: Mapping[str, np.ndarray]) -> dict:
    if set(items) != set(predictions):
        raise ValueError("prediction roster differs from evaluation roster")
    rows, all_y, all_p = {}, [], []
    grouped_y, grouped_p = {}, {}
    for session, item in items.items():
        y = np.asarray(item["Y"], np.float32)
        p = np.asarray(predictions[session], np.float32)
        row = score_arrays(y, p)
        row.update(target_sha256=array_sha(y), prediction_sha256=array_sha(p))
        rows[session] = row
        all_y.append(y)
        all_p.append(p)
        if task == "h1":
            from falcon_challenge.config import FalconConfig, FalconTask
            basename = Path(item["support_provenance"]["raw_nwb"]).stem
            group = FalconConfig(FalconTask.h1).hash_dataset(basename).split("_set_")[0]
            grouped_y.setdefault(group, []).append(y)
            grouped_p.setdefault(group, []).append(p)
    result = {
        "per_session": rows,
        "standard_equal_session_mean": float(np.mean([x["standard_variance_weighted_r2"] for x in rows.values()])),
        "legacy_equal_session_mean": float(np.mean([x["legacy_flattened_r2"] for x in rows.values()])),
        "pooled": score_arrays(np.concatenate(all_y), np.concatenate(all_p)),
        "n_sessions": len(rows), "n_windows": sum(x["n_windows"] for x in rows.values()),
        "selection_metric": "standard per-output-centered variance-weighted R²; equal source-session mean",
    }
    if task == "h1":
        grouped = {g: score_arrays(np.concatenate(grouped_y[g]), np.concatenate(grouped_p[g])) for g in grouped_y}
        result["grouped_seven"] = {
            "per_group": grouped,
            "standard_mean": float(np.mean([v["standard_variance_weighted_r2"] for v in grouped.values()])),
            "legacy_mean": float(np.mean([v["legacy_flattened_r2"] for v in grouped.values()])),
            "standard_sd_population": float(np.std([v["standard_variance_weighted_r2"] for v in grouped.values()])),
            "n_groups": len(grouped),
        }
    return result
