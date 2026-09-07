#!/usr/bin/env python3
"""Source-only grouped audit of CP-profile redundancy with frozen T4/activity.

This is a post-result diagnostic.  It never constructs the formal-test dataset,
never updates a model, and prints JSON to stdout rather than creating a result
root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ALPHAS = (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5)


def _fit_ridge(rows: list[tuple[str, np.ndarray, np.ndarray]], alpha: float):
    x = np.concatenate([row[1] for row in rows], axis=0).astype(np.float64)
    y = np.concatenate([row[2] for row in rows], axis=0).astype(np.float64)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0, ddof=0)
    x_scale[x_scale < 1.0e-8] = 1.0
    y_mean = y.mean(axis=0)
    xz = (x - x_mean) / x_scale
    gram = xz.T @ xz
    rhs = xz.T @ (y - y_mean)
    weight = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)

    def predict(value: np.ndarray) -> np.ndarray:
        return ((np.asarray(value, dtype=np.float64) - x_mean) / x_scale) @ weight + y_mean

    return predict


def _session_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.square(target - prediction).sum()
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum()
    if not total > 0.0:
        raise RuntimeError("profile target variance is degenerate")
    return float(1.0 - residual / total)


def _choose_alpha(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[float, list[dict[str, float]]]:
    sessions = sorted({row[0] for row in rows})
    folds = {session: index % 5 for index, session in enumerate(sessions)}
    trace = []
    for alpha in ALPHAS:
        values = []
        for fold in range(5):
            fit_rows = [row for row in rows if folds[row[0]] != fold]
            held_rows = [row for row in rows if folds[row[0]] == fold]
            predict = _fit_ridge(fit_rows, alpha)
            values.extend(_session_r2(y, predict(x)) for _, x, y in held_rows)
        trace.append({"alpha": float(alpha), "mean_equal_session_r2": float(np.mean(values))})
    best = max(trace, key=lambda row: (row["mean_equal_session_r2"], -row["alpha"]))
    return float(best["alpha"]), trace


def _audit_feature(
    train: list[tuple[str, np.ndarray, np.ndarray]],
    validation: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, object]:
    alpha, trace = _choose_alpha(train)
    predict = _fit_ridge(train, alpha)
    train_r2 = {session: _session_r2(y, predict(x)) for session, x, y in train}
    validation_r2 = {session: _session_r2(y, predict(x)) for session, x, y in validation}
    return {
        "selected_alpha": alpha,
        "train_inner_group_cv_trace": trace,
        "train_refit_equal_session_r2": float(np.mean(list(train_r2.values()))),
        "validation_equal_session_r2": float(np.mean(list(validation_r2.values()))),
        "validation_residual_variance_fraction": float(1.0 - np.mean(list(validation_r2.values()))),
        "validation_per_session_r2": validation_r2,
    }


def execute(repo_root: Path, seed: int) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("diagnostic requires CUDA_VISIBLE_DEVICES='' and CPU-only execution")
    repo_root = repo_root.resolve()
    import torch

    dm = data.prepare_datamodule(repo_root)
    materials = data.materialize(repo_root, dm)
    if dm.test_dataset is not None:
        raise RuntimeError("formal-test dataset unexpectedly constructed")
    student = runner._prepare_student(repo_root, seed, torch.device("cpu"))
    cache = runner._session_cache(materials, student, torch.device("cpu"), seed)

    results: dict[str, object] = {}
    for horizon, target_key in ((10, "profile10"), (30, "profile30")):
        rows: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
            "T4": [],
            "H": [],
            "T4_PLUS_H": [],
        }
        split_by_session = {}
        for session, material in sorted(materials.items()):
            state = cache[session]
            carrier = state["carrier"][0].detach().cpu().numpy().astype(np.float64)
            hidden = state["mean_feature"][0].detach().cpu().numpy().astype(np.float64)
            target = np.asarray(getattr(material, target_key)[:, :2], dtype=np.float64)
            split_by_session[session] = material.split
            rows["T4"].append((session, carrier, target))
            rows["H"].append((session, hidden, target))
            rows["T4_PLUS_H"].append((session, np.concatenate((carrier, hidden), axis=1), target))
        results[f"M{horizon}"] = {}
        for feature_name, feature_rows in rows.items():
            train = [row for row in feature_rows if split_by_session[row[0]] == "train"]
            validation = [row for row in feature_rows if split_by_session[row[0]] == "val"]
            results[f"M{horizon}"][feature_name] = _audit_feature(train, validation)

    return {
        "schema": "dandi688_cp_profile_redundancy_diagnostic_v1",
        "status": "POST_RESULT_SOURCE_ONLY_DIAGNOSTIC",
        "seed": seed,
        "profile_targets": ["robust_z_high_minus_low_mean", "robust_z_logmean_difference"],
        "hidden_authority": "frozen_B3S_pre_pool_mean_over_first30_activity_trials",
        "fit_sessions": 27,
        "validation_sessions": 6,
        "formal_test_names_only": list(dm.session_splits["test"]),
        "formal_test_files_opened": False,
        "model_updates": 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "INERT", "formal_test_files_opened": False}, sort_keys=True))
        return
    global np, data, plan, runner
    import numpy as np
    from mc_maze.dandi688_cp_film_v1 import data, plan, runner

    print(json.dumps(execute(args.repo_root, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
