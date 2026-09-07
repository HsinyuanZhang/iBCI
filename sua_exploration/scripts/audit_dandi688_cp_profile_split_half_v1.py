#!/usr/bin/env python3
"""Read-only split-half reliability audit for the DANDI 000688 CP profile.

The audit opens only the frozen source train/validation roster, performs no
model construction or update, and writes no result artifact. Each half
recomputes its own speed quartiles and robust-z normalization so the measured
correlation includes the complete deployed profile estimator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise RuntimeError("split-half profile geometry mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("split-half profile is nonfinite")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        raise RuntimeError("split-half profile column is constant")
    value = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(value):
        raise RuntimeError("split-half correlation is nonfinite")
    return value


def _profile(record, trials, profile_from_bins) -> np.ndarray:
    bins = np.concatenate(
        [np.arange(int(row["start"]), int(row["stop"]), dtype=np.int64) for row in trials]
    )
    profile, _ = profile_from_bins(record.neural[bins], record.behavior[bins])
    # The frozen V1 mask leaves only delta and log-mean-difference active.
    return np.ascontiguousarray(profile[:, :2])


def execute(repo_root: Path) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("diagnostic requires CUDA_VISIBLE_DEVICES='' and CPU-only execution")

    from mc_maze.dandi688_cp_film_v1 import core, data, plan
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    repo_root = repo_root.resolve()
    dm = data.prepare_datamodule(repo_root)
    if dm.test_dataset is not None:
        raise RuntimeError("formal-test dataset unexpectedly constructed")

    results: dict[str, object] = {}
    for horizon in (10, 30):
        rows = []
        for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
            if dataset is None:
                raise RuntimeError(f"missing {split} dataset")
            for session, record in sorted(dataset.sessions.items()):
                path = repo_root / plan.DATA_RELATIVE / f"{session}_behavior+ecephys.nwb"
                trials = list_datamodule_rewarded_trials(
                    path,
                    bin_size_ms=plan.BIN_MS,
                    window_size=plan.WINDOW,
                    trial_result_filter="R",
                )[:horizon]
                if len(trials) != horizon:
                    raise RuntimeError(f"{session}: insufficient M{horizon} rewarded trials")
                odd = _profile(record, trials[0::2], core.profile_from_bins)
                even = _profile(record, trials[1::2], core.profile_from_bins)
                column_r = [_correlation(odd[:, index], even[:, index]) for index in range(2)]
                rows.append(
                    {
                        "session": session,
                        "split": split,
                        "units": int(odd.shape[0]),
                        "column_r": column_r,
                        "mean_column_r": float(np.mean(column_r)),
                    }
                )

        values = np.asarray([value for row in rows for value in row["column_r"]], dtype=np.float64)
        mean_r = float(values.mean())
        results[f"M{horizon}"] = {
            "definition": (
                "odd_vs_even_chronological_trials; each half independently recomputes "
                "speed quartiles and robust-z; active profile columns only"
            ),
            "sessions": rows,
            "all_session_column_mean_r": mean_r,
            "all_session_column_median_r": float(np.median(values)),
            "session_mean_r_median": float(np.median([row["mean_column_r"] for row in rows])),
            "spearman_brown_from_mean_r": float(2.0 * mean_r / (1.0 + mean_r)),
            "positive_session_columns": int(np.count_nonzero(values > 0.0)),
            "total_session_columns": int(values.size),
        }

    return {
        "schema": "dandi688_cp_profile_split_half_v1",
        "status": "POST_RESULT_SOURCE_ONLY_DIAGNOSTIC",
        "profile_columns": ["robust_z_high_minus_low_mean", "robust_z_logmean_difference"],
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "INERT", "formal_test_files_opened": False}, sort_keys=True))
        return
    print(json.dumps(execute(args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
