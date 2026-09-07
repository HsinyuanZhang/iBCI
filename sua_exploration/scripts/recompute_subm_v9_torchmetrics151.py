#!/usr/bin/env python3
"""Authoritative local TorchMetrics-1.5.1 finalizer for external sub-M V9.

The remote RTX 5070 Ti machine is used only for forward inference.  This
program reopens the complete 270 prediction/target artifacts after they have
been copied to the local host, recomputes every session R2 with the project's
frozen TorchMetrics implementation, and reports the two predeclared paired
contrasts without consulting a remote metric scalar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np


# Support direct execution from any working directory.  The legacy scorer
# imports both ``sua_exploration.*`` and its historical top-level ``mc_maze``
# package name, so both roots must be importable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "sua_exploration"):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)


VIEWS = ("sua", "pseudo_mua")
ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
SEEDS = (42, 43, 44)
EXPECTED_SESSIONS = 15
EXPECTED_CELLS = EXPECTED_SESSIONS * len(VIEWS) * len(ARMS) * len(SEEDS)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 68_820_260_805


class FinalizeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


def artifact_path(root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"


def commit_path(root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return root / "commits" / asset_id / view / arm / f"seed_{seed}.json"


def load_artifact(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["predictions"]
            target = archive["targets"]
    except (OSError, ValueError, KeyError) as exc:
        raise FinalizeError(f"invalid artifact: {path}") from exc
    require(
        prediction.dtype == target.dtype == np.dtype(np.float32),
        f"artifact dtype drift: {path}",
    )
    require(
        prediction.shape == target.shape == (expected_rows, 2),
        f"artifact shape drift: {path}",
    )
    require(
        prediction.flags.c_contiguous and target.flags.c_contiguous,
        f"artifact layout drift: {path}",
    )
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), f"nonfinite artifact: {path}")
    return prediction, target


def hierarchical_bootstrap(delta: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    """Resample sessions, then seeds within each sampled session."""
    require(delta.shape == (EXPECTED_SESSIONS, len(SEEDS)), "bootstrap grid must be [15,3]")
    values = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    chunk = 10_000
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        stop = min(start + chunk, BOOTSTRAP_DRAWS)
        count = stop - start
        session_index = rng.integers(0, EXPECTED_SESSIONS, size=(count, EXPECTED_SESSIONS))
        seed_index = rng.integers(0, len(SEEDS), size=(count, EXPECTED_SESSIONS, len(SEEDS)))
        sampled_sessions = np.take(delta, session_index, axis=0)
        sampled = np.take_along_axis(sampled_sessions, seed_index, axis=2)
        values[start:stop] = sampled.mean(axis=(1, 2), dtype=np.float64)
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def summarize_contrast(
    *, left: np.ndarray, right: np.ndarray, absolute_t4: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    require(left.shape == right.shape == absolute_t4.shape == (EXPECTED_SESSIONS, len(SEEDS)), "contrast grid drift")
    delta = left - right
    seed_means = delta.mean(axis=0, dtype=np.float64)
    session_means = delta.mean(axis=1, dtype=np.float64)
    bootstrap = hierarchical_bootstrap(delta, rng)
    grand = float(delta.mean(dtype=np.float64))
    absolute_seed_means = absolute_t4.mean(axis=0, dtype=np.float64)
    gates = {
        "grand_mean_at_least_0_03": grand >= 0.03,
        "three_of_three_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "at_least_12_of_15_session_means_positive": int(np.sum(session_means > 0.0)) >= 12,
        "bootstrap_lower_95_positive": float(bootstrap["lower_95"]) > 0.0,
        "absolute_t4_grand_positive": float(absolute_t4.mean(dtype=np.float64)) > 0.0,
        "absolute_t4_three_seed_means_positive": bool(np.all(absolute_seed_means > 0.0)),
    }
    return {
        "grand_mean_delta_r2": grand,
        "seed_mean_delta_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(SEEDS)},
        "session_mean_delta_r2": [float(value) for value in session_means],
        "positive_session_count": int(np.sum(session_means > 0.0)),
        "bootstrap": bootstrap,
        "absolute_t4_grand_mean_r2": float(absolute_t4.mean(dtype=np.float64)),
        "absolute_t4_seed_mean_r2": {
            str(seed): float(absolute_seed_means[index]) for index, seed in enumerate(SEEDS)
        },
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> str:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "aggregate mode drift")
    return hashlib.sha256(raw).hexdigest()


def finalize(root: Path, output: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    run_manifest_path = root / "run_manifest.json"
    manifest = read_json(run_manifest_path)
    contract = manifest.get("contract")
    require(isinstance(contract, dict), "run manifest misses contract")
    cohort = contract.get("cohort")
    require(isinstance(cohort, list) and len(cohort) == EXPECTED_SESSIONS, "run manifest cohort drift")
    require(manifest.get("expected_cell_count") == EXPECTED_CELLS, "run manifest cell count drift")

    # Import only after the full topology and local environment are known.
    import torch
    import torchmetrics
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu

    require(torchmetrics.__version__ == "1.5.1", f"authoritative finalizer requires torchmetrics 1.5.1, got {torchmetrics.__version__}")
    grids = {
        arm: {view: np.empty((EXPECTED_SESSIONS, len(SEEDS)), dtype=np.float64) for view in VIEWS}
        for arm in ARMS
    }
    cells: list[dict[str, Any]] = []
    targets_by_asset: dict[str, bytes] = {}
    for session_index, session in enumerate(cohort):
        asset_id = str(session["asset_id"])
        session_id = str(session["session_id"])
        query_count = int(session["query_window_count"])
        for view in VIEWS:
            for arm in ARMS:
                for seed_index, seed in enumerate(SEEDS):
                    artifact = artifact_path(root, asset_id, view, arm, seed)
                    commit = commit_path(root, asset_id, view, arm, seed)
                    require(commit.is_file() and not commit.is_symlink(), f"missing/unsafe commit: {commit}")
                    commit_payload = read_json(commit)
                    prediction, target = load_artifact(artifact, query_count)
                    require(commit_payload.get("artifact", {}).get("sha256") == sha256_file(artifact), "artifact SHA drift")
                    target_bytes = target.tobytes(order="C")
                    require(
                        commit_payload.get("query_behavior_trace_sha256") == hashlib.sha256(target_bytes).hexdigest(),
                        "target trace drift",
                    )
                    peer = targets_by_asset.setdefault(asset_id, target_bytes)
                    require(peer == target_bytes, f"paired target mismatch: {asset_id}")
                    r2 = recompute_torchmetrics_r2_cpu(prediction, target)
                    require(math.isfinite(r2), "nonfinite authoritative R2")
                    grids[arm][view][session_index, seed_index] = r2
                    cells.append(
                        {
                            "asset_id": asset_id,
                            "session_id": session_id,
                            "view": view,
                            "arm": arm,
                            "seed": seed,
                            "query_window_count": query_count,
                            "r2": r2,
                        }
                    )
    require(len(cells) == EXPECTED_CELLS, "authoritative cell count drift")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    contrasts: dict[str, dict[str, Any]] = {}
    for label, right_arm in (
        ("shared_t4_minus_shared_zero4", "shared_zero4"),
        ("shared_t4_minus_shared_ts4", "shared_ts4"),
    ):
        contrasts[label] = {}
        for view in VIEWS:
            contrasts[label][view] = summarize_contrast(
                left=grids["shared_t4"][view],
                right=grids[right_arm][view],
                absolute_t4=grids["shared_t4"][view],
                rng=rng,
            )
    payload = {
        "schema": "dandi_000688_subm_co_v9_torchmetrics151_authoritative_v1",
        "status": "FULL_270_LOCAL_TORCHMETRICS_1_5_1_FINALIZED",
        "source_run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256_file(run_manifest_path),
            "contract_sha256": manifest.get("contract_sha256"),
            "input_manifest_sha256": manifest.get("input_manifest_sha256"),
        },
        "environment": {
            "torch": str(torch.__version__),
            "torchmetrics": str(torchmetrics.__version__),
            "numpy": str(np.__version__),
            "device": "cpu",
        },
        "verified_cell_count": len(cells),
        "query_windows_per_view": int(sum(int(row["query_window_count"]) for row in cohort)),
        "contrasts": contrasts,
        "cells": cells,
    }
    digest = write_exclusive(output.expanduser().resolve(), payload)
    return {"status": payload["status"], "output": str(output.expanduser().resolve()), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_root, args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
