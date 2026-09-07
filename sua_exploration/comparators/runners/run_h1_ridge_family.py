#!/usr/bin/env python3
"""Run the H1 ridge-family fairness experiment and write an immutable receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sua_exploration.comparators.core import h1_ridge_family as family


DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/h1_ridge_family"
RECEIPT_PATH = OUTPUT_DIR / "h1_ridge_family_receipt.json"
V2R2_RECEIPT = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v2r2/h1_ridge_baseline_v2r2_receipt.json"
LOADER = REPO_ROOT / "SPINT-main/src/data/h1_m4_eb_pilot.py"
IMPLEMENTATION = REPO_ROOT / "sua_exploration/mc_maze/h1_ridge_family.py"
RUNNER = Path(__file__)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(path, 0o444)


def load_splits() -> tuple[tuple[str, ...], dict[str, family.SessionSplit], list[tuple[str, int]]]:
    from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, WINDOW as PILOT_WINDOW, load_target_records

    require(PILOT_WINDOW == family.WINDOW, "H1 query-window length drift")
    records = load_target_records(DATA_DIR)
    target_names = tuple(H1_M4_FOLD0_TARGET)
    require(set(records) == set(target_names), "fold-0 target session set drift")
    splits = {name: family.build_session_split(records[name]) for name in target_names}
    scored_windows: list[tuple[str, int]] = []
    for name in target_names:
        split = splits[name]
        for output_bin in split.query_output_bins:
            scored_windows.append((name, int(output_bin - family.WINDOW + 1)))
    query_sha = family.window_manifest_sha256(scored_windows)
    require(query_sha == family.EXPECTED_QUERY_SHA256, "actual scored-query manifest SHA drift")
    return target_names, splits, scored_windows


def compute_experiment() -> dict[str, Any]:
    from src.data.h1_m4_eb_pilot import load_target_records

    target_names, splits, scored_windows = load_splits()
    records = load_target_records(DATA_DIR)
    arms: dict[str, Any] = {}
    for arm_name in family.all_arm_names():
        arms[arm_name] = family.run_arm(arm_name, splits)

    sealed = arms["ridge_v2r2_sealed"]["pooled"]["r2"]
    require(abs(sealed - family.SEALED_POOLED_R2) <= 1.0e-9, f"integrity gate failed: {sealed}")

    best_arm = max(arms, key=lambda name: float(arms[name]["pooled"]["r2"]))
    best_r2 = float(arms[best_arm]["pooled"]["r2"])
    sealed_gap_to_carrier = family.CARRIER_HSE5_POOLED_R2 - family.SEALED_POOLED_R2
    best_gap_to_carrier = family.CARRIER_HSE5_POOLED_R2 - best_r2

    return {
        "schema": "h1_ridge_family_v1",
        "status": "COMPLETED_CPU_ONLY_FAIRNESS_EXPERIMENT",
        "date": "2026-08-12",
        "definition": (
            "Fold-0 two-recording H1 ridge-family fairness experiment on the sealed 8,965-window "
            "strict post-support query.  All arms fit per session on the four-trial calibration "
            "block with closed-form ridge only.  Hyperparameter and PCA selection consume "
            "calibration-block data only."
        ),
        "scope": {
            "cuda_used": False,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "query_windows": 8965,
            "query_manifest_sha256": family.EXPECTED_QUERY_SHA256,
            "sessions": list(target_names),
            "data_root": str(DATA_DIR),
        },
        "sealed_reference": {
            "pooled_r2": family.SEALED_POOLED_R2,
            "per_session_r2": dict(family.SEALED_PER_SESSION_R2),
            "carrier_hse5_pooled_r2": family.CARRIER_HSE5_POOLED_R2,
            "source_receipt": str(V2R2_RECEIPT),
        },
        "arms": arms,
        "headline": {
            "best_arm": best_arm,
            "best_pooled_r2": best_r2,
            "sealed_pooled_r2": float(family.SEALED_POOLED_R2),
            "carrier_hse5_pooled_r2": family.CARRIER_HSE5_POOLED_R2,
            "delta_best_minus_sealed": best_r2 - float(family.SEALED_POOLED_R2),
            "gap_sealed_to_carrier": sealed_gap_to_carrier,
            "gap_best_to_carrier": best_gap_to_carrier,
            "gap_change_best_minus_sealed": sealed_gap_to_carrier - best_gap_to_carrier,
        },
        "selection_policy": {
            "pca_basis": "eval-valid neural bins from the four support trials only",
            "lambda_grid": [float(value) for value in family.LAMBDA_GRID],
            "pca_k_grid": list(family.PCA_K_GRID),
            "history_bins_grid": list(family.W_GRID),
            "cv_protocol": "leave-one-support-trial-out on calibration rows only",
            "query_influence": "none",
        },
        "input_bindings": {
            "runner_sha256": sha256_file(RUNNER),
            "implementation_sha256": sha256_file(IMPLEMENTATION),
            "loader_sha256": sha256_file(LOADER),
            "v2r2_receipt_sha256": sha256_file(V2R2_RECEIPT),
            "nwb_sha256": {name: records[name].input_sha256 for name in target_names},
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_environment": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        },
        "scored_windows": len(scored_windows),
    }


def run_mode() -> None:
    require(not RECEIPT_PATH.exists(), f"refusing to overwrite {RECEIPT_PATH}")
    started = time.monotonic()
    evidence = compute_experiment()
    evidence["elapsed_seconds"] = time.monotonic() - started
    receipt_bytes = family.canonical_json_bytes(evidence)
    write_immutable(RECEIPT_PATH, receipt_bytes)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "integrity_gate_pooled_r2": evidence["arms"]["ridge_v2r2_sealed"]["pooled"]["r2"],
                "integrity_gate_delta": evidence["arms"]["ridge_v2r2_sealed"]["pooled"]["r2"] - family.SEALED_POOLED_R2,
                "best_arm": evidence["headline"]["best_arm"],
                "best_pooled_r2": evidence["headline"]["best_pooled_r2"],
                "receipt": str(RECEIPT_PATH),
                "receipt_sha256": sha256_bytes(receipt_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run",))
    args = parser.parse_args()
    if args.mode == "run":
        run_mode()


if __name__ == "__main__":
    main()
