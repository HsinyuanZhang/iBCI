#!/usr/bin/env python3
"""Run RT classical comparators on the sealed Stage-2 outer-LOSO matrix."""
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
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "streaming_calibration_exp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sua_exploration.mc_maze import rt_classical_comparators as core


DEFAULT_DATA_DIR = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_STAGE2_CELL_ROOT = (
    REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/cells"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/rt_classical_comparators"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_immutable(path: Path, payload: bytes) -> str:
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
    return hashlib.sha256(payload).hexdigest()


def load_rt_sessions(data_dir: Path) -> dict[str, Path]:
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions

    indexed = {core.session_name_from_nwb_path(Path(path)): Path(path) for path in find_rt_sessions(data_dir)}
    require(len(indexed) == core.EXPECTED_FOLDS, f"expected {core.EXPECTED_FOLDS} RT sessions, found {len(indexed)}")
    return indexed


def run_all(
    *,
    data_dir: Path,
    stage2_cell_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from streaming_calibration_exp.src.data.rt_k4_loader import load_rt_session

    sessions = load_rt_sessions(data_dir)
    folds = []
    for fold in range(core.EXPECTED_FOLDS):
        sealed = core.load_sealed_stage2_cell(stage2_cell_root, fold)
        nwb_path = sessions[sealed["session_name"]]
        folds.append(
            core.evaluate_fold(
                fold=fold,
                nwb_path=nwb_path,
                stage2_cell_root=stage2_cell_root,
                raw_loader=load_rt_session,
            )
        )
    aggregate = core.aggregate_fold_results(folds)
    receipt = {
        "schema": "rt_classical_comparators_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": time.strftime("%Y-%m-%d"),
        "scope": {
            "cuda_used": False,
            "outer_loso_folds": core.EXPECTED_FOLDS,
            "support_trials": core.CALIBRATION_TRIALS,
            "query_start_trial": core.QUERY_START_TRIAL,
            "window_bins": core.WINDOW_BINS,
            "seed": core.SEED,
        },
        "sealed_rt_references": dict(core.SEALED_RT_REFERENCES),
        "query_identity_binding": aggregate["query_identity"],
        "results": aggregate,
        "per_fold": [
            {
                "fold": fold.fold,
                "session_name": fold.session_name,
                "query_identity_bound": fold.query_identity_bound,
                "sealed_query_identity": fold.sealed_query_identity,
                "t4d_reference_r2": fold.t4d_reference_r2,
                "ridge_w50_lambda1_r2": fold.ridge_fixed_lambda.r2,
                "ridge_w50_lambda_cv_r2": fold.ridge_cv_lambda.r2,
                "ridge_w50_lambda_cv_selected": fold.ridge_cv_lambda.normalized_lambda,
                "ridge_w50_lambda1_observations_per_parameter": fold.ridge_fixed_lambda.observations_per_parameter,
                "ridge_w50_lambda1_supervision_coordinates": fold.ridge_fixed_lambda.supervision_coordinates_consumed,
                "native_direction_audit": {
                    "unique_target_dir_count": fold.native_direction_audit.unique_target_dir_count,
                    "unique_target_dir_values_rad": list(fold.native_direction_audit.unique_target_dir_values_rad),
                    "pv_definable_on_native_field": fold.native_direction_audit.pv_definable_on_native_field,
                },
                "endpoint_pv_diagnostic": fold.endpoint_pv_diagnostic,
            }
            for fold in folds
        ],
        "input_bindings": {
            "data_dir": str(data_dir.resolve()),
            "stage2_cell_root": str(stage2_cell_root.resolve()),
            "stage2_cell_root_sha256": core.sha256_file(stage2_cell_root / ".." / "STAGE2_MATRIX_MANIFEST_v1.json")
            if (stage2_cell_root / ".." / "STAGE2_MATRIX_MANIFEST_v1.json").is_file()
            else None,
            "nwb_sha256": {name: core.sha256_file(path) for name, path in sorted(sessions.items())},
            "implementation_sha256": {
                "rt_classical_comparators.py": core.sha256_file(Path(core.__file__)),
                "run_rt_classical_comparators.py": core.sha256_file(Path(__file__)),
                "subm_v9_f0_pv_ridge.py": core.sha256_file(
                    REPO_ROOT / "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py"
                ),
            },
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    receipt_path = output_dir / "rt_classical_comparators_receipt.json"
    require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "query_identity_all_bound": aggregate["query_identity"]["all_folds_bound_to_sealed_stage2"],
        "ridge_w50_lambda1_mean_r2": aggregate["arms"]["ridge_w50_lambda1"]["mean"],
        "ridge_w50_lambda_cv_mean_r2": aggregate["arms"]["ridge_w50_lambda_cv"]["mean"],
        "t4d_reference_mean_r2": aggregate["arms"]["t4d_reference"]["mean"],
        "native_pv_verdict": aggregate["native_population_vector"]["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stage2-cell-root", type=Path, default=DEFAULT_STAGE2_CELL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_all(
        data_dir=args.data_dir.resolve(),
        stage2_cell_root=args.stage2_cell_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
