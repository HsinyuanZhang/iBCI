#!/usr/bin/env python3
"""Run population-vector applicability audits and the M2 scoring skeleton."""
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

from sua_exploration.mc_maze import population_vector_comparator as core


DEFAULT_M2_DATA_DIR = REPO_ROOT / "SPINT-main/data/000953"
DEFAULT_H1_DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/population_vector_comparator"
DEFAULT_M2_REFERENCE_SPLIT_MANIFEST = (
    REPO_ROOT
    / "sua_exploration/p1_redo_anchored_20260809/m2_scratch/artifacts"
    / "p1_redo_anchor_m2_t4_full_cpu_f1_s42_20260809_161655/split_manifest.json"
)
CORE_MODULE_PATH = Path(core.__file__)


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


def receipt_path_for(dataset: str, output_dir: Path, *, audit_only: bool) -> Path:
    suffix = "audit" if audit_only else "full"
    return output_dir / f"population_vector_comparator_{dataset}_{suffix}_receipt.json"


def build_receipt(
    *,
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    audit_only: bool,
    reference_manifest_path: Path | None,
) -> dict[str, Any]:
    applicability = core.audit_dataset_applicability(dataset, data_dir)
    results: dict[str, Any] | None = None
    query_identity_hashes: dict[str, str] | None = None
    if not audit_only:
        if dataset == core.M2_DATASET:
            require(reference_manifest_path is not None, "M2 scoring requires --reference-split-manifest")
            results = core.evaluate_m2_population_vector(
                data_dir=data_dir,
                reference_manifest_path=reference_manifest_path,
            )
            query_identity_hashes = dict(results["query_identity_hashes"])
        elif dataset == core.H1_DATASET:
            core.refuse_h1_scoring_if_inapplicable(applicability)
            raise RuntimeError("H1 scoring arm is not implemented because applicability is not PV_DEFINABLE")
        else:
            raise RuntimeError(f"unsupported dataset {dataset!r}")

    input_paths = applicability.get("input_paths", {})
    input_sha256 = {name: core.sha256_file(Path(path)) for name, path in sorted(input_paths.items())}
    implementation_paths = {
        "population_vector_comparator.py": CORE_MODULE_PATH,
        "run_population_vector_comparator.py": Path(__file__),
        "subm_v9_f0_pv_ridge.py": REPO_ROOT / "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
        "native_m2_m24_ridge_w50.py": REPO_ROOT / "sua_exploration/mc_maze/native_m2_m24_ridge_w50.py",
    }
    receipt = {
        "schema": "population_vector_comparator_v1",
        "status": "AUDIT_ONLY_CPU" if audit_only else "COMPLETED_CPU_ONLY",
        "date": time.strftime("%Y-%m-%d"),
        "dataset": dataset,
        "audit_only": bool(audit_only),
        "scope": {
            "cuda_used": False,
            "seed": core.SEED,
            "m2_calibration_trials": core.M2_CALIBRATION_TRIALS,
            "m2_window_bins": core.M2_WINDOW_BINS,
            "metric_when_scored": "torchmetrics151_variance_weighted_r2",
        },
        "applicability": applicability,
        "results": results,
        "query_identity_hashes": query_identity_hashes,
        "input_bindings": {
            "data_dir": str(data_dir.resolve()),
            "nwb_sha256": input_sha256,
            "reference_split_manifest": (
                {
                    "path": str(reference_manifest_path.resolve()),
                    "sha256": core.sha256_file(reference_manifest_path),
                }
                if reference_manifest_path is not None
                else None
            ),
            "implementation_sha256": {
                name: core.sha256_file(path) for name, path in sorted(implementation_paths.items())
            },
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    receipt_path = receipt_path_for(dataset, output_dir, audit_only=audit_only)
    require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "dataset": dataset,
        "audit_only": audit_only,
        "verdict": applicability["verdict"],
        "sessions_with_nondegenerate_native_field": applicability["sessions_with_nondegenerate_native_field"],
        "sessions_total": applicability["sessions_total"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "mean_r2_variance_weighted": None if results is None else results["aggregate"]["mean_r2_variance_weighted"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(core.M2_DATASET, core.H1_DATASET), required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reference-split-manifest", type=Path, default=DEFAULT_M2_REFERENCE_SPLIT_MANIFEST)
    args = parser.parse_args()

    default_data = DEFAULT_M2_DATA_DIR if args.dataset == core.M2_DATASET else DEFAULT_H1_DATA_DIR
    data_dir = (args.data_dir or default_data).resolve()
    output_dir = args.output_dir.resolve()
    reference_manifest = None if args.audit_only else args.reference_split_manifest.resolve()
    if args.audit_only:
        reference_manifest = None
    elif args.dataset == core.M2_DATASET:
        require(reference_manifest is not None and reference_manifest.is_file(), f"missing reference manifest: {reference_manifest}")

    summary = build_receipt(
        dataset=args.dataset,
        data_dir=data_dir,
        output_dir=output_dir,
        audit_only=args.audit_only,
        reference_manifest_path=reference_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
