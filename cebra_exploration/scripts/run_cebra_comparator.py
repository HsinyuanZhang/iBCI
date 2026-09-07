#!/usr/bin/env python3
"""CPU-only runner for the CEBRA adaptation comparator.

Reviewer path: ``--dry-run`` first.  ``--positive-control`` is a required gate
before any arm is interpreted.  Real scoring is refused unless explicitly
authorised.  This session does not authorise it.
"""
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
os.environ["PYTHONNOUSERSITE"] = "1"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VENDOR = REPO_ROOT / "cebra_exploration/third_party/cebra"
SRC = REPO_ROOT / "cebra_exploration/src"
for path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main", SRC, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

import cebra_comparator as core


DEFAULT_SUBJECT_M_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-M"
DEFAULT_RT_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_H1_DATA = REPO_ROOT / "SPINT-main/data/000954"
DEFAULT_M2_DATA = REPO_ROOT / "SPINT-main/data/000953"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cebra_exploration/results/cebra_comparator"


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


def environment_fingerprint() -> dict[str, Any]:
    cebra_version = None
    torch_version = None
    try:
        import cebra

        cebra_version = getattr(cebra, "__version__", None)
    except Exception:
        pass
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        pass
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonnousersite": os.environ.get("PYTHONNOUSERSITE"),
        "cebra": cebra_version,
        "torch": torch_version,
        "device_policy": "cpu_only",
    }


def implementation_sha256() -> dict[str, str]:
    paths = {
        "cebra_comparator.py": Path(core.__file__),
        "run_cebra_comparator.py": Path(__file__),
        "TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md": REPO_ROOT / core.PROTOCOL_PATH,
        "subm_v9_f0_pv_ridge.py": REPO_ROOT / "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
        "h1_sparse_event_endpoint.py": REPO_ROOT / "sua_exploration/mc_maze/h1_sparse_event_endpoint.py",
        "native_m2_m24_ridge_w50.py": REPO_ROOT / "sua_exploration/mc_maze/native_m2_m24_ridge_w50.py",
        "cebra_sklearn.py": VENDOR / "cebra/integrations/sklearn/cebra.py",
        "CEBRA_PROVENANCE.txt": REPO_ROOT / "cebra_exploration/third_party/CEBRA_PROVENANCE.txt",
        "rt_classical_comparators.py": REPO_ROOT / "sua_exploration/mc_maze/rt_classical_comparators.py",
    }
    return {name: core.sha256_file(path) for name, path in sorted(paths.items()) if path.is_file()}


def receipt_path_for(kind: str, output_dir: Path, dataset: str | None = None, view: str | None = None) -> Path:
    if kind == "dry_run":
        return output_dir / "cebra_comparator_dry_run_receipt.json"
    if dataset == "subject_m":
        require(view is not None, "subject-M requires --view")
        return output_dir / f"cebra_comparator_subject_m_{view}_{kind}_receipt.json"
    return output_dir / f"cebra_comparator_{dataset}_{kind}_receipt.json"


def run_dry_run(*, output_dir: Path | None, write_receipt: bool) -> dict[str, Any]:
    start = time.monotonic()
    payload = core.dry_run_all(
        repo_root=REPO_ROOT,
        data_dir_subject_m=DEFAULT_SUBJECT_M_DATA if DEFAULT_SUBJECT_M_DATA.exists() else None,
        data_dir_rt=DEFAULT_RT_DATA if DEFAULT_RT_DATA.exists() else None,
        data_dir_h1=DEFAULT_H1_DATA if DEFAULT_H1_DATA.exists() else None,
        data_dir_m2=DEFAULT_M2_DATA if DEFAULT_M2_DATA.exists() else None,
    )
    receipt = {
        "schema": "cebra_comparator_dry_run_v1",
        "status": "DRY_RUN_CPU",
        "date": time.strftime("%Y-%m-%d"),
        "protocol_date": core.PROTOCOL_DATE,
        "audit_only": True,
        "scoring_arm_executed": False,
        "scope": {
            "cuda_used": False,
            "device": core.DEFAULT_DEVICE,
            "datasets_primary": list(core.PRIMARY_DATASETS),
            "datasets_bound": list(core.BOUND_DATASETS),
            "seed_material": core.SEED_MATERIAL,
            "seed": core.seed_from_material(),
        },
        "dry_run": payload,
        "input_bindings": {
            "implementation_binding": dict(core.IMPLEMENTATION_BINDING),
            "implementation_sha256": implementation_sha256(),
        },
        "environment": environment_fingerprint(),
        "elapsed_seconds": time.monotonic() - start,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    summary = {
        "status": receipt["status"],
        "scoring_arm_executed": False,
        "device": core.DEFAULT_DEVICE,
        "part_a_structural_priors": {
            name: {
                "verdicts": item["part_a_structural_prior"]["verdicts"],
                "buildable_arms": item["part_a_structural_prior"]["buildable_arms"],
                "path_status": item.get("path_status"),
            }
            for name, item in payload["adapters"].items()
        },
        "h1_m2_decision": payload["h1_m2_decision"],
        "primary_arm": payload["primary_arm"],
        "negative_control_arm": payload["negative_control_arm"],
        "integrity_gate": "HOOK_WIRED_NOT_EXECUTED",
        "positive_control_gate": "not run during dry-run; required before any arm is interpreted",
    }
    if write_receipt:
        require(output_dir is not None, "--output-dir required to write a receipt")
        path = receipt_path_for("dry_run", output_dir)
        require(not path.exists(), f"refusing to overwrite {path}")
        receipt_sha = write_immutable(path, core.canonical_json_bytes(receipt))
        summary["receipt_path"] = str(path)
        summary["receipt_sha256"] = receipt_sha
    return summary


def run_audit(*, dataset: str, view: str | None, output_dir: Path) -> dict[str, Any]:
    start = time.monotonic()
    results = core.audit_from_structural_prior(dataset, view=view)
    receipt = {
        "schema": "cebra_comparator_audit_v1",
        "status": "AUDIT_STRUCTURAL_PRIOR_CPU",
        "date": time.strftime("%Y-%m-%d"),
        "protocol_date": core.PROTOCOL_DATE,
        "dataset": dataset,
        "view": view if dataset == "subject_m" else dataset,
        "audit_only": True,
        "scoring_arm_executed": False,
        "real_nwb_opened": False,
        "note": (
            "This audit uses structural priors (FA Part A channel-count ranges, "
            "RT degenerate direction, H1/M2 fixed N). Real NWB Part A is implemented "
            "as adapters for discovery and is unrun in this session."
        ),
        "scope": {
            "cuda_used": False,
            "device": core.DEFAULT_DEVICE,
            "seed_material": core.SEED_MATERIAL,
            "seed": core.seed_from_material(),
        },
        "part_a": results,
        "input_bindings": {
            "implementation_binding": dict(core.IMPLEMENTATION_BINDING),
            "implementation_sha256": implementation_sha256(),
        },
        "environment": environment_fingerprint(),
        "elapsed_seconds": time.monotonic() - start,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    path = receipt_path_for("audit", output_dir, dataset=dataset, view=view)
    require(not path.exists(), f"refusing to overwrite {path}")
    receipt_sha = write_immutable(path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha,
        "dataset": dataset,
        "view": receipt["view"],
        "verdicts": results["verdicts"],
        "buildable_arms": results["buildable_arms"],
        "integrity_gate": results["integrity_gate"]["status"],
        "scoring_arm_executed": False,
        "real_nwb_opened": False,
    }


def run_positive_control(*, output_dir: Path | None, write_receipt: bool) -> dict[str, Any]:
    start = time.monotonic()
    gate = core.run_positive_control_gate()
    receipt = {
        "schema": "cebra_comparator_positive_control_v1",
        "status": gate["status"],
        "date": time.strftime("%Y-%m-%d"),
        "protocol_date": core.PROTOCOL_DATE,
        "scoring_arm_executed": False,
        "real_nwb_opened": False,
        "primary_arm": core.PRIMARY_ARM,
        "negative_control_arm": core.NEGATIVE_CONTROL_ARM,
        "arm_bias": dict(core.ARM_BIAS),
        "gate": gate,
        "input_bindings": {
            "implementation_binding": dict(core.IMPLEMENTATION_BINDING),
            "implementation_sha256": implementation_sha256(),
        },
        "environment": environment_fingerprint(),
        "elapsed_seconds": time.monotonic() - start,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    summary = {
        "status": gate["status"],
        "scoring_arm_executed": False,
        "primary_arm": core.PRIMARY_ARM,
        "checks": gate["checks"],
        "arms": {
            arm: {
                "status": item["status"],
                "source_r2": item["source_r2"],
                "target_r2": item["target_r2"],
                "cost": item["cost"],
            }
            for arm, item in gate["arms"].items()
        },
        "min_target_r2": gate["min_target_r2"],
        "unaligned_max_target_r2": gate["unaligned_max_target_r2"],
    }
    if write_receipt:
        require(output_dir is not None, "--output-dir required to write a receipt")
        path = output_dir / "cebra_comparator_positive_control_receipt.json"
        require(not path.exists(), f"refusing to overwrite {path}")
        receipt_sha = write_immutable(path, core.canonical_json_bytes(receipt))
        summary["receipt_path"] = str(path)
        summary["receipt_sha256"] = receipt_sha
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CEBRA adaptation comparator (CPU)")
    parser.add_argument("--dataset", choices=core.BOUND_DATASETS, default=None)
    parser.add_argument("--view", choices=core.VIEWS, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--authorise-scoring",
        action="store_true",
        help="Refused in this session. Coordinating agent only.",
    )
    args = parser.parse_args()
    if args.authorise_scoring:
        raise SystemExit(
            "scoring is refused in this runner. A later authorised run must pass "
            "run_positive_control_gate and the sealed-reference integrity gate before "
            "any arm is interpreted. Primary arm is cebra_joint_behavior; "
            "cebra_adapt_unaligned is a declared negative control, never CEBRA's best."
        )
    if args.positive_control:
        summary = run_positive_control(
            output_dir=args.output_dir.resolve() if args.write_receipt else None,
            write_receipt=args.write_receipt,
        )
    elif args.dry_run or args.dataset is None:
        summary = run_dry_run(output_dir=args.output_dir.resolve(), write_receipt=args.write_receipt)
    elif args.audit_only:
        require(args.dataset is not None, "--audit-only requires --dataset")
        summary = run_audit(dataset=args.dataset, view=args.view, output_dir=args.output_dir.resolve())
    else:
        raise SystemExit("refusing real-data scoring; pass --dry-run, --audit-only, or --positive-control")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
