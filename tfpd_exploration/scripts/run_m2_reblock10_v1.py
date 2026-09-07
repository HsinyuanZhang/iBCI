#!/usr/bin/env python3
"""Dry by default; reserve the re-block attempt or execute the cell.

CPU-only: launch with CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE=1,
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for item in (REPO_ROOT, TFPD_ROOT, TFPD_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.m2_reblock10_v1 import plan  # noqa: E402

OWNED_RELATIVE = plan.OWNED_PATHS
PREDECESSOR_RELATIVE = plan.PREDECESSOR_RELATIVE
TESTS_RELATIVE = "tfpd_exploration/tests/test_m2_reblock10_v1.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_attempt() -> dict[str, object]:
    root = plan.result_root(REPO_ROOT)
    target = root / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "M2_REBLOCK10_V1",
        "authority": (
            "operator work order 2026-09-02: the M2 re-blocking cell -- shrink "
            "the labeled calibration pool from the first-30 positions to the "
            "FIRST-10 positions (frozen greedy D-opt support within the "
            "finite-angle candidates of those 10 positions, plus the KALL "
            "row), treat everything after position 10 as the query stream "
            "(online evidence trials); CPU-only (CUDA_VISIBLE_DEVICES='', "
            "torch threads 4, 0 workers); every comparator is re-scored on "
            "the same post10 windows, sealed post30 values never mixed in"
        ),
        "inference_only": True,
        "pre_registration": {
            "question": (
                "does the extended query stream (positions 10+ as online "
                "completed-trial evidence) pay for the smaller first-10 "
                "selection pool"
            ),
            "reblock_law": dict(plan.REBLOCK_LAW),
            "cells": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.CELLS.items()
            },
            "surface": dict(plan.SURFACE_LAW),
            "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
            "anchors": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
            "gates": dict(plan.GATES),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "cpu_binding_plan": {
            **plan.ENVIRONMENT_LAW,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "one_launch_one_process": True,
        },
        "tests": {
            "path": TESTS_RELATIVE,
            "sha256": _sha256_file(REPO_ROOT / TESTS_RELATIVE),
            "required_green_before_execute": True,
            "scope": "no-data no-CUDA (first-10 candidate/K4/KALL support "
                     "laws, post10 window partition incl. sealed post30 "
                     "parity, census counting, paired contrast breadth, gate "
                     "boundaries at 1e-12, k-row binding laws, plan bindings)",
        },
        "owned_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in OWNED_RELATIVE},
        "predecessor_sha256s": {
            item: _sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  attempt.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"status": payload["status"], "attempt": str(target), "attempt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry", "attempt", "execute"), default="dry")
    parser.add_argument("--batch-size", type=int, default=plan.BATCH_SIZE)
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": f"{plan.SCHEMA}_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
            "cells": list(plan.CELLS),
            "static_surfaces": list(plan.SURFACES_STATIC),
            "cdm_surfaces": list(plan.SURFACES_CDM),
            "device": "cpu",
            "official_contract_claim": False,
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from src.m2_reblock10_v1 import physical

    result = physical.execute(REPO_ROOT, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
