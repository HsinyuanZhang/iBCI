#!/usr/bin/env python3
"""CPU-only fail-closed preflight for A2 matched 2×2 correspondence matrix.

Verifies bindings exist, sealed sessions are absent from manifests, output roots are
empty, and existing partial results are not present. Never creates a run root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA = REPO_ROOT / "sua_exploration"
CONTRACT = SUA / "docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"
SCREEN_ID = "a2_matched_correspondence_v1"
RESULT_ROOT = SUA / "results" / SCREEN_ID
CHECKPOINT_ROOT = SUA / "checkpoints"
MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER = CHECKPOINT_ROOT / "teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
DATA_SUBC = SUA / "data/dandi_000688/sub-C"
DATA_SUBM = SUA / "data/dandi_000688/sub-M"
CACHE = SUA / "cache/dandi688_subc_co_v1"
TRAIN_SCRIPT = SUA / "scripts/train_variant_dandi688.py"
EVAL_SCRIPT = SUA / "scripts/eval_epoch_window_generic_dandi688.py"

EXPECTED_TEACHER_SHA = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
EXPECTED_MANIFEST_SHA = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SEEDS = (42, 43, 44)
CELLS = (
    "within_z4",
    "within_t4",
    "cross_z4",
    "cross_t4",
)
SEALED = {
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Override result root for isolated preflight checks",
    )
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()

    blockers: list[dict[str, str]] = []
    checks: dict[str, Any] = {
        "schema_version": 1,
        "screen_id": SCREEN_ID,
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT) if CONTRACT.is_file() else None,
        "gpu_authorized": False,
        "creates_run_root": False,
    }

    if not CONTRACT.is_file():
        blockers.append(fail("MISSING_CONTRACT", str(CONTRACT)))
    for path, label in (
        (MANIFEST, "manifest"),
        (TEACHER, "teacher"),
        (DATA_SUBC, "sub-C data"),
        (DATA_SUBM, "sub-M data"),
        (CACHE, "cache"),
        (TRAIN_SCRIPT, "train script"),
        (EVAL_SCRIPT, "eval script"),
    ):
        if not path.exists():
            blockers.append(fail("MISSING_BINDING", f"{label}: {path}"))

    if MANIFEST.is_file() and sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        blockers.append(fail("MANIFEST_SHA_DRIFT", EXPECTED_MANIFEST_SHA))
    if TEACHER.is_file() and sha256_file(TEACHER) != EXPECTED_TEACHER_SHA:
        blockers.append(fail("TEACHER_SHA_DRIFT", EXPECTED_TEACHER_SHA))

    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        splits = manifest.get("session_splits") or manifest
        for key in ("train", "val", "test"):
            sessions = splits.get(key) or []
            hit = [s for s in sessions if s in SEALED]
            if hit:
                blockers.append(fail("SEALED_SESSION_IN_MANIFEST", f"{key}: {hit}"))

    if result_root.exists():
        for seed in SEEDS:
            for cell in CELLS:
                result = result_root / f"{cell}_s{seed}.json"
                if result.exists():
                    blockers.append(fail("NONEMPTY_OUTPUT_ROOT", str(result)))
        aggregate = result_root / "aggregate.json"
        if aggregate.exists():
            blockers.append(fail("NONEMPTY_OUTPUT_ROOT", str(aggregate)))
        for seed in SEEDS:
            for cell in CELLS:
                run_name = f"{SCREEN_ID}_{cell}_dandi688_co_s{seed}"
                run_dir = CHECKPOINT_ROOT / run_name
                if run_dir.exists():
                    blockers.append(fail("EXISTING_RUN_DIRECTORY", str(run_dir)))

    status = "READY_FOR_SEPARATE_GPU_REVIEW" if not blockers else "STOP_A2_PREFLIGHT_BLOCKERS"
    receipt = {
        **checks,
        "status": status,
        "implementation_blockers": blockers,
        "expected_gpu_cells": len(CELLS) * len(SEEDS),
        "cells": list(CELLS),
        "seeds": list(SEEDS),
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text)
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
