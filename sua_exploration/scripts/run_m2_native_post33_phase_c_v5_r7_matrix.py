#!/usr/bin/env python3
"""Deterministic Stage-A-only r7 matrix launcher for repaired v5 workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    cell_paths,
    claim_cell,
    verify_cell_exact,
    write_failed_once,
    write_started,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r7 import (
    claim_authorization_nonce,
    validate_observed_gpu,
    validate_observed_host,
    validate_r7_portable_transfer_manifest,
    validate_r7_shard_manifest,
    verify_signed_authorization,
)


PIPELINE = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r7_cell_pipeline.py"
R7_RETIRED = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_launch_receipts_20260805/retirement/r7_prelaunch_matrix_contract_retirement.json"
ACTIVE: dict[str, Any] = {"process": None, "key": None, "owner_token": None, "cell_root": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-signature", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--independent-review", type=Path,
        help="required only for --execute; no r7 cell may start without an external review receipt",
    )
    return parser.parse_args()


def _signal(signum, frame) -> None:
    del frame
    process = ACTIVE["process"]
    if process is not None and process.poll() is None:
        process.terminate()
    key = ACTIVE["key"]
    if key is not None:
        write_failed_once(
            ACTIVE["cell_root"],
            key,
            owner_token=ACTIVE["owner_token"],
            failure_kind=f"signal_{signum}",
            return_code=128 + signum,
        )
    raise SystemExit(128 + signum)


def _pipeline_command(args: argparse.Namespace, key: CellKey, owner_token: str) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE),
        "--cell-root", str(args.cell_root.resolve()),
        "--workspace-root", str(args.workspace_root.resolve(strict=True)),
        "--data-root", str(args.data_root.resolve(strict=True)),
        "--arm", key.arm,
        "--fold", str(key.fold),
        "--seed", str(key.seed),
        "--owner-token", owner_token,
        "--authorization", str(args.authorization.resolve(strict=True)),
        "--authorization-signature", str(args.authorization_signature.resolve(strict=True)),
        "--shard-manifest", str(args.shard_manifest.resolve(strict=True)),
        "--program-receipt", str(args.program_receipt.resolve(strict=True)),
        "--portable-manifest", str(args.portable_manifest.resolve(strict=True)),
        "--cost-supplement", str(args.cost_supplement.resolve(strict=True)),
    ]


def _run_cell(args: argparse.Namespace, key: CellKey, *, host_id: str, gpu_id: str) -> None:
    owner_token = f"{host_id}-fold{key.fold}-seed{key.seed}-{key.arm}"
    claim_cell(args.cell_root, key, owner_token=owner_token)
    write_started(args.cell_root, key, owner_token=owner_token)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    validate_observed_gpu({"gpu_id": gpu_id}, observed_visible_devices=env["CUDA_VISIBLE_DEVICES"])
    command = _pipeline_command(args, key, owner_token)
    ACTIVE.update(
        {"key": key, "owner_token": owner_token, "cell_root": args.cell_root}
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=args.workspace_root.resolve(strict=True),
            env=env,
            text=True,
        )
        ACTIVE["process"] = process
        return_code = process.wait()
        ACTIVE["process"] = None
        if return_code != 0:
            write_failed_once(
                args.cell_root,
                key,
                owner_token=owner_token,
                failure_kind="child_nonzero",
                return_code=return_code,
            )
            raise RuntimeError(f"cell pipeline failed: {key} return_code={return_code}")
        # A zero exit without exact completed evidence is itself a failed cell.
        try:
            verify_cell_exact(args.cell_root, key)
        except Exception:
            write_failed_once(
                args.cell_root,
                key,
                owner_token=owner_token,
                failure_kind="zero_exit_without_exact_completion",
                return_code=0,
            )
            raise
    except BaseException:
        paths = cell_paths(args.cell_root, key)
        if not paths["completed"].exists() and not paths["failed"].exists():
            write_failed_once(
                args.cell_root,
                key,
                owner_token=owner_token,
                failure_kind="python_exception",
                return_code=None,
            )
        raise
    finally:
        ACTIVE.update({"process": None, "key": None, "owner_token": None, "cell_root": None})


def main() -> None:
    if R7_RETIRED.exists():
        raise PermissionError("r7 capability is append-only retired after CPU dry-run and must never launch")
    args = parse_args()
    if args.workspace_root.resolve(strict=True) != ROOT.resolve(strict=True):
        raise PermissionError("launcher --workspace-root differs from its source workspace")
    portable = validate_r7_portable_transfer_manifest(
        args.portable_manifest,
        workspace_root=args.workspace_root,
        data_root=args.data_root,
        cell_root=args.cell_root,
    )
    shard = validate_r7_shard_manifest(
        args.shard_manifest,
        portable_manifest_path=args.portable_manifest,
        cell_root=args.cell_root,
    )
    validate_observed_host(shard)
    pairs = [(fold, seed) for fold in shard["fold_allowlist"] for seed in shard["seed_allowlist"]]
    dry = {
        "schema": "m2_post33_phase_c_v5_r7_shard_dry_run_v1",
        "host_id": shard["host_id"],
        "gpu_id": shard["gpu_id"],
        "absolute_cell_root": shard["absolute_cell_root"],
        "pair_count": len(pairs),
        "ordered_cells": [
            {"fold": fold, "seed": seed, "arms_in_order": ["spint", "t4"]}
            for fold, seed in pairs
        ],
        "paired_same_host": True,
        "t4_only_launch_possible": False,
        "portable_manifest_sha256": shard["portable_transfer_manifest_sha256"],
        "execution_started": False,
    }
    if not args.execute:
        print(json.dumps(dry, sort_keys=True, indent=2))
        return
    if args.independent_review is None:
        raise PermissionError("r7 --execute is blocked until an independent review receipt is supplied")
    review = json.loads(args.independent_review.resolve(strict=True).read_text(encoding="utf-8"))
    expected_review = {
        "schema": "m2_post33_phase_c_v5_r7_independent_launch_review_v1",
        "status": "APPROVED_FOR_STAGE_A_LAUNCH",
        "program": {
            "canonical_path": str(args.program_receipt.resolve(strict=True)),
        },
        "shard": {
            "canonical_path": str(args.shard_manifest.resolve(strict=True)),
        },
    }
    if not isinstance(review, dict) or any(review.get(key) != value for key, value in expected_review.items()):
        raise PermissionError("r7 independent review receipt does not approve this exact program/shard")
    if args.authorization is None or args.authorization_signature is None:
        raise ValueError("--execute requires root-issued authorization and detached signature")
    authorization = verify_signed_authorization(
        args.authorization, args.authorization_signature,
        phase_c_program_receipt_path=args.program_receipt,
        portable_manifest_path=args.portable_manifest,
        shard_manifest_path=args.shard_manifest,
        cost_supplement_path=args.cost_supplement,
        cell_root=args.cell_root,
    )
    claim_authorization_nonce(
        root=args.cell_root,
        authorization=authorization,
        authorization_path=args.authorization,
        signature_path=args.authorization_signature,
        shard_manifest_path=args.shard_manifest,
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal)
    for fold, seed in pairs:
        _run_cell(
            args,
            CellKey(PROTOCOL_ID, "spint", fold, seed),
            host_id=shard["host_id"],
            gpu_id=shard["gpu_id"],
        )
        # Same process, host, GPU allowlist, and absolute root; T4 cannot be
        # constructed until the just-finished paired SPINT receipt revalidates.
        _run_cell(
            args,
            CellKey(PROTOCOL_ID, "t4", fold, seed),
            host_id=shard["host_id"],
            gpu_id=shard["gpu_id"],
        )


if __name__ == "__main__":
    main()
