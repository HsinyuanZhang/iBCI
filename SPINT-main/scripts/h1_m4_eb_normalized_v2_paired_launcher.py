#!/usr/bin/env python3
"""Prepare-only-by-default paired launcher for normalized V2.

``--execute`` is intentionally blocked unless an immutable root-review marker
binds the exact CPU-preflight receipt.  Running without ``--execute`` merely
prints the two future commands and never touches CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    V2_RECEIPT_SCHEMA,
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
)


PREFLIGHT_STATUS = "PASS_H1_M4_EB_NORMALIZED_V2_REAL_DATA_CPU_PREFLIGHT_NONLAUNCH"
REVIEW_STATUS = "APPROVED_H1_M4_EB_NORMALIZED_V2_GPU_LAUNCH_AFTER_CODE_REVIEW"


def _immutable(path: Path) -> None:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise NormalizedV2ContractError(f"required V2 gate input must be immutable mode 0444: {path}")


def _review_marker(path: str | Path, preflight_sha256: str) -> dict[str, Any]:
    marker = Path(path).resolve()
    _immutable(marker)
    value = assert_immutable_receipt(marker, REVIEW_STATUS)
    if value.get("preflight_receipt_sha256") != preflight_sha256:
        raise NormalizedV2ContractError("V2 root review marker does not bind this exact preflight receipt")
    return {"path": str(marker), "sha256": sha256_file(marker), "status": value["status"]}


def _verify_source_closure(project_root: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    """Bind execution to every source/config/test byte hashed by preflight."""

    root = project_root.resolve()
    expected = preflight.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise NormalizedV2ContractError("V2 preflight lacks a nonempty source_sha256 closure")
    verified: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
            raise NormalizedV2ContractError("V2 preflight source_sha256 entry is malformed")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise NormalizedV2ContractError(f"V2 source closure path escapes project root: {relative}")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise NormalizedV2ContractError(f"V2 source closure path escapes project root: {relative}") from exc
        if not candidate.is_file():
            raise NormalizedV2ContractError(f"V2 source closure file is missing: {relative}")
        observed = sha256_file(candidate)
        if observed != digest:
            raise NormalizedV2ContractError(
                f"V2 source closure SHA-256 drift at {relative}: expected {digest}, observed {observed}"
            )
        verified[relative] = observed
    return {"verified": True, "file_count": len(verified), "sha256": verified}


def build_commands(*, project_root: Path, output_root: Path, python_bin: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for arm in ("base", "joint"):
        run_dir = output_root / arm
        arm_cache = output_root / "shared_source_cache" / arm
        result[arm] = [
            str(python_bin),
            str(project_root / "src/train.py"),
            f"experiment=h1_m4_eb_normalized_v2_{arm}",
            f"hydra.run.dir={run_dir}",
            f"paths.root_dir={project_root}",
            f"paths.work_dir={project_root}",
            f"paths.data_dir={project_root / 'data'}",
            f"pilot.shared_cache_dir={arm_cache}",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            "trainer.max_epochs=50",
            "trainer.min_epochs=50",
            "model.optimizer.lr=5e-5",
            "seed=42",
            "test=false",
            "ckpt_path=null",
        ]
    return result


def prepare_or_launch(
    *,
    project_root: str | Path,
    output_root: str | Path,
    python_bin: str | Path,
    preflight_receipt: str | Path,
    root_review_marker: str | Path | None = None,
    execute: bool = False,
    base_gpu: str = "0",
    joint_gpu: str = "1",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    python = Path(python_bin).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise NormalizedV2ContractError(f"Python executable is unavailable: {python}")
    preflight_path = Path(preflight_receipt).resolve()
    preflight = assert_immutable_receipt(preflight_path, PREFLIGHT_STATUS)
    if preflight.get("schema") != V2_RECEIPT_SCHEMA or preflight.get("launch", {}).get("launch_authorized") is not False:
        raise NormalizedV2ContractError("expected a nonlaunch normalized V2 CPU preflight receipt")
    preflight_sha = sha256_file(preflight_path)
    source_closure = _verify_source_closure(root, preflight)
    commands = build_commands(project_root=root, output_root=output, python_bin=python)
    result: dict[str, Any] = {
        "schema": "h1_m4_eb_normalized_v2_paired_launcher_plan_v1",
        "mode": "execute" if execute else "prepare_print_only",
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha, "status": preflight["status"]},
        "source_closure": source_closure,
        "root_review": None,
        "output_root": str(output),
        "paired_cache_dirs": {arm: str(output / "shared_source_cache" / arm) for arm in commands},
        "run_dirs": {arm: str(output / arm) for arm in commands},
        "commands": {arm: shlex.join(command) for arm, command in commands.items()},
        "gpu_assignment": {"base": base_gpu, "joint": joint_gpu},
        "fixed_contract": {
            "seed": 42,
            "optimizer": "Adam",
            "lr": 5.0e-5,
            "weight_decay": 0.0,
            "batch_size": 32,
            "epochs": 50,
            "normalizer_formula": "s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)",
            "same_source_manifest_and_normalizer_required": True,
        },
        "launched": False,
    }
    if not execute:
        return result
    if root_review_marker is None:
        raise NormalizedV2ContractError("V2 GPU execution refuses to proceed without root review marker")
    result["root_review"] = _review_marker(root_review_marker, preflight_sha)
    if output.exists() and any((output / arm).exists() for arm in commands):
        raise FileExistsError("V2 paired launcher refuses existing base/joint run directories")
    output.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen] = {}
    logs: dict[str, Any] = {}
    try:
        for arm, gpu in (("base", base_gpu), ("joint", joint_gpu)):
            run_dir = output / arm
            run_dir.mkdir(parents=True, exist_ok=False)
            handle = (run_dir / "launcher.stdout.log").open("xb")
            logs[arm] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            processes[arm] = subprocess.Popen(
                commands[arm], cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
        result["launched"] = True
        result["pids"] = {arm: process.pid for arm, process in processes.items()}
        result["return_codes"] = {arm: process.wait() for arm, process in processes.items()}
        if any(code != 0 for code in result["return_codes"].values()):
            raise RuntimeError(f"normalized V2 paired run failed: {result['return_codes']}")
        return result
    finally:
        for handle in logs.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--root-review-marker", type=Path)
    parser.add_argument("--base-gpu", default="0")
    parser.add_argument("--joint-gpu", default="1")
    parser.add_argument("--execute", action="store_true", help="requires immutable root review marker")
    args = parser.parse_args()
    result = prepare_or_launch(
        project_root=args.project_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
        preflight_receipt=args.preflight_receipt,
        root_review_marker=args.root_review_marker,
        execute=args.execute,
        base_gpu=args.base_gpu,
        joint_gpu=args.joint_gpu,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
