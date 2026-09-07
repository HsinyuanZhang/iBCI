#!/usr/bin/env python3
"""Prepare/launch the two matched H1 M=4 EB pilot arms after all gates."""
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

from src.h1_m4_eb_pilot_contract import assert_immutable_receipt, sha256_file
from scripts.h1_m4_eb_fold0_pilot_preflight import PREFLIGHT_STATUS


REVIEW_STATUS = "APPROVED_H1_M4_EB_FOLD0_GPU_LAUNCH_AFTER_CODE_REVIEW"


def _immutable(path: Path) -> None:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ValueError(f"required gate input must be immutable mode 0444: {path}")


def _baseline_marker(path: str | Path, *, terminal: bool) -> dict[str, Any]:
    marker = Path(path).resolve()
    _immutable(marker)
    text = marker.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = {"text": text}
    upper = text.upper()
    if not any(token in upper for token in ("PASS", "COMPLETE", "SEALED")):
        raise ValueError(f"original H1 baseline marker lacks terminal/sealed status: {marker}")
    if terminal:
        epoch = value.get("checkpoint_epoch_zero_based")
        completed = value.get("epochs_completed")
        if epoch != 49 or completed != 50:
            raise ValueError("original H1 baseline terminal receipt must bind epoch 49 / 50 completed")
    return {"path": str(marker), "sha256": sha256_file(marker)}


def _review_marker(path: str | Path, preflight_sha256: str) -> dict[str, Any]:
    marker = Path(path).resolve()
    value = assert_immutable_receipt(marker, REVIEW_STATUS)
    if value.get("preflight_receipt_sha256") != preflight_sha256:
        raise ValueError("root review marker does not bind this exact preflight receipt")
    return {"path": str(marker), "sha256": sha256_file(marker), "status": value["status"]}


def build_commands(
    *,
    project_root: Path,
    output_root: Path,
    python_bin: Path,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for arm in ("base", "joint"):
        run_dir = output_root / arm
        # Separate physical caches avoid a concurrent-create race.  Their
        # manifests/schedules are content-addressed and must hash identically.
        arm_cache = output_root / "shared_schedule_cache" / arm
        result[arm] = [
            str(python_bin),
            str(project_root / "src/train.py"),
            f"experiment=h1_m4_eb_fold0_exploratory_pilot_{arm}",
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
    baseline_terminal_marker: str | Path,
    baseline_seal_marker: str | Path,
    root_review_marker: str | Path | None,
    execute: bool,
    base_gpu: str = "0",
    joint_gpu: str = "1",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    python = Path(python_bin).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python executable is unavailable: {python}")
    preflight_path = Path(preflight_receipt).resolve()
    preflight = assert_immutable_receipt(preflight_path, PREFLIGHT_STATUS)
    if preflight.get("launch", {}).get("launch_authorized") is not False:
        raise ValueError("expected a nonlaunch engineering preflight receipt")
    preflight_sha = sha256_file(preflight_path)
    terminal = _baseline_marker(baseline_terminal_marker, terminal=True)
    seal = _baseline_marker(baseline_seal_marker, terminal=False)
    commands = build_commands(project_root=root, output_root=output, python_bin=python)
    result: dict[str, Any] = {
        "schema": "h1_m4_eb_fold0_paired_launcher_plan_v1",
        "mode": "execute" if execute else "prepare_print_only",
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha, "status": preflight["status"]},
        "original_h1_baseline_terminal": terminal,
        "original_h1_baseline_seal": seal,
        "root_review": None,
        "output_root": str(output),
        "paired_cache_dirs": {
            arm: str(output / "shared_schedule_cache" / arm) for arm in commands
        },
        "run_dirs": {arm: str(output / arm) for arm in commands},
        "commands": {arm: shlex.join(command) for arm, command in commands.items()},
        "gpu_assignment": {"base": base_gpu, "joint": joint_gpu},
        "fixed_contract": {
            "seed": 42,
            "optimizer": "Adam",
            "lr": 5.0e-5,
            "batch_size": 32,
            "epochs": 50,
            "selection": None,
            "same_source_manifest_and_schedule_required": True,
        },
        "launched": False,
    }
    if not execute:
        return result
    if root_review_marker is None:
        raise ValueError("GPU execution refuses to proceed without an immutable root review marker")
    result["root_review"] = _review_marker(root_review_marker, preflight_sha)
    if output.exists() and any((output / arm).exists() for arm in commands):
        raise FileExistsError("paired launcher refuses existing base/joint run directories")
    output.mkdir(parents=True, exist_ok=True)
    processes = {}
    logs = {}
    try:
        for arm, gpu in (("base", base_gpu), ("joint", joint_gpu)):
            run_dir = output / arm
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path = run_dir / "launcher.stdout.log"
            handle = log_path.open("xb")
            logs[arm] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            processes[arm] = subprocess.Popen(
                commands[arm],
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        result["launched"] = True
        result["pids"] = {arm: process.pid for arm, process in processes.items()}
        return_codes = {arm: process.wait() for arm, process in processes.items()}
        result["return_codes"] = return_codes
        if any(code != 0 for code in return_codes.values()):
            raise RuntimeError(f"paired H1 M=4 EB run failed: {return_codes}")
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
    parser.add_argument("--baseline-terminal-marker", type=Path, required=True)
    parser.add_argument("--baseline-seal-marker", type=Path, required=True)
    parser.add_argument("--root-review-marker", type=Path)
    parser.add_argument("--base-gpu", default="0")
    parser.add_argument("--joint-gpu", default="1")
    parser.add_argument("--execute", action="store_true", help="requires all three immutable gates; omitted means print only")
    args = parser.parse_args()
    result = prepare_or_launch(
        project_root=args.project_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
        preflight_receipt=args.preflight_receipt,
        baseline_terminal_marker=args.baseline_terminal_marker,
        baseline_seal_marker=args.baseline_seal_marker,
        root_review_marker=args.root_review_marker,
        execute=args.execute,
        base_gpu=args.base_gpu,
        joint_gpu=args.joint_gpu,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
